"""
El motor completo: de un archivo de TC (y, si se quiere, su etiqueta) a todo lo que muestra la app.

    leer → inspeccionar → preparar (LPS, 1,5 mm, 256×256) → red corte a corte (pila 2.5D) → fragmentos 2D
         → enlace 3D → separación en mm → [evaluación frente a la etiqueta] → visualizadores 1, 2 y 3

`Motor.analizar()` devuelve un `Resultado`: un resumen en JSON y los volúmenes para el navegador, ya comprimidos.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from . import pipeline as P
from . import preprocesado as pre
from . import visual as V
from .red import cargar_modelo

ETAPAS = [("lectura", "Leyendo el archivo", 0.04), ("preparacion", "Orientando, remuestreando y recortando", 0.08),
          ("inferencia", "Pasando cada corte por la red", 0.55), ("enlace", "Enlazando fragmentos en 3D y midiendo", 0.08),
          ("evaluacion", "Comparando con la etiqueta", 0.05), ("mip", "Calculando la MIP del volumen crudo", 0.10),
          ("mallas", "Construyendo las mallas 3D", 0.07), ("empaquetado", "Preparando la vista", 0.03)]


@dataclass
class Resultado:
    resumen: dict
    binarios: dict = field(default_factory=dict)      # nombre → bytes gzip


def _gz(datos: bytes) -> bytes:
    return gzip.compress(datos, compresslevel=5)


class Motor:
    def __init__(self, raiz: Path):
        self.raiz = Path(raiz)
        self.config = P.Config.leer(self.raiz / "checkpoints" / "configuracion_inferencia.json")
        self.ruta_modelo = self.raiz / "checkpoints" / self.config.modelo
        self._modelos: dict[str, tuple] = {}
        self._cerrojo = threading.Lock()
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True   # cifras reproducibles

    @staticmethod
    def dispositivos() -> list[str]:
        return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

    def modelo(self, dispositivo: str):
        with self._cerrojo:
            if dispositivo not in self._modelos:
                self._modelos[dispositivo] = cargar_modelo(self.ruta_modelo, torch.device(dispositivo))
            return self._modelos[dispositivo]

    def analizar(self, ruta_imagen: Path, nombre: str, ruta_etiqueta: Path | None = None, nombre_etiqueta: str = "",
                 dispositivo: str = "cuda", progreso=None) -> Resultado:
        if dispositivo not in self.dispositivos():
            dispositivo = "cpu"
        tiempos, avisos = {}, []
        acumulado = {"v": 0.0}

        def etapa(clave):
            i = [e[0] for e in ETAPAS].index(clave)
            acumulado["v"] = sum(e[2] for e in ETAPAS[:i])
            if progreso:
                progreso(clave, ETAPAS[i][1], acumulado["v"])
            return time.perf_counter()

        def avance(clave, fraccion):
            peso = dict((e[0], e[2]) for e in ETAPAS)[clave]
            if progreso:
                progreso(clave, dict((e[0], e[1]) for e in ETAPAS)[clave], acumulado["v"] + peso * fraccion)

        t_total = time.perf_counter()
        # 1 · Leer y comprobar
        t0 = etapa("lectura")
        img = pre.leer(ruta_imagen)
        ficha = pre.inspeccionar(img, nombre)
        img = pre.aplicar_correccion(img, ficha)
        lab = None
        if ruta_etiqueta is not None:
            try:
                lab = pre.leer(ruta_etiqueta)
            except pre.ArchivoNoValido as e:
                avisos.append(f"La etiqueta no se pudo usar: {e}")
        tiempos["lectura"] = time.perf_counter() - t0

        # 2 · Preparar: los pasos de build_cache.py
        t0 = etapa("preparacion")
        prep = pre.preparar_imagen(img)
        real = None
        if lab is not None:
            try:
                real = pre.preparar_etiqueta(lab, img, prep.centro)
            except pre.ArchivoNoValido as e:
                avisos.append(f"La etiqueta no se pudo usar: {e}")
        del img, lab
        # Dónde cae el recorte en el espacio físico: el volumen isotrópico conserva el origen y la dirección del
        # orientado, y el recorte empieza en el índice (x0, y0, 0) de esa rejilla
        origen = np.array(prep.img_orientada.GetOrigin())
        direccion = np.array(prep.img_orientada.GetDirection()).reshape(3, 3)
        y0, x0 = prep.centro[0] - pre.CROP_PX // 2, prep.centro[1] - pre.CROP_PX // 2
        geometria = {"origen_recorte": (origen + direccion @ (np.array([x0, y0, 0]) * pre.ISO_MM)).round(4).tolist(),
                     "direccion": direccion.ravel().round(6).tolist()}
        tiempos["preparacion"] = time.perf_counter() - t0

        # 3 · La red, corte a corte
        t0 = etapa("inferencia")
        modelo, info_modelo = self.modelo(dispositivo)
        inf = P.inferir_volumen(modelo, prep.hu, self.config, progreso=lambda f: avance("inferencia", f))
        if dispositivo == "cuda":
            torch.cuda.synchronize()
        tiempos["inferencia"] = time.perf_counter() - t0
        n_cortes = len(prep.hu)

        # 4 · Fragmentos 3D y separación
        t0 = etapa("enlace")
        vol3d = P.enlazar_3d(inf["etiquetas"], *self.config.modo_enlace(), umbral_mm3=self.config.umbral_mm3)
        fragmentos = P.fragmentos_del_caso(vol3d)
        tiempos["enlace"] = time.perf_counter() - t0

        # 5 · Evaluación, si hay etiqueta
        evaluacion = None
        if real is not None:
            t0 = etapa("evaluacion")
            evaluacion = P.evaluar(inf, vol3d, fragmentos, real)
            tiempos["evaluacion"] = time.perf_counter() - t0

        # 6 · Visualizador 1: la MIP rotatoria del volumen crudo
        t0 = etapa("mip")
        fotogramas, angulos = V.rotacion_mip(prep.nativo, prep.espaciado_nativo, max_lado=256,
                                             dispositivo=torch.device(dispositivo))
        tiempos["mip"] = time.perf_counter() - t0

        # 7 · Visualizador 3: las mallas
        t0 = etapa("mallas")
        mallas_pred = V.mallas(vol3d, fragmentos)
        mallas_real = V.mallas(real, P.fragmentos_del_caso(real)) if real is not None else None
        tiempos["mallas"] = time.perf_counter() - t0

        # 8 · Empaquetar para el navegador
        t0 = etapa("empaquetado")
        binarios = {
            "mip": _gz(np.ascontiguousarray(fotogramas, dtype="<i2").tobytes()),
            "hu": _gz(V.ventana_hueso_uint8(prep.hu).tobytes()),
            "pred": _gz(vol3d.astype(np.uint8).tobytes()),
            "mallas": _gz(json.dumps({"pred": mallas_pred, "real": mallas_real}).encode()),
        }
        if real is not None:
            binarios["real"] = _gz(real.astype(np.uint8).tobytes())
        tiempos["empaquetado"] = time.perf_counter() - t0
        tiempos["total"] = time.perf_counter() - t_total

        con_hueso = [z for z in range(n_cortes) if vol3d[z].any()]
        por_hueso = {}
        for k in (1, 2, 3):
            fk = [f for f in fragmentos if f["hueso"] == k]
            if fk:
                por_hueso[P.HUESO[k]] = {"fragmentos": len(fk), "conminutos": len(fk) - 1,
                                        "desplazados": sum(f["separacion_mm"] > pre.ISO_MM + 1e-6 for f in fk),
                                        "max_separacion_mm": max(f["separacion_mm"] for f in fk)}
        resumen = {
            "nombre": nombre, "etiqueta": nombre_etiqueta if real is not None else None,
            "ficha": ficha.como_dict(), "avisos": ficha.avisos + avisos,
            "dispositivo": dispositivo, "dispositivo_nombre": (torch.cuda.get_device_name(0) if dispositivo == "cuda"
                                                               else "CPU"),
            "modelo": info_modelo, "config": {**self.config.__dict__, "desplazamientos": list(self.config.desplazamientos)},
            "forma": [n_cortes, pre.CROP_PX, pre.CROP_PX], "centro": list(prep.centro), "forma_iso": list(prep.forma_iso),
            "espaciado_mm": pre.ISO_MM, "geometria": geometria,
            "tiempos": {k: round(v, 3) for k, v in tiempos.items()},
            "latencia": {"ms_por_corte": round(1000 * tiempos["inferencia"] / n_cortes, 1),
                         "ms_red_por_corte": round(1000 * inf["segundos_red"] / n_cortes, 1), "cortes": n_cortes},
            "fragmentos": [{**f, "volumen_mm3": round(f["volumen_mm3"], 1), "separacion_mm": round(f["separacion_mm"], 2),
                            "color": V.color_fragmento(f["etiqueta"]), "nombre_hueso": P.HUESO[f["hueso"]]}
                           for f in fragmentos],
            "por_hueso": por_hueso,
            "rango_hueso": [con_hueso[0], con_hueso[-1]] if con_hueso else None,
            "cajas": [{str(k): [round(c, 1) for c in caja] + [round(inf["puntuaciones"][z][k], 3)] for k, caja in cj.items()}
                      for z, cj in enumerate(inf["cajas"])],
            "cajas_reales": ([{str(k): list(c) for k, c in P.cajas_por_region(real[z]).items()} for z in range(n_cortes)]
                             if real is not None else None),
            "presencia": [[round(p, 3) for p in fila] for fila in inf["presencia"]],
            "evaluacion": evaluacion,
            "mip": {"n": int(len(fotogramas)), "alto": int(fotogramas.shape[1]), "ancho": int(fotogramas.shape[2]),
                    "angulos": angulos, "umbral": V.UMBRAL_HU, "hu_max": V.HU_MAX},
            "colores": {str(k): c for k, c in V.COLOR.items()},
            "tamanos": {k: len(v) for k, v in binarios.items()},
        }
        if progreso:
            progreso("listo", "Listo", 1.0)
        return Resultado(resumen=resumen, binarios=binarios)


def leer_cabecera_rapida(ruta: Path) -> dict:
    """Tamaño y espaciado sin cargar el volumen (para listar los casos de ejemplo)."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(ruta))
    r.ReadImageInformation()
    return {"tamano": list(r.GetSize()[::-1]), "espaciado": [round(s, 3) for s in r.GetSpacing()[::-1]]}
