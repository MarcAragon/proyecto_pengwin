"""
Del archivo subido a la entrada del modelo.

Los pasos 2 a 4 son exactamente los de `build_cache.py` (orientar a LPS, remuestrear a 1,5 mm isotrópico y recortar
256×256 centrado en el hueso), con las mismas constantes: el notebook de desarrollo comprueba que este módulo da, bit a
bit, los mismos cortes que el caché con el que se entrenó el modelo. Lo que añade es lo que necesita un archivo
cualquiera subido por cualquiera: leer varios formatos, validar que es una TC y no una máscara, avisar de lo que se
sale de lo que vio el modelo y, si llega una etiqueta, comprobar que comparte rejilla con la imagen.

    leer()  →  inspeccionar()  →  preparar_imagen()  [→ preparar_etiqueta()]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ISO_MM = 1.5        # rejilla isotrópica de destino
CROP_PX = 256       # 256 px × 1,5 mm = 384 mm de campo
AIR_HU = -1024      # relleno fuera del campo: aire
BONE_HU = 200       # umbral para centrar el recorte en el hueso
EXTENSIONES = (".mha", ".nii.gz", ".nii", ".nrrd")
MAX_VOXELES = 600 * 1024 * 1024      # ~1,2 GB en int16: más que cualquier TC de pelvis razonable


class ArchivoNoValido(ValueError):
    """El archivo no se puede procesar; el mensaje se muestra tal cual en la app."""


@dataclass
class Ficha:
    """Lo que dice la cabecera del archivo y lo que se comprobó al leerlo."""
    nombre: str
    tamano: tuple                    # (z, y, x) vóxeles
    espaciado: tuple                 # (z, y, x) mm
    origen: tuple
    orientacion: str                 # código de tres letras de la matriz de dirección (p. ej. "LPS", "RAI")
    tipo: str
    hu_p1: float = 0.0
    hu_p99: float = 0.0
    hueso_pct: float = 0.0           # % de vóxeles por encima de 200 HU (en una muestra)
    correccion_hu: int = 0           # desplazamiento aplicado si los valores no venían en HU
    avisos: list = field(default_factory=list)

    def como_dict(self) -> dict:
        return {"nombre": self.nombre, "tamano": list(self.tamano), "espaciado": [round(s, 4) for s in self.espaciado],
                "origen": [round(o, 2) for o in self.origen], "orientacion": self.orientacion, "tipo": self.tipo,
                "hu_p1": self.hu_p1, "hu_p99": self.hu_p99, "hueso_pct": self.hueso_pct,
                "correccion_hu": self.correccion_hu, "avisos": list(self.avisos),
                "extension_mm": [round(n * s, 1) for n, s in zip(self.tamano, self.espaciado)]}


# ---------------------------------------------------------------------------------------------------------------------
# Los pasos de build_cache.py, sin cambiar nada
# ---------------------------------------------------------------------------------------------------------------------

def orientar(img: sitk.Image) -> sitk.Image:
    """Orientación canónica LPS. Solo permuta y voltea ejes: no interpola nada."""
    return sitk.DICOMOrient(img, "LPS")


def remuestrear(img: sitk.Image, es_etiqueta: bool) -> sitk.Image:
    """A ISO_MM en los tres ejes: lineal la imagen, vecino más cercano la etiqueta, relleno de aire fuera del campo."""
    espaciado = np.array(img.GetSpacing())
    tam = np.round(np.array(img.GetSize()) * espaciado / ISO_MM).astype(int)
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([ISO_MM] * 3)
    rs.SetSize([int(n) for n in tam])
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if es_etiqueta else sitk.sitkLinear)
    rs.SetDefaultPixelValue(0 if es_etiqueta else AIR_HU)
    return rs.Execute(img)


def centro_hueso(hu: np.ndarray) -> tuple[int, int]:
    """Centro (fila, columna) del hueso visto desde arriba: la mediana de los vóxeles > 200 HU. Solo mira la imagen."""
    vista = (hu > BONE_HU).any(axis=0)
    if not vista.any():
        return hu.shape[1] // 2, hu.shape[2] // 2
    filas, columnas = np.nonzero(vista)
    return int(np.median(filas)), int(np.median(columnas))


def recortar(vol: np.ndarray, centro: tuple[int, int], relleno: int) -> np.ndarray:
    """Recorta (o rellena, donde falte) el plano (Y, X) a CROP_PX × CROP_PX alrededor de `centro`."""
    _, alto, ancho = vol.shape
    y0, x0 = centro[0] - CROP_PX // 2, centro[1] - CROP_PX // 2
    ys = slice(max(y0, 0), min(y0 + CROP_PX, alto))
    xs = slice(max(x0, 0), min(x0 + CROP_PX, ancho))
    out = np.full((len(vol), CROP_PX, CROP_PX), relleno, dtype=vol.dtype)
    out[:, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = vol[:, ys, xs]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Leer y validar un archivo cualquiera
# ---------------------------------------------------------------------------------------------------------------------

def _mm(v: float, decimales: int = 2) -> str:
    return f"{v:.{decimales}f}".replace(".", ",")


def extension_valida(nombre: str) -> str | None:
    nombre = nombre.lower()
    return next((e for e in EXTENSIONES if nombre.endswith(e)), None)


def leer(ruta: Path) -> sitk.Image:
    """Lee el volumen y lo deja en 3D escalar; cualquier problema, como ArchivoNoValido con un mensaje legible."""
    ruta = Path(ruta)
    if extension_valida(ruta.name) is None:
        raise ArchivoNoValido(f"«{ruta.name}» no es un formato admitido. Se aceptan {', '.join(EXTENSIONES)}.")
    try:
        lector = sitk.ImageFileReader()
        lector.SetFileName(str(ruta))
        lector.ReadImageInformation()
        tam = lector.GetSize()
        if len(tam) == 4 and tam[3] == 1:          # 4D con un solo instante: se trata como 3D
            pass
        elif len(tam) != 3:
            raise ArchivoNoValido(f"El volumen tiene {len(tam)} dimensiones ({' × '.join(map(str, tam))}); "
                                  "se esperaba una TC 3D.")
        if np.prod(tam[:3], dtype=np.int64) > MAX_VOXELES:
            raise ArchivoNoValido(f"El volumen es demasiado grande ({' × '.join(map(str, tam))} vóxeles).")
        if lector.GetNumberOfComponents() != 1:
            raise ArchivoNoValido("La imagen tiene varios canales por vóxel (¿es una imagen en color?); se esperaba una TC.")
        img = lector.Execute()
    except ArchivoNoValido:
        raise
    except RuntimeError as e:
        raise ArchivoNoValido(f"No se pudo leer «{ruta.name}» como volumen médico: el archivo está dañado, incompleto "
                              "(¿se cortó la descarga?) o no es un MetaImage, NIfTI o NRRD.") from e
    if img.GetDimension() == 4:
        img = sitk.Extract(img, (*img.GetSize()[:3], 0), (0, 0, 0, 0))
    if min(img.GetSize()) < 2:
        raise ArchivoNoValido(f"El volumen tiene una sola capa en algún eje ({img.GetSize()}): no es una TC 3D.")
    return img


def codigo_orientacion(img: sitk.Image) -> str:
    try:
        return sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(img.GetDirection())
    except Exception:          # noqa: BLE001  (direcciones oblicuas)
        return "oblicua"


def inspeccionar(img: sitk.Image, nombre: str) -> Ficha:
    """Ficha de la cabecera y comprobaciones sobre una muestra de vóxeles: ¿TC en HU, máscara, hueso, espaciado?"""
    espaciado = tuple(img.GetSpacing()[::-1])
    ficha = Ficha(nombre=nombre, tamano=tuple(img.GetSize()[::-1]), espaciado=espaciado,
                  origen=tuple(img.GetOrigin()), orientacion=codigo_orientacion(img),
                  tipo=sitk.GetPixelIDValueAsString(img.GetPixelID()))
    muestra = sitk.GetArrayViewFromImage(img)[::3, ::3, ::3]
    entera = np.issubdtype(muestra.dtype, np.integer) or np.allclose(muestra[::4], np.round(muestra[::4]))
    valores = np.unique(muestra) if entera else None
    if valores is not None and valores.size <= 31 and valores.min() >= 0 and valores.max() <= 30:
        raise ArchivoNoValido(
            f"«{nombre}» parece una máscara de etiquetas (valores {valores.tolist()[:12]}{'…' if valores.size > 12 else ''}), "
            "no una tomografía. Súbela en el campo opcional «etiqueta» junto a su imagen.")
    p = np.percentile(muestra, [1, 50, 99]).astype(float)
    if p[0] > -100 and p[2] > 600:
        # Sin aire por debajo de −100 HU: casi seguro una TC guardada sin desplazar (aire ≈ 0 en lugar de −1024)
        ficha.correccion_hu = -1024
        p -= 1024
        ficha.avisos.append("Los valores no parecían unidades Hounsfield (el aire estaba cerca de 0): se restaron "
                            "1024 a toda la imagen. Comprueba que el archivo es una TC sin reescalar.")
    ficha.hu_p1, ficha.hu_p99 = round(float(p[0])), round(float(p[2]))
    ficha.hueso_pct = round(100 * float((muestra.astype(np.float32) + ficha.correccion_hu > BONE_HU).mean()), 2)
    if p[2] < 150 or ficha.hueso_pct < 0.05:
        ficha.avisos.append("Apenas hay vóxeles con densidad de hueso (> 200 HU): puede que no sea una TC o que no "
                            "contenga la pelvis; el modelo no encontrará nada.")
    if p[0] > -500 and not ficha.correccion_hu:
        ficha.avisos.append("Casi no hay aire en la imagen: ¿es un recorte muy ajustado o una TC con otro rango de valores?")
    sz, sy, sx = espaciado
    if sz > 3.0:
        ficha.avisos.append(f"Cortes gruesos ({_mm(sz)} mm): al remuestrear a 1,5 mm la imagen se interpola mucho en z, "
                            "y el modelo se entrenó con cortes de 0,6–1,25 mm.")
    if max(sx, sy) > 1.5 or min(sx, sy, sz) < 0.3:
        ficha.avisos.append(f"Espaciado poco habitual ({_mm(sx)} × {_mm(sy)} × {_mm(sz)} mm): el modelo se entrenó con "
                            "0,66–1,22 mm en el plano.")
    ext_xy = max(sx * ficha.tamano[2], sy * ficha.tamano[1])
    if ext_xy < 200:
        ficha.avisos.append(f"El campo de visión es pequeño ({ext_xy:.0f} mm): una pelvis adulta mide unos 280 mm de ancho.")
    if ficha.tamano[0] * sz < 60:
        ficha.avisos.append(f"El volumen solo cubre {ficha.tamano[0] * sz:.0f} mm en z: puede que no contenga la pelvis entera.")
    if ficha.orientacion == "oblicua":
        ficha.avisos.append("La matriz de dirección es oblicua: se reorienta a la orientación LPS más cercana.")
    return ficha


def aplicar_correccion(img: sitk.Image, ficha: Ficha) -> sitk.Image:
    if not ficha.correccion_hu:
        return img
    return sitk.Cast(img, sitk.sitkInt32) + ficha.correccion_hu


@dataclass
class Preparado:
    hu: np.ndarray            # (Z, 256, 256) int16: la entrada del modelo, en HU, orientada, isotrópica y recortada
    centro: tuple             # centro del recorte en el volumen isotrópico
    forma_iso: tuple          # (Z, Y, X) del volumen isotrópico sin recortar
    nativo: np.ndarray        # el volumen solo orientado (resolución original), int16: para el visualizador 1
    espaciado_nativo: tuple   # (z, y, x) mm del volumen nativo orientado
    img_orientada: sitk.Image


def preparar_imagen(img: sitk.Image) -> Preparado:
    """Pasos 2 a 4 de build_cache.py sobre la imagen. Devuelve también el volumen nativo orientado (visualizador 1)."""
    orientada = orientar(img)
    iso = remuestrear(orientada, es_etiqueta=False)
    arr = sitk.GetArrayFromImage(iso)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr)
    hu = np.clip(arr, -32768, 32767).astype(np.int16)
    del arr
    centro = centro_hueso(hu)
    nativo = sitk.GetArrayFromImage(orientada)
    if not np.issubdtype(nativo.dtype, np.integer):
        nativo = np.rint(nativo)
    nativo = np.clip(nativo, -32768, 32767).astype(np.int16)
    return Preparado(hu=recortar(hu, centro, AIR_HU), centro=centro, forma_iso=hu.shape, nativo=nativo,
                     espaciado_nativo=tuple(orientada.GetSpacing()[::-1]), img_orientada=orientada)


def misma_rejilla(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize() == b.GetSize() and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-4)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-3) and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4))


def preparar_etiqueta(lab: sitk.Image, img: sitk.Image, centro: tuple) -> np.ndarray:
    """La etiqueta de referencia en la misma rejilla que la entrada del modelo; tiene que compartir geometría con la imagen."""
    if not misma_rejilla(lab, img):
        raise ArchivoNoValido("La etiqueta no comparte rejilla con la imagen (tamaño, espaciado, origen o dirección "
                              "distintos): no se puede comparar con la predicción.")
    arr = sitk.GetArrayViewFromImage(lab)
    valores = np.unique(arr[::2, ::2, ::2])
    if valores.min() < 0 or valores.max() > 30:
        raise ArchivoNoValido(f"La etiqueta tiene valores fuera de la codificación de PENGWIN (0–30): "
                              f"{valores.min()}…{valores.max()}.")
    iso = remuestrear(orientar(lab), es_etiqueta=True)
    return recortar(sitk.GetArrayFromImage(iso).astype(np.uint8), centro, 0)
