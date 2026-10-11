"""
build_cache.py — de los volúmenes .mha de PENGWIN al caché de cortes 2D.

Es el único script del proyecto. Cada paso está justificado con los datos en la
parte 1 del notebook (pengwin.ipynb) y probado sobre una muestra de pacientes en
la parte 2, que importa ESTAS MISMAS funciones: lo que allí se ve funcionar es
exactamente lo que aquí se ejecuta sobre los 100 casos.

Por cada caso:
    1. leer imagen y etiqueta, y comprobar que comparten geometría
    2. orientar a LPS                    (34 casos vienen girados 180° en el plano)
    3. remuestrear a 1,5 mm isotrópico   (lineal la imagen, vecino más cercano la etiqueta)
    4. recortar 256×256 centrado en el hueso   (384 mm de campo)
    5. quedarse con los cortes con hueso ± 7 negativos

Salida en data/cache/:
    slices_hu.npy   int16 (N, 256, 256)   unidades Hounsfield crudas, sin ventanear
    labels.npy      uint8 (N, 256, 256)   codificación PENGWIN original, 0-30
    index.parquet   una fila por corte, en el mismo orden que los dos .npy
    casos.csv       una fila por caso: geometría original y verificaciones

Uso:
    python build_cache.py            construye el caché (no toca uno que ya exista)
    python build_cache.py --force    lo reconstruye
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

RAIZ = Path(__file__).resolve().parent
IMAGES = RAIZ / "data" / "raw" / "images"
LABELS = RAIZ / "data" / "raw" / "labels"
CACHE = RAIZ / "data" / "cache"

ISO_MM = 1.5        # rejilla isotrópica de destino
CROP_PX = 256       # lado del corte: 256 px × 1,5 mm = 384 mm de campo
AIR_HU = -1024      # relleno de la imagen fuera del campo: aire, no agua (0 HU)
BONE_HU = 200       # umbral para localizar el hueso al centrar el recorte
NEG_MARGIN = 7      # cortes sin hueso que se guardan a cada lado (~10 mm)
HUESOS = {1: "sa", 2: "lh", 3: "rh"}   # sacro · coxal izquierdo · coxal derecho


# ---------------------------------------------------------------------------
# Los pasos, uno por función
# ---------------------------------------------------------------------------

def orientar(img: sitk.Image) -> sitk.Image:
    """Orientación canónica LPS. Solo permuta y voltea ejes: no interpola nada."""
    return sitk.DICOMOrient(img, "LPS")


def remuestrear(img: sitk.Image, es_etiqueta: bool) -> sitk.Image:
    """
    Lleva el volumen a ISO_MM en los tres ejes, sin moverlo de sitio.

    La etiqueta va siempre por vecino más cercano: interpolar valores 0-30
    mezclaría el fragmento 12 con el 13 e inventaría uno que no existe.
    """
    espaciado = np.array(img.GetSpacing())                 # orden (x, y, z)
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
    """
    Centro (fila, columna) del hueso en el plano axial: la MEDIANA de las
    coordenadas de los vóxeles > BONE_HU vistos desde arriba. La mediana resiste
    la camilla y el contraste, que arrastrarían a la media. Solo mira la imagen,
    nunca la etiqueta, así que también sirve en inferencia.
    """
    vista = (hu > BONE_HU).any(axis=0)                      # (Y, X)
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


def preparar_caso(caso: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Pasos 1 a 5 sobre un caso.

    Devuelve el volumen ya orientado, isotrópico y recortado —imagen int16 en HU y
    etiqueta uint8, ambos (Z, 256, 256)— y un informe con la geometría original,
    los cortes que hay que guardar (`z0`:`z1`) y las verificaciones.
    """
    img = sitk.ReadImage(str(IMAGES / f"{caso}.mha"))
    lab = sitk.ReadImage(str(LABELS / f"{caso}.mha"))

    # 1 · Imagen y etiqueta tienen que compartir rejilla; si no, cada vóxel de
    #     hueso quedaría desplazado respecto al hueso real.
    geometria_ok = (img.GetSize() == lab.GetSize()
                    and np.allclose(img.GetSpacing(), lab.GetSpacing(), atol=1e-4)
                    and np.allclose(img.GetOrigin(), lab.GetOrigin(), atol=1e-3)
                    and np.allclose(img.GetDirection(), lab.GetDirection()))
    originales = set(np.unique(sitk.GetArrayViewFromImage(lab)).tolist()) - {0}
    info = {
        "case_id": caso,
        # SimpleITK da tamaño y espaciado en orden (x, y, z); aquí van en (z, y, x)
        **dict(zip(("nz", "ny", "nx"), img.GetSize()[::-1])),
        **dict(zip(("sz", "sy", "sx"), (round(s, 4) for s in img.GetSpacing()[::-1]))),
        "direccion": str(tuple(int(round(v)) for v in img.GetDirection())),
        "geometria_ok": bool(geometria_ok),
        **{f"n_frag_{n}": sum(1 for v in originales if (v - 1) // 10 + 1 == k)
           for k, n in HUESOS.items()},
    }

    # 2 y 3 · Orientar ANTES de remuestrear: así el recorte y todo lo demás
    #         trabajan sobre la misma geometría en los 100 casos.
    img = remuestrear(orientar(img), es_etiqueta=False)
    lab = remuestrear(orientar(lab), es_etiqueta=True)
    hu = np.clip(sitk.GetArrayFromImage(img), -32768, 32767).astype(np.int16)   # (Z, Y, X)
    et = sitk.GetArrayFromImage(lab).astype(np.uint8)

    # 4 · Recorte centrado en el hueso de la imagen
    centro = centro_hueso(hu)
    hu_c, et_c = recortar(hu, centro, AIR_HU), recortar(et, centro, 0)

    # 5 · Cortes con hueso, más NEG_MARGIN negativos a cada lado
    con_hueso = np.nonzero(et_c.reshape(len(et_c), -1).any(axis=1))[0]
    z0 = max(int(con_hueso.min()) - NEG_MARGIN, 0) if con_hueso.size else 0
    z1 = min(int(con_hueso.max()) + NEG_MARGIN + 1, len(et_c)) if con_hueso.size else 0

    # Verificaciones: el recorte no puede perder hueso ni el remuestreo inventar
    # o perder etiquetas. Un solo caso mal recortado invalidaría el caché.
    hueso = et > 0
    finales = set(np.unique(et_c).tolist()) - {0}
    extension = [int(np.ptp(np.nonzero(hueso.any(axis=ejes))[0])) + 1 if hueso.any() else 0
                 for ejes in ((1, 2), (0, 2), (0, 1))]
    info.update({
        "nz_iso": hu.shape[0], "ny_iso": hu.shape[1], "nx_iso": hu.shape[2],
        "centro_y": centro[0], "centro_x": centro[1],
        **{f"hueso_mm_{e}": n * ISO_MM for e, n in zip("zyx", extension)},
        "vox_hueso": int(hueso.sum()),
        "vox_perdidos": int(hueso.sum() - (et_c > 0).sum()),
        "etiquetas_perdidas": len(originales - finales),
        "etiquetas_inventadas": len(finales - originales),
        "cortes_hueso": int(con_hueso.size),
        "z0": z0, "z1": z1,
    })
    return hu_c, et_c, info


def estadisticas_corte(lab2d: np.ndarray) -> dict:
    """Una fila del índice: qué huesos y cuántos fragmentos hay en el corte."""
    valores = np.unique(lab2d)
    valores = valores[valores > 0]
    fila, presentes = {}, 0
    for k, nombre in HUESOS.items():
        n = int(((valores - 1) // 10 + 1 == k).sum())
        fila[f"n_frag_{nombre}"] = n
        presentes |= (1 << (k - 1)) if n else 0
    fila["bones_present"] = presentes          # bit 0 sacro · bit 1 coxal izq · bit 2 coxal der
    fila["n_frag_total"] = int(valores.size)
    fila["n_regions"] = bin(presentes).count("1")
    fila["area_bone_px"] = int((lab2d > 0).sum())
    fila["has_bone"] = int(presentes > 0)
    return fila


def guardar_npy(ruta: Path, trozos: list[np.ndarray]) -> None:
    """
    Escribe los trozos uno tras otro en un único .npy: primero la cabecera con la forma
    total y después los datos de cada trozo. Concatenarlos antes duplicaría la memoria.
    """
    cabecera = {"descr": np.lib.format.dtype_to_descr(trozos[0].dtype), "fortran_order": False,
                "shape": (sum(len(t) for t in trozos), CROP_PX, CROP_PX)}
    with open(ruta, "wb") as f:
        np.lib.format.write_array_header_1_0(f, cabecera)
        for t in trozos:
            t.tofile(f)


# ---------------------------------------------------------------------------
# El proceso completo
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Construye el caché de cortes 2D de PENGWIN.")
    ap.add_argument("--force", action="store_true", help="reconstruirlo aunque ya exista")
    args = ap.parse_args()

    if (CACHE / "slices_hu.npy").exists() and not args.force:
        print(f"El caché ya existe en {CACHE}; no se toca. Para reconstruirlo: "
              "python build_cache.py --force")
        return 0

    casos = sorted(p.stem for p in IMAGES.glob("*.mha"))
    if not casos:
        sys.exit(f"No hay volúmenes en {IMAGES} (ver data/README.md).")
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"{len(casos)} casos · {ISO_MM} mm isotrópico · {CROP_PX}×{CROP_PX} px · "
          f"{NEG_MARGIN} negativos a cada lado")

    trozos_hu, trozos_lab, filas, informe = [], [], [], []
    t0 = time.time()
    for i, caso in enumerate(casos, 1):
        hu, lab, info = preparar_caso(caso)
        z0, z1 = info["z0"], info["z1"]
        # .copy(): un trozo de vista mantendría vivo el volumen entero en memoria
        trozos_hu.append(hu[z0:z1].copy())
        trozos_lab.append(lab[z0:z1].copy())
        filas += [{"case_id": caso, "z": z, **estadisticas_corte(lab[z])} for z in range(z0, z1)]
        informe.append(info)
        print(f"\r  {i}/{len(casos)} casos · {len(filas):,} cortes · {time.time() - t0:.0f} s   ",
              end="", flush=True)
    print()

    guardar_npy(CACHE / "slices_hu.npy", trozos_hu)
    guardar_npy(CACHE / "labels.npy", trozos_lab)
    indice = pd.DataFrame(filas)
    indice.insert(0, "row", np.arange(len(indice), dtype=np.int32))
    indice.to_parquet(CACHE / "index.parquet", index=False)
    informe = pd.DataFrame(informe)
    informe.to_csv(CACHE / "casos.csv", index=False)

    # Resumen y verificación: si algo falla, el código de salida lo dice
    fallos = {
        "geometría distinta en imagen y etiqueta": informe.case_id[~informe.geometria_ok].tolist(),
        "vóxeles de hueso perdidos en el recorte": informe.case_id[informe.vox_perdidos > 0].tolist(),
        "etiquetas perdidas o inventadas": informe.case_id[
            (informe.etiquetas_perdidas + informe.etiquetas_inventadas) > 0].tolist(),
    }
    gb = sum(t.nbytes for t in trozos_hu + trozos_lab) / 1024**3
    print(f"\nListo en {time.time() - t0:.0f} s: {len(indice):,} cortes de {len(informe)} casos "
          f"({int(indice.has_bone.sum()):,} con hueso) · {gb:.2f} GB en {CACHE}")
    for nombre, lista in fallos.items():
        print(f"  {nombre:<40}: {'ninguno' if not lista else lista}")
    return 1 if any(fallos.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
