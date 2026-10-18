"""
Los datos de los tres visualizadores del §3.4. El navegador solo dibuja: todo lo que hay que calcular está aquí.

1. Volumen crudo: MIP rotatoria del volumen original (solo orientado, sin remuestrear), sin la camilla. Los fotogramas
   viajan en HU y el umbral se aplica en el navegador: umbralizar y proyectar conmutan, max(max(v, t)) = max(max v, t),
   así que mover el umbral no obliga a recalcular nada.
2. Corte a corte: el corte en la ventana de hueso, los fragmentos 3D predichos, las cajas y la separación de cada uno.
3. Reconstrucción 3D: una malla por fragmento con marching cubes (skimage), con su color y su separación.
"""
from __future__ import annotations

import math

import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.measure import marching_cubes

from .preprocesado import AIR_HU, ISO_MM

COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}        # sacro · coxal izquierdo · coxal derecho
ABREVIATURA = {1: "sacro", 2: "coxal izq.", 3: "coxal der."}
UMBRAL_HU, HU_MAX = 250, 2000


def color_fragmento(v: int) -> str:
    """El color de un fragmento: el de su hueso, más claro cuanto mayor es su número (el principal, el tono puro)."""
    base = np.array(mcolors.to_rgb(COLOR[(v - 1) // 10 + 1]))
    return mcolors.to_hex(tuple(base + (1 - base) * min(0.22 * ((v - 1) % 10), 0.66)))


# ---------------------------------------------------------------------------------------------------------------------
# Visualizador 1: MIP rotatoria del volumen crudo
# ---------------------------------------------------------------------------------------------------------------------

def aislar_paciente(vol: np.ndarray) -> np.ndarray:
    """Solo el cuerpo —el mayor componente conexo por encima de −300 HU—: fuera la camilla, más densa que el hueso."""
    cuerpo, n = ndimage.label(vol > -300)
    if n <= 1:
        return vol
    tam = np.bincount(cuerpo.ravel())
    tam[0] = 0
    return np.where(cuerpo == tam.argmax(), vol, AIR_HU).astype(vol.dtype)


def recortar_a_hueso(vol: np.ndarray, umbral: int = 150, margen: int = 4, pct: float = 0.5) -> np.ndarray:
    hueso = vol > umbral
    if not hueso.any():
        return vol
    caja = []
    for eje in range(3):
        acumulado = np.cumsum(hueso.sum(axis=tuple(e for e in range(3) if e != eje)))
        acumulado = acumulado / acumulado[-1]
        lo, hi = np.searchsorted(acumulado, pct / 100), np.searchsorted(acumulado, 1 - pct / 100)
        caja.append(slice(max(int(lo) - margen, 0), int(hi) + margen + 1))
    return vol[tuple(caja)]


def max_pool(vol: np.ndarray, k, dispositivo) -> np.ndarray:
    """Reducción por bloques quedándose con el máximo: en una MIP conserva el hueso fino, que saltando vóxeles se pierde."""
    x = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).to(dispositivo)[None, None]
    return F.max_pool3d(x, kernel_size=tuple(int(v) for v in k), stride=tuple(int(v) for v in k), ceil_mode=True)[0, 0]


@torch.no_grad()
def rotacion_mip(vol: np.ndarray, espaciado: tuple, n: int = 36, max_lado: int = 256,
                 dispositivo: torch.device | None = None) -> tuple[np.ndarray, list]:
    """
    MIP girando el volumen alrededor del eje craneocaudal: `n` fotogramas en HU (int16), la cabeza arriba.

    Fuera la camilla y el aire (recorte al hueso); después el volumen se lleva a isotrópico, con el lado mayor ≤ max_lado:
    primero por *max-pooling* en bloques enteros y después con interpolación trilineal hasta el tamaño exacto. Un
    engrosado de un vóxel (máximo en 3×3×3) evita que la interpolación del giro module las corticales finas, que si no
    salen tramadas. Cada corte axial se gira con grid_sample, en GPU si la hay.
    """
    dispositivo = dispositivo or torch.device("cpu")
    espaciado = np.array(espaciado, dtype=float)
    limite = 128 * 1024 * 1024
    if vol.size > limite:                               # volúmenes enormes: aislar al paciente sobre una versión reducida
        k = math.ceil((vol.size / limite) ** (1 / 3))
        vol = max_pool(vol, (k, k, k), dispositivo).cpu().numpy().astype(np.int16)
        espaciado = espaciado * k
    vol = recortar_a_hueso(aislar_paciente(vol))
    destino = np.array(vol.shape) * espaciado / espaciado.min()
    destino = destino * min(1.0, max_lado / destino.max())
    k = np.maximum(np.floor(np.array(vol.shape) / destino).astype(int), 1)
    x = max_pool(vol, k, dispositivo) if (k > 1).any() else \
        torch.from_numpy(vol.astype(np.float32)).to(dispositivo)
    x = F.interpolate(x[None, None], size=tuple(max(int(round(d)), 1) for d in destino), mode="trilinear",
                      align_corners=False)
    x = F.max_pool3d(x, 3, stride=1, padding=1)[0, 0].clamp(AIR_HU, HU_MAX)
    _, ny, nx = x.shape
    d = int(np.ceil(np.hypot(ny, nx)))
    py, px = (d - ny + 1) // 2, (d - nx + 1) // 2
    x = F.pad(x - AIR_HU, (px, px, py, py))[:, None]                     # el aire a 0: el relleno de grid_sample
    angulos = [round(360 * i / n, 1) for i in range(n)]
    fotogramas = []
    for a in angulos:
        t = math.radians(a)
        theta = torch.tensor([[math.cos(t), -math.sin(t), 0.0], [math.sin(t), math.cos(t), 0.0]],
                             dtype=torch.float32, device=dispositivo)[None].expand(len(x), 2, 3)
        rejilla = F.affine_grid(theta, x.shape, align_corners=False)
        girado = F.grid_sample(x, rejilla, mode="bilinear", padding_mode="zeros", align_corners=False)
        fotogramas.append((girado[:, 0].amax(dim=1) + AIR_HU).flip(0).round().short().cpu().numpy())   # (Z, D)
    return np.stack(fotogramas), angulos


# ---------------------------------------------------------------------------------------------------------------------
# Visualizador 2: corte a corte
# ---------------------------------------------------------------------------------------------------------------------

def ventana_hueso_uint8(hu: np.ndarray) -> np.ndarray:
    """El volumen en la ventana de hueso [−200, 1300] HU, en uint8: lo que dibuja el visor de cortes."""
    return (np.clip((hu.astype(np.float32) + 200) / 1500, 0, 1) * 255).round().astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# Visualizador 3: una malla por fragmento
# ---------------------------------------------------------------------------------------------------------------------

def malla(mascara: np.ndarray, paso: int = 2, suavizado: float = 1.0):
    """La superficie de una máscara 3D en mm (z, y, x): marching cubes al nivel 0,5 sobre la máscara suavizada."""
    caja = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in ndimage.find_objects(mascara.astype(np.uint8))[0])
    trozo = ndimage.gaussian_filter(np.pad(mascara[caja].astype(np.float32), 1), suavizado)
    vertices, caras, _, _ = marching_cubes(trozo, 0.5, spacing=(ISO_MM,) * 3, step_size=paso)
    return (vertices + (np.array([s.start for s in caja]) - 1) * ISO_MM).astype(np.float32), caras.astype(np.int32)


def mallas(vol3d: np.ndarray, fragmentos: list, paso: int = 2) -> list:
    """Una malla por fragmento, con su nombre y separación, lista para Plotly Mesh3d (x = izquierda, y = posterior, z = craneal)."""
    salida = []
    for f in sorted(fragmentos, key=lambda f: (f["hueso"], f["etiqueta"])):
        m = vol3d == f["etiqueta"]
        if m.sum() < 8:
            continue
        try:
            v, caras = malla(m, paso)
        except (ValueError, RuntimeError):                     # una máscara demasiado fina para cerrar una superficie
            continue
        cz, cy, cx = np.array(ndimage.center_of_mass(m)) * ISO_MM
        salida.append({
            "etiqueta": int(f["etiqueta"]), "hueso": int(f["hueso"]), "principal": bool(f["principal"]),
            "separacion_mm": round(float(f["separacion_mm"]), 1), "color": color_fragmento(int(f["etiqueta"])),
            "color_hueso": COLOR[int(f["hueso"])],
            "nombre": f"{ABREVIATURA[f['hueso']]} {f['etiqueta']} · " + ("principal" if f["principal"] else
                                                                         f"{f['separacion_mm']:.1f} mm".replace(".", ",")),
            "x": np.round(v[:, 2], 1).tolist(), "y": np.round(v[:, 1], 1).tolist(), "z": np.round(v[:, 0], 1).tolist(),
            "i": caras[:, 0].tolist(), "j": caras[:, 1].tolist(), "k": caras[:, 2].tolist(),
            "centro": [round(float(cx), 1), round(float(cy), 1), round(float(cz), 1)],
        })
    return salida
