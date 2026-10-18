"""
El pipeline del sistema sobre un volumen ya preparado (Z, 256, 256) en HU:

    pila 2.5D de cada corte → ventaneo → red → detección → NMS propio → mejor caja por región
        → compuerta + instancias (dos etapas) → fragmentos 2D → enlace 3D → fragmentos 3D → separación en mm

Son las funciones del notebook de desarrollo, copiadas sin cambios de comportamiento (el notebook lo comprueba). Las
decisiones que se tomaron en validación —la compuerta, el umbral de los centros, el enlace 3D y el volumen mínimo de un
fragmento— llegan en `Config`, leída de `checkpoints/configuracion_inferencia.json`, que escribe el notebook.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .preprocesado import CROP_PX, ISO_MM

HUESO = {1: "sacro", 2: "coxal izquierdo", 3: "coxal derecho"}
VENTANAS = ((-200, 1300), (-150, 250), (-1000, 2000))       # hueso · tejido blando · amplia
LO, HI = (np.array(v, np.float32) for v in zip(*VENTANAS))
GRID, CELDA, LADO_MIN = 16, 16, 8
CENTROS = (np.arange(GRID) + 0.5) * CELDA
NUCLEO_PICOS, MAX_FRAG, MARGEN_CAJA, LADO_RUIDO = 9, 4, 8, 4
VOX_MM3 = ISO_MM ** 3


@dataclass
class Config:
    modelo: str
    desplazamientos: tuple
    compuerta: bool
    umbral_centro: float
    enlace: str
    umbral_mm3: float

    @classmethod
    def leer(cls, ruta: Path) -> "Config":
        c = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls(c["modelo"], tuple(c["desplazamientos"]), bool(c["compuerta"]), float(c["umbral_centro"]),
                   c["enlace"], float(c["umbral_mm3"]))

    def modo_enlace(self) -> tuple[str, bool]:
        return {"parejas": ("parejas", False), "solape": ("solape", False),
                "parejas + una pieza conexa": ("parejas", True)}[self.enlace]


# ---------------------------------------------------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------------------------------------------------

def a_hueso(lab: np.ndarray) -> np.ndarray:
    lab = lab.astype(np.int16)
    return np.where(lab > 0, (lab - 1) // 10 + 1, 0).astype(np.uint8)


def orden_central(desplazamientos) -> tuple:
    return (0, *[d for d in desplazamientos if d != 0])


def ventanear_pila(hu, lo=LO, hi=HI):
    """Pila en HU (S, H, W) → (3·S, H, W) en [0, 1]: las tres ventanas de cada corte, el central primero."""
    x = np.clip((np.asarray(hu, np.float32)[:, None] - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1)
    return x.reshape(-1, *x.shape[2:])


def cajas_por_region(lab2d: np.ndarray) -> dict:
    h = a_hueso(lab2d)
    cajas = {}
    for k in (1, 2, 3):
        m = h == k
        filas, columnas = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        if filas.size:
            cajas[k] = (int(columnas[0]), int(filas[0]), int(columnas[-1]) + 1, int(filas[-1]) + 1)
    return cajas


# ---------------------------------------------------------------------------------------------------------------------
# Detección: decodificación de la grilla y NMS propio
# ---------------------------------------------------------------------------------------------------------------------

def iou_cajas(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area = lambda c: (c[:, 2] - c[:, 0]).clamp(min=0) * (c[:, 3] - c[:, 1]).clamp(min=0)  # noqa: E731
    esquina_sup = torch.max(a[:, None, :2], b[None, :, :2])
    esquina_inf = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (esquina_inf - esquina_sup).clamp(min=0).prod(-1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter).clamp(min=1e-9)


def nms(cajas: torch.Tensor, puntuaciones: torch.Tensor, umbral: float = 0.5) -> torch.Tensor:
    orden = puntuaciones.argsort(descending=True)
    quedan = []
    while orden.numel():
        i, resto = orden[0], orden[1:]
        quedan.append(i)
        orden = resto[iou_cajas(cajas[i:i + 1], cajas[resto])[0] <= umbral]
    return torch.stack(quedan) if quedan else orden


def decodificar(salida: dict, i: int, umbral: float = 0.3, umbral_nms: float = 0.5, max_det: int = 20,
                max_por_region: int | None = None):
    p = torch.sigmoid(salida["det_obj"][i].float())[..., None] * torch.softmax(salida["det_cls"][i].float(), -1)
    d = salida["det_ltrb"][i].float() * CELDA
    c = torch.as_tensor(CENTROS, dtype=torch.float32, device=p.device)
    cy, cx = torch.meshgrid(c, c, indexing="ij")
    cajas = torch.stack([cx - d[..., 0], cy - d[..., 1], cx + d[..., 2], cy + d[..., 3]], -1).clamp(0, CROP_PX)
    por_clase = []
    for k in range(3):
        m = p[..., k] >= umbral
        ck, pk = cajas[m], p[..., k][m]
        q = nms(ck, pk, umbral_nms)[:max_por_region]
        por_clase.append((ck[q], pk[q], torch.full((len(q),), k, device=p.device)))
    cajas, puntuaciones, clases = (torch.cat(t) for t in zip(*por_clase))
    orden = puntuaciones.argsort(descending=True)[:max_det]
    return cajas[orden], puntuaciones[orden], clases[orden]


def cajas_del_corte(salida, i) -> tuple[dict, dict]:
    """La mejor caja de cada región tras el NMS, como {hueso: caja}, y su puntuación {hueso: p}."""
    cj, pu, cl = decodificar(salida, i, max_por_region=1)
    cajas = {int(k) + 1: c for c, k in zip(cj.tolist(), cl.tolist())}
    return cajas, {int(k) + 1: float(p) for p, k in zip(pu.tolist(), cl.tolist())}


# ---------------------------------------------------------------------------------------------------------------------
# Segmentación de instancia: las dos etapas del §3.2
# ---------------------------------------------------------------------------------------------------------------------

def picos(calor: np.ndarray, umbral: float) -> np.ndarray:
    pico = (calor == ndimage.maximum_filter(calor, NUCLEO_PICOS)) & (calor >= umbral)
    grupos, n = ndimage.label(pico)
    if n == 0:
        return np.zeros((0, 2), int)
    yx = np.rint(ndimage.center_of_mass(pico, grupos, range(1, n + 1))).astype(int).reshape(-1, 2)
    return yx[np.argsort(-calor[yx[:, 0], yx[:, 1]])[:MAX_FRAG]]


def instancias(sem, calor, desp, compuerta=None, presencia=None, umbral=0.3) -> np.ndarray:
    """Primero la región de cada hueso (acotada por su caja); después sus fragmentos, votando por el centro más cercano."""
    etiqueta = np.zeros(sem.shape, np.uint8)
    for k in (1, 2, 3):
        region = sem == k
        if compuerta is not None:
            if k in compuerta:
                x0, y0, x1, y1 = (int(round(v)) for v in compuerta[k])
                dentro = np.zeros_like(region)
                dentro[max(y0 - MARGEN_CAJA, 0):y1 + MARGEN_CAJA, max(x0 - MARGEN_CAJA, 0):x1 + MARGEN_CAJA] = True
                region &= dentro
            elif presencia is not None and presencia[k - 1] < 0.5:
                continue
        ys, xs = np.nonzero(region)
        if not len(ys):
            continue
        centros = picos(calor[k - 1], umbral)
        if len(centros):
            vy, vx = ys + desp[0, ys, xs] * CELDA, xs + desp[1, ys, xs] * CELDA
            grupo = ((vy[:, None] - centros[:, 0]) ** 2 + (vx[:, None] - centros[:, 1]) ** 2).argmin(1)
        else:
            grupo = np.zeros(len(ys), int)
        j = 0
        for g in np.argsort(-np.bincount(grupo)):
            sel = grupo == g
            if sel.any() and max(np.ptp(ys[sel]), np.ptp(xs[sel])) + 1 >= LADO_RUIDO:
                etiqueta[ys[sel], xs[sel]] = 10 * (k - 1) + 1 + j
                j += 1
    return etiqueta


def separar(crudo: dict):
    seg = (crudo["seg_sem"].argmax(1).byte().cpu().numpy(), torch.sigmoid(crudo["seg_ctr"].float()).cpu().numpy(),
           crudo["seg_desp"].float().cpu().numpy())
    return {k: v.float().cpu() for k, v in crudo.items() if not k.startswith("seg_")}, seg


@torch.no_grad()
def predecir(modelo, imagen):
    con_bf16 = imagen.device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=con_bf16):
        return separar(modelo(imagen))


@torch.no_grad()
def inferir_volumen(modelo, hu: np.ndarray, config: Config, tam: int = 16, progreso=None) -> dict:
    """
    El pipeline corte a corte sobre un volumen (Z, 256, 256) en HU, sin mirar ninguna etiqueta: la pila 2.5D de cada
    corte dentro del volumen (en los bordes se repite el último corte), la red y la decodificación. En lotes de `tam`.
    """
    dispositivo = next(modelo.parameters()).device
    orden = np.array(orden_central(modelo.desplazamientos))
    etiquetas, cajas, puntuaciones, presencia = [], [], [], []
    t_red = 0.0
    for a in range(0, len(hu), tam):
        zs = np.arange(a, min(a + tam, len(hu)))
        pilas = np.asarray(hu)[np.clip(zs[:, None] + orden[None], 0, len(hu) - 1)]
        x = torch.from_numpy(np.stack([ventanear_pila(p) for p in pilas])).to(dispositivo).contiguous(
            memory_format=torch.channels_last)
        t0 = time.perf_counter()
        salida, seg = predecir(modelo, x)
        t_red += time.perf_counter() - t0
        for i in range(len(zs)):
            c, p = cajas_del_corte(salida, i)
            cajas.append(c)
            puntuaciones.append(p)
            prob = torch.sigmoid(salida["cls"][i]).numpy()
            presencia.append(prob.tolist())
            etiquetas.append(instancias(seg[0][i], seg[1][i], seg[2][i], c if config.compuerta else None, prob,
                                        config.umbral_centro))
        if progreso:
            progreso(min(a + tam, len(hu)) / len(hu))
    return {"etiquetas": np.stack(etiquetas), "cajas": cajas, "puntuaciones": puntuaciones, "presencia": presencia,
            "segundos_red": t_red}


# ---------------------------------------------------------------------------------------------------------------------
# De cortes a fragmentos 3D, y la separación en milímetros (§3.3)
# ---------------------------------------------------------------------------------------------------------------------

def enlazar_3d(etiquetas: np.ndarray, modo: str = "parejas", conexos: bool = True, umbral_mm3: float = 1600.0) -> np.ndarray:
    """Fragmentos 2D de cortes consecutivos → fragmentos 3D: parejas por IoU (húngaro), union-find, piezas conexas y filtro."""
    padre = {}

    def raiz(n):
        while padre.setdefault(n, n) != n:
            padre[n] = padre[padre[n]]
            n = padre[n]
        return n

    for z in range(len(etiquetas)):
        for v in np.unique(etiquetas[z][etiquetas[z] > 0]):
            raiz((z, int(v)))
        if z + 1 == len(etiquetas):
            break
        conjunta = np.bincount(etiquetas[z].ravel().astype(np.int64) * 31 + etiquetas[z + 1].ravel(),
                               minlength=31 * 31).reshape(31, 31)
        area_a, area_b = conjunta.sum(1), conjunta.sum(0)
        for k in (1, 2, 3):
            va = [v for v in range(10 * k - 9, 10 * k + 1) if area_a[v]]
            vb = [v for v in range(10 * k - 9, 10 * k + 1) if area_b[v]]
            if not (va and vb):
                continue
            inter = conjunta[np.ix_(va, vb)]
            if modo == "parejas":
                iou = inter / (area_a[va][:, None] + area_b[vb][None, :] - inter)
                uniones = [(a, b) for a, b in zip(*linear_sum_assignment(-iou)) if iou[a, b] >= 0.2]
            else:
                uniones = zip(*np.nonzero(inter / np.minimum(area_a[va][:, None], area_b[vb][None, :]) >= 0.5))
            for a, b in uniones:
                padre[raiz((z, va[a]))] = raiz((z + 1, vb[b]))
    grupo = {n: raiz(n) for n in padre}
    ids = {r: j + 1 for j, r in enumerate(sorted(set(grupo.values())))}
    componentes = np.zeros(etiquetas.shape, np.int32)
    for (z, v), r in grupo.items():
        componentes[z][etiquetas[z] == v] = ids[r]
    hueso_de = {ids[r]: (r[1] - 1) // 10 + 1 for r in ids}
    piezas = {k: [] for k in (1, 2, 3)}
    for c, caja in enumerate(ndimage.find_objects(componentes), 1):
        if caja is None:
            continue
        m = componentes[caja] == c
        if conexos:
            lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
            piezas[hueso_de[c]] += [(int(t), caja, lab == g + 1) for g, t in enumerate(np.bincount(lab.ravel())[1:])]
        else:
            piezas[hueso_de[c]].append((int(m.sum()), caja, m))
    salida = np.zeros(etiquetas.shape, np.uint8)
    for k, lista in piezas.items():
        lista = sorted((p for p in lista if p[0] * VOX_MM3 >= umbral_mm3), key=lambda p: -p[0])[:10]
        for j, (_, caja, m) in enumerate(lista):
            salida[caja][m] = 10 * (k - 1) + 1 + j
    return salida


def separacion_mm(lab3d, v, principal, cajas, espaciado=(ISO_MM,) * 3, margen=2):
    """Distancia mínima borde a borde, en mm, entre el fragmento `v` y el `principal` (distance_transform_edt, §3.3)."""
    caja = tuple(slice(max(min(a.start, b.start) - margen, 0), max(a.stop, b.stop) + margen)
                 for a, b in zip(cajas[v - 1], cajas[principal - 1]))
    trozo = lab3d[caja]
    distancia = ndimage.distance_transform_edt(trozo != principal, sampling=espaciado)
    return float(distancia[trozo == v].min())


def fragmentos_del_caso(lab3d: np.ndarray) -> list[dict]:
    """Una fila por fragmento 3D: hueso, volumen, si es el principal de su hueso y su separación respecto a él."""
    vox = np.bincount(lab3d.ravel(), minlength=31)
    cajas = ndimage.find_objects(lab3d)
    filas = []
    for k in (1, 2, 3):
        presentes = [v for v in range(10 * k - 9, 10 * k + 1) if vox[v]]
        if not presentes:
            continue
        principal = max(presentes, key=lambda v: vox[v])
        for v in presentes:
            zs = cajas[v - 1][0]
            filas.append({"hueso": k, "etiqueta": v, "principal": v == principal, "volumen_mm3": float(vox[v] * VOX_MM3),
                          "z0": int(zs.start), "z1": int(zs.stop - 1),
                          "separacion_mm": 0.0 if v == principal else separacion_mm(lab3d, v, principal, cajas)})
    return filas


# ---------------------------------------------------------------------------------------------------------------------
# Si llega una etiqueta: las métricas del §5 sobre este paciente y la separación frente al ground truth
# ---------------------------------------------------------------------------------------------------------------------

def emparejar(pred: np.ndarray, real: np.ndarray):
    """Fragmentos de un corte emparejados dentro de cada hueso (húngaro sobre el IoU), con la regla ignore (< 8 px)."""
    conjunta = np.bincount(pred.ravel().astype(np.int64) * 31 + real.ravel(), minlength=31 * 31).reshape(31, 31)
    area_p, area_r = conjunta.sum(1), conjunta.sum(0)
    trozos = ndimage.find_objects(real)
    filas, sin_pareja = [], []
    for k in (1, 2, 3):
        vr = [v for v in range(10 * k - 9, 10 * k + 1) if area_r[v]]
        vp = [v for v in range(10 * k - 9, 10 * k + 1) if area_p[v]]
        inter = conjunta[np.ix_(vp, vr)]
        iou = inter / (area_p[vp][:, None] + area_r[vr][None, :] - inter).clip(min=1)
        pareja = {}
        for a, b in zip(*linear_sum_assignment(-iou)):
            if inter[a, b] > 0:
                pareja[vr[b]] = vp[a]
        for v in vr:
            p = pareja.get(v)
            i = conjunta[p, v] if p else 0
            filas.append((k, v, max(s.stop - s.start for s in trozos[v - 1]) < LADO_MIN, int(area_r[v]), p or 0,
                          2 * i / (area_p[p] + area_r[v]) if p else 0.0, i / (area_p[p] + area_r[v] - i) if p else 0.0))
        sin_pareja += [(k, p) for p in vp if p not in pareja.values()]
    return filas, sin_pareja


def emparejar_3d(pred: np.ndarray, real: np.ndarray) -> dict:
    conjunta = np.bincount(pred.ravel().astype(np.int64) * 31 + real.ravel(), minlength=31 * 31).reshape(31, 31)
    area_p, area_r = conjunta.sum(1), conjunta.sum(0)
    parejas = {}
    for k in (1, 2, 3):
        vr = [v for v in range(10 * k - 9, 10 * k + 1) if area_r[v]]
        vp = [v for v in range(10 * k - 9, 10 * k + 1) if area_p[v]]
        if vr and vp:
            inter = conjunta[np.ix_(vp, vr)]
            iou = inter / (area_p[vp][:, None] + area_r[vr][None, :] - inter)
            for a, b in zip(*linear_sum_assignment(-iou)):
                if inter[a, b] > 0:
                    parejas[vr[b]] = (vp[a], float(iou[a, b]), float(2 * inter[a, b] / (area_p[vp[a]] + area_r[vr[b]])))
    return parejas


def evaluar(inferencia: dict, vol3d: np.ndarray, fragmentos: list, real: np.ndarray) -> dict:
    """Métricas de este paciente frente a su etiqueta: detección, clasificación, Dice de fragmento 2D y 3D y separación."""
    etiquetas = inferencia["etiquetas"]
    ious, cls_real, cls_pred, frag2d = [], [], [], []
    for z in range(len(real)):
        cr = cajas_por_region(real[z])
        cp = inferencia["cajas"][z]
        for k, caja in cr.items():
            if max(caja[2] - caja[0], caja[3] - caja[1]) < LADO_MIN:
                continue
            ious.append(float(iou_cajas(torch.tensor([caja], dtype=torch.float32),
                                        torch.tensor([cp[k]], dtype=torch.float32))[0, 0]) if k in cp else 0.0)
        cls_real.append([k in cr for k in (1, 2, 3)])
        cls_pred.append([p >= 0.5 for p in inferencia["presencia"][z]])
        if real[z].any() or etiquetas[z].any():
            frag2d.append(emparejar(etiquetas[z], real[z]))
    filas = [f for fs, _ in frag2d for f in fs if not f[2]]
    inventados = sum(len(s) for _, s in frag2d)
    no_encontrados = sum(1 for f in filas if not f[4])
    tp = len(filas) - no_encontrados
    cls_real, cls_pred = np.array(cls_real), np.array(cls_pred)
    acierto_cls = {HUESO[k]: float((cls_real[:, k - 1] == cls_pred[:, k - 1]).mean()) for k in (1, 2, 3)}

    reales = {f["etiqueta"]: f for f in fragmentos_del_caso(real)}
    pred = {f["etiqueta"]: f for f in fragmentos}
    parejas = emparejar_3d(vol3d, real)
    tabla = []
    for v, f in sorted(reales.items()):
        p, iou3, dice3 = parejas.get(v, (0, 0.0, 0.0))
        fp_ = pred.get(p)
        tabla.append({"hueso": f["hueso"], "etiqueta_real": v, "principal_real": f["principal"],
                      "volumen_real_mm3": f["volumen_mm3"], "separacion_real_mm": f["separacion_mm"],
                      "etiqueta_pred": p, "dice_3d": dice3,
                      "separacion_pred_mm": fp_["separacion_mm"] if fp_ else None,
                      "pareja_es_principal": bool(fp_ and fp_["principal"])})
    medidos = [t for t in tabla if not t["principal_real"] and t["etiqueta_pred"] and not t["pareja_es_principal"]]
    errores = [t["separacion_pred_mm"] - t["separacion_real_mm"] for t in medidos]
    pegado = lambda d: d <= ISO_MM + 1e-6  # noqa: E731
    return {
        "iou_deteccion": float(np.mean(ious)) if ious else None, "cajas_evaluadas": len(ious),
        "clasificacion_acierto": acierto_cls,
        "dice_fragmento": float(np.mean([f[5] for f in filas])) if filas else None,
        "iou_fragmento": float(np.mean([f[6] for f in filas])) if filas else None,
        "fragmentos_2d": len(filas), "no_encontrados_2d": no_encontrados, "inventados_2d": inventados,
        "f1_reconocimiento": 2 * tp / max(2 * tp + no_encontrados + inventados, 1),
        "dice_3d": float(np.mean([t["dice_3d"] for t in tabla])) if tabla else None,
        "fragmentos_reales": len(reales), "fragmentos_predichos": len(pred),
        "conminutos_reales": sum(not f["principal"] for f in reales.values()),
        "conminutos_medidos": len(medidos),
        "error_separacion_medio_mm": float(np.mean(np.abs(errores))) if errores else None,
        "pegado_desplazado_bien": (float(np.mean([pegado(t["separacion_real_mm"]) == pegado(t["separacion_pred_mm"])
                                                  for t in medidos])) if medidos else None),
        "tabla": tabla,
    }
