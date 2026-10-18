"""
La red del sistema: `PelvisNetCompleta` con entrada 2.5D.

Es la misma arquitectura que construye el notebook de desarrollo (capítulo 2), con los mismos nombres de módulos, así
que carga sus checkpoints tal cual. Aquí no hace falta ImageNet: los pesos vienen todos del checkpoint. La sección
final del notebook comprueba que este archivo da exactamente las mismas salidas que sus propias clases.

    FundidoraPelvis (tronco con CBAM)  ─┬─ p5 → CabezaClasificacion  → cls       (3 logits: sacro, coxal izq., coxal der.)
           │                            ├─ p4 → CabezaDeteccion      → det_*     (grilla 16×16: objectness, clase, ltrb)
           └─ FPNLite ───────────────── └─ p2 + c1 → CabezaInstancia → seg_*     (256×256: semántica, centros, desplazamientos)
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """Atención de canal y espacial, cada una con su compuerta γ aprendible (γ = 0: el bloque es la identidad)."""

    def __init__(self, canales: int, reduccion: int = 16, nucleo: int = 7):
        super().__init__()
        oculto = max(canales // reduccion, 8)
        self.mlp = nn.Sequential(nn.Conv2d(canales, oculto, 1), nn.ReLU(inplace=True), nn.Conv2d(oculto, canales, 1))
        self.conv = nn.Conv2d(2, 1, nucleo, padding=nucleo // 2, bias=False)
        self.gamma_canal = nn.Parameter(torch.zeros(1))
        self.gamma_espacial = nn.Parameter(torch.zeros(1))

    def atencion_canal(self, x):
        return torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))

    def atencion_espacial(self, x):
        return torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))

    def forward(self, x):
        x = x * (1 + self.gamma_canal * (self.atencion_canal(x) - 1))
        return x * (1 + self.gamma_espacial * (self.atencion_espacial(x) - 1))


class BloqueResidual(nn.Module):
    def __init__(self, c_in, c_out, paso=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, paso, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.downsample = (nn.Sequential(nn.Conv2d(c_in, c_out, 1, paso, bias=False), nn.BatchNorm2d(c_out))
                           if paso != 1 or c_in != c_out else nn.Identity())

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + self.downsample(x))


class FundidoraPelvis(nn.Module):
    """Tronco 2.5D: la primera convolución partida (corte central | vecinos) y cuatro etapas residuales con un CBAM cada una."""

    ANCHOS = (64, 128, 256, 512)

    def __init__(self, cbam=True, cortes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.conv1_vecinos = nn.Conv2d(3 * (cortes - 1), 64, 7, 2, 3, bias=False) if cortes > 1 else None
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        c_prev = 64
        for i, c in enumerate(self.ANCHOS, 1):
            setattr(self, f"layer{i}", nn.Sequential(BloqueResidual(c_prev, c, 1 if i == 1 else 2), BloqueResidual(c, c)))
            setattr(self, f"cbam{i}", CBAM(c) if cbam else nn.Identity())
            c_prev = c

    def forward(self, x):
        y = self.conv1(x[:, :3])
        if self.conv1_vecinos is not None:
            y = y + self.conv1_vecinos(x[:, 3:])
        c1 = F.relu(self.bn1(y))
        x = self.maxpool(c1)
        niveles = {"c1": c1}
        for i in range(1, 5):
            x = getattr(self, f"cbam{i}")(getattr(self, f"layer{i}")(x))
            niveles[f"c{i + 1}"] = x
        return niveles


class FPNLite(nn.Module):
    def __init__(self, entradas=FundidoraPelvis.ANCHOS, canales=128):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, canales, 1, bias=False) for c in entradas)
        self.suavizado = nn.ModuleDict({n: nn.Sequential(nn.Conv2d(canales, canales, 3, padding=1, bias=False),
                                                          nn.BatchNorm2d(canales), nn.ReLU(inplace=True))
                                         for n in ("p2", "p4", "p5")})

    def forward(self, c):
        p = [conv(c[f"c{i}"]) for i, conv in zip(range(2, 6), self.lateral)]
        for i in (2, 1, 0):
            p[i] = p[i] + F.interpolate(p[i + 1], size=p[i].shape[-2:], mode="nearest")
        p = dict(zip(("p2", "p3", "p4", "p5"), p))
        return {n: capa(p[n]) for n, capa in self.suavizado.items()}


class Backbone(nn.Module):
    def __init__(self, cbam=True, cortes=1):
        super().__init__()
        self.tronco = FundidoraPelvis(cbam, cortes)
        self.fpn = FPNLite()

    def forward(self, x):
        c = self.tronco(x)
        return {**self.fpn(c), "c1": c["c1"]}


class CabezaClasificacion(nn.Module):
    def __init__(self, canales=128, oculto=256):
        super().__init__()
        self.red = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(canales, oculto),
                                 nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(oculto, 3))

    def forward(self, p5):
        return self.red(p5)


class CabezaDeteccion(nn.Module):
    def __init__(self, canales=128, prior=0.01):
        super().__init__()
        bloque = lambda: nn.Sequential(nn.Conv2d(canales, canales, 3, padding=1, bias=False),  # noqa: E731
                                       nn.GroupNorm(32, canales), nn.ReLU(inplace=True))
        self.convs = nn.Sequential(bloque(), bloque())
        self.obj, self.cls, self.ltrb = (nn.Conv2d(canales, n, 1) for n in (1, 3, 4))
        nn.init.constant_(self.obj.bias, -math.log((1 - prior) / prior))
        nn.init.ones_(self.ltrb.bias)

    def forward(self, p4):
        f = self.convs(p4)
        return {"obj": self.obj(f)[:, 0], "cls": self.cls(f).permute(0, 2, 3, 1), "ltrb": self.ltrb(f).permute(0, 2, 3, 1)}


class CabezaInstancia(nn.Module):
    def __init__(self, canales=128, canales_c1=64, oculto=64, prior=0.01):
        super().__init__()
        bloque = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False),  # noqa: E731
                                            nn.GroupNorm(32, b), nn.ReLU(inplace=True))
        self.reducir = bloque(canales, oculto)
        self.fundir = bloque(oculto + canales_c1, oculto)
        self.rama_sem = bloque(oculto, oculto)
        self.rama_inst = bloque(oculto, oculto)
        self.sem, self.ctr, self.desp = (nn.Conv2d(oculto, n, 1) for n in (4, 3, 2))
        nn.init.constant_(self.ctr.bias, -math.log((1 - prior) / prior))

    def forward(self, p2, c1):
        x = F.interpolate(self.reducir(p2), size=c1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fundir(torch.cat([x, c1], 1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        s, i = self.rama_sem(x), self.rama_inst(x)
        return {"sem": self.sem(s), "ctr": self.ctr(i), "desp": self.desp(i)}


class PelvisNetCompleta(nn.Module):
    """Backbone compartido y tres cabezas: clasificación, detección y segmentación de instancia."""

    def __init__(self, cbam=True, desplazamientos=(0,)):
        super().__init__()
        self.desplazamientos = tuple(desplazamientos)
        self.backbone = Backbone(cbam, len(self.desplazamientos))
        self.cabeza_cls = CabezaClasificacion()
        self.cabeza_det = CabezaDeteccion()
        self.cabeza_inst = CabezaInstancia()

    def forward(self, x):
        f = self.backbone(x)
        return {"cls": self.cabeza_cls(f["p5"]),
                **{f"det_{k}": v for k, v in self.cabeza_det(f["p4"]).items()},
                **{f"seg_{k}": v for k, v in self.cabeza_inst(f["p2"], f["c1"]).items()}}


def cargar_modelo(ruta: Path, dispositivo: torch.device) -> tuple[nn.Module, dict]:
    """Un checkpoint del proyecto, listo para inferir en `dispositivo`, y su configuración (con la pila 2.5D)."""
    g = torch.load(ruta, map_location="cpu", weights_only=False)
    config = g["config"]
    modelo = PelvisNetCompleta(config.get("cbam", True), tuple(config.get("desplazamientos", (0,))))
    modelo.load_state_dict(g["modelo"])
    modelo = modelo.to(dispositivo).to(memory_format=torch.channels_last).eval()
    mejor = max(g["historial"], key=lambda f: f["puntuacion"])
    info = {"archivo": Path(ruta).name, "desplazamientos": list(modelo.desplazamientos), "epoca": mejor["epoca"],
            "epocas": len(g["historial"]), "iou_val": mejor["iou"], "dice_val": mejor["dice"],
            "parametros": sum(p.numel() for p in modelo.parameters()), "fecha": g.get("fecha")}
    return modelo, info
