"""
Servidor de la app PENGWIN 2.5D.

    python app/servidor.py                    → http://localhost:8000
    python app/servidor.py --puerto 8080 --host 0.0.0.0

Sirve la página (app/web/) y una API pequeña:

    GET  /api/estado                          dispositivos, modelo, casos de ejemplo y cola
    POST /api/subidas                         abre una subida {nombre, tamano} → {id, trozo}
    PUT  /api/subidas/{id}/trozo/{n}          un trozo del archivo, en orden; X-Comprimido: gzip si viene comprimido
    POST /api/subidas/{id}/fin                cierra la subida y comprueba el tamaño
    POST /api/trabajos                        {imagen: id | demo: caso, etiqueta: id?, dispositivo} → {id}
    GET  /api/trabajos/{id}                   estado, etapa y progreso; con el resumen cuando termina
    GET  /api/trabajos/{id}/datos/{nombre}    volúmenes para el navegador (gzip): mip, hu, pred, real, mallas
    GET  /api/trabajos/{id}/descarga/{arch}   segmentacion.mha · fragmentos.csv

La subida va en trozos (16 MB antes de comprimir) porque el túnel de Cloudflare corta las peticiones de más de 100 MB
y una TC de pelvis pesa hasta 420 MB. Los trabajos se procesan de uno en uno en un hilo propio: hay una sola GPU.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))

from motor import preprocesado as pre  # noqa: E402
from motor.servicio import Motor, leer_cabecera_rapida  # noqa: E402

TROZO = 16 * 1024 * 1024
TROZO_MAX = 80 * 1024 * 1024               # lo que puede ocupar un trozo, comprimido o no
TAMANO_MAX = 2 * 1024 ** 3
MAX_EN_COLA = 4
TRABAJOS_GUARDADOS = 8
CADUCIDAD_S = 2 * 3600
TEMPORAL = Path(tempfile.gettempdir()) / "pengwin_app"
AVISO = ("Uso exclusivamente académico. Este sistema no es un dispositivo médico, no ha sido validado clínicamente y no "
         "debe utilizarse para apoyar decisiones quirúrgicas reales.")

motor = Motor(RAIZ)
subidas: dict[str, dict] = {}
trabajos: dict[str, dict] = {}
cerrojo = threading.Lock()
cola: "queue.Queue[str]" = queue.Queue()


def error(mensaje: str, codigo: int = 400) -> JSONResponse:
    return JSONResponse({"error": mensaje}, status_code=codigo)


def nombre_seguro(nombre: str) -> str:
    base = Path(nombre).name
    ext = pre.extension_valida(base) or ""
    tallo = re.sub(r"[^A-Za-z0-9_.-]", "_", base[: len(base) - len(ext)])[:60] or "volumen"
    return tallo + ext


def casos_demo() -> list[dict]:
    imagenes, etiquetas = RAIZ / "data" / "raw" / "images", RAIZ / "data" / "raw" / "labels"
    config = json.loads((RAIZ / "checkpoints" / "configuracion_inferencia.json").read_text(encoding="utf-8"))
    salida = []
    for caso in config.get("casos_prueba", []):
        ruta = imagenes / f"{caso}.mha"
        if ruta.exists():
            try:
                cab = leer_cabecera_rapida(ruta)
            except RuntimeError:
                continue
            salida.append({"caso": caso, "mb": round(ruta.stat().st_size / 1e6), **cab,
                           "con_etiqueta": (etiquetas / f"{caso}.mha").exists()})
    return salida


DEMOS = casos_demo()


# ---------------------------------------------------------------------------------------------------------------------
# El hilo que procesa los trabajos, de uno en uno
# ---------------------------------------------------------------------------------------------------------------------

def procesar():
    while True:
        tid = cola.get()
        t = trabajos.get(tid)
        if t is None:
            continue
        t.update(estado="procesando", inicio=time.time())

        def progreso(clave, texto, fraccion):
            t.update(etapa=clave, texto=texto, progreso=round(fraccion, 4))

        try:
            r = motor.analizar(t["ruta_imagen"], t["nombre"], t.get("ruta_etiqueta"), t.get("nombre_etiqueta", ""),
                               t["dispositivo"], progreso)
            t.update(estado="listo", resumen=r.resumen, binarios=r.binarios, fin=time.time(), progreso=1.0)
        except pre.ArchivoNoValido as e:
            t.update(estado="error", error=str(e), fin=time.time())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            t.update(estado="error", fin=time.time(),
                     error=f"Fallo inesperado al procesar el volumen ({type(e).__name__}: {str(e)[:300]}).")
        finally:
            for clave in ("ruta_imagen", "ruta_etiqueta"):
                ruta = t.get(clave)
                if ruta and t.get("borrar_al_terminar"):
                    shutil.rmtree(Path(ruta).parent, ignore_errors=True)
            limpiar()


def limpiar():
    """Olvida los trabajos terminados más antiguos y borra las subidas abandonadas."""
    with cerrojo:
        hechos = sorted((t for t in trabajos.values() if t["estado"] in ("listo", "error")), key=lambda t: t["creado"])
        for t in hechos[:-TRABAJOS_GUARDADOS]:
            trabajos.pop(t["id"], None)
        ahora = time.time()
        for sid, s in list(subidas.items()):
            if ahora - s["creado"] > CADUCIDAD_S and not s.get("en_uso"):
                shutil.rmtree(s["carpeta"], ignore_errors=True)
                subidas.pop(sid, None)


# ---------------------------------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------------------------------

async def estado(request: Request):
    import torch
    m = motor.config
    return JSONResponse({
        "aviso": AVISO, "dispositivos": motor.dispositivos(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "modelo": {"archivo": m.modelo, "desplazamientos": list(m.desplazamientos), "compuerta": m.compuerta,
                   "umbral_centro": m.umbral_centro, "enlace": m.enlace, "umbral_mm3": m.umbral_mm3},
        "demos": DEMOS, "en_cola": sum(t["estado"] in ("en_cola", "procesando") for t in trabajos.values()),
        "formatos": list(pre.EXTENSIONES), "tamano_max": TAMANO_MAX, "trozo": TROZO,
    })


async def nueva_subida(request: Request):
    datos = await request.json()
    nombre, tamano = str(datos.get("nombre", "")), int(datos.get("tamano", 0))
    if pre.extension_valida(nombre) is None:
        return error(f"«{nombre}» no es un formato admitido. Se aceptan {', '.join(pre.EXTENSIONES)}.")
    if not 0 < tamano <= TAMANO_MAX:
        return error(f"Tamaño no admitido ({tamano / 1e6:.0f} MB; máximo {TAMANO_MAX / 1e9:.0f} GB).")
    sid = uuid.uuid4().hex
    carpeta = TEMPORAL / "subidas" / sid
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre_seguro(nombre)
    ruta.touch()
    with cerrojo:
        subidas[sid] = {"id": sid, "nombre": Path(nombre).name, "ruta": ruta, "carpeta": carpeta, "tamano": tamano,
                        "recibido": 0, "siguiente": 0, "cerrada": False, "creado": time.time()}
    return JSONResponse({"id": sid, "trozo": TROZO})


async def recibir_trozo(request: Request):
    s = subidas.get(request.path_params["sid"])
    if s is None:
        return error("Subida desconocida o caducada: vuelve a seleccionar el archivo.", 404)
    n = int(request.path_params["n"])
    if s["cerrada"]:
        return error("La subida ya estaba cerrada.")
    if n < s["siguiente"]:                                  # un reintento de un trozo que ya llegó
        return JSONResponse({"recibido": s["recibido"]})
    if n != s["siguiente"]:
        return error(f"Trozo fuera de orden: llegó el {n} y se esperaba el {s['siguiente']}.", 409)
    if int(request.headers.get("content-length", 0)) > TROZO_MAX:
        return error("Trozo demasiado grande.", 413)
    cuerpo = await request.body()
    if len(cuerpo) > TROZO_MAX:
        return error("Trozo demasiado grande.", 413)
    if request.headers.get("x-comprimido") == "gzip":
        try:
            cuerpo = await run_in_threadpool(zlib.decompress, cuerpo, 31)
        except zlib.error:
            return error("El trozo comprimido llegó dañado.", 422)
    if s["recibido"] + len(cuerpo) > s["tamano"]:
        return error("El archivo es más grande de lo anunciado.", 422)

    def escribir():
        with open(s["ruta"], "ab") as f:
            f.write(cuerpo)
    await run_in_threadpool(escribir)
    s["recibido"] += len(cuerpo)
    s["siguiente"] = n + 1
    return JSONResponse({"recibido": s["recibido"]})


async def cerrar_subida(request: Request):
    s = subidas.get(request.path_params["sid"])
    if s is None:
        return error("Subida desconocida o caducada.", 404)
    if s["recibido"] != s["tamano"]:
        return error(f"Faltan datos: llegaron {s['recibido']:,} de {s['tamano']:,} bytes.", 422)
    s["cerrada"] = True
    return JSONResponse({"id": s["id"], "nombre": s["nombre"], "tamano": s["tamano"]})


async def nuevo_trabajo(request: Request):
    datos = await request.json()
    pendientes = sum(t["estado"] in ("en_cola", "procesando") for t in trabajos.values())
    if pendientes >= MAX_EN_COLA:
        return error("Hay demasiados estudios en cola; inténtalo en un par de minutos.", 503)
    dispositivo = datos.get("dispositivo", "cuda")
    t = {"id": uuid.uuid4().hex, "estado": "en_cola", "etapa": None, "texto": "En cola", "progreso": 0.0,
         "creado": time.time(), "dispositivo": dispositivo if dispositivo in motor.dispositivos() else "cpu"}
    if datos.get("demo"):
        caso = str(datos["demo"])
        if caso not in {d["caso"] for d in DEMOS}:
            return error("Caso de ejemplo desconocido.")
        t.update(ruta_imagen=RAIZ / "data" / "raw" / "images" / f"{caso}.mha", nombre=f"{caso}.mha", demo=caso)
        etiqueta = RAIZ / "data" / "raw" / "labels" / f"{caso}.mha"
        if datos.get("con_etiqueta", True) and etiqueta.exists():
            t.update(ruta_etiqueta=etiqueta, nombre_etiqueta=f"{caso}.mha (etiqueta)")
    else:
        s = subidas.get(str(datos.get("imagen")))
        if s is None or not s["cerrada"]:
            return error("La imagen no se terminó de subir.")
        s["en_uso"] = True
        t.update(ruta_imagen=s["ruta"], nombre=s["nombre"], borrar_al_terminar=True)
        if datos.get("etiqueta"):
            e = subidas.get(str(datos["etiqueta"]))
            if e is None or not e["cerrada"]:
                return error("La etiqueta no se terminó de subir.")
            e["en_uso"] = True
            t.update(ruta_etiqueta=e["ruta"], nombre_etiqueta=e["nombre"])
    with cerrojo:
        trabajos[t["id"]] = t
    cola.put(t["id"])
    return JSONResponse({"id": t["id"]})


def publico(t: dict) -> dict:
    delante = sum(1 for o in trabajos.values() if o["estado"] in ("en_cola", "procesando") and o["creado"] < t["creado"])
    salida = {k: t.get(k) for k in ("id", "estado", "etapa", "texto", "progreso", "error", "nombre", "dispositivo")}
    salida["delante"] = delante if t["estado"] == "en_cola" else 0
    if t.get("inicio"):
        salida["segundos"] = round((t.get("fin") or time.time()) - t["inicio"], 1)
    if t["estado"] == "listo":
        salida["resumen"] = t["resumen"]
    return salida


async def ver_trabajo(request: Request):
    t = trabajos.get(request.path_params["tid"])
    if t is None:
        return error("Ese análisis ya no está disponible (el servidor guarda solo los últimos).", 404)
    return JSONResponse(publico(t))


async def datos_trabajo(request: Request):
    t = trabajos.get(request.path_params["tid"])
    nombre = request.path_params["nombre"]
    if t is None or t["estado"] != "listo" or nombre not in t["binarios"]:
        return error("No disponible.", 404)
    return Response(t["binarios"][nombre], media_type="application/octet-stream",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "private, max-age=3600"})


def segmentacion_mha(t: dict) -> bytes:
    """Los fragmentos 3D predichos como .mha, en la rejilla del modelo (1,5 mm, LPS) y en su posición física real."""
    r = t["resumen"]
    vol = np.frombuffer(gzip.decompress(t["binarios"]["pred"]), np.uint8).reshape(r["forma"])
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing([r["espaciado_mm"]] * 3)
    img.SetDirection(r["geometria"]["direccion"])
    img.SetOrigin(r["geometria"]["origen_recorte"])
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "segmentacion.mha"
        sitk.WriteImage(img, str(ruta), useCompression=True)
        return ruta.read_bytes()


async def descarga(request: Request):
    t = trabajos.get(request.path_params["tid"])
    archivo = request.path_params["archivo"]
    if t is None or t["estado"] != "listo":
        return error("No disponible.", 404)
    base = Path(t["nombre"]).name.split(".")[0]
    if archivo == "segmentacion.mha":
        datos = await run_in_threadpool(segmentacion_mha, t)
        return Response(datos, media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{base}_fragmentos.mha"'})
    if archivo == "fragmentos.csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["hueso", "etiqueta", "principal", "volumen_mm3", "separacion_mm", "z0_mm", "z1_mm"])
        for f in t["resumen"]["fragmentos"]:
            w.writerow([f["nombre_hueso"], f["etiqueta"], f["principal"], f["volumen_mm3"], f["separacion_mm"],
                        round(f["z0"] * 1.5, 1), round(f["z1"] * 1.5, 1)])
        return Response(buf.getvalue().encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{base}_fragmentos.csv"'})
    return error("Descarga desconocida.", 404)


async def pagina(request: Request):
    return FileResponse(AQUI / "web" / "index.html", headers={"Cache-Control": "no-cache"})


rutas = [
    Route("/", pagina),
    Route("/api/estado", estado),
    Route("/api/subidas", nueva_subida, methods=["POST"]),
    Route("/api/subidas/{sid}/trozo/{n:int}", recibir_trozo, methods=["PUT"]),
    Route("/api/subidas/{sid}/fin", cerrar_subida, methods=["POST"]),
    Route("/api/trabajos", nuevo_trabajo, methods=["POST"]),
    Route("/api/trabajos/{tid}", ver_trabajo),
    Route("/api/trabajos/{tid}/datos/{nombre}", datos_trabajo),
    Route("/api/trabajos/{tid}/descarga/{archivo}", descarga),
    Mount("/static", StaticFiles(directory=AQUI / "web"), name="static"),
]
app = Starlette(routes=rutas)


def main():
    ap = argparse.ArgumentParser(description="App PENGWIN 2.5D: fragmentos óseos pélvicos en TC")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--puerto", type=int, default=8000)
    args = ap.parse_args()
    TEMPORAL.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(TEMPORAL / "subidas", ignore_errors=True)          # lo que quedara de una ejecución anterior
    threading.Thread(target=procesar, daemon=True, name="pipeline").start()
    print(f"Modelo: {motor.ruta_modelo.name} · dispositivos: {', '.join(motor.dispositivos())} · "
          f"{len(DEMOS)} casos de ejemplo")
    print(f"App en http://localhost:{args.puerto}   (Ctrl+C para parar)")
    uvicorn.run(app, host=args.host, port=args.puerto, log_level="warning")


if __name__ == "__main__":
    main()
