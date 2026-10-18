# La app PENGWIN 2.5D

Una aplicación web que recibe una tomografía de pelvis, la procesa con el modelo del proyecto y muestra el resultado en los tres visualizadores del enunciado. Está hecha para que **cualquiera con el enlace suba su propio archivo**: no enseña resultados ya calculados, ejecuta el pipeline completo sobre lo que le den.

> ⚠️ **Uso exclusivamente académico.** No es un dispositivo médico, no ha sido validado clínicamente y no debe utilizarse para apoyar decisiones quirúrgicas reales. El aviso está siempre visible en la página.

## Arrancarla

Desde la carpeta de la entrega, con el entorno de `requirements.txt`:

```bash
python app/servidor.py                       # http://localhost:8000
python app/servidor.py --puerto 8080         # en otro puerto
python app/servidor.py --host 0.0.0.0        # accesible desde otros equipos de la red local
```

Al arrancar dice qué modelo carga, en qué dispositivos puede inferir y cuántos casos de ejemplo encontró. Para publicarla en internet sin abrir puertos, un túnel rápido de Cloudflare (`cloudflared` se instala aparte):

```bash
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` imprime una dirección `https://….trycloudflare.com` que funciona en cualquier navegador, también en el celular.

**Qué necesita para funcionar:** `checkpoints/modelo_final_25d.pth` y `checkpoints/configuracion_inferencia.json` (los dos los deja el notebook). `data/raw/` es opcional: si está, la app ofrece los 15 pacientes de prueba como casos de ejemplo; si no, solo acepta archivos subidos.

## Cómo se usa

1. **Subir una TC** en `.mha`, `.nii`, `.nii.gz` o `.nrrd` —arrastrándola o eligiéndola—, y si se quiere su **máscara de fragmentos** en el campo opcional: entonces la app compara la predicción con ese ground truth. O elegir uno de los **15 pacientes de prueba**, que se analizan con su etiqueta.
2. Elegir **GPU o CPU** (si el servidor tiene GPU) y pulsar **Analizar**. La página enseña la subida (MB, velocidad), la posición en la cola si hay otros estudios delante y cada etapa del procesamiento.
3. El resultado: unos indicadores (regiones detectadas, fragmentos, conminutos desplazados, la mayor separación, la latencia y, con etiqueta, el Dice de fragmento) y seis pestañas:

| Pestaña | Qué muestra |
|---|---|
| **1 · Volumen crudo** | Visualizador 1: la MIP del volumen original girando alrededor del eje craneocaudal, sin la camilla. Umbral en HU ajustable en vivo (100–700, por defecto 250); se gira con el deslizador, con ▶ o arrastrando sobre la imagen |
| **2 · Corte a corte** | Visualizador 2: cada corte en la ventana de hueso con los fragmentos coloreados por su hueso (un tono por fragmento, bordes blancos), la caja de cada región con su puntuación y, sobre cada fragmento, **su número y su separación en mm** respecto al principal de su hueso. Axial, coronal o sagital; *slider*, rueda del ratón, flechas o arrastre vertical; opacidad; la etiqueta real al lado, si se subió |
| **3 · Reconstrucción 3D** | Visualizador 3: una malla por fragmento (*marching cubes*) en Plotly `Mesh3d`, con el color de su hueso y un rótulo con su separación; vistas de frente, atrás, arriba y lateral; principales semitransparentes para ver los conminutos; la etiqueta real, si se subió |
| **Fragmentos** | La tabla de fragmentos 3D: hueso, número, principal o conminuto (en contacto o desplazado), volumen, separación y extensión en z. Una fila lleva al visualizador 2, al corte donde el fragmento es mayor |
| **Frente al ground truth** | Solo con etiqueta: Dice e IoU de fragmento corte a corte, F1 de reconocimiento, Dice 3D, IoU de detección, clasificación del corte y la separación predicha frente a la real de cada fragmento |
| **Ficha técnica** | La cabecera del archivo, cada paso del preprocesado que se le aplicó, el modelo y las decisiones de validación, y el tiempo de cada etapa con la latencia por corte |

Se pueden descargar la **segmentación** (`.mha` con los fragmentos 3D en la codificación de PENGWIN, en la rejilla de 1,5 mm del modelo y en su posición física real: se superpone a la TC original en 3D Slicer o ITK-SNAP) y la **tabla de fragmentos** (`.csv`). La dirección de la página lleva el análisis (`#analisis=…`): recargarla lo recupera mientras el servidor lo guarde (los 8 últimos).

## Cómo está hecha

```
app/
├── servidor.py              Starlette + uvicorn: la página, la API, la cola y el hilo que procesa
├── motor/                   el pipeline, sin interfaz (se puede usar desde Python)
│   ├── preprocesado.py      leer, validar, inspeccionar; orientar, remuestrear, recortar
│   ├── red.py               PelvisNetCompleta y cargar_modelo()
│   ├── pipeline.py          pila 2.5D, ventaneo, NMS, decodificación en dos etapas, enlace 3D, separación, evaluación
│   ├── visual.py            MIP rotatoria, ventana de hueso, mallas
│   └── servicio.py          Motor.analizar(): todas las etapas, con progreso
└── web/
    ├── index.html, estilo.css
    ├── js/app.js            la página: subida, cola, progreso, indicadores, tablas, ficha
    ├── js/subida.js         subida en trozos comprimidos, con reintentos
    ├── js/visor1.js         MIP con umbral en vivo (paleta «bone»)
    ├── js/visor2.js         visor de cortes en canvas: tres planos, cajas, rótulos con la separación
    ├── js/visor3.js         mallas con Plotly
    └── vendor/plotly.min.js Plotly.js 4.1.1, local: la app no depende de ningún CDN
```

**Por qué así y no Streamlit.** Lo que la app tiene que hacer bien es recibir archivos de cientos de megas a través de un túnel, procesarlos de uno en uno en la GPU y dejar que el navegador recorra cientos de cortes sin esperar al servidor en cada movimiento. Un servidor pequeño con una API propia lo resuelve directamente: la subida va en trozos, el trabajo pesado va en un hilo con su cola, y el navegador recibe los volúmenes una sola vez, comprimidos, y dibuja él mismo cada corte en un `<canvas>`. Mover el *slider* o el umbral no cuesta ninguna petición.

**El motor es autocontenido.** `motor/` lleva su propia copia de la arquitectura, del preprocesado y del pipeline: no lee el notebook ni depende de que esté abierto. Para que esa copia no se separe del desarrollo, el notebook (sección 5.2) la comprueba en cada ejecución con igualdad exacta: el `.mha` crudo preparado por la app da bit a bit los cortes del caché; la red de la app da las mismas salidas; el pipeline, las mismas etiquetas, el mismo volumen 3D y las mismas separaciones. Las decisiones tomadas en validación —la pila de cortes, la compuerta, el umbral de los centros, el enlace 3D y el volumen mínimo de un fragmento— las lee de `checkpoints/configuracion_inferencia.json`, que escribe el notebook (5.1).

### El recorrido de un estudio

| Etapa | Dónde | Qué hace |
|---|---|---|
| Subida | `web/js/subida.js` → `PUT /api/subidas/{id}/trozo/{n}` | Trozos de 16 MB, comprimidos con `CompressionStream("gzip")` si el navegador lo permite (una TC sin comprimir ocupa unas 3,5 veces menos); cada trozo se reintenta hasta 5 veces si la red falla; el servidor los acepta en orden y un reintento de uno ya recibido no se duplica |
| Lectura y validación | `preprocesado.leer()`, `inspeccionar()` | Formato admitido, volumen 3D escalar (un 4D de un solo instante se acepta), tamaño máximo; rechaza las máscaras subidas como TC; corrige y avisa si los valores no están en HU; avisa de cortes gruesos, espaciados raros, campos pequeños, volúmenes cortos o sin hueso |
| Preparación | `preprocesado.preparar_imagen()` | Los pasos de `build_cache.py`: `DICOMOrient` a LPS, remuestreo lineal a 1,5 mm con relleno de aire, recorte 256×256 en la mediana del hueso > 200 HU |
| Etiqueta (opcional) | `preprocesado.preparar_etiqueta()` | Tiene que compartir rejilla con la imagen y estar en 0–30; se remuestrea por vecino más cercano y se recorta igual |
| Inferencia | `pipeline.inferir_volumen()` | Todos los cortes del volumen, en lotes de 16: pila de 5 cortes dentro del volumen, tres ventanas, red en bf16 (GPU) o fp32 (CPU), mejor caja por región tras el NMS, compuerta e instancias |
| Fragmentos 3D | `pipeline.enlazar_3d()`, `fragmentos_del_caso()` | Enlace por parejas con una pieza conexa por fragmento, filtro de 1.600 mm³, separación con `distance_transform_edt` a 1,5 mm |
| Evaluación | `pipeline.evaluar()` | Con etiqueta: las métricas de este paciente y la separación frente a la real |
| Visualizadores | `visual.rotacion_mip()`, `visual.mallas()` | 36 fotogramas de MIP en HU (giro con `grid_sample`, en GPU si la hay); una malla por fragmento |
| Empaquetado | `servicio.Motor.analizar()` | Los volúmenes para el navegador, en gzip: MIP (int16), cortes en la ventana de hueso (uint8), etiquetas predichas y reales (uint8) y mallas (JSON). Unos 5–8 MB por estudio |

### La API

| Método y ruta | Qué hace |
|---|---|
| `GET /api/estado` | Dispositivos, GPU, configuración del modelo, casos de ejemplo, estudios en cola, formatos y tamaño máximo |
| `POST /api/subidas` | Abre una subida: `{nombre, tamano}` → `{id, trozo}` |
| `PUT /api/subidas/{id}/trozo/{n}` | Un trozo, en orden; `X-Comprimido: gzip` si viene comprimido |
| `POST /api/subidas/{id}/fin` | Cierra la subida y comprueba el tamaño |
| `POST /api/trabajos` | `{imagen: id, etiqueta: id?, dispositivo}` o `{demo: caso, dispositivo}` → `{id}` |
| `GET /api/trabajos/{id}` | Estado, etapa, progreso, posición en la cola, error; el resumen completo cuando termina |
| `GET /api/trabajos/{id}/datos/{mip,hu,pred,real,mallas}` | Los volúmenes para el navegador (`Content-Encoding: gzip`) |
| `GET /api/trabajos/{id}/descarga/{segmentacion.mha,fragmentos.csv}` | Las descargas |

### Robustez

| Situación | Qué hace la app |
|---|---|
| Archivo de más de 100 MB detrás del túnel | Lo sube en trozos de 16 MB, comprimidos |
| Corte de red durante la subida | Reintenta el mismo trozo con espera creciente; el servidor no duplica un trozo repetido |
| Una máscara subida como si fuera la TC | La rechaza y explica que va en el campo «etiqueta» |
| Archivo dañado, truncado o que no es un volumen | Lo rechaza con un mensaje claro |
| Formato no admitido | Lo rechaza antes de subirlo |
| TC guardada sin desplazar a HU (aire ≈ 0) | Resta 1024 y avisa |
| Cortes gruesos, espaciado o campo de visión inusuales, volumen corto, sin hueso | Procesa y avisa de que se sale de lo que vio el modelo |
| Etiqueta de otro paciente o con otra rejilla | Procesa la imagen, descarta la etiqueta y avisa |
| Orientación distinta de LPS (p. ej. RAI, o girada 180°) | La reorienta antes de nada, como en el entrenamiento |
| Varios estudios a la vez | Cola de un solo hilo (una GPU), con posición visible; como mucho 4 en espera |
| Sin GPU | Todo en CPU, más despacio (unos 90 ms por corte frente a ~20 ms) |
| Celular | Diseño adaptable; el visor de cortes se mueve arrastrando; el 3D con el dedo |

Todas estas situaciones se probaron con archivos reales o fabricados a partir de los de PENGWIN (ver la documentación técnica). El servidor guarda los 8 últimos análisis en memoria y borra los archivos subidos al terminar de procesarlos; las subidas abandonadas se borran a las dos horas.

## Usar el motor sin la página

```python
import sys
from pathlib import Path
sys.path.insert(0, "app")
from motor.servicio import Motor

motor = Motor(Path("."))                     # la carpeta de la entrega
r = motor.analizar(Path("data/raw/images/005.mha"), "005.mha", Path("data/raw/labels/005.mha"), "005.mha", "cuda")
print(r.resumen["fragmentos"])               # una fila por fragmento 3D, con su separación en mm
print(r.resumen["evaluacion"]["dice_fragmento"])
```
