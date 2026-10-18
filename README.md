# Fragmentos óseos pélvicos en TC — Sistema PENGWIN 2.5D

**Detección, segmentación de instancia, clasificación y medición en milímetros de fragmentos óseos pélvicos en tomografía computarizada, con entrada 2.5D, y una aplicación web que lo aplica a cualquier `.mha`**

| | |
|---|---|
| Integrantes | Marco Aragon, Juan David Reyes, Maria Paula Pinillo, María José Suárez Toro y Laura Astudillo |
| Dataset | PENGWIN Task 1 (CT), MICCAI 2024 · [Zenodo 10927452](https://zenodo.org/records/10927452) · CC-BY 4.0 |
| Las dos partes | `pengwin_desarrollo.ipynb` (el desarrollo técnico, paso a paso) y `app/` (la aplicación) |
| Fecha | 2026-10-17 |

> ⚠️ **Uso exclusivamente académico.** Este sistema no es un dispositivo médico, no ha sido validado clínicamente y no debe utilizarse para apoyar decisiones quirúrgicas reales.

---

## Qué hay aquí

El proyecto tiene dos partes, y esta carpeta las separa:

1. **El desarrollo** — `pengwin_desarrollo.ipynb`. El trabajo técnico completo, en el orden en que se toman las decisiones: los datos, el diseño del modelo, su entrenamiento, su evaluación y el paso a la aplicación. Cinco capítulos y 28 decisiones de diseño numeradas, cada una con el dato que la justifica.
2. **La aplicación** — `app/`. Una aplicación web que recibe una TC en `.mha` (o `.nii`, `.nii.gz`, `.nrrd`), la preprocesa con las mismas funciones del notebook, la pasa por el modelo y muestra los tres visualizadores que pide el enunciado. Está pensada para que quien reciba el enlace suba sus propios archivos.

```
entrega final/
├── pengwin_desarrollo.ipynb       el desarrollo: datos → modelo → entrenamiento → evaluación → app
├── build_cache.py                 volúmenes .mha → caché de cortes (lo usa el notebook; la app lleva su propia copia de los pasos)
├── app/                           la aplicación web
│   ├── servidor.py                el servidor: python app/servidor.py → http://localhost:8000
│   ├── motor/                     el pipeline sin interfaz: preprocesado, red, inferencia, visualizadores
│   ├── web/                       la página (HTML, CSS, JavaScript y Plotly.js local)
│   └── README.md                  cómo funciona la app, por dentro
├── checkpoints/
│   ├── modelo_final_25d.pth       el modelo final: tres cabezas, pila de 5 cortes (lo usa la app)
│   ├── configuracion_inferencia.json   lo que se decidió en validación, para la app (lo escribe el notebook, 5.1)
│   ├── sam_vit_b_01ec64.pth       pesos de SAM ViT-B, solo para la comparación de 4.7
│   └── experimentos/              los demás modelos entrenados: las otras pilas (3.3) y la ablación (4.8)
├── data/                          raw/ (los .mha de Zenodo) y cache/ (lo construye build_cache.py); ver data/README.md
├── splits.json                    partición fija por paciente
├── requirements.txt               dependencias exactas
├── MODEL_CARD.md                  el model card del modelo final
├── DOCUMENTACION_TECNICA.md       referencia: parámetros, funciones, cifras, la API de la app, reproducción
└── DOCUMENTACION_NARRATIVA.md     el recorrido completo, capítulo a capítulo, con las salidas del notebook
```

## El sistema, en una línea

Cada corte axial entra al modelo con cuatro vecinos del mismo paciente (±6 mm), en tres ventanas de Hounsfield: 15 canales. Un tronco ResNet-18 con un CBAM por etapa y un FPN alimenta tres cabezas —clasificación del corte, detección con una grilla propia de 16×16 y NMS propio, y segmentación de instancia por centros y desplazamientos—; la caja de cada región acota sus fragmentos; los fragmentos se enlazan en 3D y se mide en milímetros la separación de cada uno al principal de su hueso.

## Resultados

Las siete métricas del §5 en el conjunto de **prueba** (15 pacientes que el modelo no vio ni para entrenar ni para elegir nada):

| | F1 macro | AUC macro | IoU | mAP@0,50 | mAP@[0,50:0,95] | Dice de fragmento | IoU de fragmento |
|---|---|---|---|---|---|---|---|
| Objetivo (§5) | 0,85 | 0,85 | 0,65 | 0,65 | 0,40 | 0,85 | 0,70 |
| **Prueba** | 0,994 | 0,999 | 0,884 | 0,978 | 0,767 | **0,848** ✗ (a 0,0025) | 0,797 |
| Validación | 0,992 | 0,999 | 0,883 | 0,979 | 0,766 | 0,855 | 0,809 |

| | Resultado |
|---|---|
| **Separación en mm** (§3.3, prueba) | De los 42 conminutos se miden 33; el **94 %** se lee bien como pegado o desplazado; error mediano 0 mm, el 88 % dentro de un vóxel |
| **SAM ViT-B** (§4.2, con nuestras cajas) | Silueta: 0,865 frente a 0,946 del modelo propio · fragmentos: 0,658 frente a 0,847 |
| **Ablación** (§7, en 2.5D, validación) | El CBAM mejora el ajuste de las cajas con las dos semillas (mAP@[0,50:0,95] +0,014, el triple del ruido entre semillas) y no cambia la segmentación; sin ImageNet la detección sale mejor (+0,021) |
| **Latencia** (§4.3, un corte) | 22,6 ms en GPU (RTX 5060 Ti, 44 por segundo) · 74,1 ms en CPU (Ryzen 7 8700F, 13,5 por segundo) |
| **Volumen entero** (la app) | Sobre los 15 `.mha` de prueba completos, ni un vóxel predicho fuera de la pelvis y el mismo Dice que sobre el caché |

## Cómo reproducirlo

```powershell
# 1 · Entorno (Python 3.12; PyTorch con CUDA 13, ver requirements.txt)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2 · Datos: descargar de Zenodo y extraer en data/raw/ (ver data/README.md)

# 3 · Caché: una sola vez, ~1,5 min (si falta, el notebook lo lanza solo)
python build_cache.py

# 4 · Pesos de SAM ViT-B para 4.7, si no están ya en checkpoints/ (375 MB):
#     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth → checkpoints/sam_vit_b_01ec64.pth

# 5 · El notebook de principio a fin, desde esta carpeta (con los modelos ya entrenados, unos 20 min)
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 pengwin_desarrollo.ipynb

# 6 · La app, en http://localhost:8000
python app/servidor.py

# 7 · Publicarla con un túnel rápido de Cloudflare (cloudflared instalado aparte)
cloudflared tunnel --url http://localhost:8000
```

**Cada modelo se entrena una sola vez.** Si falta un checkpoint (o está incompleto), la celda que lo necesita lo entrena y lo guarda —unos 65 minutos por modelo en una RTX 5060 Ti—; si existe, lo carga y muestra su registro. El entrenamiento en GPU no es reproducible bit a bit; toda la evaluación sobre un modelo guardado sí lo es. El notebook usa como raíz la carpeta desde la que se ejecuta (`Path.cwd()`).

## La app

```bash
python app/servidor.py                 # http://localhost:8000
python app/servidor.py --puerto 8080   # otro puerto
```

En la página se sube una TC —y, si se quiere, su máscara de fragmentos, para comparar con el ground truth— o se elige uno de los 15 pacientes de prueba. El servidor:

1. **recibe** el archivo en trozos de 16 MB, comprimidos en el navegador (el túnel de Cloudflare corta las peticiones de más de 100 MB, y una TC pesa hasta 420 MB);
2. **lo valida**: que sea un volumen 3D legible, que sea una TC y no una máscara, que esté en unidades Hounsfield (si no, lo corrige y avisa), y avisa de espaciados, campos de visión o grosores de corte poco habituales;
3. **lo prepara** con los pasos de `build_cache.py`: orientar a LPS, remuestrear a 1,5 mm isotrópico y recortar 256×256 centrado en el hueso;
4. **lo pasa por el modelo** corte a corte, con la pila de 5 cortes de cada uno, en GPU o en CPU;
5. **enlaza** los fragmentos en 3D y **mide** la separación de cada uno en milímetros;
6. si llegó una etiqueta, **evalúa**: Dice de fragmento, IoU de detección y la separación predicha frente a la real;
7. y prepara los **tres visualizadores**: la MIP rotatoria del volumen crudo, el visor corte a corte con la separación escrita sobre cada fragmento y la reconstrucción 3D con una malla por fragmento.

Se puede descargar la segmentación (`.mha`, en su posición física) y la tabla de fragmentos (`.csv`). Los detalles, en [`app/README.md`](app/README.md).

## Documentación

| Documento | Para qué |
|---|---|
| [`pengwin_desarrollo.ipynb`](pengwin_desarrollo.ipynb) | El desarrollo, ejecutado, con todas sus salidas |
| [`DOCUMENTACION_TECNICA.md`](DOCUMENTACION_TECNICA.md) | Referencia: cumplimiento del enunciado, parámetros, funciones, cifras, la API de la app y reproducción |
| [`DOCUMENTACION_NARRATIVA.md`](DOCUMENTACION_NARRATIVA.md) | El recorrido completo: qué se hace en cada capítulo, por qué y cómo se llega a cada resultado |
| [`MODEL_CARD.md`](MODEL_CARD.md) | El modelo final: arquitectura, datos, métricas, limitaciones y uso no clínico |
| [`app/README.md`](app/README.md) | La app por dentro: arquitectura, flujo de un estudio, robustez y despliegue |
| [`data/README.md`](data/README.md) | Los datos: qué hay, cómo obtenerlos y el esquema del caché |

## Declaraciones

- **Arquitectura propia.** Ningún framework de detección o segmentación de alto nivel (YOLO, Detectron2, Mask R-CNN preentrenado). El tronco tiene las formas de una ResNet-18 para poder cargar sus pesos de ImageNet (transfer learning solo en el tronco, §4.1); CBAM, *neck*, las tres cabezas, los objetivos, el NMS y el decodificador son propios. torchvision solo aporta los pesos de ImageNet y un oráculo para comprobar el NMS.
- **2.5D.** El modelo sigue siendo de convoluciones 2D y predice corte a corte, como pide el §5; la entrada lleva vecinos del mismo paciente (nunca de otro: la partición es por paciente).
- **Formato.** El enunciado habla de NIfTI; PENGWIN se distribuye en MetaImage (`.mha`). El espaciado se lee de la cabecera igual. La app acepta los dos.
- **SAM**, solo como referencia en el conjunto de prueba, con las cajas de nuestra detección como *prompt* (§4.2).
- **Partición de prueba** congelada desde la primera semana y abierta una sola vez, con todo ya decidido en validación. No se ha ajustado nada para corregir el Dice de prueba.
- **Atribución:** PENGWIN Task 1 (CT), MICCAI 2024 — Zenodo 10927452, CC-BY 4.0. SAM: Kirillov et al. (2023), pesos oficiales de Meta (Apache 2.0). Plotly.js (MIT).

