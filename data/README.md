# `data/` — los datos del proyecto

Todo lo que leen el notebook (`pengwin.ipynb`) y el script (`build_cache.py`) está aquí. Son unos 42 GB entre los volúmenes originales y el caché, así que **no viajan con la entrega**: quien la reciba los descarga de Zenodo y reconstruye el caché con una orden.

## Qué hay

```
data/
├── README.md                     este archivo
├── raw/                          los datos originales, tal como vienen de Zenodo
│   ├── images/NNN.mha            100 volúmenes de TC en unidades Hounsfield   18,8 GB
│   ├── labels/NNN.mha            100 máscaras de fragmentos (0–30)            12,6 GB
│   └── _zips/                    los tres zips descargados (opcional)          7,1 GB
└── cache/                        lo construye build_cache.py a partir de raw/
    ├── slices_hu.npy             15.626 cortes 256×256 en HU crudo (int16)     1,9 GB
    ├── labels.npy                sus máscaras, codificación 0–30 (uint8)       1,0 GB
    ├── index.parquet             una fila por corte (esquema abajo)           145 KB
    └── casos.csv                 una fila por caso: geometría y verificaciones  14 KB
```

`raw/_zips/` no lo usa nada del proyecto: es la descarga original, útil solo para volver a extraer los `.mha` sin descargarlos de nuevo. Puede borrarse si hace falta espacio.

## Si no tienes los datos

1. Descarga los tres zips de [Zenodo 10927452](https://zenodo.org/records/10927452) (7,1 GiB, licencia CC-BY 4.0) y comprueba sus MD5:

   | Archivo | Tamaño (B) | MD5 |
   |---|---|---|
   | `PENGWIN_CT_train_labels.zip` | 33.326.051 | `77b3a29d528c719a917e6f90eb9b20d4` |
   | `PENGWIN_CT_train_images_part1.zip` | 3.944.991.549 | `1d7cd8dc537ba12c61165f1be3e86022` |
   | `PENGWIN_CT_train_images_part2.zip` | 3.635.398.992 | `308a619a9c24fbcfbd197127c4cb39fb` |

2. Extrae los `.mha` de imágenes en `data/raw/images/` y los de etiquetas en `data/raw/labels/` (los nombres, `001.mha` … `100.mha`, coinciden entre ambas carpetas).
3. Construye el caché desde la carpeta del proyecto. Tarda en torno a minuto y medio, con un pico de 3,5 GB de RAM:

   ```powershell
   python build_cache.py
   ```

   Si el caché ya existe no lo toca; para reconstruirlo, `python build_cache.py --force` (con el notebook cerrado: Windows no deja reescribir un archivo que otro proceso tiene mapeado en memoria).

## Esquema de `cache/index.parquet`

Una fila por corte, en el mismo orden que `slices_hu.npy` y `labels.npy`: la columna `row` es el índice en esos arrays.

| Columna | Tipo | Significado |
|---|---|---|
| `row` | int32 | Índice del corte en `slices_hu.npy` / `labels.npy` |
| `case_id` | str | Paciente (`001`–`100`). **La partición se hace sobre esta columna** |
| `z` | int | Posición del corte en el volumen isotrópico del caso |
| `n_frag_sa` | int | Fragmentos de sacro presentes en el corte |
| `n_frag_lh` | int | Fragmentos de coxal izquierdo presentes en el corte |
| `n_frag_rh` | int | Fragmentos de coxal derecho presentes en el corte |
| `bones_present` | int | Máscara de bits: bit 0 sacro · bit 1 coxal izquierdo · bit 2 coxal derecho. Es el objetivo de la cabeza de clasificación multi-etiqueta |
| `n_frag_total` | int | Suma de las tres columnas `n_frag_*` |
| `n_regions` | int | Regiones anatómicas presentes (0–3) |
| `area_bone_px` | int | Píxeles con etiqueta > 0 |
| `has_bone` | int | 1 si el corte contiene hueso; 0 si es uno de los negativos contiguos |

Las filas de un mismo caso son contiguas y van en orden de `z`: apilando `labels.npy[r0:r1]` se recupera el volumen 3D completo del paciente, que es lo que usa el análisis 3D del notebook.

## Esquema de `cache/casos.csv`

Una fila por caso. Es el informe de verificación del script: el notebook lo usa para comprobar los 100 casos sin volver a leer los `.mha`.

| Columnas | Significado |
|---|---|
| `case_id` | Paciente |
| `nz`, `ny`, `nx` · `sz`, `sy`, `sx` | Tamaño y espaciado (mm) originales, en orden (z, y, x) |
| `direccion` | Matriz de dirección de la cabecera (identidad, o girada 180° en el plano) |
| `geometria_ok` | Imagen y etiqueta comparten tamaño, espaciado, origen y dirección |
| `n_frag_sa`, `n_frag_lh`, `n_frag_rh` | Fragmentos de cada hueso en la etiqueta original |
| `nz_iso`, `ny_iso`, `nx_iso` | Tamaño tras remuestrear a 1,5 mm isotrópico |
| `centro_y`, `centro_x` | Centro del recorte 256×256 (mediana del hueso por encima de 200 HU) |
| `hueso_mm_z`, `hueso_mm_y`, `hueso_mm_x` | Extensión del hueso anotado, en mm |
| `vox_hueso`, `vox_perdidos` | Vóxeles de hueso antes del recorte y cuántos se perdieron en él (debe ser 0) |
| `etiquetas_perdidas`, `etiquetas_inventadas` | Valores de etiqueta que desaparecieron o aparecieron al remuestrear y recortar (deben ser 0) |
| `cortes_hueso` · `z0`, `z1` | Cortes con hueso y rango guardado en el caché (`z0` incluido, `z1` excluido) |

## Al empaquetar la entrega

Los datos **no van en el zip**. Excluye `raw/` entero y los dos `.npy` de `cache/`. `index.parquet` y `casos.csv` pesan unos pocos KB y conviene conservarlos: describen el dataset curado sin necesidad de descargar nada.
