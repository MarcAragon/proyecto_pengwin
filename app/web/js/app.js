// La página: elegir un estudio (subido o de ejemplo), seguir su procesamiento y mostrar el resultado.

import { ABREV, COLOR, HUESO, api, binario, el, entero, mb, num, segundos } from "./util.js";
import { subir } from "./subida.js";
import { crearVisor1 } from "./visor1.js";
import { crearVisor2 } from "./visor2.js";
import { crearVisor3 } from "./visor3.js";

const $ = (id) => document.getElementById(id);
const estado = { servidor: null, imagen: null, etiqueta: null, dispositivo: null, ocupado: false, visores: {} };

const ETAPAS = [
  ["subida", "Subiendo el archivo"], ["cola", "En cola"], ["lectura", "Leyendo y comprobando el archivo"],
  ["preparacion", "Orientando (LPS), remuestreando a 1,5 mm y recortando"], ["inferencia", "Red corte a corte (pila de 5 cortes)"],
  ["enlace", "Fragmentos 3D y separación en mm"], ["evaluacion", "Comparando con la etiqueta"],
  ["mip", "MIP del volumen crudo"], ["mallas", "Mallas 3D"], ["empaquetado", "Preparando la vista"],
];

// ------------------------------------------------------------------------------------------------ arranque
async function iniciar() {
  try {
    const s = await api("/api/estado");
    estado.servidor = s;
    const chip = $("estado-servidor");
    chip.replaceChildren(el("span", { class: "punto punto--ok" }),
      el("span", {}, s.gpu ? `GPU · ${s.gpu}` : "Solo CPU"));
    pintarDispositivos(s);
    pintarDemos(s.demos);
  } catch (e) {
    $("estado-servidor").replaceChildren(el("span", { class: "punto punto--error" }), el("span", {}, "Sin conexión con el servidor"));
  }
  prepararSoltar("soltar-imagen", "archivo-imagen", "nombre-imagen", "imagen");
  prepararSoltar("soltar-etiqueta", "archivo-etiqueta", "nombre-etiqueta", "etiqueta");
  $("boton-analizar").onclick = analizarSubida;
  prepararPestanas();
  const previo = new URLSearchParams(location.hash.slice(1)).get("analisis");
  if (previo) seguir(previo, false).catch(() => history.replaceState(null, "", location.pathname));
}

function pintarDispositivos(s) {
  const cont = $("dispositivo");
  const opciones = s.dispositivos.map((d) => [d, d === "cuda" ? "GPU" : "CPU"]);
  estado.dispositivo = s.dispositivos[0];
  cont.replaceChildren(...opciones.map(([d, t]) => el("button", {
    "data-valor": d, "aria-pressed": d === estado.dispositivo, title: d === "cuda" ? s.gpu : "Procesador",
    onclick: (e) => {
      estado.dispositivo = d;
      cont.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b === e.currentTarget));
    },
  }, t)));
  cont.hidden = opciones.length < 2;
}

function pintarDemos(demos) {
  const cont = $("demos");
  if (!demos?.length) {
    cont.replaceChildren(el("span", { class: "nota" }, "No hay casos de ejemplo en este servidor (falta data/raw/)."));
    return;
  }
  cont.replaceChildren(...demos.map((d) => el("button", {
    class: "demo", onclick: () => analizarDemo(d.caso),
    title: `${d.tamano.join(" × ")} vóxeles · ${d.espaciado.join(" × ")} mm`,
  }, el("b", {}, `Caso ${d.caso}`), el("span", {}, `${d.mb} MB · ${d.tamano[0]} cortes`))));
}

function prepararSoltar(zonaId, inputId, nombreId, clave) {
  const zona = $(zonaId), input = $(inputId);
  const aceptar = (archivo) => {
    if (!archivo) return;
    const ok = (estado.servidor?.formatos || [".mha", ".nii.gz", ".nii", ".nrrd"]).some((e) => archivo.name.toLowerCase().endsWith(e));
    if (!ok) { $(nombreId).textContent = `«${archivo.name}» no es un formato admitido`; estado[clave] = null; }
    else { estado[clave] = archivo; $(nombreId).textContent = `${archivo.name} · ${mb(archivo.size)}`; }
    $("boton-analizar").disabled = !estado.imagen || estado.ocupado;
  };
  input.onchange = () => aceptar(input.files[0]);
  zona.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
  zona.ondragover = (e) => { e.preventDefault(); zona.classList.add("soltar--encima"); };
  zona.ondragleave = () => zona.classList.remove("soltar--encima");
  zona.ondrop = (e) => { e.preventDefault(); zona.classList.remove("soltar--encima"); aceptar(e.dataTransfer.files[0]); };
}

// ------------------------------------------------------------------------------------------------ progreso
function mostrarProgreso(titulo, conSubida, conEtiqueta) {
  $("progreso").hidden = false;
  $("progreso-titulo").textContent = titulo;
  $("progreso-error").hidden = true;
  $("progreso-tiempo").textContent = "";
  $("progreso-barra").style.width = "0%";
  requestAnimationFrame(() => $("progreso").scrollIntoView({ behavior: "smooth", block: "nearest" }));
  $("etapas").replaceChildren(...ETAPAS.filter(([k]) => (conSubida || k !== "subida") && (conEtiqueta || k !== "evaluacion"))
    .map(([k, t]) => el("li", { "data-etapa": k }, t)));
}

function marcarEtapa(clave, texto) {
  let pasada = true;
  for (const li of $("etapas").children) {
    if (li.dataset.etapa === clave) { li.className = "activa"; pasada = false; if (texto) li.textContent = texto; }
    else li.className = pasada ? "hecha" : "";
  }
}

function fallo(mensaje) {
  const p = $("progreso-error");
  p.hidden = false;
  p.textContent = mensaje;
  estado.ocupado = false;
  $("boton-analizar").disabled = !estado.imagen;
}

// ------------------------------------------------------------------------------------------------ lanzar
async function analizarSubida() {
  if (!estado.imagen || estado.ocupado) return;
  estado.ocupado = true;
  $("boton-analizar").disabled = true;
  mostrarProgreso(`Analizando ${estado.imagen.name}`, true, !!estado.etiqueta);
  try {
    const archivos = [estado.imagen, estado.etiqueta].filter(Boolean);
    const total = archivos.reduce((s, a) => s + a.size, 0);
    let previo = 0;
    const ids = [];
    for (const a of archivos) {
      ids.push(await subir(a, (p) => {
        const hecho = previo + p.hecho;
        $("progreso-barra").style.width = `${(100 * hecho) / total}%`;
        marcarEtapa("subida", `Subiendo ${a.name}: ${mb(hecho)} de ${mb(total)} · ${mb(p.velocidad)}/s`);
      }));
      previo += a.size;
    }
    const { id } = await api("/api/trabajos", {
      method: "POST", body: JSON.stringify({ imagen: ids[0], etiqueta: ids[1] || null, dispositivo: estado.dispositivo }),
    });
    await seguir(id, true);
  } catch (e) { fallo(e.message); }
}

async function analizarDemo(caso) {
  if (estado.ocupado) return;
  estado.ocupado = true;
  $("boton-analizar").disabled = true;
  mostrarProgreso(`Analizando el caso de prueba ${caso}`, false, true);
  try {
    const { id } = await api("/api/trabajos", {
      method: "POST", body: JSON.stringify({ demo: caso, con_etiqueta: true, dispositivo: estado.dispositivo }),
    });
    await seguir(id, true);
  } catch (e) { fallo(e.message); }
}

async function seguir(id, conProgreso) {
  const espera = (ms) => new Promise((r) => setTimeout(r, ms));
  let fallos = 0;
  for (;;) {
    let t;
    try { t = await api(`/api/trabajos/${id}`); fallos = 0; }
    catch (e) {
      if (e.codigo === 404 || ++fallos > 8) throw e;
      await espera(1500);
      continue;
    }
    if (conProgreso) {
      if (t.estado === "en_cola") marcarEtapa("cola", t.delante ? `En cola: ${t.delante} estudio${t.delante > 1 ? "s" : ""} delante` : "En cola");
      else if (t.etapa && t.etapa !== "listo") marcarEtapa(t.etapa);
      $("progreso-barra").style.width = `${100 * (t.progreso || 0)}%`;
      if (t.segundos) $("progreso-tiempo").textContent = segundos(t.segundos);
    }
    if (t.estado === "error") { if (!conProgreso) mostrarProgreso("Error", false, false); throw new Error(t.error); }
    if (t.estado === "listo") {
      if (conProgreso) { marcarEtapa("__fin"); $("progreso-titulo").textContent = "Cargando la vista…"; }
      await mostrarResultado(id, t.resumen);
      if (conProgreso) $("progreso").hidden = true;
      estado.ocupado = false;
      $("boton-analizar").disabled = !estado.imagen;
      return;
    }
    await espera(t.estado === "en_cola" ? 1200 : 500);
  }
}

// ------------------------------------------------------------------------------------------------ resultado
async function mostrarResultado(id, r) {
  const base = `/api/trabajos/${id}/datos/`;
  const [mip, hu, pred, real, mallas] = await Promise.all([
    binario(base + "mip"), binario(base + "hu"), binario(base + "pred"),
    r.tamanos.real ? binario(base + "real") : null, binario(base + "mallas"),
  ]);
  const vol = { hu: new Uint8Array(hu), pred: new Uint8Array(pred), real: real ? new Uint8Array(real) : null };
  const datosMallas = JSON.parse(new TextDecoder().decode(mallas));
  const pedida = new URLSearchParams(location.hash.slice(1)).get("pestana");
  history.replaceState(null, "", `#analisis=${id}`);

  $("resultados").hidden = false;
  $("resultado-titulo").textContent = `Resultado · ${r.nombre}`;
  $("resultado-subtitulo").textContent = `${r.forma[0]} cortes de 256 × 256 a 1,5 mm · ${r.dispositivo === "cuda" ? "GPU" : "CPU"}` +
    ` (${r.dispositivo_nombre}) · ${segundos(r.tiempos.total)} en el servidor` + (r.etiqueta ? ` · comparado con ${r.etiqueta}` : "");
  $("descarga-mha").href = `/api/trabajos/${id}/descarga/segmentacion.mha`;
  $("descarga-csv").href = `/api/trabajos/${id}/descarga/fragmentos.csv`;

  const avisos = $("avisos");
  avisos.hidden = !r.avisos.length;
  avisos.replaceChildren(...r.avisos.map((a) => el("li", {}, a)));

  pintarKpis(r);
  pintarTablaFragmentos(r);
  pintarEvaluacion(r);
  pintarFicha(r);

  estado.visores.v1?.parar();
  estado.visores.v2?.destruir();
  estado.visores.v1 = crearVisor1(r.mip, new Int16Array(mip));
  estado.visores.v2 = crearVisor2(r, vol);
  estado.visores.v3 = crearVisor3(datosMallas, !!vol.real);
  activarPestana(pedida && $(pedida) ? pedida : "panel-v1");
  $("resultados").scrollIntoView({ behavior: "smooth", block: "start" });
}

function kpi(valor, etiqueta, detalle) {
  return el("div", { class: "kpi" }, el("div", { class: "kpi__valor" }, valor), el("div", { class: "kpi__etiqueta" }, etiqueta),
    detalle ? el("div", { class: "kpi__detalle" }, detalle) : null);
}

function pintarKpis(r) {
  const f = r.fragmentos;
  const regiones = [1, 2, 3].filter((k) => f.some((x) => x.hueso === k));
  const desplazados = f.filter((x) => !x.principal && x.separacion_mm > 1.5 + 1e-6).sort((a, b) => b.separacion_mm - a.separacion_mm);
  const tiles = [
    kpi(`${regiones.length} de 3`, "regiones detectadas", regiones.map((k) => ABREV[k]).join(" · ") || "ninguna"),
    kpi(entero(f.length), "fragmentos 3D", regiones.map((k) => `${ABREV[k]} ${f.filter((x) => x.hueso === k).length}`).join(" · ")),
    kpi(entero(f.filter((x) => !x.principal).length), "fragmentos conminutos",
      `${desplazados.length} desplazado${desplazados.length === 1 ? "" : "s"} (> 1,5 mm)`),
    kpi(desplazados.length ? `${num(desplazados[0].separacion_mm, 1)} mm` : "–", "mayor separación",
      desplazados.length ? `${HUESO[desplazados[0].hueso]}, fragmento ${desplazados[0].etiqueta}` : "todos en contacto"),
    kpi(`${num(r.latencia.ms_por_corte, 0)} ms`, `por corte en ${r.dispositivo === "cuda" ? "GPU" : "CPU"}`,
      `${r.latencia.cortes} cortes · ${segundos(r.tiempos.total)} en total`),
  ];
  if (r.evaluacion?.dice_fragmento !== undefined && r.evaluacion?.dice_fragmento !== null) {
    tiles.push(kpi(num(r.evaluacion.dice_fragmento, 3), "Dice de fragmento", `frente a la etiqueta · 3D ${num(r.evaluacion.dice_3d, 3)}`));
  }
  $("kpis").replaceChildren(...tiles);
}

function pintarTablaFragmentos(r) {
  const t = $("tabla-fragmentos");
  const filas = r.fragmentos.map((f) => el("tr", { onclick: () => { activarPestana("panel-v2"); estado.visores.v2.irAFragmento(f.etiqueta); } },
    el("td", {}, el("span", { class: "etiqueta-hueso" }, el("span", { class: "muestra", style: `background:${f.color}` }), HUESO[f.hueso])),
    el("td", { class: "num" }, f.etiqueta),
    el("td", {}, f.principal ? el("span", { class: "chip" }, "principal")
      : f.separacion_mm > 1.5 + 1e-6 ? el("span", { class: "chip chip--desplazado" }, "conminuto · desplazado")
        : el("span", { class: "chip" }, "conminuto · en contacto")),
    el("td", { class: "num" }, num(f.volumen_mm3 / 1000, 1)),
    el("td", { class: "num" }, f.principal ? "—" : num(f.separacion_mm, 1)),
    el("td", { class: "num" }, `${num(f.z0 * 1.5, 0)} – ${num(f.z1 * 1.5, 0)}`)));
  t.replaceChildren(el("thead", {}, el("tr", {}, el("th", {}, "Hueso"), el("th", { class: "num" }, "Fragmento"), el("th", {}, "Tipo"),
    el("th", { class: "num" }, "Volumen (cm³)"), el("th", { class: "num" }, "Separación (mm)"), el("th", { class: "num" }, "Extensión en z (mm)"))),
  el("tbody", {}, filas.length ? filas : el("tr", {}, el("td", { colspan: 6 }, "No se encontró ningún fragmento óseo pélvico."))));
}

function pintarEvaluacion(r) {
  const ev = r.evaluacion;
  $("pestana-evaluacion").hidden = !ev;
  if (!ev) return;
  const acierto = Object.entries(ev.clasificacion_acierto).map(([k, v]) => `${k} ${num(100 * v, 1)} %`).join(" · ");
  $("kpis-evaluacion").replaceChildren(
    kpi(num(ev.dice_fragmento, 3), "Dice de fragmento", `corte a corte, ${entero(ev.fragmentos_2d)} fragmentos 2D`),
    kpi(num(ev.iou_fragmento, 3), "IoU de fragmento", "con la regla ignore (< 8 px)"),
    kpi(num(ev.f1_reconocimiento, 3), "F1 de reconocimiento", `${ev.no_encontrados_2d} no encontrados · ${ev.inventados_2d} inventados`),
    kpi(num(ev.dice_3d, 3), "Dice de fragmento 3D", `${ev.fragmentos_predichos} predichos · ${ev.fragmentos_reales} reales`),
    kpi(num(ev.iou_deteccion, 3), "IoU de detección", `${entero(ev.cajas_evaluadas)} cajas reales`),
    kpi(ev.error_separacion_medio_mm === null ? "–" : `${num(ev.error_separacion_medio_mm, 1)} mm`, "error medio de la separación",
      `${ev.conminutos_medidos} de ${ev.conminutos_reales} conminutos medidos`),
    kpi(ev.pegado_desplazado_bien === null ? "–" : `${num(100 * ev.pegado_desplazado_bien, 0)} %`, "pegado o desplazado, bien leído",
      `de los ${ev.conminutos_medidos} conminutos medidos`),
    kpi(`${num(100 * Math.min(...Object.values(ev.clasificacion_acierto)), 1)} %`, "clasificación del corte (peor hueso)", acierto));
  const filas = ev.tabla.map((t) => {
    const dif = t.separacion_pred_mm === null || t.principal_real ? null : t.separacion_pred_mm - t.separacion_real_mm;
    return el("tr", {},
      el("td", {}, el("span", { class: "etiqueta-hueso" }, el("span", { class: "muestra", style: `background:${COLOR[t.hueso]}` }), HUESO[t.hueso])),
      el("td", { class: "num" }, t.etiqueta_real), el("td", {}, t.principal_real ? "principal" : "conminuto"),
      el("td", { class: "num" }, num(t.volumen_real_mm3 / 1000, 1)),
      el("td", { class: "num" }, t.principal_real ? "—" : num(t.separacion_real_mm, 1)),
      el("td", { class: "num" }, t.etiqueta_pred || "no encontrado"), el("td", { class: "num" }, num(t.dice_3d, 3)),
      el("td", { class: "num" }, t.principal_real ? "—" : t.pareja_es_principal ? "fundido con el principal" : num(t.separacion_pred_mm, 1)),
      el("td", { class: "num" }, dif === null || t.pareja_es_principal ? "—" : `${dif >= 0 ? "+" : ""}${num(dif, 1)}`));
  });
  $("tabla-evaluacion").replaceChildren(el("thead", {}, el("tr", {}, ...["Hueso", "Fragmento real", "Tipo", "Volumen real (cm³)",
    "Separación real (mm)", "Fragmento predicho", "Dice 3D", "Separación predicha (mm)", "Diferencia (mm)"].map((h, i) =>
    el("th", { class: i === 0 || i === 2 ? "" : "num" }, h)))), el("tbody", {}, filas));
}

function pintarFicha(r) {
  const f = r.ficha, m = r.modelo, c = r.config, t = r.tiempos;
  const bloque = (titulo, pares) => el("div", { class: "tarjeta ficha" }, el("h3", {}, titulo),
    el("dl", {}, pares.flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, v)])));
  const ori = f.orientacion === "LPS" ? "LPS (sin cambios)" : `${f.orientacion} → LPS`;
  $("ficha").replaceChildren(
    bloque("El archivo", [["Nombre", f.nombre], ["Vóxeles (z × y × x)", f.tamano.join(" × ")],
      ["Espaciado (mm)", f.espaciado.map((s) => num(s, 3)).join(" × ")], ["Extensión (mm)", f.extension_mm.map((s) => num(s, 0)).join(" × ")],
      ["Orientación", f.orientacion], ["Tipo de dato", f.tipo], ["HU (p1 – p99)", `${f.hu_p1} – ${f.hu_p99}`],
      ["Vóxeles > 200 HU", `${num(f.hueso_pct, 2)} %`], ["Corrección de HU", f.correccion_hu ? `${f.correccion_hu}` : "ninguna"]]),
    bloque("Preprocesado (como build_cache.py)", [["1 · Orientar", ori],
      ["2 · Remuestrear", `a 1,5 mm isotrópico: ${r.forma_iso.join(" × ")} vóxeles (imagen lineal${r.etiqueta ? ", etiqueta por vecino más cercano" : ""})`],
      ["3 · Recortar", `256 × 256 (384 mm) centrado en la mediana del hueso > 200 HU, fila ${r.centro[0]}, columna ${r.centro[1]}`],
      ["4 · Entrada", `cada corte con sus vecinos ${c.desplazamientos.filter((d) => d).map((d) => (d > 0 ? "+" : "") + d).join(", ")}` +
        ` (±${num(1.5 * Math.max(...c.desplazamientos), 0)} mm), en 3 ventanas: 15 canales`], ["Cortes procesados", `${r.forma[0]} (el volumen entero)`]]),
    bloque("Modelo y decisiones de validación", [["Pesos", m.archivo], ["Parámetros", `${num(m.parametros / 1e6, 2)} M`],
      ["Época guardada", `${m.epoca} de ${m.epocas}`], ["En validación", `IoU ${num(m.iou_val, 3)} · Dice de fragmento ${num(m.dice_val, 3)}`],
      ["Decodificación", `caja como compuerta: ${c.compuerta ? "sí" : "no"} · umbral de centros ${num(c.umbral_centro, 1)}`],
      ["Enlace 3D", c.enlace], ["Fragmento mínimo", `${entero(c.umbral_mm3)} mm³`]]),
    bloque("Tiempos de este análisis", [["Dispositivo", `${r.dispositivo === "cuda" ? "GPU" : "CPU"} · ${r.dispositivo_nombre}`],
      ["Latencia por corte", `${num(r.latencia.ms_por_corte, 1)} ms (la red sola: ${num(r.latencia.ms_red_por_corte, 1)} ms)`],
      ...Object.entries({ lectura: "Lectura", preparacion: "Preprocesado", inferencia: "Red + decodificación", enlace: "Enlace 3D y separación",
        evaluacion: "Evaluación", mip: "MIP", mallas: "Mallas", empaquetado: "Empaquetado", total: "Total" })
        .filter(([k]) => t[k] !== undefined).map(([k, n]) => [n, segundos(t[k])])]));
}

// ------------------------------------------------------------------------------------------------ pestañas
function prepararPestanas() {
  for (const b of $("pestanas").querySelectorAll("button")) b.onclick = () => activarPestana(b.dataset.panel);
}
function activarPestana(panel) {
  for (const b of $("pestanas").querySelectorAll("button")) b.setAttribute("aria-selected", b.dataset.panel === panel);
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== panel;
  if (panel === "panel-v3") estado.visores.v3?.mostrar();
  if (panel === "panel-v2") estado.visores.v2?.actualizar();
  if (panel !== "panel-v1") estado.visores.v1?.parar();
}

iniciar();
