// Utilidades comunes: llamadas a la API, formato de números en español y colores de los fragmentos.

export const HUESO = { 1: "sacro", 2: "coxal izquierdo", 3: "coxal derecho" };
export const ABREV = { 1: "sacro", 2: "coxal izq.", 3: "coxal der." };
export const COLOR = { 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a" };

export async function api(ruta, opciones = {}) {
  const r = await fetch(ruta, {
    headers: opciones.body && typeof opciones.body === "string" ? { "Content-Type": "application/json" } : undefined,
    ...opciones,
  });
  let datos = null;
  try { datos = await r.json(); } catch { /* respuesta vacía */ }
  if (!r.ok) {
    const e = new Error((datos && datos.error) || `El servidor respondió ${r.status}`);
    e.codigo = r.status;
    throw e;
  }
  return datos;
}

export async function binario(ruta) {
  const r = await fetch(ruta);
  if (!r.ok) throw new Error(`No se pudo descargar ${ruta} (${r.status})`);
  return r.arrayBuffer();          // el navegador ya deshace el gzip (Content-Encoding)
}

const fmt = {};
export function num(v, decimales = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const clave = decimales;
  fmt[clave] ??= new Intl.NumberFormat("es-ES", { minimumFractionDigits: decimales, maximumFractionDigits: decimales });
  return fmt[clave].format(v);
}
export const entero = (v) => num(v, 0);
export function mb(bytes) { return `${num(bytes / 1e6, bytes > 1e8 ? 0 : 1)} MB`; }
export function segundos(s) { return s < 60 ? `${num(s, s < 10 ? 1 : 0)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`; }

function rgb(hex) { return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)); }

// El color de un fragmento: el de su hueso, más claro cuanto mayor es su número (como en el motor)
const cacheColor = new Map();
export function colorFragmento(v) {
  if (cacheColor.has(v)) return cacheColor.get(v);
  const base = rgb(COLOR[Math.floor((v - 1) / 10) + 1]).map((c) => c / 255);
  const t = Math.min(0.22 * ((v - 1) % 10), 0.66);
  const c = base.map((b) => Math.round(255 * (b + (1 - b) * t)));
  cacheColor.set(v, c);
  return c;
}
export const hex = (c) => "#" + c.map((x) => x.toString(16).padStart(2, "0")).join("");
export const huesoDe = (v) => Math.floor((v - 1) / 10) + 1;

export function el(etiqueta, atributos = {}, ...hijos) {
  const n = document.createElement(etiqueta);
  for (const [k, v] of Object.entries(atributos)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k.startsWith("aria-")) n.setAttribute(k, String(v));
    else if (v !== false && v !== null && v !== undefined) n.setAttribute(k, v === true ? "" : v);
  }
  for (const h of hijos.flat()) if (h !== null && h !== undefined && h !== false) n.append(h.nodeType ? h : document.createTextNode(h));
  return n;
}

export function textoSeparacion(f) {
  return f.principal ? "principal" : `${num(f.separacion_mm, 1)} mm`;
}

// Componente conexa mayor de un valor en una imagen 2D: su centroide (para colocar el rótulo, como el notebook)
export function centroMayorPieza(img, ancho, alto, valor, visitado) {
  let mejor = null, mejorN = 0;
  visitado.fill(0);
  const pila = new Int32Array(ancho * alto);
  for (let p = 0; p < ancho * alto; p++) {
    if (img[p] !== valor || visitado[p]) continue;
    let n = 0, sx = 0, sy = 0, tope = 0;
    pila[tope++] = p; visitado[p] = 1;
    while (tope) {
      const q = pila[--tope];
      const x = q % ancho, y = (q / ancho) | 0;
      n++; sx += x; sy += y;
      if (x > 0 && img[q - 1] === valor && !visitado[q - 1]) { visitado[q - 1] = 1; pila[tope++] = q - 1; }
      if (x < ancho - 1 && img[q + 1] === valor && !visitado[q + 1]) { visitado[q + 1] = 1; pila[tope++] = q + 1; }
      if (y > 0 && img[q - ancho] === valor && !visitado[q - ancho]) { visitado[q - ancho] = 1; pila[tope++] = q - ancho; }
      if (y < alto - 1 && img[q + ancho] === valor && !visitado[q + ancho]) { visitado[q + ancho] = 1; pila[tope++] = q + ancho; }
    }
    if (n > mejorN) { mejorN = n; mejor = [sx / n, sy / n]; }
  }
  return mejor ? { x: mejor[0], y: mejor[1], n: mejorN } : null;
}
