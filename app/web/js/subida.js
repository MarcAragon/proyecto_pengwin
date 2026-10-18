// Subida de un archivo en trozos. El túnel de Cloudflare corta cualquier petición de más de 100 MB y una TC de pelvis
// pesa hasta 420 MB, así que el archivo viaja en trozos de 16 MB, cada uno comprimido en el navegador con gzip
// (CompressionStream) cuando se puede: una TC sin comprimir ocupa unas 3,5 veces menos. Cada trozo se reintenta si falla.

import { api } from "./util.js";

const comprimir = typeof CompressionStream !== "undefined";
const espera = (ms) => new Promise((r) => setTimeout(r, ms));

async function gzip(blob) {
  return new Response(blob.stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer();
}

export async function subir(archivo, alAvanzar, senal) {
  const { id, trozo } = await api("/api/subidas", {
    method: "POST", body: JSON.stringify({ nombre: archivo.name, tamano: archivo.size }),
  });
  const n = Math.ceil(archivo.size / trozo);
  const yaComprimido = /\.gz$/i.test(archivo.name);
  let enviados = 0;
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    const pedazo = archivo.slice(i * trozo, Math.min(archivo.size, (i + 1) * trozo));
    const usarGzip = comprimir && !yaComprimido;
    const cuerpo = usarGzip ? await gzip(pedazo) : pedazo;
    for (let intento = 0; ; intento++) {
      try {
        const r = await fetch(`/api/subidas/${id}/trozo/${i}`, {
          method: "PUT", body: cuerpo, signal: senal,
          headers: { "Content-Type": "application/octet-stream", ...(usarGzip ? { "X-Comprimido": "gzip" } : {}) },
        });
        if (!r.ok) {
          const datos = await r.json().catch(() => ({}));
          const e = new Error(datos.error || `El servidor respondió ${r.status} al trozo ${i + 1} de ${n}`);
          e.definitivo = r.status >= 400 && r.status < 500 && r.status !== 408 && r.status !== 429;
          throw e;
        }
        break;
      } catch (e) {
        if (senal?.aborted || e.definitivo || intento >= 4) throw e;
        await espera(800 * 2 ** intento);              // red caída un momento: reintentar el mismo trozo
      }
    }
    enviados += cuerpo.byteLength ?? cuerpo.size;
    const hecho = Math.min(archivo.size, (i + 1) * trozo);
    alAvanzar?.({ fraccion: hecho / archivo.size, hecho, total: archivo.size, enviados,
                  velocidad: hecho / ((performance.now() - t0) / 1000) });
  }
  await api(`/api/subidas/${id}/fin`, { method: "POST" });
  return id;
}
