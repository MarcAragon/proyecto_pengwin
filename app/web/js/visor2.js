// Visualizador 2: inferencia corte a corte. Sobre cada corte (en la ventana de hueso), los fragmentos 3D predichos
// coloreados por su hueso, la caja de la detección de cada región y, escrita sobre cada fragmento, su separación en mm
// respecto al principal de su hueso. Axial, coronal o sagital (la rejilla es isotrópica: 1,5 mm en los tres ejes).

import { ABREV, COLOR, HUESO, centroMayorPieza, colorFragmento, el, huesoDe, num } from "./util.js";

export function crearVisor2(resumen, vol) {
  const [Z, H, W] = resumen.forma;
  const mm = resumen.espaciado_mm;
  const conReal = !!vol.real;
  const sepPred = new Map(resumen.fragmentos.map((f) => [f.etiqueta, f]));
  const sepReal = new Map((resumen.evaluacion?.tabla || []).map((t) => [t.etiqueta_real,
    { principal: t.principal_real, separacion_mm: t.separacion_real_mm }]));

  const ui = {
    lienzo: document.getElementById("v2-lienzo"), lienzoReal: document.getElementById("v2-lienzo-real"),
    figuraReal: document.getElementById("v2-figura-real"), contenedor: document.getElementById("v2-lienzos"),
    corte: document.getElementById("v2-corte"), corteTxt: document.getElementById("v2-corte-txt"),
    alfa: document.getElementById("v2-alfa"), cajas: document.getElementById("v2-cajas"),
    rotulos: document.getElementById("v2-rotulos"), encuadre: document.getElementById("v2-encuadre"),
    real: document.getElementById("v2-real"), info: document.getElementById("v2-info"), pie: document.getElementById("v2-pie"),
  };
  document.getElementById("v2-real-opcion").hidden = !conReal;
  ui.real.checked = false;
  ui.figuraReal.hidden = true;
  ui.contenedor.classList.remove("v2-lienzos--dos");

  // El corte inicial: el axial con más fragmentos distintos (a igualdad, el más cercano al centro de la pelvis)
  const inicial = (() => {
    const centro = resumen.rango_hueso ? (resumen.rango_hueso[0] + resumen.rango_hueso[1]) / 2 : Z / 2;
    let mejor = Math.round(centro), puntos = -1;
    for (let z = 0; z < Z; z++) {
      const vistos = new Set();
      const s = vol.pred.subarray(z * H * W, (z + 1) * H * W);
      for (let p = 0; p < s.length; p += 2) if (s[p]) vistos.add(s[p]);
      const valor = vistos.size * 1000 - Math.abs(z - centro);
      if (valor > puntos) { puntos = valor; mejor = z; }
    }
    return mejor;
  })();
  const estado = { plano: "axial", indice: inicial };

  // Dimensiones de cada plano: (ancho, alto, número de cortes)
  const PLANOS = { axial: [W, H, Z], coronal: [W, Z, H], sagital: [H, Z, W] };

  // La caja de todo el hueso (predicho o real), en cada plano, para encuadrar
  const caja3d = (() => {
    let z0 = Z, z1 = -1, y0 = H, y1 = -1, x0 = W, x1 = -1;
    const fuentes = [vol.pred, vol.real].filter(Boolean);
    for (let z = 0; z < Z; z++) for (let y = 0; y < H; y++) {
      const base = (z * H + y) * W;
      for (let x = 0; x < W; x++) {
        if (fuentes.some((f) => f[base + x])) {
          if (z < z0) z0 = z; if (z > z1) z1 = z; if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x;
        }
      }
    }
    return z1 < 0 ? null : { z0, z1, y0, y1, x0, x1 };
  })();

  function encuadre(plano) {
    const [w, h] = PLANOS[plano];
    if (!ui.encuadre.checked || !caja3d) return { x0: 0, y0: 0, w, h };
    const m = 12;
    let a, b, c, d;           // columnas a..b, filas c..d en coordenadas del plano
    if (plano === "axial") [a, b, c, d] = [caja3d.x0, caja3d.x1, caja3d.y0, caja3d.y1];
    else if (plano === "coronal") [a, b, c, d] = [caja3d.x0, caja3d.x1, Z - 1 - caja3d.z1, Z - 1 - caja3d.z0];
    else [a, b, c, d] = [caja3d.y0, caja3d.y1, Z - 1 - caja3d.z1, Z - 1 - caja3d.z0];
    // cuadrado, para que no salte el encuadre al cambiar de plano
    const lado = Math.max(b - a, d - c) + 2 * m, cx = (a + b) / 2, cy = (c + d) / 2;
    const x0 = Math.round(Math.max(0, Math.min(w - lado, cx - lado / 2))), y0 = Math.round(Math.max(0, Math.min(h - lado, cy - lado / 2)));
    return { x0, y0, w: Math.min(lado, w), h: Math.min(lado, h) };
  }

  function extraer(volumen, plano, i) {
    const [w, h] = PLANOS[plano];
    const out = new Uint8Array(w * h);
    if (plano === "axial") out.set(volumen.subarray(i * H * W, (i + 1) * H * W));
    else if (plano === "coronal") {
      for (let r = 0; r < h; r++) { const z = Z - 1 - r; out.set(volumen.subarray((z * H + i) * W, (z * H + i + 1) * W), r * w); }
    } else {
      for (let r = 0; r < h; r++) { const z = Z - 1 - r; for (let c = 0; c < w; c++) out[r * w + c] = volumen[(z * H + c) * W + i]; }
    }
    return out;
  }

  const visitado = new Uint8Array(Math.max(W * H, W * Z, H * Z));

  function dibujar(lienzo, etiquetas, separaciones, cajasCorte, discontinuas) {
    const { plano, indice } = estado;
    const [w, h] = PLANOS[plano];
    const gris = extraer(vol.hu, plano, indice), lab = extraer(etiquetas, plano, indice);
    const e = encuadre(plano);
    const alfa = +ui.alfa.value / 100;
    // 1 · El corte a su resolución, con los fragmentos y su borde en blanco
    const base = document.createElement("canvas");
    base.width = e.w; base.height = e.h;
    const bctx = base.getContext("2d");
    const img = bctx.createImageData(e.w, e.h);
    for (let r = 0; r < e.h; r++) for (let c = 0; c < e.w; c++) {
      const p = (e.y0 + r) * w + (e.x0 + c), q = 4 * (r * e.w + c);
      const g = gris[p], v = lab[p];
      if (v) {
        const borde = (c + e.x0 > 0 && lab[p - 1] !== v) || (c + e.x0 < w - 1 && lab[p + 1] !== v) ||
                      (r + e.y0 > 0 && lab[p - w] !== v) || (r + e.y0 < h - 1 && lab[p + w] !== v);
        if (borde) { img.data[q] = img.data[q + 1] = img.data[q + 2] = 255; }
        else {
          const col = colorFragmento(v);
          img.data[q] = (1 - alfa) * g + alfa * col[0]; img.data[q + 1] = (1 - alfa) * g + alfa * col[1]; img.data[q + 2] = (1 - alfa) * g + alfa * col[2];
        }
      } else { img.data[q] = img.data[q + 1] = img.data[q + 2] = g; }
      img.data[q + 3] = 255;
    }
    bctx.putImageData(img, 0, 0);

    // 2 · Escalado al tamaño en pantalla, sin suavizar (cada píxel es un vóxel de 1,5 mm)
    const ancho = Math.max(240, Math.round(lienzo.parentElement.getBoundingClientRect().width));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const S = (ancho * dpr) / e.w;
    lienzo.width = Math.round(e.w * S); lienzo.height = Math.round(e.h * S);
    const ctx = lienzo.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(base, 0, 0, lienzo.width, lienzo.height);

    // 3 · Las cajas de la detección (solo en axial: la red detecta en cada corte axial)
    if (ui.cajas.checked && plano === "axial" && cajasCorte) {
      ctx.lineWidth = 2 * dpr;
      for (const [k, c] of Object.entries(cajasCorte)) {
        ctx.strokeStyle = COLOR[k];
        ctx.setLineDash(discontinuas ? [6 * dpr, 4 * dpr] : []);
        ctx.strokeRect((c[0] - e.x0) * S, (c[1] - e.y0) * S, (c[2] - c[0]) * S, (c[3] - c[1]) * S);
        if (!discontinuas && c[4] !== undefined) {
          const t = num(c[4], 2);
          ctx.font = `${11 * dpr}px system-ui, sans-serif`;
          const tw = ctx.measureText(t).width + 6 * dpr;
          ctx.fillStyle = COLOR[k];
          ctx.fillRect((c[0] - e.x0) * S, (c[1] - e.y0) * S - 15 * dpr, tw, 15 * dpr);
          ctx.fillStyle = "#fff";
          ctx.fillText(t, (c[0] - e.x0) * S + 3 * dpr, (c[1] - e.y0) * S - 4 * dpr);
        }
      }
      ctx.setLineDash([]);
    }

    // 4 · El rótulo de cada fragmento, en el centro de su mayor pieza: número y separación
    const presentes = [...new Set(lab)].filter((v) => v).sort((a, b) => a - b);
    const ocupados = [];
    if (ui.rotulos.checked) {
      ctx.font = `600 ${12 * dpr}px system-ui, sans-serif`;
      for (const v of presentes) {
        const c = centroMayorPieza(lab, w, h, v, visitado);
        if (!c || c.n < 6) continue;
        const f = separaciones.get(v);
        const texto = f ? (f.principal ? `${v} · principal` : `${v} · ${num(f.separacion_mm, 1)} mm`) : `${v}`;
        const tw = ctx.measureText(texto).width + 10 * dpr, th = 18 * dpr;
        let x = (c.x - e.x0 + 0.5) * S - tw / 2, y = (c.y - e.y0 + 0.5) * S - th / 2;
        x = Math.max(2, Math.min(lienzo.width - tw - 2, x));
        for (let k = 0; k < 6 && ocupados.some((o) => Math.abs(o[0] - x) < tw && Math.abs(o[1] - y) < th); k++) y += th + 2;
        ocupados.push([x, y]);
        ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.strokeStyle = COLOR[huesoDe(v)];
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath(); ctx.roundRect(x, y, tw, th, 4 * dpr); ctx.fill(); ctx.stroke();
        ctx.fillStyle = "#111";
        ctx.fillText(texto, x + 5 * dpr, y + 13 * dpr);
      }
    }
    return presentes;
  }

  function actualizar() {
    const { plano, indice } = estado;
    const [, , n] = PLANOS[plano];
    ui.corte.max = n - 1;
    ui.corte.value = indice;
    const pos = plano === "axial" ? `z = ${num(indice * mm, 1)} mm` : plano === "coronal" ? `y = ${num(indice * mm, 1)} mm` : `x = ${num(indice * mm, 1)} mm`;
    ui.corteTxt.textContent = `${indice + 1} de ${n} · ${pos}`;
    const presentes = dibujar(ui.lienzo, vol.pred, sepPred, plano === "axial" ? resumen.cajas[indice] : null, false);
    if (conReal && ui.real.checked) {
      dibujar(ui.lienzoReal, vol.real, sepReal, plano === "axial" ? resumen.cajas_reales[indice] : null, true);
    }
    ui.pie.textContent = conReal && ui.real.checked ? "Predicción" : `Predicción · ${plano}`;
    // Lo que dice el modelo de este corte
    const filas = [];
    if (plano === "axial") {
      const p = resumen.presencia[indice];
      filas.push(el("div", {}, "Clasificación del corte: ", [1, 2, 3].map((k, j) =>
        el("span", { style: `color:${COLOR[k]};font-weight:600` }, `${ABREV[k]} ${num(p[j], 2)}${j < 2 ? " · " : ""}`))));
    }
    if (presentes.length) {
      filas.push(el("div", {}, `${presentes.length} fragmento${presentes.length > 1 ? "s" : ""} en este corte:`));
      for (const v of presentes) {
        const f = sepPred.get(v);
        filas.push(el("div", { class: "leyenda__fila" }, el("span", { class: "muestra", style: `background:rgb(${colorFragmento(v)})` }),
          `${HUESO[huesoDe(v)]} ${v} — ${f ? (f.principal ? "principal" : `${num(f.separacion_mm, 1)} mm`) : ""}`));
      }
    } else filas.push(el("div", {}, "Sin fragmentos predichos en este corte."));
    ui.info.replaceChildren(...filas);
  }

  const irA = (i) => {
    const n = PLANOS[estado.plano][2];
    estado.indice = Math.max(0, Math.min(n - 1, i));
    actualizar();
  };

  // Controles
  ui.corte.oninput = () => irA(+ui.corte.value);
  document.getElementById("v2-anterior").onclick = () => irA(estado.indice - 1);
  document.getElementById("v2-siguiente").onclick = () => irA(estado.indice + 1);
  document.getElementById("v2-centro").onclick = () => {
    if (!caja3d) return irA(PLANOS[estado.plano][2] >> 1);
    const c = { axial: (caja3d.z0 + caja3d.z1) / 2, coronal: (caja3d.y0 + caja3d.y1) / 2, sagital: (caja3d.x0 + caja3d.x1) / 2 };
    irA(Math.round(c[estado.plano]));
  };
  ui.alfa.oninput = () => { document.getElementById("v2-alfa-txt").textContent = `${ui.alfa.value} %`; actualizar(); };
  for (const c of [ui.cajas, ui.rotulos, ui.encuadre]) c.onchange = actualizar;
  ui.real.onchange = () => {
    ui.figuraReal.hidden = !ui.real.checked;
    ui.contenedor.classList.toggle("v2-lienzos--dos", ui.real.checked);
    actualizar();
  };
  for (const b of document.querySelectorAll("#v2-plano button")) {
    b.onclick = () => {
      document.querySelectorAll("#v2-plano button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      const previo = estado.plano;
      estado.plano = b.dataset.valor;
      if (previo !== estado.plano) {
        document.getElementById("v2-centro").click();
        return;
      }
      actualizar();
    };
  }
  // Rueda, flechas y arrastre vertical recorren los cortes
  for (const lz of [ui.lienzo, ui.lienzoReal]) {
    lz.onwheel = (e) => { e.preventDefault(); irA(estado.indice + (e.deltaY > 0 ? 1 : -1)); };
    let y0 = null, i0 = 0;
    lz.onpointerdown = (e) => { y0 = e.clientY; i0 = estado.indice; lz.setPointerCapture(e.pointerId); };
    lz.onpointermove = (e) => { if (y0 !== null) irA(i0 + Math.round((e.clientY - y0) / 6)); };
    lz.onpointerup = lz.onpointercancel = () => { y0 = null; };
  }
  const teclas = (e) => {
    if (document.getElementById("panel-v2").hidden || e.target.closest?.("input[type=range]")) return;
    if (e.key === "ArrowUp" || e.key === "ArrowRight") { irA(estado.indice + 1); e.preventDefault(); }
    if (e.key === "ArrowDown" || e.key === "ArrowLeft") { irA(estado.indice - 1); e.preventDefault(); }
  };
  document.addEventListener("keydown", teclas);
  const observador = new ResizeObserver(() => { if (!document.getElementById("panel-v2").hidden) actualizar(); });
  observador.observe(ui.contenedor);

  const leyenda = document.getElementById("v2-leyenda");
  leyenda.replaceChildren(...[1, 2, 3].map((k) => el("div", { class: "leyenda__fila" },
    el("span", { class: "muestra", style: `background:${COLOR[k]}` }), HUESO[k])),
    el("div", { class: "nota" }, "Dentro de un hueso, cada fragmento un tono: el principal, el más intenso. Cajas: la mejor de cada región tras el NMS, con su puntuación" +
      (conReal ? "; discontinuas, las reales." : ".")));

  return {
    actualizar,
    irAFragmento(v) {
      // El corte axial donde el fragmento es mayor
      let mejor = 0, z = estado.indice;
      for (let k = 0; k < Z; k++) {
        let n = 0;
        const s = vol.pred.subarray(k * H * W, (k + 1) * H * W);
        for (let p = 0; p < s.length; p++) if (s[p] === v) n++;
        if (n > mejor) { mejor = n; z = k; }
      }
      document.querySelectorAll("#v2-plano button").forEach((o) => o.setAttribute("aria-pressed", o.dataset.valor === "axial"));
      estado.plano = "axial";
      irA(z);
    },
    destruir() { document.removeEventListener("keydown", teclas); observador.disconnect(); },
  };
}
