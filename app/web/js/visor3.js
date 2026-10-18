// Visualizador 3: reconstrucción 3D final. Una malla Plotly Mesh3d por fragmento (marching cubes en el servidor), con
// el color de su hueso y, junto a cada una, su rótulo con la separación al principal. Coordenadas en mm: x hacia la
// izquierda del paciente, y hacia atrás, z hacia la cabeza.

import { COLOR, HUESO, el } from "./util.js";

const VISTAS = {
  frente: { eye: { x: 0, y: -2.1, z: 0.35 }, up: { x: 0, y: 0, z: 1 } },
  atras: { eye: { x: 0, y: 2.1, z: 0.35 }, up: { x: 0, y: 0, z: 1 } },
  arriba: { eye: { x: 0, y: -0.05, z: 2.3 }, up: { x: 0, y: -1, z: 0 } },
  lado: { eye: { x: 2.2, y: 0, z: 0.3 }, up: { x: 0, y: 0, z: 1 } },
};

export function crearVisor3(mallas, conReal) {
  const div = document.getElementById("v3-grafico");
  const rotulos = document.getElementById("v3-rotulos");
  const transparente = document.getElementById("v3-transparente");
  const selector = document.getElementById("v3-fuente");
  selector.hidden = !conReal;
  let fuente = "pred", camara = VISTAS.frente;
  const movil = matchMedia("(max-width: 700px)").matches;

  function trazas() {
    return (mallas[fuente] || []).map((m) => ({
      type: "mesh3d", x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k, color: m.color, name: m.nombre,
      opacity: transparente.checked && m.principal ? 0.3 : 1, flatshading: false, showlegend: true,
      hovertemplate: `${m.nombre}<extra></extra>`,
      lighting: { ambient: 0.45, diffuse: 0.8, specular: 0.15, roughness: 0.6 }, lightposition: { x: 0, y: -1e5, z: 1e5 },
    }));
  }

  function anotaciones() {
    if (!rotulos.checked) return [];
    return (mallas[fuente] || []).map((m) => ({
      x: m.centro[0], y: m.centro[1], z: m.centro[2], text: m.nombre, showarrow: true, arrowcolor: m.color_hueso,
      arrowwidth: 1.5, ax: 0, ay: -36, font: { size: movil ? 9 : 11, color: "#111" }, bgcolor: "rgba(255,255,255,0.88)",
      bordercolor: m.color_hueso, borderwidth: 1, borderpad: 2,
    }));
  }

  function layout() {
    const oscuro = matchMedia("(prefers-color-scheme: dark)").matches;
    return {
      margin: { l: 0, r: 0, t: 8, b: 0 }, paper_bgcolor: "rgba(0,0,0,0)", showlegend: !movil,
      legend: { font: { size: 11, color: oscuro ? "#ddd" : "#222" }, itemsizing: "constant", x: 0, y: 1,
                bgcolor: oscuro ? "rgba(29,31,34,0.75)" : "rgba(255,255,255,0.75)" },
      scene: {
        aspectmode: "data", annotations: anotaciones(), camera: camara, dragmode: "orbit",
        xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false }, bgcolor: "rgba(0,0,0,0)",
      },
    };
  }

  const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["toImage", "resetCameraLastSave3d"] };
  function dibujar() {
    if (!window.Plotly) { div.textContent = "No se pudo cargar Plotly."; return; }
    Plotly.react(div, trazas(), layout(), config);
  }
  rotulos.onchange = () => Plotly.relayout(div, { "scene.annotations": anotaciones() });
  transparente.onchange = dibujar;
  for (const b of selector.querySelectorAll("button")) {
    b.onclick = () => {
      selector.querySelectorAll("button").forEach((o) => o.setAttribute("aria-pressed", o === b));
      fuente = b.dataset.valor;
      dibujar();
    };
  }
  for (const b of document.querySelectorAll("[data-vista]")) {
    b.onclick = () => { camara = VISTAS[b.dataset.vista]; Plotly.relayout(div, { "scene.camera": camara }); };
  }
  const leyenda = document.getElementById("v3-leyenda");
  leyenda.replaceChildren(...[1, 2, 3].map((k) => el("div", { class: "leyenda__fila" },
    el("span", { class: "muestra", style: `background:${COLOR[k]}` }), HUESO[k])),
    el("div", { class: "nota" }, `${(mallas.pred || []).length} fragmentos predichos` +
      (conReal ? ` · ${(mallas.real || []).length} en la etiqueta real` : "") + ". Pulsa un fragmento en la leyenda para ocultarlo."));

  let dibujado = false;
  return {
    mostrar() {
      if (!dibujado) { dibujar(); dibujado = true; } else Plotly.Plots.resize(div);
    },
    reiniciar() { dibujado = false; },
  };
}
