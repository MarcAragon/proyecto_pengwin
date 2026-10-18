// Visualizador 1: reconstrucción 3D inicial del volumen crudo por MIP rotatoria.
// Los fotogramas llegan en HU (int16). El umbral se aplica aquí, en cada fotograma: umbralizar y proyectar conmutan,
// así que moverlo no exige recalcular la rotación. Paleta «bone» de matplotlib, la misma del notebook.

const LUT_HEX = "00000001010102020203030404030504040605050706060907070a08080b09090c0a0a0d0a0a0f0b0b100c0c110d0d120e0e130f0f1510101611111712111812121a13131b14141c15151d16161e1717201818211818221919231a1a251b1b261c1c271d1d281e1e291f1f2b201f2c20202d21212e22222f2323312424322525332626342626362727372828382929392a2a3a2b2b3c2c2c3d2d2d3e2e2d3f2e2e412f2f4230304331314432324533334734344834344935354a36364b37374d38384e39394f3a3a503b3b523c3b533c3c543d3d553e3e563f3f5840405941415a42425b42425d43435e44445f4545604646614747634848644949654a49664a4a674b4b694c4c6a4d4d6b4e4e6c4f4f6e50506f505070515171525372535473545574555675565776575976575a77585b78595c795a5d7a5b5f7b5c607c5d617d5e627d5e637e5f657f606680616781626882636983646b84656c84666d85666e86676f876871886972896a738a6b748b6c758b6c778c6d788d6e798e6f7a8f707b90717d91727e92737f927380937481947583957684967785977886987987997a89997a8a9a7b8b9b7c8c9c7d8d9d7e8f9e7f909f8091a08192a08194a18295a28396a38497a48598a5869aa6879ba7889ca7889da8899ea98aa0aa8ba1ab8ca2ac8da3ad8ea4ae8fa6ae8fa7af90a8b091a9b192aab293acb394adb495aeb596afb596b0b697b2b798b3b899b4b99ab5ba9bb6bb9cb8bc9db9bc9dbabd9ebbbe9fbcbfa0bec0a1bfc1a2c0c2a3c1c3a4c2c3a4c4c4a5c5c5a6c6c6a7c7c7a9c8c8aac9c9accacaadcacaaecbcbb0ccccb1cdcdb2ceceb4cfcfb5d0d0b7d1d1b8d2d1b9d2d2bbd3d3bcd4d4bdd5d5bfd6d6c0d7d7c1d8d8c3d8d8c4d9d9c6dadac7dbdbc8dcdccaddddcbdedeccdfdfcee0dfcfe0e0d1e1e1d2e2e2d3e3e3d5e4e4d6e5e5d7e6e6d9e6e6dae7e7dbe8e8dde9e9deeaeae0ebebe1ecece2edede4eeede5eeeee6efefe8f0f0e9f1f1eaf2f2ecf3f3edf4f4eff4f4f0f5f5f1f6f6f3f7f7f4f8f8f5f9f9f7fafaf8fbfbfafcfbfbfcfcfcfdfdfefefeffffff";
const LUT = new Uint8Array(768);
for (let i = 0; i < 768; i++) LUT[i] = parseInt(LUT_HEX.slice(2 * i, 2 * i + 2), 16);

export function crearVisor1(meta, fotogramas) {
  const lienzo = document.getElementById("v1-lienzo");
  const ctx = lienzo.getContext("2d");
  const { n, alto, ancho, angulos, hu_max: huMax } = meta;
  lienzo.width = ancho;
  lienzo.height = alto;
  const imagen = ctx.createImageData(ancho, alto);
  const angulo = document.getElementById("v1-angulo");
  const umbral = document.getElementById("v1-umbral");
  const play = document.getElementById("v1-play");
  angulo.max = n - 1;
  let i = 0, t = meta.umbral, girando = null;

  function dibujar() {
    const base = i * alto * ancho, d = imagen.data, escala = 255 / (huMax - t);
    for (let p = 0; p < alto * ancho; p++) {
      let v = (fotogramas[base + p] - t) * escala;
      v = v < 0 ? 0 : v > 255 ? 255 : v | 0;
      d[4 * p] = LUT[3 * v]; d[4 * p + 1] = LUT[3 * v + 1]; d[4 * p + 2] = LUT[3 * v + 2]; d[4 * p + 3] = 255;
    }
    ctx.putImageData(imagen, 0, 0);
    document.getElementById("v1-angulo-txt").textContent = `${angulos[i]}°`;
    document.getElementById("v1-umbral-txt").textContent = `${t} HU`;
    document.getElementById("v1-rotulo").textContent = `MIP · ${angulos[i]}° · umbral ${t} HU`;
  }
  const irA = (k) => { i = ((k % n) + n) % n; angulo.value = i; dibujar(); };

  angulo.oninput = () => irA(+angulo.value);
  umbral.oninput = () => { t = +umbral.value; dibujar(); };
  document.getElementById("v1-defecto").onclick = () => { umbral.value = t = meta.umbral; dibujar(); };
  function parar() { clearInterval(girando); girando = null; play.textContent = "▶ Girar"; }
  play.onclick = () => {
    if (girando) return parar();
    girando = setInterval(() => irA(i + 1), 90);
    play.textContent = "⏸ Pausa";
  };

  // Arrastrar sobre la imagen la hace girar (ratón o dedo)
  let x0 = null, i0 = 0;
  lienzo.onpointerdown = (e) => { parar(); x0 = e.clientX; i0 = i; lienzo.setPointerCapture(e.pointerId); };
  lienzo.onpointermove = (e) => {
    if (x0 === null) return;
    const paso = lienzo.getBoundingClientRect().width / n;
    irA(i0 - Math.round((e.clientX - x0) / Math.max(paso * 0.6, 4)));
  };
  lienzo.onpointerup = lienzo.onpointercancel = () => { x0 = null; };

  umbral.value = t;
  irA(0);
  return { parar };
}
