"""Del píxel al volumen — aplicación web (computador y celular).

Estima el volumen de un envase a partir de una foto, con una hoja de referencia
impresa (marcadores ArUco) bajo el envase. Cada sección corresponde a una etapa o
requisito del enunciado del laboratorio.

Ejecutar en local:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from retroenvases import integracion as integ
from retroenvases.foto import Foto, cargar
from retroenvases.plantilla import HOJAS, distancia_control_mm, pdf_plantilla
from retroenvases.sintetico import BOTELLA_600, generar_con_hoja
from retroenvases.vision import bordes_por_fila, segmentar_grabcut, segmentar_otsu
from retroenvases.vision_aruco import (SinMarcadores, detectar, estimar_camara,
                                       rectangulo_envase, reconstruir, segmentar_con_hoja)

st.set_page_config(page_title="Del píxel al volumen", page_icon="🧴", layout="wide",
                   initial_sidebar_state="collapsed")

LIMITE = 5.0                      # criterio del reto: error relativo máximo [%]
ESTRATEGIAS = {
    "Automática con la hoja (recomendada)": "hoja",
    "GrabCut con rectángulo manual": "rect",
    "Umbral de Otsu (solo fondo liso)": "otsu",
}


# ════════════════════════════════════════════════════════════ utilidades ══
@st.cache_data(show_spinner=False)
def _cargar(datos: bytes) -> Foto:
    return cargar(datos, lado_max=1800)


@st.cache_data(show_spinner=False)
def _plantilla(nombre: str) -> bytes:
    return pdf_plantilla(HOJAS[nombre])


@st.cache_data(show_spinner="Generando la foto de ejemplo…")
def _ejemplo() -> tuple[bytes, dict]:
    """Escena sintética: botella de volumen exacto sobre la hoja, cámara a 50 cm."""
    img, v = generar_con_hoja(BOTELLA_600, distancia_cm=50, altura_camara_cm=28)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes(), {"volumen_cm3": v["volumen_cm3"], "altura_cm": v["altura_cm"]}


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _dibujar_referencia(bgr, marcadores, cam, hoja, escala):
    """Marcadores detectados, borde de la hoja, círculos guía y eje según la pose."""
    out = bgr.copy()
    g = max(2, bgr.shape[0] // 400)
    for i, c in marcadores.items():
        cv2.polylines(out, [c.astype(np.int32)], True, (0, 220, 0), g)
        cv2.circle(out, tuple(c[0].astype(int)), 3 * g, (0, 0, 255), -1)   # esquina 0
        cv2.putText(out, f"ID {i}", tuple((c.mean(axis=0) + [-30, 10]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * g, (0, 0, 255), g)
    w2, h2 = hoja.ancho_mm / 20 * escala, hoja.alto_mm / 20 * escala
    borde = cam.proyectar(np.array([[-w2, -h2, 0], [w2, -h2, 0], [w2, h2, 0], [-w2, h2, 0]]))
    cv2.polylines(out, [borde.astype(np.int32)], True, (255, 160, 0), g)
    ang = np.linspace(0, 2 * np.pi, 60)
    for rr in (2.0, 4.0):
        circ = cam.proyectar(np.column_stack([rr * np.cos(ang), rr * np.sin(ang), 0 * ang]))
        cv2.polylines(out, [circ.astype(np.int32)], True, (255, 0, 255), g)
    eje = cam.proyectar(np.array([[0, 0, 0], [0, 0, 30.0]]))
    cv2.line(out, tuple(eje[0].astype(int)), tuple(eje[1].astype(int)), (255, 0, 255), g)
    return _rgb(out)


def _con_contorno(bgr, mascara, color=(0, 200, 0)):
    out = bgr.copy()
    m = mascara > 0
    out[m] = (0.55 * out[m] + 0.45 * np.array(color)).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 255, 255), max(2, bgr.shape[0] // 500))
    return _rgb(out)


def _segmentar(estrategia, img, cam, rect, altura):
    if estrategia == "hoja":
        return segmentar_con_hoja(img, cam, altura_cm=altura)
    if estrategia == "rect":
        return segmentar_grabcut(img, rect=rect)
    return segmentar_otsu(img)


def _perfil(img, cam, mascara, altura):
    filas, bi, bd, frac = bordes_por_fila(mascara, img=img)
    return reconstruir(cam, filas, bi, bd, altura), frac


def _metodos(z, r, m_gauss, paneles):
    """Los 4 métodos sobre la misma tabla, con tiempo y parámetros; y los avisos."""
    n, dz = len(z) - 1, float(z[1] - z[0])
    cortes = list(np.linspace(z[0], z[-1], paneles + 1)[1:-1])
    filas, avisos = [], []
    for nombre, fn, orden, evals, delta in [
        ("Trapecio", lambda: integ.trapecio(z, r), "O(Δz²)", n + 1, dz),
        ("Simpson 1/3", lambda: integ.simpson(z, r), "O(Δz⁴)", n + 1, dz),
        ("Newton-Cotes 3/8", lambda: integ.newton_cotes(z, r, 3), "O(Δz⁴)", n + 1, dz),
        (f"Gauss-Legendre ({m_gauss} pts × {paneles} paneles)",
         lambda: integ.gauss(z, r, m_gauss, cortes),
         f"exacta hasta grado {2 * m_gauss - 1}", m_gauss * paneles, None),
    ]:
        try:
            t0 = time.perf_counter()
            I = fn()
            ms = 1000 * (time.perf_counter() - t0)
        except integ.CondicionNoSatisfecha as e:
            avisos.append(f"**{nombre}** no se puede aplicar: {e}")
            continue
        filas.append({"Método": nombre, "n (subintervalos)": n, "Puntos usados": evals,
                      "a [cm]": float(z[0]), "b [cm]": float(z[-1]), "Δz [cm]": delta,
                      "Orden / exactitud": orden, "Integral ∫r²dz [cm³]": I,
                      "Volumen π∫r²dz [cm³]": math.pi * I, "Tiempo [ms]": ms})
    return pd.DataFrame(filas), avisos


def _fmt(df):
    return {c: st.column_config.NumberColumn(format="%.4f")
            for c in df.columns if df[c].dtype.kind == "f"}


# ═══════════════════════════════════════════════════════════════ página ══
st.title("🧴 Del píxel al volumen")
st.markdown("Estima el volumen de un envase a partir de una foto, modelándolo como un "
            "**sólido de revolución**:")
st.latex(r"V = \pi \int_a^b [r(z)]^2 \, dz")
st.caption("La foto se convierte en una tabla de puntos (zᵢ, rᵢ) en centímetros, y la "
           "integral se calcula con cuatro métodos numéricos.")

# ── antes de empezar ─────────────────────────────────────────────────────
with st.expander("📄 Antes de empezar: imprime la hoja de referencia y lee cómo tomar la foto"):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**1. Imprime la hoja**")
        tam = st.radio("Tamaño de papel", list(HOJAS), horizontal=True, key="papel_pdf")
        st.download_button(f"Descargar la hoja ({tam}, PDF)", _plantilla(tam),
                           file_name=f"hoja_referencia_{tam.lower()}.pdf",
                           mime="application/pdf", width="stretch")
        st.markdown(
            "- Imprime al **100 %**, sin «ajustar a la página».\n"
            "- Con una regla, mide la distancia **de borde exterior a borde exterior entre "
            f"los marcadores ID 0 e ID 1** (debería ser {distancia_control_mm(HOJAS[tam]):.1f} mm). "
            "La vas a escribir abajo: es la referencia de escala de toda la medición.")
    with c2:
        st.markdown("**2. Protocolo de captura**")
        st.markdown(
            "- **Hoja** plana sobre una mesa, con el borde de los marcadores ID 0 e ID 1 "
            "hacia la cámara.\n"
            "- **Envase** de pie, **centrado** en los círculos.\n"
            "- **Cámara** a 40–70 cm, un poco por encima del envase (a 20–35 cm de la mesa), "
            "apuntando a él. Deben verse completos al menos los marcadores ID 0 e ID 1.\n"
            "- **Orientación** vertical u horizontal, la que deje ver mejor envase y hoja.\n"
            "- **Iluminación** pareja, sin reflejos fuertes sobre la hoja ni sobre el envase.\n"
            "- **Fondo** detrás del envase lo más liso posible.\n"
            "- En el celular, para máxima resolución usa **Subir foto → Cámara** (la cámara "
            "nativa del teléfono) en vez de la cámara del navegador.")

# ── 1. foto ───────────────────────────────────────────────────────────────
st.header("1. Foto del envase")
# "Subir foto" va primero y por defecto: la cámara del navegador solo se enciende
# cuando el usuario la elige (si el permiso ya estaba dado, mostrarla por defecto la
# encendería apenas carga la página).
origen = st.radio("¿De dónde sale la foto?",
                  ["Subir foto", "Tomar foto con la cámara", "Ejemplo sintético"],
                  horizontal=True)
datos, verdad, nombre = None, None, None
if origen == "Tomar foto con la cámara":
    st.caption("La cámara del navegador da poca resolución. Si puedes, usa «Subir foto», "
               "que en el celular abre la cámara nativa con toda su resolución.")
    cam_in = st.camera_input("Encuadra el envase y la hoja")
    if cam_in:
        datos, nombre = cam_in.getvalue(), "foto de la cámara"
elif origen == "Subir foto":
    arch = st.file_uploader("Foto (JPG, PNG o WEBP). En el celular puedes elegir «Cámara» "
                            "para tomarla en el momento", type=["jpg", "jpeg", "png", "webp"])
    if arch:
        datos, nombre = arch.getvalue(), arch.name
else:
    datos, verdad = _ejemplo()
    nombre = "ejemplo sintético"
    st.info(f"Foto sintética de una botella de volumen exacto **{verdad['volumen_cm3']:.2f} cm³** "
            "sobre la hoja carta, tomada con una cámara a 50 cm. Sirve para comprobar que todo "
            "funciona sin imprimir nada.")
if not datos:
    st.stop()

foto = _cargar(datos)
img = foto.bgr
H_img, W_img = img.shape[:2]
clave = str(hash((nombre, datos)))

# ── 2. referencia de escala ────────────────────────────────────────────────
st.header("2. Referencia de escala")
c1, c2 = st.columns([1, 1])
with c1:
    tipo_hoja = st.radio("Hoja usada", list(HOJAS), horizontal=True, key=f"hoja_{clave}")
    hoja = HOJAS[tipo_hoja]
    nominal = distancia_control_mm(hoja)
    medida = st.number_input(
        "Distancia medida entre los bordes exteriores de ID 0 e ID 1 [mm]",
        min_value=100.0, max_value=300.0, value=float(nominal), step=0.5, format="%.1f",
        key=f"medida_{clave}",
        help=f"Lo nominal es {nominal:.1f} mm. Si la impresora agrandó o achicó la hoja, aquí "
             "se corrige. Su error se multiplica ×3 en el volumen.")
    escala = medida / nominal
    if abs(escala - 1) > 0.002:
        st.info(f"La hoja salió impresa al **{100 * escala:.1f} %**: se corrige con tu medida.")
    altura_medida = st.number_input(
        "Altura del envase medida con regla [cm] (opcional, sirve de control)",
        min_value=0.0, max_value=80.0, value=0.0, step=0.1, format="%.1f",
        key=f"altura_{clave}",
        help="No fija la escala: si la escribes, el perfil se cierra a esa altura y se compara "
             "con la que estima la foto. 0 = estimarla de la foto.") or None

marcadores = detectar(img)
cam, error_cam = None, None
try:
    cam = estimar_camara(img.shape, marcadores, hoja, lado_medido_mm=hoja.lado_mm * escala,
                         focal_px_inicial=foto.focal_px())
except SinMarcadores as e:
    error_cam = str(e)

with c2:
    if cam is None:
        st.image(_rgb(img), caption="Foto", width="stretch")
    else:
        st.image(_dibujar_referencia(img, marcadores, cam, hoja, escala),
                 caption="Verde: marcadores detectados (punto rojo = su primera esquina). "
                         "Naranja: borde de la hoja reconstruido. Magenta: círculos guía y eje "
                         "del envase según la pose de la cámara.", width="stretch")
if cam is None:
    st.error(f"No se puede fijar la escala: {error_cam}")
    st.stop()
if len(cam.marcadores) < 2:
    st.warning("Solo se detectó un marcador: la focal de la cámara no se puede medir y se "
               "supone la típica de un celular. Repite la foto con ID 0 e ID 1 visibles.")

C = cam.centro
with st.expander("¿Cómo se pasa de píxeles a centímetros? (referencia, coordenadas, conversión "
                 "y supuestos)", expanded=True):
    st.markdown(f"""
**Referencia física.** Los cuatro cuadrados negros (marcadores ArUco) de la hoja
{tipo_hoja.lower()}. Sus esquinas tienen posiciones conocidas sobre la hoja.

**Longitud conocida.** La distancia entre los bordes exteriores de ID 0 e ID 1, que mediste:
**{medida:.1f} mm** (nominal {nominal:.1f} mm). Todas las posiciones de la hoja se escalan con
ella; cada cuadrado mide {hoja.lado_mm * escala:.2f} mm.

**Coordenadas en píxeles.** El programa detecta automáticamente los marcadores
({", ".join(f"ID {i}" for i in cam.marcadores)}) y ubica sus esquinas con precisión de
décimas de píxel. Así tiene {4 * len(cam.marcadores)} pares (píxel ↔ posición en la hoja).

**De píxeles a centímetros.** Con esos pares se resuelve el modelo de la cámara (cámara
estenopeica o *pinhole*). Se obtienen:
- la **focal** de la cámara: **{cam.focal_px:.0f} px** ({"medida en la propia foto" if cam.focal_estimada else "supuesta"});
- su **posición**: a **{math.hypot(C[0], C[1]):.1f} cm** del envase en horizontal y a
  **{C[2]:.1f} cm** sobre la mesa, mirando **{cam.inclinacion_grados:.1f}°** hacia abajo;
- el **error de reproyección** de las esquinas: **{cam.error_px:.2f} px**.

Con eso, cada píxel de la foto es un **rayo** en el espacio, en centímetros. El borde del
envase en cada fila es un rayo tangente a su superficie: su punto más cercano al eje da el
radio r y la altura z reales. Es una conversión en 3D, no un simple factor px → cm, porque
la parte del envase más cercana a la cámara se ve más grande (perspectiva).

**Supuestos.** La hoja está plana sobre la mesa; el envase está de pie sobre ella, con su
eje vertical y centrado en los círculos; el envase es un sólido de revolución; el lente no
deforma la imagen (los celulares ya corrigen la distorsión) y su centro óptico está en el
centro de la foto.
""")

# ── 3. extracción del perfil ───────────────────────────────────────────────
st.header("3. Extracción del perfil")
c1, c2 = st.columns([1, 1])
with c1:
    nombre_estr = st.radio("Estrategia para extraer el contorno", list(ESTRATEGIAS))
    estrategia = ESTRATEGIAS[nombre_estr]
    rect_auto = rectangulo_envase(cam, img.shape, altura_cm=altura_medida)
    rect = None
    if estrategia == "rect":
        x, y, w, h = rect_auto
        rx = st.slider("Rectángulo: horizontal [%]", 0, 100,
                       (int(100 * x / W_img), int(100 * (x + w) / W_img)), key=f"rx_{clave}")
        ry = st.slider("Rectángulo: vertical [%]", 0, 100,
                       (int(100 * y / H_img), int(100 * (y + h) / H_img)), key=f"ry_{clave}")
        rect = (int(rx[0] / 100 * W_img), int(ry[0] / 100 * H_img),
                int((rx[1] - rx[0]) / 100 * W_img), int((ry[1] - ry[0]) / 100 * H_img))
    comparar = st.checkbox("Comparar las tres estrategias", key=f"cmp_{clave}")
    st.caption(
        "**Automática con la hoja:** GrabCut (modelos de color del objeto y del fondo, "
        "separados por cortes de grafo) iniciado con la geometría: lo que está delante del "
        "envase sobre la hoja es fondo seguro y el eje es objeto seguro. "
        "**Rectángulo manual:** GrabCut a partir de un rectángulo que ajustas. "
        "**Otsu:** un umbral elegido automáticamente sobre el histograma; solo sirve con "
        "fondo liso.")

with st.spinner("Extrayendo el contorno y reconstruyendo el perfil…"):
    try:
        mascara = _segmentar(estrategia, img, cam, rect, altura_medida)
        perfil, frac_sub = _perfil(img, cam, mascara, altura_medida)
    except Exception as e:                           # contorno inservible
        st.error(f"No se pudo extraer el perfil con esta estrategia: {e}. Prueba otra, o "
                 "repite la foto siguiendo el protocolo.")
        st.stop()

with c2:
    st.image(_con_contorno(img, mascara), caption="Contorno usado", width="stretch")

z_full, r_full = perfil.z_cm, perfil.r_cm
H = perfil.altura_cm
fig, ax = plt.subplots(figsize=(5.5, 6))
ax.fill_betweenx(z_full, -r_full, r_full, color="#1f77b4", alpha=.18)
ax.plot(r_full, z_full, "-", color="#1f77b4", lw=1.2)
ax.plot(-r_full, z_full, "-", color="#1f77b4", lw=1.2)
paso = max(1, len(z_full) // 60)
ax.plot(r_full[::paso], z_full[::paso], "o", ms=3, color="#d62728", label="puntos (zᵢ, rᵢ)")
ax.axvline(0, ls="--", color="gray", lw=1)
ax.set(xlabel="r [cm]", ylabel="z [cm]", title="Perfil en coordenadas físicas")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=.3)
fig.tight_layout()
c1, c2 = st.columns([1, 1])
with c1:
    st.pyplot(fig)
with c2:
    st.metric("Altura del envase (de la foto)", f"{perfil.altura_estimada_cm:.2f} cm")
    if altura_medida:
        dif = perfil.altura_estimada_cm - altura_medida
        st.metric("Altura medida con regla", f"{altura_medida:.2f} cm", f"{dif:+.2f} cm vs foto",
                  delta_color="off")
    st.metric("Radio máximo", f"{r_full.max():.2f} cm")
    st.metric("Asimetría izquierda/derecha", f"{100 * perfil.asimetria:.1f} %")
    st.metric("Bordes con precisión subpíxel", f"{100 * frac_sub:.0f} %")
    st.caption(f"Puntos del contorno: {len(z_full)} (uno por fila de píxeles). Eje ubicado a "
               f"{10 * math.hypot(*perfil.eje_cm):.1f} mm del centro del círculo.")

if comparar:
    st.subheader("Comparación de estrategias")
    cols = st.columns(3)
    for col, (nom, est) in zip(cols, ESTRATEGIAS.items()):
        with col:
            try:
                m = _segmentar(est, img, cam, rect or rect_auto, altura_medida)
                p, _ = _perfil(img, cam, m, altura_medida)
                zz, rr = integ.remuestrear(p.z_cm, p.r_cm, integ.n_compatible(len(p.z_cm) - 1))
                v = math.pi * integ.trapecio(zz, rr)
                st.image(_con_contorno(img, m), width="stretch")
                st.markdown(f"**{nom.split(' (')[0]}**: {v:.1f} cm³ · asimetría "
                            f"{100 * p.asimetria:.1f} %")
            except Exception as e:
                st.markdown(f"**{nom.split(' (')[0]}**: falló ({e})")

# ── 4. intervalo y discretización ─────────────────────────────────────────
st.header("4. Intervalo y discretización")
c1, c2, c3 = st.columns(3)
with c1:
    ab = st.slider("Intervalo [a, b] en z [cm]", 0.0, float(round(H, 2)),
                   (0.0, float(round(H, 2))), step=0.05, key=f"ab_{clave}",
                   help="Por defecto, todo el envase. Un intervalo menor da el volumen de ese tramo.")
with c2:
    n_auto = integ.n_compatible(len(z_full) - 1)
    n = int(st.number_input("Subintervalos n", 1, 20000, n_auto, step=1, key=f"n_{clave}",
                            help="Por defecto, uno por fila de píxeles, redondeado a múltiplo de 6 "
                                 "(Simpson necesita n par y Newton-Cotes 3/8, múltiplo de 3)."))
with c3:
    m_gauss = st.slider("Puntos de Gauss por panel (orden)", 2, 12, 5)
    paneles = st.slider("Paneles de Gauss", 1, 12, 6)
a, b = ab
if b - a < 0.1:
    st.error("El intervalo es demasiado corto.")
    st.stop()
dentro = (z_full > a) & (z_full < b)
zs = np.concatenate([[a], z_full[dentro], [b]])
rs = np.interp(zs, z_full, r_full)
z, r = integ.remuestrear(zs, rs, n)
dz = (b - a) / n
st.markdown(f"**Δz = (b − a)/n = ({b:.2f} − {a:.2f}) / {n} = {dz:.5f} cm** · malla uniforme de "
            f"{n + 1} puntos.")
st.caption("Los puntos del contorno salen de las filas de píxeles y no quedan equiespaciados "
           "en z (la perspectiva estira unas filas más que otras). Por eso se **remuestrean** "
           "a una malla uniforme por interpolación lineal, que no inventa oscilaciones cerca de "
           "las aristas del envase. Así los cuatro métodos usan exactamente la misma tabla.")

# ── 5. tabla de datos ───────────────────────────────────────────────────────
st.header("5. Tabla de datos")
tabla_datos = pd.DataFrame({"i": np.arange(len(z)), "zᵢ [cm]": z, "rᵢ [cm]": r,
                            "rᵢ² [cm²]": r ** 2, "f(zᵢ) = π·rᵢ² [cm²]": math.pi * r ** 2})
st.dataframe(tabla_datos, hide_index=True, width="stretch", height=260,
             column_config=_fmt(tabla_datos))

# ── 6. integración y comparación ─────────────────────────────────────────────
st.header("6. Integración numérica")
tabla, avisos = _metodos(z, r, m_gauss, paneles)
for av in avisos:
    st.warning(av)
if tabla.empty:
    st.error("Ningún método se pudo aplicar con estos parámetros.")
    st.stop()
V_principal = float(tabla.iloc[0]["Volumen π∫r²dz [cm³]"])
c1, c2, c3 = st.columns(3)
c1.metric("Volumen (trapecio)", f"{V_principal:.1f} cm³")
c2.metric("En mililitros", f"{V_principal:.1f} mL")
c3.metric("Diferencia entre métodos",
          f"{tabla['Volumen π∫r²dz [cm³]'].max() - tabla['Volumen π∫r²dz [cm³]'].min():.2f} cm³")
st.dataframe(tabla, hide_index=True, width="stretch", column_config=_fmt(tabla))

fig, ax = plt.subplots(figsize=(8, 2.6))
ax.barh(tabla["Método"], tabla["Volumen π∫r²dz [cm³]"], color="#4c72b0", alpha=.85)
vmin, vmax = tabla["Volumen π∫r²dz [cm³]"].min(), tabla["Volumen π∫r²dz [cm³]"].max()
ax.set_xlim(vmin - max(0.03 * vmin, 1), vmax + max(0.03 * vmax, 1))
ax.set(xlabel="volumen [cm³]", title="Volumen por método")
ax.grid(axis="x", alpha=.3)
fig.tight_layout()
st.pyplot(fig)

with st.expander("Análisis de cada método (efecto de Δz, condiciones sobre n, orden de Gauss)"):
    ns = sorted({max(12, int(12 * round(k / 12))) for k in np.geomspace(12, max(24, n), 9)})
    filas_conv = []
    for nn in ns:
        zz, rr = integ.remuestrear(zs, rs, nn)
        fila = {"n": nn, "Δz [cm]": (b - a) / nn}
        for nom, fn in (("Trapecio", integ.trapecio), ("Simpson 1/3", integ.simpson),
                        ("Newton-Cotes 3/8", lambda q, w: integ.newton_cotes(q, w, 3)),
                        ("Boole (Newton-Cotes 4)", lambda q, w: integ.newton_cotes(q, w, 4))):
            fila[nom] = math.pi * fn(zz, rr)
        filas_conv.append(fila)
    conv = pd.DataFrame(filas_conv)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for nom in ("Trapecio", "Simpson 1/3", "Newton-Cotes 3/8", "Boole (Newton-Cotes 4)"):
        axs[0].plot(conv["Δz [cm]"], conv[nom], "o-", ms=4, label=nom)
    axs[0].set(xscale="log", xlabel="Δz [cm]", ylabel="volumen [cm³]",
               title="Efecto de Δz (Newton-Cotes)")
    axs[0].legend(fontsize=7)
    ms_ = list(range(1, 13))
    cortes_g = list(np.linspace(z[0], z[-1], paneles + 1)[1:-1])
    vg = [math.pi * integ.gauss(z, r, mm, cortes_g) for mm in ms_]
    vg1 = [math.pi * integ.gauss(z, r, mm, None) for mm in ms_]
    axs[1].plot(ms_, vg, "o-", ms=4, label=f"{paneles} paneles")
    axs[1].plot(ms_, vg1, "s--", ms=4, label="1 panel (global)")
    axs[1].set(xlabel="puntos de Gauss por panel", ylabel="volumen [cm³]",
               title="Influencia del orden de cuadratura")
    axs[1].legend(fontsize=7)
    for a_ in axs:
        a_.grid(alpha=.3)
    fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(conv, hide_index=True, width="stretch", column_config=_fmt(conv))
    st.markdown(
        "- **Trapecio:** su error baja como Δz²; aquí el volumen apenas cambia al refinar porque "
        "con un punto por fila de píxeles Δz ya es muy pequeño.\n"
        "- **Simpson 1/3** necesita **n par**; **Newton-Cotes 3/8**, n múltiplo de 3; **Boole**, "
        "múltiplo de 4. Con aristas en el perfil (hombro, base) su orden alto no se cumple y "
        "quedan cerca del trapecio.\n"
        "- **Gauss:** con un solo panel global, un polinomio debe describir todo el envase y el "
        "resultado oscila con el orden; por paneles converge enseguida.")

# ── 7. validación y fuentes de error ─────────────────────────────────────────
st.header("7. Validación y fuentes de error")
ref = st.number_input("Volumen de referencia del envase [cm³] (opcional)", 0.0, 100000.0,
                      float(verdad["volumen_cm3"]) if verdad else 0.0, step=1.0,
                      key=f"ref_{clave}",
                      help="Un volumen conocido de forma independiente: por ejemplo, un cilindro "
                           "medido con regla (π·r²·h). 0 = no comparar.")
intervalo_completo = a <= 0.01 and b >= round(H, 2) - 0.01
if ref > 0 and not intervalo_completo:
    st.info(f"El intervalo elegido, [{a:.2f}, {b:.2f}] cm, no cubre todo el envase: su volumen "
            "es el de ese tramo y no se compara con la referencia, que es del envase completo.")
if ref > 0 and intervalo_completo:
    tabla_val = tabla[["Método", "Volumen π∫r²dz [cm³]"]].copy()
    tabla_val["Diferencia absoluta [cm³]"] = tabla_val["Volumen π∫r²dz [cm³]"] - ref
    tabla_val["Error relativo [%]"] = 100 * tabla_val["Diferencia absoluta [cm³]"].abs() / ref
    st.dataframe(tabla_val, hide_index=True, width="stretch", column_config=_fmt(tabla_val))
    err = float(tabla_val.iloc[0]["Error relativo [%]"])
    (st.success if err < LIMITE else st.error)(
        f"Error del trapecio frente a la referencia: **{err:.2f} %** "
        f"({'dentro' if err < LIMITE else 'fuera'} del criterio del {LIMITE:.0f} %).")

# presupuesto de error de ESTA medición
px_cm = float(np.linalg.norm(np.diff(
    cam.proyectar(np.array([[0, 0, H / 2], [1, 0, H / 2]])), axis=0)))
lado_px = float(np.median([np.linalg.norm(c[1] - c[0]) for c in marcadores.values()]))
r_med = float(np.median(r_full))
fuentes = pd.DataFrame([
    {"Fuente": "Escala: medida de la hoja (±0.5 mm)",
     "Error estimado en V [%]": 3 * 100 * 0.5 / medida,
     "Cómo se reduce": "Medir con cuidado la distancia ID 0 – ID 1"},
    {"Fuente": "Pose de la cámara (reproyección)",
     "Error estimado en V [%]": 3 * 100 * cam.error_px / max(lado_px, 1),
     "Cómo se reduce": "Marcadores nítidos, más marcadores visibles"},
    {"Fuente": "Extracción del contorno (borde)",
     "Error estimado en V [%]": 2 * 100 * (0.3 if frac_sub > 0.5 else 0.7) / max(r_med * px_cm, 1),
     "Cómo se reduce": "Fondo liso, buen contraste, más resolución"},
    {"Fuente": "Forma: asimetría izq./der.",
     "Error estimado en V [%]": 100 * perfil.asimetria,
     "Cómo se reduce": "Envase de revolución, bien centrado"},
    {"Fuente": "Discretización y método numérico",
     "Error estimado en V [%]": 100 * (tabla["Volumen π∫r²dz [cm³]"].max()
                                      - tabla["Volumen π∫r²dz [cm³]"].min()) / V_principal,
     "Cómo se reduce": "Más puntos (n); en la práctica ya es despreciable"},
])
total = float(np.sqrt((fuentes["Error estimado en V [%]"] ** 2).sum()))
st.markdown(f"**Presupuesto de error de esta medición** · combinado (raíz de la suma de "
            f"cuadrados): **± {total:.1f} %**")
st.dataframe(fuentes, hide_index=True, width="stretch", column_config=_fmt(fuentes))
dom = fuentes.loc[fuentes["Error estimado en V [%]"].idxmax(), "Fuente"]
st.caption(f"Fuente dominante: **{dom}**. La escala pesa ×3 (afecta a r y a z), el borde ×2 "
           "(solo a r²). El método numérico es, con diferencia, lo que menos pesa. Además hay "
           "un sesgo de ~+1 % en envases con base acampanada vista desde arriba (medido en "
           "pruebas sintéticas).")
alertas = []
if perfil.asimetria > 0.05:
    alertas.append("La silueta es muy asimétrica: el contorno puede incluir fondo.")
if math.hypot(*perfil.eje_cm) > 0.5:
    alertas.append("El envase parece descentrado más de 5 mm: céntralo en los círculos.")
if cam.error_px > 1.5:
    alertas.append("Los marcadores se reproyectan con error alto: puede que la hoja no esté plana.")
for al in alertas:
    st.warning(al)

# ── 8. datos para el informe ─────────────────────────────────────────────────
st.header("8. Datos para el informe")
st.caption("Cada bloque tiene un botón para copiar.")
with st.expander("Resultados de los métodos (CSV)"):
    st.code(tabla.to_csv(index=False, float_format="%.6f"), language="text")
with st.expander("Tabla de datos zᵢ, rᵢ, rᵢ², πrᵢ² (CSV)"):
    st.code(tabla_datos.to_csv(index=False, float_format="%.6f"), language="text")
with st.expander("Resumen de la medición"):
    st.code(
        f"Hoja: {tipo_hoja} · distancia ID0–ID1 medida {medida:.1f} mm (escala {100 * escala:.2f} %)\n"
        f"Cámara: focal {cam.focal_px:.0f} px · distancia {math.hypot(C[0], C[1]):.1f} cm · "
        f"altura {C[2]:.1f} cm · inclinación {cam.inclinacion_grados:.1f}° · "
        f"reproyección {cam.error_px:.2f} px\n"
        f"Contorno: {nombre_estr} · asimetría {100 * perfil.asimetria:.1f} % · "
        f"subpíxel {100 * frac_sub:.0f} %\n"
        f"Altura estimada {perfil.altura_estimada_cm:.2f} cm"
        + (f" · medida {altura_medida:.2f} cm" if altura_medida else "") + "\n"
        f"Intervalo [{a:.2f}, {b:.2f}] cm · n = {n} · Δz = {dz:.5f} cm\n"
        f"Volumen (trapecio) {V_principal:.2f} cm³ · presupuesto de error ±{total:.1f} %"
        + (f"\nReferencia {ref:.2f} cm³ · error {abs(V_principal - ref) / ref * 100:.2f} %"
           if ref > 0 and intervalo_completo else ""),
        language="text")
