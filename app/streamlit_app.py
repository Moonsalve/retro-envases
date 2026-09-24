"""Del píxel al volumen — aplicación web.

Ejecutar desde la raíz del repo:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from retroenvases import integracion as integ
from retroenvases.foto import Foto, cargar
from retroenvases.pipeline import estimar_volumen
from retroenvases.sintetico import BOTELLA_600, Escena, generar_perspectiva, volumen_exacto

st.set_page_config(page_title="Del píxel al volumen", page_icon="🧴", layout="wide")

LIMITE = 5.0      # criterio del reto: error relativo máximo [%]
DESCARGAS = pathlib.Path.home() / "Downloads"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}


def _fotos_en_descargas(maximo: int = 30) -> list[pathlib.Path]:
    """Imágenes de ~/Downloads, de la más reciente a la más antigua."""
    if not DESCARGAS.is_dir():
        return []
    fotos = [f for f in DESCARGAS.iterdir() if f.suffix.lower() in EXTENSIONES and f.is_file()]
    return sorted(fotos, key=lambda f: f.stat().st_mtime, reverse=True)[:maximo]


# ─────────────────────────────────────────────────────────── utilidades ──
@st.cache_data(show_spinner=False)
def _cargar(datos: bytes) -> Foto:
    return cargar(datos)


@st.cache_data(show_spinner=False)
def _ejemplo() -> tuple[bytes, dict]:
    """Foto sintética de una botella de volumen exacto, con cámara a 50 cm."""
    img, v = generar_perspectiva(BOTELLA_600, Escena(), distancia_cm=50)
    ok, buf = cv2.imencode(".png", img)
    h, w = img.shape[:2]
    v["focal_35mm"] = v["focal_px"] * 43.27 / math.hypot(w, h)
    return buf.tobytes(), v


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _con_rectangulo(bgr: np.ndarray, rect) -> np.ndarray:
    out = bgr.copy()
    if rect:
        x, y, w, h = rect
        grosor = max(2, bgr.shape[0] // 300)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), grosor)
    return _rgb(out)


def _con_contorno(bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    m = mascara > 0
    out[m] = (0.55 * out[m] + 0.45 * np.array([0, 200, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (0, 255, 255), max(2, bgr.shape[0] // 400))
    return _rgb(out)


def _metodos(z, r, m_gauss, cortes):
    """Los 4 métodos con su tiempo y sus parámetros; los que no aplican, con el motivo."""
    n, dz = len(z) - 1, float(z[1] - z[0])
    tabla, avisos = [], []
    for nombre, fn, orden, evals, delta in [
        ("Trapecio", lambda: integ.trapecio(z, r), "O(Δz²)", n + 1, dz),
        ("Simpson", lambda: integ.simpson(z, r), "O(Δz⁴)", n + 1, dz),
        ("Newton-Cotes 3/8", lambda: integ.newton_cotes(z, r, 3), "O(Δz⁴)", n + 1, dz),
        (f"Gauss-Legendre m={m_gauss}", lambda: integ.gauss(z, r, m_gauss, cortes),
         f"exacto hasta grado {2 * m_gauss - 1}", m_gauss * (len(cortes) + 1), None),
    ]:
        try:
            t0 = time.perf_counter()
            I = fn()
            ms = 1000 * (time.perf_counter() - t0)
        except integ.CondicionNoSatisfecha as e:
            avisos.append(f"**{nombre}** no aplica: {e}")
            continue
        tabla.append({"Método": nombre, "n": n, "Evaluaciones": evals,
                      "a [cm]": float(z[0]), "b [cm]": float(z[-1]),
                      "Δz [cm]": delta, "Orden / exactitud": orden,
                      "I = ∫r²dz [cm³]": I, "V = πI [cm³]": math.pi * I,
                      "Tiempo [ms]": ms})
    return pd.DataFrame(tabla), avisos


# ─────────────────────────────────────────────────────────────── página ──
st.title("🧴 Del píxel al volumen")
st.caption("Estimación del volumen de un envase a partir de una foto, modelándolo como "
           "sólido de revolución:  V = π ∫ r(z)² dz")

with st.sidebar:
    st.header("📸 Cómo tomar la foto")
    st.markdown(
        "1. **Distancia 40–100 cm**, sin zoom.\n"
        "2. **Cámara entre la base y la boca** del envase, idealmente a media altura. "
        "Nunca por encima.\n"
        "3. **Celular vertical** (usa el nivel de la cámara), o anota su inclinación.\n"
        "4. **Fondo liso** que contraste con el envase; el envase no debe tocar los "
        "bordes de la foto.\n"
        "5. **Mide la altura del envase al medio milímetro.**\n\n"
        "Si la foto llega por WhatsApp pierde la focal (EXIF): se usa la típica de "
        "la cámara principal de un celular, 26 mm. Mándala como *Documento* para "
        "conservarla.")
    st.divider()
    st.header("⚙️ Parámetros numéricos")
    n_usuario = st.number_input(
        "Subintervalos n (0 = automático)", 0, 5000, 0, step=6,
        help="Automático: una muestra por fila de píxeles, redondeado a múltiplo de 6 "
             "(Simpson pide n par y Newton-Cotes 3/8, múltiplo de 3).")
    m_gauss = st.slider("Nodos de Gauss por panel (m)", 2, 10, 5)
    paneles = st.slider("Paneles de Gauss", 1, 12, 6,
                        help="Gauss se aplica por paneles: sobre un perfil con aristas, "
                             "la regla global falla.")

# 1 ── foto ────────────────────────────────────────────────────────────────
st.subheader("1. La foto")
fuente = st.radio("Origen", ["Foto de Descargas", "Otro archivo", "Ejemplo sintético"],
                  horizontal=True)
datos, verdad = None, None
if fuente == "Foto de Descargas":
    fotos = _fotos_en_descargas()
    if not fotos:
        st.warning(f"No hay imágenes (.jpg, .png, .webp) en {DESCARGAS}.")
    else:
        elegida = st.selectbox(
            "Foto (la más reciente primero)", fotos,
            format_func=lambda f: f"{f.name}  ·  "
                                  f"{time.strftime('%d/%m %H:%M', time.localtime(f.stat().st_mtime))}")
        datos = elegida.read_bytes()
elif fuente == "Otro archivo":
    arch = st.file_uploader("Foto del envase", type=["jpg", "jpeg", "png", "webp"])
    datos = arch.getvalue() if arch else None
else:
    datos, verdad = _ejemplo()
    st.info(f"Botella sintética de volumen exacto **{verdad['volumen_cm3']:.2f} cm³**, "
            f"fotografiada con una cámara a {verdad['distancia_cm']:.0f} cm. "
            "Sirve para comprobar que todo funciona.")

if not datos:
    st.stop()

foto = _cargar(datos)
H_img, W_img = foto.bgr.shape[:2]
# Los datos que dependen de la foto (altura, focal, referencia) llevan una clave
# ligada a ella: al cambiar de foto vuelven a su valor inicial en vez de arrastrar
# los de la anterior.
clave = str(hash(datos))

# 2 ── escala y selección ───────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 1])
with col_izq:
    st.subheader("2. Escala")
    altura = st.number_input(
        "Altura real del envase [cm]", 1.0, 80.0,
        float(verdad["altura_cm"]) if verdad else 20.0, step=0.05, format="%.2f",
        key=f"altura_{clave}",
        help="Es la única referencia de escala. Su error se multiplica ×3 en el volumen: "
             "1 mm de error en una lata de 12 cm son 2.5 %.")
    st.caption("La escala se obtiene de esta altura: los píxeles que mide la silueta "
               "corresponden a estos centímetros. Supuesto: el envase es un sólido de "
               "revolución con su eje vertical.")

    st.subheader("3. Selección del envase")
    metodo = st.radio("Segmentación", ["GrabCut con rectángulo (recomendado)",
                                       "GrabCut automático", "Otsu (solo fondo liso)"])
    rect = None
    if metodo.startswith("GrabCut con"):
        rx = st.slider("Rectángulo: horizontal [%]", 0, 100, (25, 75), key=f"rx_{clave}")
        ry = st.slider("Rectángulo: vertical [%]", 0, 100, (5, 95), key=f"ry_{clave}")
        rect = (int(rx[0] / 100 * W_img), int(ry[0] / 100 * H_img),
                int((rx[1] - rx[0]) / 100 * W_img), int((ry[1] - ry[0]) / 100 * H_img))
        st.caption("Ajusta el rectángulo para que contenga todo el envase y lo menos "
                   "posible del resto.")

    st.subheader("4. Perspectiva")
    corregir = st.checkbox("Corregir perspectiva", True,
                           help="Sin corregir, a 50 cm el volumen sale ~10 % bajo.")
    f35 = pitch = None
    if corregir:
        if verdad:
            st.success(f"Ejemplo sintético: focal conocida, "
                       f"**{verdad['focal_35mm']:.1f} mm** equivalente 35 mm.")
        elif foto.focal_35mm:
            st.success(f"Focal leída de la foto: **{foto.focal_35mm:.0f} mm** "
                       f"(equivalente 35 mm){' · ' + foto.modelo if foto.modelo else ''}")
        else:
            st.info("La foto no trae la focal (EXIF), lo normal si llegó por WhatsApp. "
                    "Se usa **26 mm**, la típica de la cámara principal de un celular; "
                    "si conoces la de tu teléfono, cámbiala. Un error de ±10 % en la "
                    "focal mete hasta 2.4 % en el volumen.")
        f35 = st.number_input("Focal equivalente 35 mm [mm]", 10.0, 200.0,
                              float(foto.focal_35mm or (verdad or {}).get("focal_35mm") or 26.0),
                              step=0.5, format="%.1f", key=f"focal_{clave}")
        pitch = st.number_input("Inclinación del celular [°] (+ = apunta hacia abajo)",
                                -30.0, 30.0, 0.0, step=0.5, key=f"pitch_{clave}")
    referencia = st.number_input(
        "Volumen de referencia [cm³] (opcional, para comparar)", 0.0, 100000.0,
        float(verdad["volumen_cm3"]) if verdad else 0.0, step=1.0, key=f"ref_{clave}",
        help="Un volumen conocido del mismo envase: por ejemplo, calculado con regla para "
             "un cilindro (π r² h). 0 = no comparar.")

with col_der:
    st.image(_con_rectangulo(foto.bgr, rect), caption=f"Foto de trabajo · {W_img}×{H_img} px",
             width="stretch")

calcular = st.button("Calcular volumen", type="primary", width="stretch")
if calcular:
    seg = "otsu" if metodo.startswith("Otsu") else "grabcut"
    try:
        with st.spinner("Segmentando y reconstruyendo el perfil…"):
            est = estimar_volumen(
                foto.bgr, altura, seg, n=int(n_usuario) or None, m_gauss=m_gauss,
                paneles_gauss=paneles, rect=rect,
                focal_px=foto.focal_px(f35) if corregir else None,
                pitch_grados=pitch or 0.0)
        st.session_state["est"] = est
        st.session_state["ref"] = referencia
        st.session_state["clave_est"] = clave
        st.session_state["rect_est"] = rect
    except Exception as e:           # un fallo aquí es de datos, no del usuario
        st.error(f"No se pudo estimar el volumen: {e}")
        st.session_state.pop("est", None)

est = st.session_state.get("est")
if est is None or st.session_state.get("clave_est") != clave:
    st.stop()

# 5 ── resultados ─────────────────────────────────────────────────────────
st.divider()
st.subheader("5. Resultados")
z, r = est.z_cm, est.r_cm
cortes = list(np.linspace(z[0], z[-1], paneles + 1)[1:-1])
tabla, avisos = _metodos(z, r, m_gauss, cortes)
ref = st.session_state.get("ref") or 0.0
if ref > 0:
    tabla["Diferencia [cm³]"] = tabla["V = πI [cm³]"] - ref
    tabla["Error relativo [%]"] = 100 * tabla["Diferencia [cm³]"].abs() / ref

p = est.perfil
alertas = []
if p.asimetria > 0.03:
    alertas.append("La silueta es asimétrica (> 3 %): puede no ser un sólido de revolución, "
                   "o el contorno tomó parte del fondo. Revisa el contorno y el rectángulo.")
if p.subpixel < 0.5:
    alertas.append("Menos de la mitad de los bordes se refinaron con precisión subpíxel "
                   "(fondo no uniforme). El borde tiene precisión de ~1 px.")
ys, xs = np.nonzero(p.mascara)
if ys.size and (ys.min() <= 1 or xs.min() <= 1 or ys.max() >= p.mascara.shape[0] - 2
                or xs.max() >= p.mascara.shape[1] - 2):
    alertas.append("El contorno toca el borde de la foto: probablemente incluye fondo, o el "
                   "envase quedó cortado.")
if est.perspectiva:
    if est.perspectiva.altura_camara_cm > altura:
        alertas.append("La cámara estaba por encima de la boca del envase: la corrección de "
                       "perspectiva pierde precisión (hasta ~5 %). Repite la foto más abajo.")
    if est.perspectiva.distancia_cm < 35:
        alertas.append("La cámara estaba a menos de 35 cm: aléjate a 40–100 cm.")
no_confiable = []
rect_usado = st.session_state.get("rect_est")
if rect_usado and est.perfil.rotacion is None:
    x, y, w, h = rect_usado
    ys_m, xs_m = np.nonzero(p.mascara)
    toca = (ys_m.min() <= y + 1 or ys_m.max() >= y + h - 2 or
            xs_m.min() <= x + 1 or xs_m.max() >= x + w - 2) if ys_m.size else False
    if toca:
        no_confiable.append("el contorno toca el rectángulo: probablemente el rectángulo "
                            "corta el envase. Agrándalo para que lo contenga entero")
if p.asimetria > 0.05:
    no_confiable.append(f"la silueta es muy asimétrica ({100 * p.asimetria:.0f} %): el "
                        "contorno casi seguro incluye fondo. Ajusta el rectángulo")
if est.perspectiva and est.perspectiva.distancia_cm < 25:
    no_confiable.append("la cámara estaba muy cerca (< 25 cm)")
if no_confiable:
    st.error("⚠️ **Resultado no confiable:** " + "; ".join(no_confiable) + ".")

principal = tabla.iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Volumen (trapecio)", f"{principal['V = πI [cm³]']:.1f} cm³")
c2.metric("= mililitros", f"{principal['V = πI [cm³]']:.1f} mL")
c3.metric("Δz", f"{principal['Δz [cm]']:.4f} cm")
if ref > 0:
    err = principal["Error relativo [%]"]
    c4.metric("Error vs referencia", f"{err:.2f} %",
              "dentro del 5 %" if err < LIMITE else "fuera del 5 %",
              delta_color="normal" if err < LIMITE else "inverse")

g1, g2 = st.columns(2)
with g1:
    mascara = est.perfil.mascara
    if est.perfil.rotacion is not None:          # deshacer el enderezado para superponer
        inv = cv2.invertAffineTransform(est.perfil.rotacion)
        mascara = cv2.warpAffine(mascara, inv, (W_img, H_img), flags=cv2.INTER_NEAREST)
    st.image(_con_contorno(foto.bgr, mascara), caption="Contorno detectado",
             width="stretch")
with g2:
    fig, ax = plt.subplots(2, 1, figsize=(6, 6.4))
    ax[0].fill_between(z, -r, r, alpha=.3, color="#1f77b4")
    ax[0].plot(z, r, z, -r, color="#1f77b4", lw=1.6)
    ax[0].axhline(0, ls="--", color="#d62728", lw=1)
    ax[0].set(xlabel="z [cm]", ylabel="radio [cm]", title="Perfil en coordenadas físicas")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].fill_between(z, 0, math.pi * r ** 2, alpha=.3, color="#2ca02c")
    ax[1].plot(z, math.pi * r ** 2, color="#2ca02c", lw=1.6)
    ax[1].set(xlabel="z [cm]", ylabel="π·r(z)² [cm²]",
              title="Integrando: el volumen es el área bajo la curva")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("**Comparación de los cuatro métodos** (todos sobre la misma tabla de datos)")
st.dataframe(tabla, hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.4f")
                            for c in tabla.columns if tabla[c].dtype.kind == "f"})
for a in avisos:
    st.warning(a)

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.barh(tabla["Método"], tabla["V = πI [cm³]"], color="#4c72b0", alpha=.85)
if ref > 0:
    ax.axvline(ref, color="k", lw=1.5, label="referencia")
    ax.axvspan(ref * (1 - LIMITE / 100), ref * (1 + LIMITE / 100), color="green", alpha=.1,
               label="±5 %")
    ax.legend(fontsize=8)
vmin = tabla["V = πI [cm³]"].min()
ax.set_xlim(vmin * 0.9, max(tabla["V = πI [cm³]"].max(), ref) * 1.08)
ax.set(xlabel="volumen [cm³]", title="Volumen por método")
ax.grid(axis="x", alpha=.3)
fig.tight_layout()
st.pyplot(fig)

# 6 ── diagnóstico ────────────────────────────────────────────────────────
st.subheader("6. Diagnóstico de la medición")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Escala (sin perspectiva)", f"{p.px_por_cm:.2f} px/cm")
d2.metric("Inclinación corregida", f"{p.inclinacion_grados:+.2f}°")
d3.metric("Asimetría izq./der.", f"{100 * p.asimetria:.1f} %")
d4.metric("Bordes con precisión subpíxel", f"{100 * p.subpixel:.0f} %")
if est.perspectiva:
    e1, e2 = st.columns(2)
    e1.metric("Distancia estimada de la cámara", f"{est.perspectiva.distancia_cm:.1f} cm")
    e2.metric("Altura estimada de la cámara", f"{est.perspectiva.altura_camara_cm:.1f} cm")

for a in alertas:
    st.warning(a)
if not alertas:
    st.success("Sin alertas: la foto cumple las condiciones del modelo.")

# 7 ── datos para el informe ──────────────────────────────────────────────
with st.expander("Tabla de datos (zᵢ, rᵢ, rᵢ², π·rᵢ²) — Etapa 4"):
    datos_tabla = pd.DataFrame({"i": np.arange(len(z)), "z_i [cm]": z, "r_i [cm]": r,
                                "r_i² [cm²]": r ** 2, "π·r_i² [cm²]": math.pi * r ** 2})
    st.dataframe(datos_tabla, hide_index=True, width="stretch", height=320)
    st.caption("Selecciona celdas y copia (Ctrl/Cmd + C) para pegarlas en el informe.")

with st.expander("Resultados en texto, para copiar al informe"):
    st.code(tabla.to_csv(index=False, float_format="%.6f"), language="text")

with st.expander("Fuentes de error y cómo se atienden"):
    st.markdown(
        "| Fuente | Efecto en el volumen | Cómo se atiende |\n"
        "|---|---|---|\n"
        "| Altura medida con regla | **×3** su error relativo | Medir al medio milímetro |\n"
        "| Perspectiva de la cámara | −5 % a −25 % sin corregir | Corrección con la focal "
        "(residuo ≤ 1 % a 35–200 cm) |\n"
        "| Celular inclinado | hasta 6 % a ±10° | Informar la inclinación, o nivelar |\n"
        "| Focal equivocada | hasta 2.4 % con ±10 % | Subir la foto original (EXIF) |\n"
        "| Contorno (1 px de borde) | ~2× el error de radio | Fondo liso, rectángulo, subpíxel |\n"
        "| Envase no simétrico | variable | Se reporta la asimetría |\n"
        "| Método de integración | < 0.05 % | Irrelevante frente a lo anterior |")
