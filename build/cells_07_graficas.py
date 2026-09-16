# -*- coding: utf-8 -*-
"""Celdas: graficas del perfil, sensibilidad, exportacion y cierre."""

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

M(r"""---
## 9. Representación gráfica — §8 del PDF

El enunciado pide *"representación gráfica del perfil y, opcionalmente, de la función
πr²"*. Hacemos las dos, más dos visualizaciones adicionales que ayudan a interpretar.

**Qué hace esta celda:** cuatro paneles.

1. **Perfil $r(z)$** con los nodos de la malla y los puntos de quiebre marcados. Es la
   silueta del envase que, en la fase de cámara, saldrá del contorno de la foto.
2. **Integrando $f(z) = \pi r(z)^2$** — el área del disco a cada altura. **El volumen es
   literalmente el área sombreada bajo esta curva.** Es la traducción visual de la
   integral.
3. **Sección longitudinal completa**, reflejando el perfil sobre el eje. Es la forma en
   que reconoces el envase a simple vista, y sirve para validar de un vistazo que las
   proporciones editadas en `TRAMOS` son las que pretendías.
4. **Aporte acumulado al volumen**: $V(z) = \pi\int_a^z r^2\,d\zeta$, normalizado. Dice
   **qué fracción del volumen total se ha acumulado a cada altura**, y por tanto dónde
   es crítico medir bien.""")

C(r'''z_fino = np.linspace(ENVASE.a, ENVASE.b, 1500)
r_fino = ENVASE.radio(z_fino)
f_fino = math.pi * r_fino ** 2
quiebres = ENVASE.puntos_de_quiebre()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# --- Panel 1: perfil r(z) ---------------------------------------------
ax = axes[0, 0]
ax.plot(z_fino, r_fino, "-", color="#1f77b4", lw=2, label="perfil r(z) continuo")
ax.plot(Z_BASE, R_BASE, "o", color="#d62728", ms=4.5,
        label=f"nodos de la malla (n={N_BASE})")
for q in quiebres:
    ax.axvline(q, color="gray", ls=":", lw=1.2, alpha=0.8)
ax.annotate("puntos de quiebre\n(no derivable)", xy=(quiebres[1], float(ENVASE.radio(quiebres[1])[0])),
            xytext=(quiebres[1] - 6.5, 2.0), fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
ax.set_xlabel("z  [cm]"); ax.set_ylabel("r(z)  [cm]")
ax.set_title("1. Perfil del envase — lo que dará el contorno de la foto", fontsize=10)
ax.legend(fontsize=8); ax.set_ylim(0, None)

# --- Panel 2: integrando pi*r^2 ---------------------------------------
ax = axes[0, 1]
# El área bajo π·r² es el VOLUMEN: [cm²]·[cm] = [cm³].
ax.fill_between(z_fino, 0, f_fino, alpha=0.28, color="#2ca02c",
                label=f"área bajo la curva = V = {V_EXACTO:.2f} cm³")
ax.plot(z_fino, f_fino, "-", color="#2ca02c", lw=2)
ax.plot(Z_BASE, math.pi * R_BASE ** 2, "o", color="#d62728", ms=4)
for q in quiebres:
    ax.axvline(q, color="gray", ls=":", lw=1.2, alpha=0.8)
ax.set_xlabel("z  [cm]"); ax.set_ylabel("f(z) = π·r(z)²  [cm²]")
ax.set_title("2. Integrando: área del disco a cada altura", fontsize=10)
ax.legend(fontsize=8)

# --- Panel 3: seccion longitudinal ------------------------------------
ax = axes[1, 0]
ax.fill_between(z_fino, -r_fino, r_fino, alpha=0.30, color="#1f77b4")
ax.plot(z_fino,  r_fino, "-", color="#1f77b4", lw=1.8)
ax.plot(z_fino, -r_fino, "-", color="#1f77b4", lw=1.8)
ax.axhline(0, color="#d62728", ls="--", lw=1.3, label="eje de revolución")
_r_max = float(r_fino.max())
for t in ENVASE.tramos:
    ax.axvline(t.z_fin, color="gray", ls=":", lw=0.9, alpha=0.7)
    # Solo etiquetamos los tramos con hueco suficiente; si no, se solapan.
    if t.altura >= 0.10 * ENVASE.altura:
        ax.text(0.5 * (t.z_ini + t.z_fin), _r_max * 1.18, t.nombre,
                ha="center", va="bottom", fontsize=7.5, color="dimgray")
ax.set_xlabel("z  [cm]"); ax.set_ylabel("radio  [cm]")
ax.set_title("3. Sección longitudinal del sólido de revolución", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(-_r_max * 1.45, _r_max * 1.55)
ax.set_aspect("equal", adjustable="box")

# --- Panel 4: volumen acumulado ---------------------------------------
ax = axes[1, 1]
V_acum = math.pi * integrate.cumulative_trapezoid(r_fino ** 2, z_fino, initial=0.0)
ax.plot(z_fino, 100 * V_acum / V_EXACTO, "-", color="#9467bd", lw=2.2)
for t in ENVASE.tramos:
    ax.axvline(t.z_fin, color="gray", ls=":", lw=0.9, alpha=0.7)
# Marcar el tramo que más aporta
_res = ENVASE.resumen()
_dom = _res.loc[_res["V tramo [cm³]"].idxmax()]
ax.axvspan(_dom["z_ini [cm]"], _dom["z_fin [cm]"], color="#9467bd", alpha=0.12)
ax.text(0.5 * (_dom["z_ini [cm]"] + _dom["z_fin [cm]"]), 45,
        f"{_dom['tramo']}\n{_dom['% del total']:.1f} % del volumen",
        ha="center", fontsize=8.5, color="#5b3a8e", weight="bold")
ax.set_xlabel("z  [cm]"); ax.set_ylabel("volumen acumulado  [% del total]")
ax.set_title("4. ¿Dónde está el volumen? — dónde hay que medir bien", fontsize=10)
ax.set_ylim(0, 105)

fig.suptitle(f"{ENVASE.nombre} — V exacto = {V_EXACTO:.4f} cm³ = {V_EXACTO:.2f} mL",
             fontsize=12.5)
fig.tight_layout()
plt.show()''')

M(r"""**Cómo leer las gráficas:**

- **Panel 1** — los puntos rojos son la tabla de datos; la línea azul es el envase real.
  La **distancia entre puntos rojos es Δz**. Si imaginas que cada punto rojo es una fila
  de píxeles, estás viendo exactamente lo que entregará la fase de cámara. Las líneas
  punteadas verticales son los quiebres: observa que en general **no coinciden con
  puntos rojos** — esa desalineación es la causa de todo lo visto en la Sección 8.
- **Panel 2** — el volumen **es** el área verde sombreada. Nota el escalón brusco en el
  hombro: $\pi r^2$ se desploma de ~36 a ~5.7 cm² en apenas 3.5 cm. Esa caída abrupta es
  la zona donde los métodos numéricos sufren.
- **Panel 3** — la silueta reconocible. Úsalo como control visual tras editar `TRAMOS`:
  si no parece un envase, las proporciones están mal.
- **Panel 4 — el panel con la moraleja práctica.** La curva es empinada donde el envase
  es ancho y casi plana en el cuello. La banda sombreada marca el tramo dominante. Como
  el cuerpo cilíndrico aporta **~82 %** del volumen, la precisión del resultado final
  depende sobre todo de medir bien **el radio del cuerpo**. Un error de 1 mm ahí cuesta
  ~18 cm³ (3 % del total); el mismo error en el cuello cuesta menos de 0.5 cm³. En la
  prueba de campo, esto dice **dónde enfocar la cámara y dónde poner la referencia de
  escala**.""")

M(r"""---
## 9.1 Análisis de sensibilidad: por qué la escala manda sobre el método

**Qué hace esta celda:** cuantifica y compara las dos fuentes de error que competirán en
el proyecto real.

1. **Error del método numérico** — el que acabamos de medir, con datos perfectos.
2. **Error de escala** — el que aparecerá al convertir píxeles a centímetros.

Simulamos un error de calibración introduciendo un factor $s$ en todos los radios:
$r \to (1+s)\,r$. Como $V \propto r^2$:

$$\frac{V(s)}{V} = (1+s)^2 \;\approx\; 1 + 2s$$

**Este es el argumento que cierra el diseño del proyecto**, y conviene tenerlo con
números concretos antes de escribir una sola línea de código de visión por computador.""")

C(r'''errores_escala = np.array([0.001, 0.005, 0.01, 0.02, 0.025, 0.05])
filas_sens = []
for s in errores_escala:
    # Reutilizamos nuestra propia regla del trapecio sobre la malla fina:
    # coherente con el resto del notebook y sin dependencias de versión de numpy.
    V_s = regla_trapecio(z_fino, (1 + s) * r_fino).volumen_cm3
    filas_sens.append({
        "Error de escala [%]": 100 * s,
        "Factor (1+s)²": (1 + s) ** 2,
        "V estimado [cm³]": V_s,
        "Error en V [cm³]": V_s - V_EXACTO,
        "Error en V [%]": 100 * (V_s - V_EXACTO) / V_EXACTO,
        "¿Pasa el 5 %?": "✓ sí" if abs(100 * (V_s - V_EXACTO) / V_EXACTO) < 5 else "✗ NO",
    })
SENS = pd.DataFrame(filas_sens)

print("=" * 78)
print("SENSIBILIDAD AL ERROR DE ESCALA  (el radio entra al CUADRADO)")
print("=" * 78)
display(SENS)

# Separamos las Newton-Cotes de Gauss: son regímenes muy distintos.
_nc = TABLA_COMP[TABLA_COMP["Familia"].str.contains("Newton-Cotes")]
_err_nc_mejor = _nc["Err. rel. [%]"].min()
_err_nc_peor = _nc["Err. rel. [%]"].max()
_err_escala_1 = 100 * ((1.01) ** 2 - 1)

print("=" * 78)
print("COMPARACIÓN DE LAS DOS FUENTES DE ERROR")
print("-" * 78)
print(f"  (A) Error del MÉTODO numérico, con n = {N_BASE} y datos PERFECTOS:")
print(f"        mejor Newton-Cotes : {_err_nc_mejor:.3e} %")
print(f"        peor  Newton-Cotes : {_err_nc_peor:.3e} %")
print(f"  (B) Error por ESCALA mal calibrada al 1 %:")
print(f"        resulta            : {_err_escala_1:.3f} %")
print("-" * 78)
print(f"  Un error de escala de apenas 1 % pesa:")
print(f"     {_err_escala_1 / _err_nc_peor:,.0f}× más que el PEOR Newton-Cotes")
print(f"     {_err_escala_1 / _err_nc_mejor:,.0f}× más que el MEJOR Newton-Cotes")
print("-" * 78)
print(f"  Y el error del método se puede reducir GRATIS aumentando n:")
for _n in (60, 240, 1200):
    _zt, _rt = muestrear_perfil(ENVASE, _n)
    _e = 100 * abs(regla_trapecio(_zt, _rt).volumen_cm3 - V_EXACTO) / V_EXACTO
    print(f"     n = {_n:>5}  →  trapecio con {_e:.3e} %   "
          f"({_err_escala_1 / _e:,.0f}× menor que el error de escala del 1 %)")
print("=" * 78)
print("  → El presupuesto de error del 5 % se gasta casi íntegramente en la")
print("    calibración píxel→cm y en la extracción del contorno, NO en la cuadratura.")
print("    Escala con error < 2.5 %  ⇒  volumen con error < 5 %.")
print("=" * 78)''')

M(r"""**Cómo leer el resultado:**

- **La columna `Error en V [%]` es aproximadamente el doble de la columna
  `Error de escala [%]`.** Es la propagación $\Delta V/V = 2\,\Delta r/r$ visible en
  datos, no en una fórmula.
- **La fila del 2.5 % es la frontera:** ahí el error de volumen llega justo al 5 %. Ese
  es el **presupuesto de calibración** del proyecto: la escala píxel→cm debe tener un
  error menor que 2.5 %, y con margen de seguridad, menor que 1 %.
- **Los multiplicadores** son la conclusión cuantitativa del notebook. Con $n = 60$, un
  error de escala del 1 % pesa unas **40 veces más** que el peor Newton-Cotes y unas
  **200 veces más** que el mejor. Y el bloque final muestra algo todavía más decisivo:
  **el error del método se reduce gratis subiendo $n$** — con $n = 1200$ el trapecio baja
  a ~3e-5 %, unas 76 000 veces por debajo del error de escala.
- **Por qué esa asimetría lo decide todo:** subir $n$ no cuesta nada (son microsegundos
  de CPU y la foto ya tiene esos píxeles). Mejorar la calibración de la escala cuesta
  imprimir bien un marcador, medirlo con regla, montar la escena y controlar la
  perspectiva. **Una de las dos fuentes de error es gratis de eliminar y la otra no**, y
  resulta ser justo la cara la que domina.

> **Qué implica para lo que viene.** El esfuerzo de la fase 2 debe concentrarse en:
> (a) que el marcador de referencia esté **medido con precisión real** — imprimirlo y
> verificar con regla, porque las impresoras escalan sin avisar; (b) que esté
> **coplanar con el eje del envase**; (c) que la foto se tome **de lejos con zoom**,
> para aproximar la proyección ortográfica que supone nuestro modelo; y (d) que el
> contorno se extraiga limpio. El motor de integración ya está resuelto y sobra de
> precisión.""")

M(r"""---
## 10. Exportación de datos y resultados — §8 del PDF

El enunciado pide *"exportación o posibilidad de copiar los datos/resultados para el
informe"*.

**Qué hace esta celda:** genera cuatro archivos y los ofrece para descarga:

| Archivo | Contenido |
|---|---|
| `perfil_datos.csv` | la tabla $(i, z_i, r_i, r_i^2, \pi r_i^2)$ — §7 Etapa 4 |
| `comparacion_metodos.csv` | la tabla comparativa completa — §7 Etapa 5 y 6 |
| `convergencia.csv` | el barrido de errores frente a $\Delta z$ — §6 |
| `resumen_informe.md` | un resumen en Markdown, listo para pegar |

En Colab aparecerán en el panel de archivos (📁 a la izquierda) y se dispara la descarga
automática. Fuera de Colab, se escriben en el directorio de trabajo.""")

C(r'''import io, os, json

def _a_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """DataFrame -> tabla Markdown, con respaldo si falta el paquete `tabulate`.

    pandas.DataFrame.to_markdown() delega en `tabulate`, que no siempre está
    instalado. En vez de dejar que la exportación falle por una dependencia
    opcional, generamos la tabla a mano. El objetivo de esta celda es entregar
    archivos al usuario; no puede caerse por un detalle de empaquetado.
    """
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except (ImportError, AttributeError):
        def _fmt(v):
            if isinstance(v, float):
                return f"{v:{floatfmt}}"
            return str(v)
        cabecera = "| " + " | ".join(str(c) for c in df.columns) + " |"
        separador = "|" + "|".join("---" for _ in df.columns) + "|"
        filas = ["| " + " | ".join(_fmt(v) for v in fila) + " |"
                 for fila in df.itertuples(index=False, name=None)]
        return "\n".join([cabecera, separador, *filas])


ARCHIVOS: dict[str, str] = {}

# --- 1. Tabla de datos del perfil --------------------------------------
TABLA_BASE.to_csv("perfil_datos.csv", index=False, encoding="utf-8-sig")
ARCHIVOS["perfil_datos.csv"] = f"{len(TABLA_BASE)} filas — tabla (z, r, r², πr²)"

# --- 2. Comparación de métodos -----------------------------------------
TABLA_COMP.to_csv("comparacion_metodos.csv", index=False, encoding="utf-8-sig")
ARCHIVOS["comparacion_metodos.csv"] = f"{len(TABLA_COMP)} métodos con todos sus parámetros"

# --- 3. Estudio de convergencia ----------------------------------------
_conv = pd.concat([
    CONV_CIEGA.assign(Malla="ciega"),
    CONV_ALIN.rename(columns={"n_total": "n"}).assign(Malla="alineada"),
], ignore_index=True)
_conv.to_csv("convergencia.csv", index=False, encoding="utf-8-sig")
ARCHIVOS["convergencia.csv"] = f"{len(_conv)} puntos de convergencia (ambas mallas)"

# --- 4. Resumen en Markdown --------------------------------------------
_buf = io.StringIO()
_buf.write(f"# Resultados — {ENVASE.nombre}\n\n")
_buf.write("## Parámetros del envase\n\n")
_buf.write(f"- Intervalo de integración: [{ENVASE.a:.4f}, {ENVASE.b:.4f}] cm\n")
_buf.write(f"- Altura total: {ENVASE.altura:.4f} cm\n")
_buf.write(f"- Puntos de quiebre: {[round(float(q), 4) for q in quiebres]} cm\n")
_buf.write(f"- Volumen exacto de referencia: **{V_EXACTO:.6f} cm³ = {V_EXACTO:.2f} mL**\n")
_buf.write(f"- Integral exacta ∫r²dz: {I_EXACTA:.6f} cm³\n\n")
_buf.write("### Composición por tramos\n\n")
_buf.write(_a_markdown(ENVASE.resumen(), ".4f"))
_buf.write("\n\n## Discretización\n\n")
_buf.write(f"- n (subintervalos): {N_BASE}\n")
_buf.write(f"- Puntos: {N_BASE + 1}\n")
_buf.write(f"- **Δz = (b − a)/n = {DZ_BASE:.6f} cm**\n")
_buf.write(f"- Malla uniforme; n múltiplo de 12 (compatible con los cuatro métodos)\n\n")
_buf.write("## Comparación de métodos\n\n")
_buf.write(_a_markdown(TABLA_COMP[["Método", "n", "Evals", "Δz [cm]",
                                   "Orden / exactitud", "I = ∫r²dz [cm³]",
                                   "V = πI [cm³]", "Err. abs. [cm³]",
                                   "Err. rel. [%]", "Tiempo [ms]"]], ".6g"))
_buf.write("\n\n## Órdenes de convergencia observados\n\n")
_buf.write("| Método | Teórico | Malla ciega | Malla alineada |\n|---|---|---|---|\n")
def _fmt_orden(v: float) -> str:
    """NaN aquí significa 'exacto salvo redondeo', no 'no se pudo calcular'."""
    return "exacto (suelo de redondeo)" if math.isnan(v) else f"{v:.2f}"

for met, p_teo in TEORICO.items():
    _buf.write(f"| {met} | {p_teo} | {_fmt_orden(orden_observado(CONV_CIEGA, met))} | "
               f"{_fmt_orden(orden_observado(CONV_ALIN, met))} |\n")
_buf.write("\n## Sensibilidad al error de escala\n\n")
_buf.write(_a_markdown(SENS, ".4f"))
_buf.write("\n")

with open("resumen_informe.md", "w", encoding="utf-8") as fh:
    fh.write(_buf.getvalue())
ARCHIVOS["resumen_informe.md"] = "resumen en Markdown listo para el informe"

# --- Reporte y descarga ------------------------------------------------
print("=" * 74)
print("ARCHIVOS GENERADOS")
print("=" * 74)
for nombre, desc in ARCHIVOS.items():
    print(f"  ✓ {nombre:<28} {os.path.getsize(nombre):>8,} bytes   {desc}")
print("=" * 74)

try:
    from google.colab import files as _colab_files
    print("\nEntorno Colab detectado — iniciando descarga...")
    for nombre in ARCHIVOS:
        _colab_files.download(nombre)
except ImportError:
    print(f"\nFuera de Colab: los archivos están en {os.getcwd()}")''')

M(r"""**Cómo leer el resultado:** las cuatro marcas ✓ con sus tamaños confirman la
escritura. En Colab se dispara la descarga de cada archivo.

`resumen_informe.md` está pensado para pegarse directamente en el documento del
laboratorio: contiene los parámetros del envase, la discretización con $\Delta z$
explícito, la tabla comparativa, los órdenes de convergencia y el análisis de
sensibilidad, todo en tablas Markdown ya formateadas.""")

M(r"""---
## 11. Fuentes de error y discusión — §9 MVP

El MVP exige *"explicar las principales fuentes de error"*. Con lo medido en este
notebook podemos ordenarlas **por magnitud real**, no por intuición.

### 11.1 Inventario de fuentes de error

| # | Fuente | Magnitud típica en $V$ | ¿Medida aquí? | Cómo se ataca |
|---|---|---|---|---|
| 1 | **Calibración de escala px→cm** | **2× el error de escala** | Simulada (§9.1) | Marcador ArUco medido con regla; verificar la impresión |
| 2 | **Proyección perspectiva** | 2·(R/d), ~3-7 % a 1 m | No (fase 2) | Cámara lejos + zoom; homografía coplanar |
| 3 | **Extracción del contorno** | 1-3 px por borde | No (fase 2) | Segmentación robusta; comparar Otsu vs GrabCut |
| 4 | **Distorsión de lente** | 2-5 % en los bordes | No (fase 2) | Calibrar con ChArUco y aplicar `undistort` |
| 5 | **Asimetría del envase** | 0-5 % según el envase | No (fase 2) | Promediar radios izquierdo y derecho; reportar la asimetría |
| 6 | **Discretización ($\Delta z$)** | **0.01–0.05 %** con n=60 | **Sí (§7, §8)** | Subir $n$ (gratis), o alinear la malla con los quiebres |
| 7 | **Elección del método (Newton-Cotes)** | **0.01–0.05 %** | **Sí (§7)** | Indiferente entre los cuatro |
| 7b | **Gauss de bajo orden** | **0.6 % con m=5; >5 % con m=2,3** | **Sí (§5.7)** | ⚠️ No usar como método principal |
| 8 | **Redondeo `float64`** | ~1e-14 % | **Sí (§8)** | Irrelevante; es el suelo de la aritmética |

### 11.2 La conclusión que ordena el resto del proyecto

Las fuentes **6, 7 y 8 son despreciables**: consumen menos del 1 % del margen del 5 %, y
además **el error de discretización se reduce gratis subiendo $n$** (con $n=1200$ el
trapecio baja a ~3e-5 %).

La fuente **7b es la única excepción numérica y es una decisión de diseño, no un error
inevitable**: Gauss-Legendre global de bajo orden falla el criterio del 5 % por sí solo,
incluso con datos perfectos. Se reporta porque el enunciado lo exige, pero no se usa como
valor de salida.

Todo el presupuesto de error se lo van a llevar las fuentes **1 a 5**, que son de
**adquisición y procesamiento de imagen**. El motor de integración está resuelto y le
sobra precisión.

Esto no le resta valor al laboratorio: se lo da. Un ingeniero que mide dónde está el
error antes de optimizar es un ingeniero que no desperdicia esfuerzo. La respuesta a la
pregunta orientadora del enunciado —

> *¿Cómo convertir una imagen capturada por un celular en datos numéricos confiables
> para estimar un volumen mediante integración numérica?* —

es, a la luz de estos resultados: **el reto no está en la integración; está en el
"confiables"**.

### 11.3 Decisiones cerradas para la fase 2

| Aspecto | Decisión | Justificación |
|---|---|---|
| **Método a producción** | Simpson 1/3, con trapecio de respaldo | Mejor relación precisión/robustez; el trapecio no tiene restricciones sobre $n$ |
| **Estrategia de malla** | Remuestrear a $n$ múltiplo de 12 e integrar por tramos si se detectan quiebres | Permite ejecutar los cuatro métodos y recupera el orden teórico |
| **Gauss** | Se reporta, pero sobre spline; no se usa como valor principal | Su ventaja teórica no sobrevive a datos ruidosos con interpolación |
| **Validación** | Volumen **exterior** por desplazamiento de agua | Única cantidad físicamente comparable con una silueta |
| **Presupuesto de escala** | Error < 1 % (objetivo), < 2.5 % (límite) | $\Delta V/V = 2\,\Delta r/r$, medido en §9.1 |

### 11.4 Qué falta — hoja de ruta

1. **Escala:** detección ArUco + homografía, con dos clics manuales como respaldo.
2. **Extracción del contorno:** Otsu/Canny y GrabCut, comparados (cubre el requisito de
   la Etapa 1 de investigar al menos dos estrategias).
3. **Eje de revolución:** detección por centroide con regresión robusta; corrección de
   inclinación.
4. **Perfil:** radio como semiancho por fila, promediando ambos lados; **reportar la
   asimetría como indicador de calidad de la medición**.
5. **Remuestreo:** a malla uniforme con $n$ múltiplo de 12, por spline.
6. **Integración:** este motor, sin cambios.
7. **Aplicación Streamlit:** carga de imagen, `st.camera_input()`, y toda la trazabilidad
   que ya produce este notebook.
8. **Validación de campo:** desplazamiento de agua sobre 3-5 envases reales.

---

### Cierre

El enunciado dice que el reto *"no consiste únicamente en calcular un volumen, consiste
en demostrar cómo aplicas todo lo que has aprendido en tu disciplina para resolver retos
de ingeniería"*.

La contribución de ingeniería de esta fase no es haber implementado cuatro fórmulas de
cuadratura — eso es la parte fácil, y lo hemos demostrado midiendo que su error es
despreciable. Es haber **medido el presupuesto de error antes de construir**, y haber
salido con un número concreto: *la escala debe ser mejor que el 2.5 %*. Ese número es lo
que va a dirigir todas las decisiones de la fase 2.""")
