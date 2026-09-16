# -*- coding: utf-8 -*-
"""Genera la version CORTA del notebook: lo que exige el PDF, sin material extra."""
import json, pathlib, sys

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

# ============================================================ 1
M(r"""# Reto: del píxel al volumen — Integración numérica

**Fase 1:** motor de cálculo validado sobre un envase de proporciones conocidas.
Todavía sin cámara.

## El modelo

El envase se modela como **sólido de revolución**: si $r(z)$ es el radio a la altura $z$,
cada rebanada de espesor $dz$ es un disco de área $\pi r^2$, y

$$V = \pi \int_a^b [r(z)]^2\,dz$$

Como no tendremos $r(z)$ como fórmula sino como **puntos $(z_i, r_i)$** sacados de una
foto, hay que aproximar esa integral numéricamente. Eso es lo que hace este notebook.

## Lo que importa entender antes de empezar

El radio entra **al cuadrado**, así que un error relativo $\varepsilon$ al medir el radio
se convierte en $2\varepsilon$ en el volumen:

$$\frac{\Delta V}{V} = 2\,\frac{\Delta r}{r}$$

Con el criterio del 5 % del reto, eso significa que **la escala píxel→cm debe tener menos
del 2.5 % de error**. Ahí se gana o se pierde el reto, no en elegir el método de
integración.

*Ejecuta las celdas en orden (`Ctrl+F9`). No hay que instalar nada.*""")

# ============================================================ 2
C(r"""import math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
print("Listo.")""")

# ============================================================ 3
M(r"""---
## 1. El envase

Lo definimos por **tramos**: cada tramo va de una altura a otra y el radio cambia
linealmente entre dos valores. Geométricamente cada tramo es un **tronco de cono**
(un cilindro si los dos radios son iguales).

La ventaja de modelarlo así: el volumen exacto se conoce por fórmula cerrada,

$$V_{\text{tronco}} = \frac{\pi h}{3}\left(R_1^2 + R_1R_2 + R_2^2\right)$$

y el total es la suma de los tramos. Ese número **no es otra aproximación**: es la
respuesta exacta, y sirve como valor de referencia para medir el error de cada método
(lo que pide la Etapa 6 del enunciado).

**✏️ Edita `TRAMOS` para probar otras proporciones.** Todo se recalcula solo.""")

# ============================================================ 4
C(r'''# (z_inicio, z_fin, radio_inicio, radio_fin, nombre)   medidas en cm
TRAMOS = [
    ( 0.0,  1.0, 2.50, 3.40, "base (pie)"),
    ( 1.0, 15.0, 3.40, 3.40, "cuerpo cilíndrico"),
    (15.0, 18.5, 3.40, 1.35, "hombro"),
    (18.5, 20.0, 1.35, 1.35, "cuello"),
    (20.0, 21.0, 1.35, 1.45, "boca / rosca"),
]
A = TRAMOS[0][0]        # límite inferior de integración
B = TRAMOS[-1][1]       # límite superior


def radio(z):
    """Radio r(z) del envase [cm]. Interpola linealmente dentro de cada tramo."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z = np.clip(z, A, B)
    r = np.zeros_like(z)
    for zi, zf, ri, rf, _ in TRAMOS:
        m = (z >= zi) & (z <= zf)
        r[m] = ri + (rf - ri) * (z[m] - zi) / (zf - zi)
    return r


def volumen_exacto():
    """Volumen analítico [cm³] = suma de los troncos de cono. VALOR DE REFERENCIA."""
    return sum(math.pi * (zf - zi) / 3 * (ri**2 + ri*rf + rf**2)
               for zi, zf, ri, rf, _ in TRAMOS)


V_EXACTO = volumen_exacto()
QUIEBRES = [t[1] for t in TRAMOS[:-1]]      # alturas donde el perfil tiene una arista

print(f"Intervalo [a, b]  : [{A}, {B}] cm   (altura {B-A} cm)")
print(f"Radio máximo      : {radio(np.linspace(A,B,500)).max():.2f} cm")
print(f"Puntos de quiebre : {QUIEBRES} cm")
print(f"\nVOLUMEN EXACTO    : {V_EXACTO:.6f} cm³  =  {V_EXACTO:.2f} mL")
print(f"Capacidad nominal :       600.00 mL   (diferencia: "
      f"{100*(V_EXACTO-600)/600:+.1f} % — pared, cuello macizo, base cóncava)")

display(pd.DataFrame(
    [{"tramo": n, "z [cm]": f"{zi}–{zf}", "r [cm]": f"{ri}–{rf}",
      "V [cm³]": math.pi*(zf-zi)/3*(ri**2+ri*rf+rf**2),
      "% del total": 100*math.pi*(zf-zi)/3*(ri**2+ri*rf+rf**2)/V_EXACTO}
     for zi, zf, ri, rf, n in TRAMOS]))''')

# ============================================================ 5
M(r"""**Nota sobre qué volumen es este.** El perfil describe la superficie **exterior**, así
que el resultado (~617 cm³) es el **volumen exterior**, no la capacidad de 600 mL de la
etiqueta. Por eso la validación del proyecto será por **desplazamiento de agua**: es lo
único comparable con lo que se ve en una silueta. Comparar contra la etiqueta arrancaría
con ~3 % de error regalado, sobre un margen de 5 %.

Observa también que el **cuerpo cilíndrico aporta el 82 %** del volumen: en la foto, medir
bien el radio del cuerpo importa mucho más que medir bien el cuello.

---
## 2. De la función a los datos — Etapa 4 del enunciado

Ahora tiramos la función y nos quedamos solo con una **tabla de puntos**, que es lo que
dará la foto. Muestreamos con espaciamiento constante:

$$\Delta z = \frac{b-a}{n}, \qquad z_i = a + i\,\Delta z$$

**Cuidado con el conteo:** $n$ = subintervalos, $n+1$ = puntos. Simpson pide "$n$ par"
refiriéndose a subintervalos.

**Por qué $n = 60$:** cada método exige algo distinto ($n$ par, múltiplo de 3, múltiplo
de 4). Para que los cuatro funcionen sobre **la misma tabla**, $n$ debe ser múltiplo de
$\text{mcm}(2,3,4) = 12$. Y 60 = 5×12.""")

# ============================================================ 6
C(r'''N = 60                                   # ✏️ subintervalos (múltiplo de 12)
Z = np.linspace(A, B, N + 1)             # n+1 nodos
R = radio(Z)
DZ = (B - A) / N

TABLA = pd.DataFrame({
    "i": np.arange(N + 1),
    "z_i [cm]": Z,
    "r_i [cm]": R,
    "r_i² [cm²]": R**2,
    "f(z_i) = π·r_i² [cm²]": math.pi * R**2,
})

print(f"n (subintervalos) : {N}")
print(f"puntos (n+1)      : {N+1}")
print(f"Δz = (b−a)/n      : {DZ:.6f} cm     ← EL DIFERENCIAL")
print(f"compatibilidad    : par={N%2==0}  múltiplo de 3={N%3==0}  múltiplo de 4={N%4==0}")
display(TABLA.head(6))
display(TABLA.tail(3))''')

# ============================================================ 7
M(r"""---
## 3. Los cuatro métodos — §6 del enunciado

Todos aproximan $I = \int_a^b r^2\,dz$ como una **suma ponderada** de los valores
tabulados, $I \approx \Delta z \sum w_i f_i$. Lo único que cambia son los pesos.

| Método | Ajusta | Fórmula del bloque | Pesos compuestos | Requiere | Error |
|---|---|---|---|---|---|
| **Trapecio** | recta | $\frac{\Delta z}{2}(f_0+f_1)$ | $[1,2,2,\dots,2,1]$ | nada | $O(\Delta z^2)$ |
| **Simpson 1/3** | parábola | $\frac{\Delta z}{3}(f_0+4f_1+f_2)$ | $[1,4,2,4,\dots,4,1]$ | $n$ par | $O(\Delta z^4)$ |
| **Simpson 3/8** | cúbica | $\frac{3\Delta z}{8}(f_0+3f_1+3f_2+f_3)$ | $[1,3,3,2,\dots,3,3,1]$ | $n$ múlt. 3 | $O(\Delta z^4)$ |
| **Boole** | grado 4 | $\frac{2\Delta z}{45}(7f_0+32f_1+12f_2+32f_3+7f_4)$ | $[7,32,12,32,14,\dots,7]$ | $n$ múlt. 4 | $O(\Delta z^6)$ |

**Gauss-Legendre** funciona distinto: en vez de usar los nodos donde están, **elige dónde
muestrear** (las raíces del polinomio de Legendre) y con $m$ nodos es exacto hasta grado
$2m-1$. Requiere el cambio de variable $z = \frac{b-a}{2}x + \frac{a+b}{2}$, con el
jacobiano $\frac{b-a}{2}$ al frente.

⚠️ **Dos problemas prácticos de Gauss:**

1. Sus nodos caen en posiciones irracionales donde la foto no tiene píxeles, así que hay
   que **interpolar** (spline). El resultado es la integral del spline, no la de los datos.
2. Es una regla **global**: intenta capturar los 21 cm del envase con un solo polinomio.
   Si el perfil tiene aristas, eso falla. Por eso lo calculamos **de las dos formas** —
   global y aplicándolo por separado en cada tramo — y comparamos.

*Cada función valida sus condiciones y lanza un mensaje explicando qué falla y cómo
arreglarlo — lo pide la §8 del enunciado.*""")

# ============================================================ 8
C(r'''class CondicionNoSatisfecha(Exception):
    """Los datos no cumplen los requisitos del método. Error esperado, no un bug."""


def _preparar(z, r, requiere=1, nombre=""):
    """Valida malla uniforme y la condición sobre n. Devuelve (n, Δz, f=r²)."""
    dz = np.diff(z)
    if not np.allclose(dz, dz[0]):
        raise CondicionNoSatisfecha(
            f"{nombre}: la malla NO es uniforme (el espaciamiento varía). "
            f"Las fórmulas de Newton-Cotes lo exigen. "
            f"Solución: remuestrear el perfil por interpolación antes de integrar.")
    n = len(z) - 1
    if n % requiere:
        raise CondicionNoSatisfecha(
            f"{nombre} requiere n múltiplo de {requiere}, y n = {n} "
            f"(n % {requiere} = {n % requiere}). "
            f"Solución: usar n = {n - n % requiere} o n = {n + requiere - n % requiere}.")
    return n, float(dz[0]), np.asarray(r, float)**2


def trapecio(z, r):
    n, dz, f = _preparar(z, r, 1, "Trapecio")
    w = np.full(n + 1, 2.0); w[0] = w[-1] = 1.0
    return dz / 2 * w @ f


def simpson13(z, r):
    n, dz, f = _preparar(z, r, 2, "Simpson 1/3")
    w = np.ones(n + 1); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return dz / 3 * w @ f


def simpson38(z, r):
    n, dz, f = _preparar(z, r, 3, "Simpson 3/8")
    w = np.zeros(n + 1)
    for i in range(0, n, 3):                 # bloques de 3 subintervalos
        w[i:i+4] += [1, 3, 3, 1]
    return 3 * dz / 8 * w @ f


def boole(z, r):
    n, dz, f = _preparar(z, r, 4, "Boole")
    w = np.zeros(n + 1)
    for i in range(0, n, 4):                 # bloques de 4 subintervalos
        w[i:i+5] += [7, 32, 12, 32, 7]       # los extremos compartidos suman 14
    return 2 * dz / 45 * w @ f


def gauss(z, r, m=5, cortes=None):
    """Gauss-Legendre de m nodos sobre un spline cúbico ajustado a los datos.

    cortes=None  -> una sola cuadratura sobre todo [a, b]  (regla GLOBAL)
    cortes=[...] -> una cuadratura independiente en cada subintervalo (COMPUESTA)

    La versión compuesta es la correcta cuando el perfil tiene aristas: dentro de
    cada tramo el integrando es suave y Gauss lo integra casi exactamente.
    """
    if len(z) < 4:
        raise CondicionNoSatisfecha("Gauss (spline cúbico) requiere al menos 4 puntos.")
    spl = interpolate.InterpolatedUnivariateSpline(z, r, k=3)
    x, w = np.polynomial.legendre.leggauss(m)          # nodos y pesos en [-1, 1]
    bordes = [z[0], z[-1]] if cortes is None else [z[0], *cortes, z[-1]]
    total = 0.0
    for izq, der in zip(bordes, bordes[1:]):
        semi, centro = (der - izq) / 2, (der + izq) / 2   # semi = jacobiano
        total += semi * w @ spl(semi * x + centro)**2
    return total


M_GAUSS = 5
print("Métodos definidos. Comprobación rápida de Gauss:")
_x, _w = np.polynomial.legendre.leggauss(M_GAUSS)
print(f"  suma de pesos = {_w.sum():.12f}  (debe ser 2 = longitud de [-1,1]) ✓")
print(f"  nodos en [-1,1] = {np.round(_x, 4)}   (no equiespaciados)")''')

# ============================================================ 9
M(r"""---
## 4. Resultados y comparación — Etapas 5 y 6

Los cinco métodos se aplican a **la misma tabla** y se reporta todo lo que pide el
enunciado: $n$, $[a,b]$, $\Delta z$, orden, integral, volumen, tiempo, diferencia
absoluta y error relativo.""")

# ============================================================ 10
C(r'''METODOS = [
    # nombre, función, orden teórico, nº de evaluaciones
    ("Trapecio",           trapecio,  "O(Δz²)", N + 1),
    ("Simpson 1/3",        simpson13, "O(Δz⁴)", N + 1),
    ("Simpson 3/8",        simpson38, "O(Δz⁴)", N + 1),
    ("Boole",              boole,     "O(Δz⁶)", N + 1),
    (f"Gauss global m={M_GAUSS}",
     lambda z, r: gauss(z, r, M_GAUSS),
     f"grado ≤ {2*M_GAUSS-1}", M_GAUSS),
    (f"Gauss por tramos m={M_GAUSS}",
     lambda z, r: gauss(z, r, M_GAUSS, cortes=QUIEBRES),
     f"grado ≤ {2*M_GAUSS-1}", M_GAUSS * (len(QUIEBRES) + 1)),
]

filas = []
for nombre, fn, orden, evals in METODOS:
    try:
        t0 = time.perf_counter()
        I = float(fn(Z, R))
        ms = (time.perf_counter() - t0) * 1000
    except CondicionNoSatisfecha as e:
        print(f"⚠ {nombre} NO aplicable: {e}\n")
        continue
    V = math.pi * I
    filas.append({
        "Método": nombre, "n": N, "Evals": evals,
        "a [cm]": A, "b [cm]": B,
        # Gauss no usa malla uniforme: no tiene un Δz que reportar.
        "Δz [cm]": DZ if "Gauss" not in nombre else np.nan,
        "Orden": orden,
        "I = ∫r²dz [cm³]": I,
        "V = πI [cm³]": V,
        "Error abs [cm³]": abs(V - V_EXACTO),
        "Error rel [%]": 100 * abs(V - V_EXACTO) / V_EXACTO,
        "Tiempo [ms]": ms,
    })

RES = pd.DataFrame(filas)

print(f"VALOR DE REFERENCIA (exacto): {V_EXACTO:.6f} cm³")
print(f"Δz = {DZ:.6f} cm   |   n = {N}   |   intervalo [{A}, {B}] cm\n")
display(RES[["Método", "n", "Evals", "Δz [cm]", "Orden",
             "I = ∫r²dz [cm³]", "V = πI [cm³]"]])
print()
display(RES[["Método", "V = πI [cm³]", "Error abs [cm³]", "Error rel [%]", "Tiempo [ms]"]]
        .sort_values("Error abs [cm³]").reset_index(drop=True))

margen = 0.05 * V_EXACTO
print(f"\nCriterio del reto: 5 % de {V_EXACTO:.2f} cm³ = ±{margen:.2f} cm³\n")
for _, f in RES.iterrows():
    print(f"  {f['Método']:<24} usa el {100*f['Error abs [cm³]']/margen:7.3f} % del margen")''')

# ============================================================ 11
M(r"""**Cómo leer esto — hay dos cosas que no se esperan.**

**1. Entre los Newton-Cotes, el ranking sale invertido: el trapecio (orden 2) gana y
Boole (orden 6) pierde.**

No es un error de implementación. Las cotas de error ($O(\Delta z^4)$, $O(\Delta z^6)$)
suponen que el integrando es **suave**, y nuestro envase tiene **aristas** en el pie y el
hombro. Ahí las derivadas de orden alto no existen y la cota deja de aplicar. Como los
bloques de Boole son los más anchos (4 subintervalos), son los que más aristas cruzan, y
por eso acumula más error.

**2. Gauss global es el peor de todos; Gauss por tramos es el mejor.**

Es la **misma fórmula y el mismo número de nodos**, y sin embargo hay un factor **150×**
entre las dos filas (≈ 0.49 % frente a ≈ 0.003 %). Lo único que cambia es *dónde se
aplica*:

- **Global:** un solo polinomio de grado 9 intentando describir un envase con 4 aristas.
  Falla, y de forma errática: añadirle nodos a veces lo empeora.
- **Por tramos:** dentro de cada tramo $f = r^2$ es un polinomio de grado 2, y Gauss con 5
  nodos es exacto hasta grado 9. Lo clava, con solo 25 evaluaciones — **menos de la mitad
  de los 61 puntos** que usan los Newton-Cotes.

El error que le queda (≈ 0.003 %) **no viene de la cuadratura sino del spline**: para
evaluar en los nodos de Gauss hay que interpolar, y el spline redondea ligeramente las
aristas del envase. Es un error de *modelado*, y es exactamente el que aparecerá con datos
de una foto.

> **La lección de las dos sorpresas:** lo que decide la precisión no es el orden de la
> fórmula, sino **si la aplicas respetando la estructura del problema**. Partir el
> intervalo en los quiebres valió más que cualquier salto de orden.

**Y lo más importante, la lista del final:** todos los métodos, incluso el peor, usan una
fracción pequeña del margen del 5 %. En la práctica **la elección del método da casi
igual** — el cuello de botella está en otra parte, y la sección 6 lo mide.

---
## 5. Gráficas — §8 del enunciado""")

# ============================================================ 12
C(r'''zf = np.linspace(A, B, 1200)
rf = radio(zf)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# 1) perfil r(z) con los nodos de la malla
ax[0].plot(zf, rf, lw=2, color="#1f77b4", label="perfil r(z)")
ax[0].plot(Z, R, "o", ms=4, color="#d62728", label=f"datos (n={N})")
for q in QUIEBRES:
    ax[0].axvline(q, ls=":", color="gray", lw=1)
ax[0].set(xlabel="z [cm]", ylabel="r(z) [cm]", title="1. Perfil del envase", ylim=(0, None))
ax[0].legend(fontsize=8)

# 2) el integrando: el volumen es el área sombreada
ax[1].fill_between(zf, 0, math.pi*rf**2, alpha=.3, color="#2ca02c",
                   label=f"área = V = {V_EXACTO:.1f} cm³")
ax[1].plot(zf, math.pi*rf**2, lw=2, color="#2ca02c")
ax[1].plot(Z, math.pi*R**2, "o", ms=3.5, color="#d62728")
ax[1].set(xlabel="z [cm]", ylabel="f(z) = π·r(z)² [cm²]",
          title="2. Integrando: el volumen es esta área")
ax[1].legend(fontsize=8)

# 3) sección longitudinal: el envase reconocible
ax[2].fill_between(zf, -rf, rf, alpha=.3, color="#1f77b4")
ax[2].plot(zf, rf, zf, -rf, lw=1.8, color="#1f77b4")
ax[2].axhline(0, ls="--", color="#d62728", lw=1.2, label="eje de revolución")
ax[2].set(xlabel="z [cm]", ylabel="radio [cm]", title="3. Sección del sólido")
ax[2].legend(fontsize=8); ax[2].set_aspect("equal", adjustable="box")

plt.tight_layout(); plt.show()

# Comparación visual de los métodos frente al valor exacto
fig, ax = plt.subplots(figsize=(9, 3.6))
_ord = RES.sort_values("Error rel [%]", ascending=False)   # peor arriba, mejor abajo
ax.barh(_ord["Método"], _ord["Error rel [%]"], color="#4c72b0", alpha=.85)
ax.axvline(5, color="crimson", ls="--", lw=1.5, label="criterio del reto: 5 %")
ax.set(xlabel="error relativo [%]  (escala log)", xscale="log",
       title="Error de cada método frente al volumen exacto")
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", which="both", alpha=.3)
plt.tight_layout(); plt.show()''')

# ============================================================ 13
M(r"""**La gráfica de barras lo resume todo:** la línea roja es el criterio del 5 % y todas
las barras están **muy** a su izquierda (escala logarítmica). Hay dos órdenes de magnitud
de margen sobrante.

---
## 6. ¿Dónde está realmente el error? — §9 del enunciado

La celda siguiente compara las dos fuentes de error que competirán en el proyecto real:
el **método numérico** (que acabamos de medir) y la **escala píxel→cm** (que aparecerá al
usar la cámara). Se simula un error de calibración multiplicando todos los radios por
$(1+s)$.""")

# ============================================================ 14
C(r'''filas = []
for s in [0.005, 0.01, 0.02, 0.025, 0.05]:
    V_s = math.pi * trapecio(Z, (1 + s) * R)
    rel = 100 * (V_s - V_EXACTO) / V_EXACTO
    filas.append({"Error de escala [%]": 100*s, "V estimado [cm³]": V_s,
                  "Error en V [%]": rel,
                  "¿Pasa el 5 %?": "✓ sí" if abs(rel) < 5 else "✗ NO"})
display(pd.DataFrame(filas))

# Comparamos contra los Newton-Cotes, que son los candidatos reales para producción.
nc = RES[~RES["Método"].str.contains("Gauss")]
peor = nc["Error rel [%]"].max()
esc1 = 100 * (1.01**2 - 1)
print(f"Error del PEOR Newton-Cotes (n={N})      : {peor:.4f} %")
print(f"Error por una escala mal calibrada al 1 % : {esc1:.4f} %")
print(f"\n→ Un error de escala del 1 % pesa {esc1/peor:.0f}× más que el peor método.")
print(f"→ Y el error del método baja gratis subiendo n:")
for n in (60, 240, 1200):
    z2 = np.linspace(A, B, n+1)
    e = 100*abs(math.pi*trapecio(z2, radio(z2)) - V_EXACTO)/V_EXACTO
    print(f"     n = {n:>4}  →  {e:.2e} %")''')

# ============================================================ 15
M(r"""**Conclusión — lo que este notebook deja fijado para la fase de cámara.**

El error en el volumen es **el doble** del error en la escala. De ahí salen los números
que dirigen todo el proyecto:

| Error de escala px→cm | Error en el volumen | |
|---|---|---|
| 1 % | 2.0 % | ✅ objetivo |
| 2.5 % | 5.1 % | ⚠️ frontera |
| 5 % | 10.3 % | ❌ |

**Fuentes de error, ordenadas por lo que realmente pesan:**

| Fuente | Magnitud | ¿Resuelta? |
|---|---|---|
| Calibración de escala px→cm | 2× el error de escala | ❌ fase 2 |
| Perspectiva de la foto | ~2·(R/distancia) | ❌ fase 2 |
| Extracción del contorno | 1–3 px por borde | ❌ fase 2 |
| Asimetría del envase | 0–5 % | ❌ fase 2 |
| **Método de integración** | **~0.05 %** | ✅ **medido aquí** |
| **Discretización (Δz)** | **~0.01 %, y baja gratis con n** | ✅ **medido aquí** |

Las dos fuentes numéricas son **despreciables**: entre 40 y 200 veces menores que un
error de escala del 1 %. El motor de cálculo está resuelto y le sobra precisión.

**Lo que falta (fase 2):** capturar con la cámara, detectar el marcador de escala
(ArUco + homografía), extraer el contorno del envase (Otsu y GrabCut), convertir a
$(z_i, r_i)$ en cm, y montar la app en Streamlit. La integración ya no hay que tocarla.

---
## 7. Exportar para el informe — §8""")

# ============================================================ 16
C(r'''TABLA.to_csv("perfil_datos.csv", index=False, encoding="utf-8-sig")
RES.to_csv("comparacion_metodos.csv", index=False, encoding="utf-8-sig")
print("✓ perfil_datos.csv        — tabla (i, z, r, r², πr²)")
print("✓ comparacion_metodos.csv — los 5 métodos con todos sus parámetros")

try:
    from google.colab import files
    files.download("perfil_datos.csv")
    files.download("comparacion_metodos.csv")
except ImportError:
    import os
    print(f"\n(fuera de Colab: guardados en {os.getcwd()})")''')

# ------------------------------------------------------------------ build
def a_fuente(t):
    l = t.split("\n")
    return [x + "\n" for x in l[:-1]] + [l[-1]]

celdas, errores, nc = [], [], 0
for tipo, txt in CELLS:
    if tipo == "markdown":
        celdas.append({"cell_type": "markdown", "metadata": {}, "source": a_fuente(txt)})
    else:
        nc += 1
        try:
            compile(txt, f"<celda {nc}>", "exec")
        except SyntaxError as e:
            errores.append(f"  celda {nc}: {e}")
        celdas.append({"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": a_fuente(txt)})

if errores:
    print("ERRORES DE SINTAXIS:", *errores, sep="\n", file=sys.stderr)
    sys.exit(1)

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"colab": {"provenance": [], "toc_visible": True,
                             "name": "Lab_RetroEnvases.ipynb"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "cells": celdas}

out = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "Lab_RetroEnvases.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
md = sum(1 for c in celdas if c["cell_type"] == "markdown")
print(f"Notebook CORTO escrito: {out}")
print(f"  {len(celdas)} celdas ({md} markdown + {nc} codigo)  |  {out.stat().st_size:,} bytes")
print("  Sintaxis OK")
