# -*- coding: utf-8 -*-
"""Genera Lab_RetroEnvases.ipynb — version con los 4 metodos del enunciado."""
import json, pathlib, sys

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

# ═════════════════════════════════════════════════════ CELDA 1 (texto)
M(r"""# Reto: del píxel al volumen — Integración numérica

**Fase 1:** motor de cálculo validado sobre un envase de proporciones conocidas.
Todavía sin cámara.

## El modelo

El envase se modela como **sólido de revolución**: si $r(z)$ es el radio a la altura $z$,
cada rebanada de espesor $dz$ es un disco de área $\pi r^2$, y sumando todas las rebanadas:

$$V = \pi \int_a^b [r(z)]^2\,dz$$

Como no tendremos $r(z)$ como fórmula sino como **puntos $(z_i, r_i)$** sacados de una
foto, esa integral hay que aproximarla numéricamente.

## Los cuatro métodos

| Método | Idea |
|---|---|
| **Regla del Trapecio** | Une los puntos con rectas |
| **Regla de Simpson** | Ajusta parábolas cada 2 subintervalos |
| **Newton–Cotes (3/8)** | Ajusta cúbicas cada 3 subintervalos |
| **Cuadratura de Gauss** | Elige *dónde* muestrear, no solo con qué peso |

## Lo que importa entender antes de empezar

El volumen es muy sensible a los errores de medición, y hay que distinguir dos tipos:

- **Error en el borde del contorno** (afecta solo al radio). Como el radio entra al
  cuadrado: $\;\Delta V/V = 2\,\Delta r/r$.
- **Error en la escala píxel→cm** (afecta al radio **y** a la altura a la vez, porque
  ambos se convierten con la misma escala). Entonces $V \propto s^3$:

$$\frac{\Delta V}{V} \approx 3\,\frac{\Delta s}{s}$$

Con el criterio del 5 % del reto, **la escala debe tener menos del 1.6 % de error**
($1.05^{1/3}-1$). Ahí se gana o se pierde el reto, no en elegir el método.

*Ejecuta las celdas en orden (`Ctrl+F9`). No hay que instalar nada.*""")

# ═════════════════════════════════════════════════════ CELDA 2 (código)
C(r"""import math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
print("Listo.")""")

# ═════════════════════════════════════════════════════ CELDA 3 (texto)
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

# ═════════════════════════════════════════════════════ CELDA 4 (código)
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
    """Radio r(z) del envase, en cm. Interpola linealmente dentro de cada tramo.

    Acepta un número o un array y siempre devuelve un array, para poder evaluar
    miles de alturas de una sola vez.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z = np.clip(z, A, B)                      # fuera del envase, saturamos al borde
    r = np.zeros_like(z)
    for zi, zf, ri, rf, _ in TRAMOS:
        m = (z >= zi) & (z <= zf)             # máscara: qué alturas caen en este tramo
        r[m] = ri + (rf - ri) * (z[m] - zi) / (zf - zi)
    return r


def volumen_exacto():
    """Volumen analítico del envase [cm³]. Es el VALOR DE REFERENCIA.

    Suma cerrada de los troncos de cono: V = (πh/3)(R1² + R1R2 + R2²).
    No es una aproximación numérica — es la respuesta exacta.
    """
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

# ═════════════════════════════════════════════════════ CELDA 5 (texto)
M(r"""**Nota sobre qué volumen es este.** El perfil describe la superficie **exterior**, así
que el resultado (~617 cm³) es el **volumen exterior**, no la capacidad de 600 mL de la
etiqueta. Esa diferencia (~3 %) es pared, cuello macizo y base cóncava: una silueta no
puede verla. Si la prueba de campo compara contra la etiqueta, hará falta estimar el
espesor de pared para pasar de volumen exterior a capacidad.

Observa también que el **cuerpo cilíndrico aporta el 82 %** del volumen: en la foto, medir
bien el radio del cuerpo importa mucho más que medir bien el cuello.

---
## 2. De la función a los datos — Etapa 4 del enunciado

Ahora tiramos la función y nos quedamos solo con una **tabla de puntos**, que es lo que
dará la foto. Muestreamos con espaciamiento constante:

$$\Delta z = \frac{b-a}{n}, \qquad z_i = a + i\,\Delta z$$

**Cuidado con el conteo:** $n$ = subintervalos, $n+1$ = puntos. Simpson pide "$n$ par"
refiriéndose a subintervalos.

**Por qué $n = 60$:** cada método exige algo distinto ($n$ par para Simpson, múltiplo de 3
para Newton-Cotes 3/8). Para que todos funcionen sobre **la misma tabla**, tomamos $n$
múltiplo de $\text{mcm}(2,3) = 6$. Usamos 60, que además es múltiplo de 4 por si quieres
probar Boole.""")

# ═════════════════════════════════════════════════════ CELDA 6 (código)
C(r'''N = 60                                   # ✏️ subintervalos (múltiplo de 6)
Z = np.linspace(A, B, N + 1)             # n+1 nodos, de a hasta b
R = radio(Z)                             # el radio en cada nodo
DZ = (B - A) / N                         # el diferencial de discretización

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
print(f"compatibilidad    : n par = {N%2==0}   |   n múltiplo de 3 = {N%3==0}")
display(TABLA.head(6))
display(TABLA.tail(3))''')

# ═════════════════════════════════════════════════════ CELDA 7 (texto)
M(r"""---
## 3. Los cuatro métodos — §6 del enunciado

Los tres primeros son **Newton–Cotes**: usan los nodos tal como están (equiespaciados) y
solo eligen los **pesos**. Todos tienen la forma $I \approx \Delta z \sum w_i f_i$, donde
$f_i = r_i^2$. Lo único que cambia es el patrón de pesos.

| Método | Ajusta | Bloque | Pesos compuestos | Requiere | Error |
|---|---|---|---|---|---|
| **Trapecio** | recta (grado 1) | $\frac{\Delta z}{2}(f_0+f_1)$ | $[1,2,2,\dots,2,1]$ | nada | $O(\Delta z^2)$ |
| **Simpson** | parábola (grado 2) | $\frac{\Delta z}{3}(f_0+4f_1+f_2)$ | $[1,4,2,4,\dots,4,1]$ | $n$ par | $O(\Delta z^4)$ |
| **Newton–Cotes 3/8** | cúbica (grado 3) | $\frac{3\Delta z}{8}(f_0+3f_1+3f_2+f_3)$ | $[1,3,3,2,\dots,3,3,1]$ | $n$ múlt. 3 | $O(\Delta z^4)$ |

Los pesos "2" en el patrón de Simpson y de 3/8 son las **costuras**: nodos compartidos
entre dos bloques consecutivos, que reciben 1 de cada lado.

**Un detalle que sorprende:** Simpson ajusta parábolas (grado 2) pero integra exactamente
hasta **cúbicas** (grado 3). Gana un grado gratis por una cancelación de simetría. Por eso
3/8, que usa polinomios de grado *mayor*, no mejora el orden: sigue siendo $O(\Delta z^4)$
y su constante de error es peor ($1/80$ frente a $1/180$).

### La cuadratura de Gauss funciona distinto

Las reglas anteriores tienen los nodos fijos y solo eligen los pesos: con $m$ nodos hay
$m$ grados de libertad, y se logra exactitud hasta grado $m-1$.

Gauss también **elige dónde muestrear**. Con $m$ nodos hay $2m$ grados de libertad
($m$ posiciones + $m$ pesos), y se alcanza exactitud hasta grado:

$$\boxed{2m-1}$$

Los nodos óptimos son las **raíces del polinomio de Legendre** $P_m(x)$ en $[-1,1]$ — no
son equiespaciados, se agrupan hacia los extremos. Hay que llevarlos a $[a,b]$ con

$$z = \frac{b-a}{2}x + \frac{a+b}{2}, \qquad dz = \underbrace{\frac{b-a}{2}}_{\text{jacobiano}}dx$$

⚠️ **Dos problemas prácticos de Gauss:**

1. Sus nodos caen en posiciones irracionales donde la foto no tiene píxeles, así que hay
   que **interpolar** (spline cúbico). El resultado es la integral del spline, no la de los
   datos crudos.
2. Es una regla **global**. Si se aplica de golpe a todo el envase, un solo polinomio
   tiene que describir las 4 aristas del perfil, y falla. Lo aplicamos **por tramos**: una
   cuadratura independiente en cada tramo, donde el integrando sí es suave.

*Cada función valida sus condiciones y, si no se cumplen, lanza un mensaje explicando qué
falla y cómo arreglarlo — lo pide la §8 del enunciado.*""")

# ═════════════════════════════════════════════════════ CELDA 8 (código)
C(r'''class CondicionNoSatisfecha(Exception):
    """Los datos no cumplen los requisitos del método. Error esperado, no un bug."""


def _preparar(z, r, requiere=1, nombre=""):
    """Comprobación común a las tres reglas de Newton-Cotes.

    Verifica que (1) la malla sea uniforme y (2) n sea múltiplo de lo que exige el
    método. Devuelve (n, Δz, f=r²) para que la regla solo tenga que poner los pesos.

    Se valida aquí, una sola vez, en lugar de confiar en quien llame: aplicar estas
    fórmulas a una malla no uniforme no falla, devuelve un número INCORRECTO — que
    es el peor tipo de error, porque nadie sospecha de un volumen verosímil.
    """
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


# ─────────────────────────────────────────────── 1. REGLA DEL TRAPECIO
def trapecio(z, r):
    """Regla del trapecio compuesta.

    Une cada par de puntos consecutivos con una recta y suma las áreas de los
    trapecios resultantes. Los nodos interiores aparecen en dos trapecios, de ahí
    el patrón de pesos [1, 2, 2, ..., 2, 1].

    Sin restricciones sobre n: es el método de respaldo universal.
    """
    n, dz, f = _preparar(z, r, 1, "Trapecio")
    w = np.full(n + 1, 2.0)
    w[0] = w[-1] = 1.0                       # los extremos solo cuentan una vez
    return dz / 2 * w @ f


# ─────────────────────────────────────────────── 2. REGLA DE SIMPSON
def simpson(z, r):
    """Regla de Simpson 1/3 compuesta.

    Ajusta una parábola por cada PAR de subintervalos (3 puntos definen una
    parábola) y la integra analíticamente. Los índices impares son centros de
    parábola (peso 4) y los pares interiores son costuras compartidas (peso 2).

    Requiere n PAR: cada parábola consume 2 subintervalos y con n impar sobraría uno.
    """
    n, dz, f = _preparar(z, r, 2, "Simpson")
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0                          # índices impares → centro de parábola
    w[2:-1:2] = 2.0                          # índices pares interiores → costuras
    return dz / 3 * w @ f


# ─────────────────────────────────── 3. NEWTON-COTES CERRADA (3/8 o Boole)
def newton_cotes(z, r, grado=3):
    """Fórmula de Newton-Cotes cerrada de grado 3 (regla 3/8) o 4 (regla de Boole).

    grado=3 → Simpson 3/8: cúbica cada 3 subintervalos, coeficiente 3Δz/8,
              pesos de bloque [1, 3, 3, 1].
    grado=4 → Boole: polinomio de grado 4 cada 4 subintervalos, coeficiente
              2Δz/45, pesos de bloque [7, 32, 12, 32, 7].

    Los pesos se acumulan con += para que los nodos compartidos entre bloques
    consecutivos sumen su aporte de ambos lados (1+1=2 en 3/8, 7+7=14 en Boole).
    """
    bloques = {
        3: (np.array([1., 3., 3., 1.]),            3/8,   "Newton-Cotes 3/8"),
        4: (np.array([7., 32., 12., 32., 7.]),     2/45,  "Newton-Cotes (Boole)"),
    }
    if grado not in bloques:
        raise CondicionNoSatisfecha(
            f"grado={grado} no soportado. Usa grado=3 (regla 3/8) o grado=4 (Boole).")
    pesos_bloque, coef, nombre = bloques[grado]
    n, dz, f = _preparar(z, r, grado, nombre)
    w = np.zeros(n + 1)
    for i in range(0, n, grado):             # avanzamos de bloque en bloque
        w[i:i + grado + 1] += pesos_bloque
    return coef * dz * w @ f


# ─────────────────────────────────────────────── 4. CUADRATURA DE GAUSS
def gauss(z, r, m=5, cortes=None):
    """Cuadratura de Gauss-Legendre de m nodos sobre un spline cúbico de los datos.

    Gauss no evalúa en los nodos de la tabla sino en las raíces del polinomio de
    Legendre, que caen en posiciones irracionales. Como la tabla no tiene valores
    ahí, se ajusta un spline cúbico y se evalúa sobre él.

    cortes=None  → una sola cuadratura en todo [a, b]     (regla GLOBAL)
    cortes=[...] → una cuadratura en cada subintervalo    (regla COMPUESTA)

    La versión compuesta es la correcta cuando el perfil tiene aristas: dentro de
    cada tramo el integrando es un polinomio de grado 2 y Gauss lo integra casi
    exactamente. Pasar cortes=QUIEBRES mejora el resultado unas 150 veces.
    """
    if len(z) < 4:
        raise CondicionNoSatisfecha("Gauss (spline cúbico) requiere al menos 4 puntos.")
    spl = interpolate.InterpolatedUnivariateSpline(z, r, k=3)
    x, w = np.polynomial.legendre.leggauss(m)        # nodos y pesos en [-1, 1]
    bordes = [z[0], z[-1]] if cortes is None else [z[0], *cortes, z[-1]]
    total = 0.0
    for izq, der in zip(bordes, bordes[1:]):
        semi, centro = (der - izq) / 2, (der + izq) / 2     # semi = jacobiano
        total += semi * w @ spl(semi * x + centro)**2       # cambio de variable
    return total


M_GAUSS = 5
print("Los cuatro métodos están definidos.\n")
_x, _w = np.polynomial.legendre.leggauss(M_GAUSS)
print(f"Comprobación de Gauss con m = {M_GAUSS} nodos:")
print(f"  suma de pesos  = {_w.sum():.12f}   (debe ser 2 = longitud de [-1,1]) ✓")
print(f"  nodos en [-1,1] = {np.round(_x, 4)}")
print(f"  → no equiespaciados: se agrupan hacia los extremos")
print(f"  → exacta para polinomios de grado ≤ 2m−1 = {2*M_GAUSS-1}")''')

# ═════════════════════════════════════════════════════ CELDA 9 (texto)
M(r"""---
## 4. Resultados y comparación — Etapas 5 y 6

Los cuatro métodos se aplican a **la misma tabla** de 61 puntos y se reporta todo lo que
pide el enunciado: $n$, $[a,b]$, $\Delta z$, orden o número de nodos, integral, volumen,
tiempo de cálculo, diferencia absoluta y error relativo porcentual.""")

# ═════════════════════════════════════════════════════ CELDA 10 (código)
C(r'''# (nombre, función, orden o exactitud, nº de evaluaciones del integrando)
METODOS = [
    ("Trapecio",         trapecio,                         "O(Δz²)", N + 1),
    ("Simpson",          simpson,                          "O(Δz⁴)", N + 1),
    ("Newton-Cotes 3/8", lambda z, r: newton_cotes(z, r, 3), "O(Δz⁴)", N + 1),
    (f"Gauss m={M_GAUSS}",
     lambda z, r: gauss(z, r, M_GAUSS, cortes=QUIEBRES),
     f"grado ≤ {2*M_GAUSS-1}", M_GAUSS * (len(QUIEBRES) + 1)),
]

filas = []
for nombre, fn, orden, evals in METODOS:
    try:
        t0 = time.perf_counter()
        I = float(fn(Z, R))                   # I = ∫ r² dz
        ms = (time.perf_counter() - t0) * 1000
    except CondicionNoSatisfecha as e:
        print(f"⚠ {nombre} NO aplicable: {e}\n")
        continue
    V = math.pi * I                           # V = π · I
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
    print(f"  {f['Método']:<18} usa el {100*f['Error abs [cm³]']/margen:6.3f} % del margen")

# ¿Cuánto se pierde si Gauss se aplica de golpe, sin partir por tramos?
V_gg = math.pi * gauss(Z, R, M_GAUSS)                      # sin cortes = global
V_gt = math.pi * gauss(Z, R, M_GAUSS, cortes=QUIEBRES)     # por tramos
print(f"\nGauss global (sin partir por tramos): {100*abs(V_gg-V_EXACTO)/V_EXACTO:.4f} % de error")
print(f"Gauss por tramos (el usado arriba)  : {100*abs(V_gt-V_EXACTO)/V_EXACTO:.4f} % de error")
print(f"→ misma fórmula, mismos {M_GAUSS} nodos, "
      f"{abs(V_gg-V_EXACTO)/abs(V_gt-V_EXACTO):.0f}× de diferencia")''')

# ═════════════════════════════════════════════════════ CELDA 11 (texto)
M(r"""**Cómo leer esto — hay dos cosas que no se esperan.**

**1. Entre los Newton–Cotes, el ranking sale invertido respecto al orden teórico.**
El trapecio ($O(\Delta z^2)$) resulta **más preciso que Simpson** ($O(\Delta z^4)$), pese
a que Simpson promete converger mucho más rápido.

No es un error de implementación. Las cotas de error de orden alto suponen que el
integrando es **suave**, y nuestro envase tiene **aristas** en el pie y el hombro. Ahí las
derivadas de orden alto no existen y la cota deja de aplicar. Además, cada parábola de
Simpson abarca 2 subintervalos, así que cruza aristas con el doble de frecuencia que un
trapecio — y ajustar una parábola a tres puntos repartidos entre dos curvas distintas da
peor resultado que dos rectas.

**2. La misma fórmula de Gauss cambia 150× según dónde se aplique.**
Las dos últimas líneas lo muestran: global (un polinomio de grado 9 para todo el envase)
da ~0.49 % de error; aplicada por tramos da ~0.003 % y es el método más preciso de los
cuatro, con solo 25 evaluaciones frente a las 61 de los demás.

El residuo que le queda **no viene de la cuadratura sino del spline**, que redondea
ligeramente las aristas al interpolar. Es un error de *modelado*, y es exactamente el que
aparecerá con datos de una foto.

> **Conclusión sobre los métodos:** lo que decide la precisión no es el orden de la
> fórmula, sino **si se aplica respetando la estructura del problema**. Partir el intervalo
> en los quiebres valió más que cualquier salto de orden.

**Y lo más importante, la lista del margen:** los cuatro métodos usan **menos del 1 %**
del margen del 5 %. En la práctica la elección da casi igual — el cuello de botella está
en otra parte, y la sección 6 lo mide.

---
## 5. Gráficas — §8 del enunciado""")

# ═════════════════════════════════════════════════════ CELDA 12 (código)
C(r'''zf = np.linspace(A, B, 1200)
rf = radio(zf)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# 1) perfil r(z) con los nodos de la malla superpuestos
ax[0].plot(zf, rf, lw=2, color="#1f77b4", label="perfil r(z)")
ax[0].plot(Z, R, "o", ms=4, color="#d62728", label=f"datos (n={N})")
for q in QUIEBRES:
    ax[0].axvline(q, ls=":", color="gray", lw=1)
ax[0].set(xlabel="z [cm]", ylabel="r(z) [cm]", title="1. Perfil del envase", ylim=(0, None))
ax[0].legend(fontsize=8)

# 2) el integrando: el volumen es literalmente el área sombreada
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

# Error de cada método frente al criterio del reto, en escala logarítmica
fig, ax = plt.subplots(figsize=(9, 3))
_ord = RES.sort_values("Error rel [%]", ascending=False)
ax.barh(_ord["Método"], _ord["Error rel [%]"], color="#4c72b0", alpha=.85)
ax.axvline(5, color="crimson", ls="--", lw=1.5, label="criterio del reto: 5 %")
ax.set(xlabel="error relativo [%]  (escala log)", xscale="log",
       title="Error de cada método frente al volumen exacto")
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", which="both", alpha=.3)
plt.tight_layout(); plt.show()''')

# ═════════════════════════════════════════════════════ CELDA 13 (texto)
M(r"""**La gráfica de barras lo resume todo:** la línea roja es el criterio del 5 % y las
cuatro barras están **muy** a su izquierda (ojo, escala logarítmica). Hay dos órdenes de
magnitud de margen sobrante.

---
## 6. ¿Dónde está realmente el error? — §9 del enunciado

La celda siguiente compara las dos fuentes de error que competirán en el proyecto real: el
**método numérico** (que acabamos de medir) y la **escala píxel→cm** (que aparecerá al
usar la cámara). Se simula un error de calibración $s$: como la escala convierte **tanto
los radios como las alturas**, ambos se multiplican por $(1+s)$.""")

# ═════════════════════════════════════════════════════ CELDA 14 (código)
C(r'''filas = []
for s in [0.005, 0.01, 0.015, 0.02, 0.03]:
    # Un error de escala afecta a z y a r a la vez → el volumen escala como (1+s)³
    V_s = math.pi * trapecio((1 + s) * Z, (1 + s) * R)
    rel = 100 * (V_s - V_EXACTO) / V_EXACTO
    filas.append({"Error de escala [%]": 100*s, "V estimado [cm³]": V_s,
                  "Error en V [%]": rel,
                  "¿Pasa el 5 %?": "✓ sí" if abs(rel) < 5 else "✗ NO"})
display(pd.DataFrame(filas))

peor = RES["Error rel [%]"].max()
esc1 = 100 * (1.01**3 - 1)                        # error en V si la escala falla un 1 %
print(f"Error del PEOR método (n={N})            : {peor:.4f} %")
print(f"Error por una escala mal calibrada al 1 % : {esc1:.4f} %")
print(f"\n→ Un error de escala del 1 % pesa {esc1/peor:.0f}× más que el peor método.")
print(f"→ Límite de escala para el 5 %: {100*(1.05**(1/3)-1):.2f} %")
print(f"→ Y el error del método baja gratis subiendo n:")
for n in (60, 240, 1200):
    z2 = np.linspace(A, B, n + 1)
    e = 100 * abs(math.pi*trapecio(z2, radio(z2)) - V_EXACTO) / V_EXACTO
    print(f"     n = {n:>4}  →  {e:.2e} %")''')

# ═════════════════════════════════════════════════════ CELDA 15 (texto)
M(r"""**Conclusión — lo que este notebook deja fijado para la fase de cámara.**

El error en el volumen es **el triple** del error en la escala, porque la escala convierte
a la vez radios y alturas. De ahí salen los números que dirigen todo el proyecto:

| Error de escala px→cm | Error en el volumen | |
|---|---|---|
| 0.5 % | 1.5 % | ✅ objetivo |
| 1.0 % | 3.0 % | ✅ |
| 1.6 % | 5.0 % | ⚠️ frontera |
| 2.0 % | 6.1 % | ❌ |

Y hay una asimetría que lo decide todo: **el error del método se reduce gratis** subiendo
$n$, mientras que mejorar la escala exige trabajo físico real (medir bien la referencia,
controlar la perspectiva). La fuente de error barata de
eliminar no es la que domina.

**Lo que falta (fase 2):** capturar con la cámara, fijar la escala con la altura real del
envase medida con regla, extraer el contorno (Otsu y GrabCut), convertir a $(z_i, r_i)$
en cm, y montar la app en Streamlit. La integración ya no hay que tocarla.""")

# ═════════════════════════════════════════════════════ CELDA 16 (texto, resumen)
M(r"""---
# Resumen de resultados — para copiar al informe

*Todas las cifras corresponden al envase definido por defecto. Si editas `TRAMOS`, los
números cambian: los verás actualizados en las salidas de las celdas de arriba.*

## 1. Datos del problema

| Concepto | Valor |
|---|---|
| Envase | Botella PET, 600 mL nominales |
| Intervalo de integración $[a,b]$ | $[0,\;21]$ cm |
| Altura total | 21 cm |
| Radio máximo | 3.40 cm |
| Puntos de quiebre (aristas) | 1.0, 15.0, 18.5, 20.0 cm |
| Subintervalos $n$ | 60 |
| Puntos ($n+1$) | 61 |
| **Diferencial $\Delta z = (b-a)/n$** | **0.350000 cm** |
| Malla | uniforme; $n$ múltiplo de 6 |
| **Volumen exacto de referencia** | **616.608244 cm³ = 616.61 mL** |
| Integral exacta $\int r^2 dz$ | 196.272500 cm³ |
| Margen del criterio del 5 % | ±30.83 cm³ |

El volumen calculado es el **exterior**, un **+2.8 %** sobre los 600 mL de la etiqueta: la
diferencia es el espesor de pared, el cuello macizo y la base cóncava, que una silueta no
puede ver.

## 2. Composición del envase por tramos

| Tramo | z [cm] | r [cm] | Forma | V [cm³] | % del total |
|---|---|---|---|---|---|
| base (pie) | 0.0 – 1.0 | 2.50 → 3.40 | tronco de cono | 27.55 | 4.5 % |
| **cuerpo cilíndrico** | 1.0 – 15.0 | 3.40 | cilindro | **508.44** | **82.5 %** |
| hombro | 15.0 – 18.5 | 3.40 → 1.35 | tronco de cono | 65.87 | 10.7 % |
| cuello | 18.5 – 20.0 | 1.35 | cilindro | 8.59 | 1.4 % |
| boca / rosca | 20.0 – 21.0 | 1.35 → 1.45 | tronco de cono | 6.16 | 1.0 % |

El cuerpo cilíndrico concentra el **82.5 %** del volumen: un error de 1 mm en el radio del
cuerpo cuesta ~18 cm³ (3 %); el mismo error en el cuello cuesta menos de 0.5 cm³.

## 3. Comparación de los cuatro métodos

Todos sobre la **misma tabla** de 61 puntos, $\Delta z = 0.35$ cm, $[a,b]=[0,21]$ cm.

| Método | Evals | $\Delta z$ [cm] | Orden / exactitud | $I=\int r^2dz$ [cm³] | $V=\pi I$ [cm³] | Error abs [cm³] | Error rel [%] | % del margen |
|---|---|---|---|---|---|---|---|---|
| **Gauss, m=5 por tramos** | 25 | — | grado ≤ 9 | 196.2661 | 616.5881 | 0.0201 | **0.0033** | 0.07 % |
| Trapecio | 61 | 0.35 | $O(\Delta z^2)$ | 196.2524 | 616.5450 | 0.0632 | 0.0102 | 0.20 % |
| Newton–Cotes 3/8 | 61 | 0.35 | $O(\Delta z^4)$ | 196.2500 | 616.5377 | 0.0705 | 0.0114 | 0.23 % |
| Simpson | 61 | 0.35 | $O(\Delta z^4)$ | 196.3562 | 616.8713 | 0.2631 | 0.0427 | 0.85 % |

**Los cuatro cumplen el criterio del 5 % con enorme holgura:** ninguno llega al 1 % del
margen permitido.

### Dos resultados que contradicen la intuición

**(a) Entre los Newton–Cotes el ranking sale invertido.** El trapecio ($O(\Delta z^2)$)
resulta **4.2 veces más preciso** que Simpson ($O(\Delta z^4)$). Las cotas de error de
orden alto suponen un integrando suave, y el envase tiene aristas donde las derivadas de
orden alto no existen. Cada parábola de Simpson abarca 2 subintervalos y cruza aristas con
el doble de frecuencia que un trapecio.

**(b) La misma fórmula de Gauss, con los mismos 5 nodos, cambia 150× según dónde se
aplique.** Global (un polinomio de grado 9 para todo el envase): 0.49 % de error.
Por tramos (una cuadratura en cada tramo, donde $f=r^2$ es de grado 2): 0.003 %, el mejor
de los cuatro, con solo 25 evaluaciones frente a 61. El residuo viene del **spline**, que
redondea las aristas al interpolar, no de la cuadratura.

> **Conclusión:** lo que decide la precisión no es el orden de la fórmula, sino **si se
> aplica respetando la estructura del problema**.

## 4. Condiciones de aplicabilidad

| Método | Requiere | Si no se cumple |
|---|---|---|
| Trapecio | nada | siempre aplicable — es el respaldo universal |
| Simpson | $n$ par | usar $n\pm1$, o 3/8 en los últimos 3 subintervalos |
| Newton–Cotes 3/8 | $n$ múltiplo de 3 | usar el múltiplo de 3 más cercano |
| Los tres anteriores | malla uniforme | remuestrear por interpolación |
| Gauss | poder evaluar entre los datos | interpolar (spline cúbico) |

Con $n$ múltiplo de $\text{mcm}(2,3)=6$ los tres Newton–Cotes son aplicables a la vez. En
la fase de cámara $n$ vendrá impuesto por la altura en píxeles del envase, así que habrá
que **remuestrear a un $n$ múltiplo de 6** antes de integrar.

## 5. Sensibilidad al error de escala — lo que decide el proyecto

La escala convierte radios **y** alturas, así que $V \propto s^3$ y
$\;\Delta V/V \approx 3\,\Delta s/s$. (Un error solo en el borde del contorno afecta
únicamente al radio: $\Delta V/V = 2\,\Delta r/r$.)

| Error de escala | V estimado [cm³] | Error en V | ¿Pasa el 5 %? |
|---|---|---|---|
| 0.5 % | 625.84 | +1.50 % | ✅ |
| 1.0 % | 635.23 | +3.02 % | ✅ |
| 1.5 % | 644.71 | +4.56 % | ✅ |
| **2.0 %** | 654.28 | **+6.11 %** | ❌ |
| 3.0 % | 673.72 | +9.26 % | ❌ |

**Límite de escala para el 5 %: $1.05^{1/3}-1 = 1.64\,\%$.**
**Un error de escala del 1 % pesa 71 veces más que el peor método numérico.**

Con la escala tomada de la altura del envase medida con regla, 1 mm de error de lectura
es 0.48 % de escala en una botella de 21 cm (1.4 % de volumen) y 0.82 % en una lata de
12 cm (2.5 % de volumen). **Hay que medir la altura con cuidado, al medio milímetro.**

El error del método se reduce gratis subiendo $n$ — $1.0\times10^{-2}$ % con $n$=60,
$2.6\times10^{-3}$ % con $n$=240, $2.6\times10^{-5}$ % con $n$=1200 — mientras que mejorar
la escala exige trabajo físico real.

## 6. Fuentes de error, por lo que realmente pesan

| Fuente | Magnitud | Estado |
|---|---|---|
| Calibración de escala px→cm | 3× el error de escala | ❌ fase 2 |
| Perspectiva de la fotografía | ~2·(R/distancia) | ❌ fase 2 |
| Extracción del contorno | 1–3 px por borde | ❌ fase 2 |
| Asimetría del envase | 0–5 % | ❌ fase 2 |
| Elección del método | ~0.04 % | ✅ medido |
| Discretización ($\Delta z$) | ~0.01 %, y baja con $n$ | ✅ medido |

Las dos fuentes numéricas son **despreciables**. El motor de cálculo está resuelto y le
sobra precisión.

## 7. Presupuesto de error para la fase 2

| Parámetro | Objetivo | Límite |
|---|---|---|
| Error de escala px→cm | < 0.5 % | < 1.64 % |
| Error resultante en el volumen | < 1.5 % | < 5 % |

**Decisiones que se derivan:**

- **Método a producción:** trapecio o Simpson sobre los datos crudos — robustos ante
  aristas y ruido, y el trapecio no impone condiciones sobre $n$.
- **Gauss:** solo por tramos, nunca global. Exige detectar antes los quiebres del perfil.
- **Malla:** remuestrear a $n$ múltiplo de 6; subir $n$ es gratis.
- **Validación:** fotos sintéticas de volumen exacto, más un envase de geometría conocida
  (una lata) medido con regla.
- **Escala:** la altura real del envase, medida con regla al medio milímetro. Está sobre
  el eje de revolución, así que no hay problema de coplanaridad. Respaldo: dos clics
  sobre una regla visible en la foto.
- **Captura:** cámara lejos y con zoom, para aproximar la proyección ortográfica que
  supone el modelo.

## 8. Cobertura del enunciado

| Requisito del PDF | Dónde |
|---|---|
| §5 Contexto matemático — $V=\pi\int r^2dz$ | encabezado |
| §6 Trapecio, Simpson, Gauss, Newton–Cotes | sección 3 |
| §6 Efecto de $\Delta z$ y del nº de puntos | secciones 4 y 6 |
| §7 Etapa 4 — tabla $(z_i, r_i, r_i^2, \pi r_i^2)$ | sección 2 |
| §7 Etapa 5 — parámetros de cada método | sección 4 |
| §7 Etapa 6 — diferencia absoluta y error relativo | sección 4 |
| §8 Reporte explícito de $\Delta z$ | secciones 2 y 4 |
| §8 Mensajes ante condiciones no cumplidas | sección 3 |
| §8 Gráfica del perfil y de $\pi r^2$ | sección 5 |
| §9 MVP — explicar las fuentes de error | sección 6 |

Pendiente para la fase 2: §7 Etapas 2 y 3 (captura, escalamiento, extracción del perfil) y
los requisitos de interfaz de la §8.""")

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
print(f"Notebook escrito: {out}")
print(f"  {len(celdas)} celdas ({md} texto + {nc} codigo)  |  {out.stat().st_size:,} bytes")
print("  Sintaxis OK")
