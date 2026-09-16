# -*- coding: utf-8 -*-
"""Celdas: Boole (Newton-Cotes orden superior) y cuadratura de Gauss-Legendre."""

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

# ----------------------------------------------------------------- BOOLE
M(r"""### 5.5 Regla de Boole — Newton-Cotes cerrada de orden superior

Cubre la otra opción que ofrece la §6 del PDF: *"...o una de orden superior"*.

#### La idea

Grado 4: ajustamos un polinomio de **cuarto grado** por cada **cuatro** subintervalos
(5 puntos determinan una cuártica):

$$\int_{z_i}^{z_{i+4}} f\,dz \;\approx\; \frac{2\,\Delta z}{45}\bigl[7f_i + 32f_{i+1} + 12f_{i+2} + 32f_{i+3} + 7f_{i+4}\bigr]$$

Los pesos $[7, 32, 12, 32, 7]$ ya no son "bonitos" como los de Simpson — son el
resultado de integrar los polinomios de Lagrange de grado 4 sobre 5 nodos equiespaciados.
Al componer, los extremos compartidos entre bloques consecutivos suman $7+7 = 14$:

$$[\,7,\;32,\;12,\;32,\;\mathbf{14},\;32,\;12,\;32,\;\mathbf{14},\;\dots,\;32,\;12,\;32,\;7\,]$$

#### Condición: n múltiplo de 4

#### Error

$$E \;=\; -\frac{2(b-a)\,\Delta z^{6}}{945}\,f^{(6)}(\xi)$$

**$O(\Delta z^6)$**: al duplicar los puntos, el error cae a **1/64**. Y el **grado de
exactitud es 5**, no 4 — otra vez el grado extra de regalo, porque 4 es **par**. Esto
confirma el patrón anunciado en la sección anterior.

#### Por qué Newton-Cotes se detiene aquí en la práctica

Podríamos seguir: grado 6, 8, 10... Pero a partir de grado ≈ 8 los pesos de
Newton-Cotes **empiezan a tomar signos alternados y magnitudes crecientes**. Eso hace la
fórmula **numéricamente inestable**: se restan números grandes para obtener uno pequeño,
y la cancelación catastrófica amplifica el error de redondeo hasta destruir el
resultado. Es una manifestación del **fenómeno de Runge**, la razón por la que
interpolar con polinomios de alto grado sobre nodos *equiespaciados* es una mala idea.

Ese callejón sin salida es precisamente **lo que motiva la cuadratura de Gauss**: si los
nodos equiespaciados limitan el grado alcanzable, la salida es **dejar de exigir que los
nodos sean equiespaciados**. Es el tema de la siguiente sección.

#### Predicción para nuestro envase

Grado de exactitud 5 ≥ 2 = grado de $f = r^2$. Igual que Simpson, Boole debería ser
**exacto dentro de cada tramo**, y todo su error debería venir de los bloques que cruzan
un quiebre. Como sus bloques son más anchos (4 subintervalos frente a 2), **cruzan
quiebres con más frecuencia**. Es perfectamente posible que Boole salga **peor que
Simpson 1/3** aquí, pese a ser de orden muy superior. Si ocurre, no es un error: es la
consecuencia directa de aplicar una herramienta de alto orden a una función no suave.""")

C(r'''def regla_boole(z: np.ndarray, r: np.ndarray) -> ResultadoIntegracion:
    """Regla de Boole compuesta (Newton-Cotes cerrada de grado 4).

    Pesos [7,32,12,32,14,32,12,32,14,...,32,12,32,7] · 2Δz/45.
    Condición: n múltiplo de 4.
    """
    n, dz = _validar_malla(z, r)
    if n % 4 != 0:
        faltan = 4 - (n % 4)
        raise CondicionNoSatisfecha(
            f"La regla de Boole requiere n múltiplo de 4 y se recibió n = {n} "
            f"(n % 4 = {n % 4}).\n"
            f"  Motivo: cada polinomio de grado 4 se ajusta sobre 4 subintervalos "
            f"consecutivos (5 nodos).\n"
            f"  Soluciones: usar n = {n - (n % 4)} o n = {n + faltan}, o remuestrear "
            f"el perfil a una malla con n múltiplo de 12 (compatible con los cuatro "
            f"métodos a la vez)."
        )
    f = np.asarray(r, dtype=float) ** 2

    def _calcular() -> float:
        w = np.zeros(n + 1)
        for i in range(0, n, 4):          # bloques de 4 subintervalos
            w[i]     +=  7.0
            w[i + 1] += 32.0
            w[i + 2] += 12.0
            w[i + 3] += 32.0
            w[i + 4] +=  7.0              # los extremos compartidos acumulan 7+7=14
        return float(2.0 * dz / 45.0 * np.dot(w, f))

    integral, ms = _cronometrar(_calcular)
    return ResultadoIntegracion(
        metodo="Boole compuesto",
        familia="Newton-Cotes cerrada (grado 4)",
        n_subintervalos=n,
        n_evaluaciones=n + 1,
        a=float(z[0]), b=float(z[-1]),
        delta_z=dz,
        orden_error="O(Δz⁶)",
        grado_exactitud=5,
        integral_r2=integral,
        volumen_cm3=math.pi * integral,
        tiempo_ms=ms,
        notas="Requiere n múltiplo de 4. Bloques anchos: sensible a puntos de quiebre.",
    )


res_boole = regla_boole(Z_BASE, R_BASE)
print("=" * 74)
print("MÉTODO 4 — REGLA DE BOOLE  (Newton-Cotes de orden superior)")
print("=" * 74)
print(res_boole.informe(V_EXACTO))
print("=" * 74)
print("  Comparación entre las Newton-Cotes, con los MISMOS 61 puntos:")
for _r in (res_trap, res_s13, res_s38, res_boole):
    _ea, _er = _r.error_vs(V_EXACTO)
    print(f"     {_r.metodo:<24} {_r.orden_error:<9} error = {_ea:.6e} cm³  ({_er:.3e} %)")''')

M(r"""**Cómo leer el resultado — la tabla comparativa al final de la celda es la clave.**

El ranking que verás, de mejor a peor, es este:

| Puesto | Método | Orden teórico | Error [cm³] |
|---|---|---|---|
| 1 | **Trapecio** | $O(\Delta z^2)$ | ≈ 0.063 |
| 2 | Simpson 3/8 | $O(\Delta z^4)$ | ≈ 0.071 |
| 3 | Simpson 1/3 | $O(\Delta z^4)$ | ≈ 0.263 |
| 4 | **Boole** | $O(\Delta z^6)$ | ≈ 0.301 |

**El ranking está exactamente invertido respecto al orden teórico.** El método de orden
2 gana; el de orden 6 pierde. Y no por poco: Boole acumula casi 5 veces más error que el
trapecio.

- **La causa es la anticipada:** los bloques de Boole abarcan 4 subintervalos, los más
  anchos de las cuatro reglas, así que **cruzan codos con más frecuencia que ninguna
  otra**. En un codo, la hipótesis "$f^{(6)}$ existe y es acotada" no se cumple — ni
  siquiera existe $f''$ — y la cota $O(\Delta z^6)$ sencillamente **no es aplicable**.
  Lo que queda es un polinomio de grado 4 intentando ajustarse a cinco puntos repartidos
  entre dos curvas distintas.
- **Hay una regularidad interesante en el ranking:** ordena los métodos por **ancho de
  bloque** (1, 3, 2, 4 subintervalos) más que por orden teórico. Cuanto más ancho el
  bloque, más codos cruza y más error acumula. Sobre integrandos quebrados, **el ancho
  del estarcido predice el error mejor que el orden de la fórmula**.
- **Esta es la observación más valiosa de toda la sección**, y merece ir textualmente al
  informe: *el orden teórico de un método solo se materializa si el integrando cumple
  las hipótesis de suavidad de su cota de error*. Sobre datos extraídos de una fotografía
  — con ruido de píxel y aristas del envase — **los métodos de orden alto no solo pierden
  su ventaja: pueden quedar por detrás de los más simples**.
- **Los cuatro tiempos de cálculo son del mismo orden** (microsegundos). El coste
  computacional **no** es un criterio de selección aquí. Lo que decide es la robustez
  frente a datos imperfectos, y en ese criterio **gana el método más humilde**.
- **Perspectiva, para no exagerar:** los cuatro errores están entre 0.06 y 0.30 cm³, es
  decir, entre el **0.2 % y el 1.0 % del margen del 5 %** (que son 30.8 cm³). Los cuatro
  son perfectamente utilizables. La discusión sobre cuál gana es **académicamente
  fascinante y prácticamente irrelevante** — y saber distinguir esas dos cosas también
  es parte del oficio.""")

# ----------------------------------------------------------------- GAUSS
M(r"""### 5.6 Cuadratura de Gauss-Legendre

#### El cambio de perspectiva

Todas las reglas anteriores comparten una restricción autoimpuesta: **los nodos están
fijos y equiespaciados**, y lo único que se elige son los pesos. Con $m$ nodos hay $m$
pesos libres, lo que permite imponer exactitud hasta grado $m-1$.

Gauss se pregunta: *¿y si también elegimos dónde poner los nodos?*

Con $m$ nodos hay entonces $2m$ parámetros libres ($m$ posiciones + $m$ pesos), y se
puede lograr exactitud hasta grado:

$$\boxed{\;2m - 1\;}$$

**Con 3 nodos, Gauss integra exactamente polinomios de grado 5** — algo para lo que
Newton-Cotes necesita 5 nodos (Boole). Es prácticamente el doble de rendimiento por
evaluación.

#### Dónde van los nodos

Los nodos óptimos son las **raíces del polinomio de Legendre** $P_m(x)$ en $[-1, 1]$.
No son equiespaciados: se **agrupan hacia los extremos** del intervalo, que es
precisamente el patrón que evita el fenómeno de Runge.

Para $m = 3$: $x = -\sqrt{3/5},\; 0,\; +\sqrt{3/5}$, con pesos $5/9,\; 8/9,\; 5/9$.

#### El cambio de variable

Los nodos de Legendre viven en $[-1, 1]$ y nuestro envase en $[a, b]$. La transformación
afín es:

$$z = \frac{b-a}{2}x + \frac{a+b}{2}, \qquad dz = \frac{b-a}{2}\,dx$$

y por tanto:

$$\int_a^b f(z)\,dz \;=\; \frac{b-a}{2}\int_{-1}^{1} f\!\left(\tfrac{b-a}{2}x + \tfrac{a+b}{2}\right)dx \;\approx\; \frac{b-a}{2}\sum_{k=1}^{m} w_k\, f(z_k)$$

El factor $\frac{b-a}{2}$ es el **jacobiano** del cambio de variable. Olvidarlo es el
error de implementación más común en Gauss, y es fácil de detectar: el resultado sale
escalado por un factor constante, en nuestro caso $21/2 = 10.5$.

#### ⚠️ El problema práctico, y por qué importa tanto para este proyecto

Los nodos de Gauss caen en posiciones **irracionales** como $z = 10.5 - 10.5\sqrt{3/5}$.

De una fotografía **jamás** obtendremos el radio exactamente ahí: los datos vienen en
filas de píxeles equiespaciadas. **Gauss exige poder evaluar el integrando donde él
decida, y con datos tabulados eso es imposible.**

La solución obligada: **interpolar**. Construimos un spline cúbico sobre los datos
$(z_i, r_i)$ y lo evaluamos en los nodos de Gauss. Y aquí está el punto crítico:

> El resultado de Gauss deja de ser "la integral de los datos" y pasa a ser
> **"la integral del spline que ajustamos a los datos"**. Hemos introducido una segunda
> aproximación — el modelo de interpolación — encima de la cuadratura.

Por eso implementamos **dos variantes**, y la comparación entre ambas es material de
primera para el informe:

| Variante | Evalúa el radio con | Situación que representa |
|---|---|---|
| `gauss_desde_funcion` | el perfil analítico exacto | ideal irreal: el "techo" de Gauss |
| `gauss_desde_datos` | spline cúbico sobre la tabla | **lo que realmente ocurrirá con una foto** |

La §7 Etapa 5 del PDF exige *"exactamente el mismo perfil de entrada, o una
representación equivalente"*: el spline sobre la misma tabla **es** esa representación
equivalente, y la variante analítica queda como referencia diagnóstica.""")

C(r'''def _nodos_pesos_legendre(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodos y pesos de Gauss-Legendre de m puntos en [-1, 1].

    numpy los calcula como raíces del polinomio de Legendre P_m mediante el
    autovalor de la matriz de Jacobi asociada — estable incluso para m grande.
    """
    if m < 1:
        raise CondicionNoSatisfecha(f"El número de nodos de Gauss debe ser >= 1 (m={m}).")
    x, w = np.polynomial.legendre.leggauss(m)
    return x, w


def gauss_desde_funcion(perfil: PerfilEnvase, m: int) -> ResultadoIntegracion:
    """Gauss-Legendre de m nodos evaluando el perfil ANALÍTICO.

    Es el caso ideal: podemos pedir el radio en cualquier z con precisión exacta.
    En la fase de cámara esto NO estará disponible; sirve como cota superior del
    rendimiento alcanzable por Gauss.
    """
    x, w = _nodos_pesos_legendre(m)
    a, b = perfil.a, perfil.b
    semi = (b - a) / 2.0            # jacobiano del cambio de variable
    centro = (a + b) / 2.0
    z_nodos = semi * x + centro     # mapeo [-1,1] → [a,b]

    def _calcular() -> float:
        f = perfil.radio_cuadrado(z_nodos)
        return float(semi * np.dot(w, f))

    integral, ms = _cronometrar(_calcular)
    return ResultadoIntegracion(
        metodo=f"Gauss-Legendre {m} nodos (perfil analítico)",
        familia="Cuadratura gaussiana",
        n_subintervalos=0,              # no hay subintervalos: no es una regla compuesta
        n_evaluaciones=m,
        a=float(a), b=float(b),
        delta_z=None,                   # nodos NO equiespaciados
        orden_error=f"exacta para grado ≤ {2*m - 1}",
        grado_exactitud=2 * m - 1,
        integral_r2=integral,
        volumen_cm3=math.pi * integral,
        tiempo_ms=ms,
        notas=f"Nodos en las raíces de P_{m}(x), reescalados a [a,b]. Caso ideal.",
    )


def gauss_desde_datos(z: np.ndarray, r: np.ndarray, m: int,
                      tipo_spline: str = "cubico") -> ResultadoIntegracion:
    """Gauss-Legendre de m nodos sobre un SPLINE ajustado a la tabla (z, r).

    Es el escenario realista: los datos vienen tabulados en posiciones fijas y hay
    que interpolar para evaluar el integrando en los nodos de Gauss.

    IMPORTANTE: el resultado es la integral del SPLINE, no la de los datos. La
    calidad del spline pasa a formar parte del error total.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if len(z) < 4 and tipo_spline == "cubico":
        raise CondicionNoSatisfecha(
            f"Un spline cúbico requiere al menos 4 puntos y se recibieron {len(z)}.\n"
            f"  Solución: aumentar n, o usar tipo_spline='lineal'."
        )
    grado = {"lineal": 1, "cuadratico": 2, "cubico": 3}[tipo_spline]
    spline = interpolate.InterpolatedUnivariateSpline(z, r, k=grado, ext="const")

    x, w = _nodos_pesos_legendre(m)
    a, b = float(z[0]), float(z[-1])
    semi = (b - a) / 2.0
    centro = (a + b) / 2.0
    z_nodos = semi * x + centro

    def _calcular() -> float:
        f = spline(z_nodos) ** 2
        return float(semi * np.dot(w, f))

    integral, ms = _cronometrar(_calcular)
    return ResultadoIntegracion(
        metodo=f"Gauss-Legendre {m} nodos (spline sobre datos)",
        familia="Cuadratura gaussiana",
        n_subintervalos=len(z) - 1,
        n_evaluaciones=m,
        a=a, b=b,
        delta_z=None,
        orden_error=f"exacta para grado ≤ {2*m - 1} (del spline)",
        grado_exactitud=2 * m - 1,
        integral_r2=integral,
        volumen_cm3=math.pi * integral,
        tiempo_ms=ms,
        notas=(f"Spline {tipo_spline} (k={grado}) sobre {len(z)} puntos, evaluado en "
               f"los {m} nodos de Gauss. Este es el escenario de la fase de cámara."),
    )


# --- Ejecución con m = 5 nodos ----------------------------------------
M_GAUSS = 5
res_gauss_fn = gauss_desde_funcion(ENVASE, M_GAUSS)
res_gauss_dt = gauss_desde_datos(Z_BASE, R_BASE, M_GAUSS)

_x5, _w5 = _nodos_pesos_legendre(M_GAUSS)
_z5 = (ENVASE.b - ENVASE.a) / 2 * _x5 + (ENVASE.a + ENVASE.b) / 2

print("=" * 74)
print(f"MÉTODO 5 — CUADRATURA DE GAUSS-LEGENDRE ({M_GAUSS} nodos)")
print("=" * 74)
print(f"  Nodos en [-1, 1] : {_x5}")
print(f"  Pesos            : {_w5}")
print(f"  Suma de pesos    : {_w5.sum():.12f}   (debe ser exactamente 2 = longitud de [-1,1])")
print(f"  Nodos en [a, b]  : {_z5}  cm")
print(f"  Jacobiano (b−a)/2: {(ENVASE.b - ENVASE.a) / 2:.6f}")
print("=" * 74)
print("\n--- Variante A: evaluando el perfil ANALÍTICO (ideal) ---")
print(res_gauss_fn.informe(V_EXACTO))
print("\n--- Variante B: evaluando un SPLINE sobre la tabla (realista) ---")
print(res_gauss_dt.informe(V_EXACTO))
print("=" * 74)''')

M(r"""**Cómo leer el resultado:**

- **`Suma de pesos = 2.000000000000`** — verificación imprescindible. Los pesos de
  Gauss-Legendre deben sumar la longitud del intervalo canónico $[-1,1]$, que es 2.
  Equivale a decir que el método integra exactamente la función constante $f=1$. Si esta
  suma no diera 2, habría un error en la generación de nodos y **todo lo demás sería
  inválido**. Es la comprobación de sanidad más barata y más útil de Gauss.
- **Los nodos NO están equiespaciados**: observa cómo se agrupan hacia $\pm 1$ y dejan
  un hueco mayor en el centro. Esa distribución es lo que evita el fenómeno de Runge y
  hace posible el grado de exactitud $2m-1$.
- **Solo 5 evaluaciones** frente a las 61 de los métodos anteriores. Gauss es
  **doce veces más barato en datos**. Si cada evaluación costara un experimento de
  laboratorio, esa diferencia sería decisiva.
- **⚠️ Gauss con 5 nodos es, con diferencia, el PEOR método de todo el notebook:**
  ≈ 3.9 cm³ de error, un **0.63 %** — unas **13 veces peor** que Boole y **62 veces peor**
  que el trapecio. Consume él solo el **12.7 % del margen del 5 %**, mientras que los
  cuatro Newton-Cotes se quedan por debajo del 1 %.
- **Por qué falla tan estrepitosamente.** Con 5 nodos, Gauss es exacto hasta grado 9 —
  muy por encima del grado 2 de $f = r^2$. Pero **nuestro integrando no es un polinomio
  global**: es cuadrático *por tramos*, con cuatro esquinas. Ningún polinomio de grado 9,
  por bien elegidos que estén sus nodos, puede reproducir una función quebrada. Y Gauss
  es una regla **global**, no compuesta: no subdivide el intervalo, intenta capturar los
  21 cm de envase con **un solo polinomio de grado 9** y 5 muestras. Con 5 puntos es
  literalmente imposible saber que hay un hombro en $z=15$.
- **La comparación entre las dos variantes da un resultado inesperado:** la variante B
  (spline, ≈ 3.01 cm³) sale **algo MEJOR** que la variante A (perfil exacto, ≈ 3.91 cm³),
  pese a que B añade un error de interpolación que A no tiene.
- **Esto no significa que interpolar mejore las cosas.** Es una **cancelación
  fortuita**: el spline cúbico *suaviza* los codos del envase, redondeando el hombro y
  el pie. Ese sesgo de modelado resulta tener signo opuesto al error de Gauss sobre la
  función quebrada, y los dos se compensan parcialmente. Nada lo garantiza — con otro
  envase, otro $m$ u otro tipo de spline, los dos errores podrían sumarse en vez de
  restarse.
- **La lección correcta:** cuando dos errores independientes son del mismo orden de
  magnitud, su suma es impredecible en signo y magnitud. **Que un resultado salga mejor
  no lo hace más fiable.** Aquí ambas variantes están en ~0.5-0.6 %, y esa es la cifra
  honesta del rendimiento de Gauss-5 sobre este problema; cuál de las dos sale encima es
  ruido.
- **Conclusión para el proyecto — y va contra la intuición:** con datos de imagen, Gauss
  **no solo no es superior: es el peor candidato**. Su ventaja teórica se paga con dos
  facturas que los Newton-Cotes no pagan: (1) depende de una interpolación, porque no
  puede evaluar donde no hay datos; y (2) al ser una regla global, es **especialmente
  vulnerable a la falta de suavidad**. Con 61 puntos disponibles y una foto ruidosa,
  **el trapecio o Simpson sobre los datos crudos son la opción sensata**.
- **Gauss brilla** cuando evaluar el integrando es caro (cada evaluación es una
  simulación o un experimento) y la función es suave. **Nuestro escenario es exactamente
  el contrario**: evaluaciones gratis, función quebrada. Reportarlo es obligatorio por el
  enunciado; usarlo como valor principal sería un error de ingeniería.""")

M(r"""### 5.7 ¿Y si le damos más nodos a Gauss?

La reacción natural al resultado anterior es: *"5 nodos son pocos, subamos a 20 o 50"*.

**Qué hace esta celda:** barre $m$ desde 2 hasta 160 nodos y mide el error.

La §6 del PDF pide analizar la *"influencia del orden de cuadratura"*. Esto lo hace, y
el resultado es instructivo precisamente porque **no se comporta como uno espera**.""")

C(r"""filas_g = []
for m in [2, 3, 5, 8, 12, 20, 40, 80, 160]:
    rg = gauss_desde_funcion(ENVASE, m)
    e_abs, e_rel = rg.error_vs(V_EXACTO)
    filas_g.append({
        "m (nodos)": m,
        "Grado exacto (2m−1)": 2 * m - 1,
        "V [cm³]": rg.volumen_cm3,
        "Error abs. [cm³]": e_abs,
        "Error rel. [%]": e_rel,
        "% del margen del 5%": 100 * e_abs / (0.05 * V_EXACTO),
        "¿Pasa el 5%?": "✓" if e_rel < 5 else "✗ NO",
    })
GAUSS_BARRIDO = pd.DataFrame(filas_g)

print("=" * 78)
print("INFLUENCIA DEL ORDEN DE CUADRATURA EN GAUSS-LEGENDRE")
print("=" * 78)
display(GAUSS_BARRIDO)

# ¿Es monótono el error al aumentar m?
_errs = GAUSS_BARRIDO['Error abs. [cm³]'].to_numpy()
_subidas = [(int(GAUSS_BARRIDO['m (nodos)'].iloc[i]), int(GAUSS_BARRIDO['m (nodos)'].iloc[i+1]))
            for i in range(len(_errs) - 1) if _errs[i+1] > _errs[i]]
print("=" * 78)
if _subidas:
    print('  ⚠ El error NO decrece monótonamente al añadir nodos. Empeora al pasar de:')
    for m1, m2 in _subidas:
        print(f'       m = {m1}  →  m = {m2}')
else:
    print('  El error decrece monótonamente.')

_trap_err = res_trap.error_vs(V_EXACTO)[0]
_m_necesarios = GAUSS_BARRIDO[GAUSS_BARRIDO['Error abs. [cm³]'] < _trap_err]
print('=' * 78)
if len(_m_necesarios):
    _m = int(_m_necesarios['m (nodos)'].iloc[0])
    print(f'  Gauss necesita m ≥ {_m} nodos para igualar al TRAPECIO con 61 puntos')
    print(f'  (error del trapecio: {_trap_err:.4e} cm³)')
else:
    print(f'  Ni con 160 nodos Gauss alcanza la precisión del trapecio con 61 puntos')
    print(f'  (error del trapecio: {_trap_err:.4e} cm³)')
print('=' * 78)""")

M(r"""**Cómo leer el resultado — es contundente:**

- **Con $m = 2$ y $m = 3$ nodos, Gauss FALLA el criterio del 5 % del reto** (≈ 5.1 % y
  ≈ 5.3 % de error). Si alguien implementara Gauss de bajo orden como método principal,
  **suspendería la prueba de campo aun con una fotografía perfecta**.
- **El error NO decrece monótonamente.** Añadir nodos a veces **empeora** el resultado:
  de $m=2$ a $m=3$ sube, de $m=5$ a $m=8$ sube, de $m=12$ a $m=20$ sube. Esto es
  desconcertante si uno espera que "más puntos = mejor", y es la firma inequívoca de un
  método **fuera de su régimen de validez**. La convergencia de Gauss está garantizada
  para funciones analíticas; para una función con esquinas, degenera a una convergencia
  lenta y errática.
- **Gauss necesita ≈ 80 nodos para igualar lo que el trapecio consigue con 61 puntos.**
  Es decir: el método teóricamente más eficiente en evaluaciones necesita **más
  evaluaciones que el más elemental** para llegar al mismo sitio. La ventaja de
  "grado $2m-1$ con $m$ nodos", que sobre una función suave sería aplastante, aquí se
  evapora por completo.
- **La moraleja, que aplica mucho más allá de este laboratorio:**

> Un método numérico no es "bueno" o "malo" en abstracto: es bueno **para una clase de
> problemas**. Gauss-Legendre es extraordinario para integrandos suaves y evaluaciones
> caras. Aplicado a una función quebrada muestreada densamente, es la peor elección
> disponible. **Elegir el método sin mirar la estructura del problema es el error de
> ingeniería que este notebook existe para prevenir.**

- **Cómo se arreglaría** (fuera del alcance de esta fase): aplicar Gauss **por tramos**,
  es decir, una cuadratura gaussiana independiente en cada tramo suave del envase. Eso
  convierte una regla global en una compuesta y recupera su precisión. Es la misma idea
  de "alinear la malla con la estructura del problema" que la Sección 8 desarrolla para
  los Newton-Cotes, y es exactamente lo que hace `scipy.integrate.quad` con su argumento
  `points`.""")
