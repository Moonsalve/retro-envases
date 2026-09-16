# -*- coding: utf-8 -*-
"""Celdas: muestreo, tabla de datos (Etapa 4) e infraestructura de resultados."""

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

M(r"""---
## 4. Construcción del conjunto de datos numéricos — §7 Etapa 4

### 4.1 El paso conceptual clave de todo el laboratorio

Hasta ahora teníamos $r(z)$ como **función**: podíamos evaluarla en cualquier $z$, con
precisión infinita. Eso es un lujo que en el proyecto real **no vamos a tener**.

De una fotografía no sale una función; sale una **lista finita de pares** $(z_i, r_i)$,
uno por fila de píxeles del contorno. Así que aquí hacemos deliberadamente el mismo
sacrificio: **tiramos la función y nos quedamos solo con la tabla**.

$$r(z) \;\;\xrightarrow{\text{muestreo}}\;\; \{(z_0, r_0), (z_1, r_1), \dots, (z_n, r_n)\}$$

A partir de este punto, los métodos de integración verán **exactamente el mismo tipo de
dato que verán en la fase de cámara**. Eso es lo que hace que las conclusiones de este
notebook sean transferibles: no estamos midiendo el error de integrar una función bonita,
estamos midiendo el error de integrar *una tabla*.

### 4.2 Malla uniforme y el diferencial Δz

Muestreamos con **espaciamiento constante**, como pide la §5 del enunciado:

$$\Delta z = \frac{b-a}{n}, \qquad z_i = a + i\,\Delta z, \quad i = 0,1,\dots,n$$

Dos advertencias sobre el conteo, porque es fuente constante de errores de programación:

- $n$ = número de **subintervalos** (los huecos entre puntos).
- $n+1$ = número de **puntos** (nodos).

Simpson pide "$n$ par" refiriéndose a **subintervalos**, es decir, un número *impar* de
puntos. Confundirlos es el error clásico. En el código, `n_subintervalos` y
`n_puntos` son campos separados y explícitos precisamente para que nadie tenga que
recordar cuál es cuál.

### 4.3 Por qué elegimos n = 60 como caso base

Los cuatro métodos imponen restricciones distintas sobre $n$:

| Método | Requiere |
|---|---|
| Trapecio | cualquier $n \ge 1$ |
| Simpson 1/3 | $n$ **par** (múltiplo de 2) |
| Simpson 3/8 | $n$ múltiplo de **3** |
| Boole | $n$ múltiplo de **4** |

Para que **los cuatro operen sobre la misma tabla** — lo cual exige literalmente la §7
Etapa 5 del PDF: *"utilizar exactamente el mismo perfil de entrada"* — necesitamos que
$n$ sea múltiplo del **mínimo común múltiplo**:

$$\mathrm{mcm}(2, 3, 4) = 12$$

$n = 60 = 5 \times 12$ cumple las cuatro condiciones, y da $\Delta z = 0.35$ cm, una
resolución comparable a la que dará una foto de celular. Más adelante, la Sección 8
barrerá muchos valores de $n$; aquí fijamos uno para poder inspeccionar la tabla a
mano.""")

C(r'''def mcm(a: int, b: int) -> int:
    """Mínimo común múltiplo de dos enteros."""
    return a * b // math.gcd(a, b)


# mcm(2,3,4) = 12 → todo múltiplo de 12 sirve para los cuatro métodos a la vez.
N_COMPATIBLE = mcm(mcm(2, 3), 4)


def n_compatible_mas_cercano(n_deseado: int) -> int:
    """Devuelve el múltiplo de 12 más cercano a n_deseado (mínimo 12).

    Sirve para responder al usuario cuando pide un n que no sirve para todos los
    métodos: en lugar de solo rechazarlo, le proponemos el valor válido más próximo.
    Rechazar sin sugerir es un mal mensaje de error.
    """
    if n_deseado <= N_COMPATIBLE:
        return N_COMPATIBLE
    abajo = (n_deseado // N_COMPATIBLE) * N_COMPATIBLE
    arriba = abajo + N_COMPATIBLE
    return abajo if (n_deseado - abajo) <= (arriba - n_deseado) else arriba


def muestrear_perfil(perfil: PerfilEnvase, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Convierte el perfil continuo en una tabla discreta (z_i, r_i).

    Este es el punto donde renunciamos a la función y nos quedamos con datos,
    replicando lo que ocurrirá al extraer el contorno de una fotografía.

    Parámetros
    ----------
    perfil : PerfilEnvase   El envase a muestrear.
    n      : int            Número de SUBINTERVALOS (se generan n+1 puntos).

    Devuelve
    --------
    (z, r) : dos arrays de longitud n+1, en cm.
    """
    if n < 1:
        raise ValueError(f"n debe ser >= 1, se recibió {n}.")
    z = np.linspace(perfil.a, perfil.b, n + 1)   # linspace incluye ambos extremos
    r = perfil.radio(z)
    return z, r


def tabla_datos(z: np.ndarray, r: np.ndarray) -> pd.DataFrame:
    """Construye la tabla que exige la §7 Etapa 4 del enunciado.

    Columnas: i | z_i | r_i | r_i² | f(z_i) = π·r_i²

    La columna r² es el integrando real; la columna π·r² es el área del disco a esa
    altura. Ambas se muestran para que el cálculo sea verificable a mano, que es
    justamente lo que pide el enunciado.
    """
    return pd.DataFrame({
        "i": np.arange(len(z)),
        "z_i [cm]": z,
        "r_i [cm]": r,
        "r_i² [cm²]": r ** 2,
        "f(z_i) = π·r_i² [cm²]": math.pi * r ** 2,
    })


# =======================================================================
#  CASO BASE
# =======================================================================
N_BASE = 60                                   # ✏️ editable (múltiplo de 12)
Z_BASE, R_BASE = muestrear_perfil(ENVASE, N_BASE)
DZ_BASE = (ENVASE.b - ENVASE.a) / N_BASE
TABLA_BASE = tabla_datos(Z_BASE, R_BASE)

print("=" * 74)
print("CONJUNTO DE DATOS NUMÉRICOS  (§7 Etapa 4)")
print("=" * 74)
print(f"  Intervalo [a, b]          : [{ENVASE.a:.4f}, {ENVASE.b:.4f}] cm")
print(f"  n (subintervalos)         : {N_BASE}")
print(f"  Puntos (nodos) = n + 1    : {N_BASE + 1}")
print(f"  Δz = (b − a)/n            : {DZ_BASE:.6f} cm   ← EL DIFERENCIAL")
print(f"  Malla uniforme            : sí")
print("-" * 74)
print(f"  Compatibilidad de n = {N_BASE}:")
for nombre, req in (("Trapecio", 1), ("Simpson 1/3", 2), ("Simpson 3/8", 3), ("Boole", 4)):
    ok = "✓" if N_BASE % req == 0 else "✗"
    print(f"     {ok} {nombre:<14} requiere n múltiplo de {req}"
          f"   →  {N_BASE} % {req} = {N_BASE % req}")
print("=" * 74)

print("\nPrimeras 8 filas:")
display(TABLA_BASE.head(8))
print("Últimas 5 filas:")
display(TABLA_BASE.tail(5))''')

M(r"""**Cómo leer el resultado:**

- **`Δz = 0.35 cm`** es *el diferencial de discretización* que la §5 del PDF exige
  hacer explícito. Es la distancia física entre dos filas consecutivas de la tabla. En
  la fase de cámara, Δz será literalmente **la altura de un píxel expresada en
  centímetros**.
- **61 puntos para 60 subintervalos.** Compruébalo en la tabla: el índice `i` llega
  hasta 60, así que hay 61 filas. Esta distinción es la que rige las condiciones de
  Simpson y Boole.
- **Las cuatro marcas ✓** confirman que $n=60$ sirve para los cuatro métodos, de modo
  que los compararemos sobre datos idénticos — sin esto, cualquier diferencia entre
  métodos sería incomparable, porque estarían integrando tablas distintas.
- **La columna `r_i`** en las primeras filas crece de 2.50 a 3.40: es el pie del envase
  ensanchándose. En las últimas filas ronda 1.35-1.45: el cuello y la boca.
- **La columna `f(z_i) = π·r_i²`** es el área del disco a esa altura, en cm². El volumen
  es el área bajo *esta* curva. Nota que $\pi r^2$ amplifica: pasar de $r$=3.40 a
  $r$=1.35 (una caída del 60 %) hace caer $f$ de 36.3 a 5.7 cm² (una caída del 84 %).
  Esa es la no linealidad del cuadrado, visualmente.""")

M(r"""---
## 5. Los cuatro métodos de integración numérica — §6

### 5.1 Infraestructura común

**Qué hace esta celda:** define dos piezas de soporte que usarán los cuatro métodos.

**`ResultadoIntegracion`** — un contenedor inmutable con *todo* lo que la §7 Etapa 5 del
enunciado obliga a reportar:

> *número de subintervalos o puntos utilizados; intervalo [a,b]; Δz o la discretización
> equivalente; orden o número de puntos de cuadratura cuando aplique; valor de la
> integral estimada; volumen estimado; tiempo de cálculo.*

**Por qué un dataclass y no imprimir dentro de cada función.** Separar el **cálculo** de
la **presentación** es una regla básica de diseño, y aquí tiene un beneficio concreto e
inmediato: en la Sección 8 ejecutaremos los métodos ~200 veces para el estudio de
convergencia. Si cada uno imprimiera, tendríamos 200 bloques de texto inútiles. Al
devolver objetos, los acumulamos en una lista, los volcamos a un DataFrame y los
graficamos. **Una función que calcula e imprime a la vez no se puede reutilizar.**

**`CondicionNoSatisfecha`** — una excepción propia. El PDF (§8) pide explícitamente
*"mensajes claros cuando los datos no cumplen las condiciones requeridas por un
método"*. Una excepción específica permite a la interfaz distinguir *"este método no
aplica a estos datos"* (recuperable: se avisa y se sigue con los demás) de *"el código
tiene un bug"* (no recuperable). Sus mensajes explican **qué falló, por qué, y cuál es
el valor válido más cercano** — los tres ingredientes de un buen mensaje de error.""")

C(r'''class CondicionNoSatisfecha(Exception):
    """Los datos no cumplen los requisitos del método de integración solicitado.

    Es un error ESPERADO y recuperable, no un bug: significa que este método
    concreto no es aplicable a esta tabla concreta. La interfaz debe capturarlo,
    informar al usuario y continuar con los demás métodos.
    """
    pass


@dataclass(frozen=True)
class ResultadoIntegracion:
    """Resultado completo de aplicar un método de cuadratura.

    Contiene todos los campos que la §7 Etapa 5 del enunciado exige reportar, de
    modo que el objeto por sí solo es un registro auditable del cálculo.
    """
    metodo: str                 # nombre legible
    familia: str                # "Newton-Cotes cerrada" | "Cuadratura gaussiana"
    n_subintervalos: int        # n
    n_evaluaciones: int         # cuántas veces se evaluó el integrando
    a: float                    # límite inferior [cm]
    b: float                    # límite superior [cm]
    delta_z: float | None       # Δz [cm]; None si la malla no es uniforme (Gauss)
    orden_error: str            # orden teórico del error global
    grado_exactitud: int        # máximo grado polinómico integrado exactamente
    integral_r2: float          # I = ∫ r² dz  [cm³]
    volumen_cm3: float          # V = π·I      [cm³]
    tiempo_ms: float            # tiempo de cálculo [ms]
    notas: str = ""             # observaciones (p. ej. composición de reglas)

    @property
    def volumen_ml(self) -> float:
        """Volumen en mL. 1 cm³ = 1 mL exactamente, por definición del litro."""
        return self.volumen_cm3

    def error_vs(self, v_referencia: float) -> tuple[float, float]:
        """Error absoluto [cm³] y error relativo porcentual frente a una referencia.

        Son las dos métricas que pide la §7 Etapa 6 del enunciado.
        """
        abs_err = abs(self.volumen_cm3 - v_referencia)
        rel_err = 100.0 * abs_err / abs(v_referencia)
        return abs_err, rel_err

    def informe(self, v_referencia: float | None = None) -> str:
        """Bloque de texto con todos los parámetros del método. Para el informe."""
        dz = "no uniforme" if self.delta_z is None else f"{self.delta_z:.6f} cm"
        lineas = [
            f"  Método                  : {self.metodo}",
            f"  Familia                 : {self.familia}",
            f"  Intervalo [a, b]        : [{self.a:.4f}, {self.b:.4f}] cm",
            f"  n (subintervalos)       : {self.n_subintervalos}",
            f"  Evaluaciones de r(z)    : {self.n_evaluaciones}",
            f"  Δz                      : {dz}",
            f"  Orden del error global  : {self.orden_error}",
            f"  Grado de exactitud      : {self.grado_exactitud} "
            f"(integra exactamente polinomios de grado ≤ {self.grado_exactitud})",
            f"  Integral I = ∫r²dz      : {self.integral_r2:.10f} cm³",
            f"  Volumen  V = π·I        : {self.volumen_cm3:.10f} cm³  = {self.volumen_ml:.4f} mL",
            f"  Tiempo de cálculo       : {self.tiempo_ms:.4f} ms",
        ]
        if v_referencia is not None:
            e_abs, e_rel = self.error_vs(v_referencia)
            lineas += [
                f"  Error absoluto          : {e_abs:.3e} cm³",
                f"  Error relativo          : {e_rel:.3e} %",
            ]
        if self.notas:
            lineas.append(f"  Notas                   : {self.notas}")
        return "\n".join(lineas)


def _cronometrar(funcion: Callable[[], float], repeticiones: int = 7) -> tuple[float, float]:
    """Ejecuta `funcion` varias veces y devuelve (valor, tiempo_mínimo_en_ms).

    Se toma el MÍNIMO, no el promedio. Razón: el tiempo medido es el tiempo real
    de cálculo MÁS el ruido del sistema (el planificador del SO interrumpe, otro
    proceso usa la CPU, entra el recolector de basura). Ese ruido es siempre
    positivo — nunca hace que algo vaya más rápido de lo que puede. Por tanto el
    mínimo de varias corridas es la mejor estimación del coste real.
    Es la misma razón por la que el módulo `timeit` de Python reporta el mínimo.
    """
    mejor = float("inf")
    valor = 0.0
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        valor = funcion()
        t1 = time.perf_counter()
        mejor = min(mejor, (t1 - t0) * 1000.0)
    return valor, mejor


def _validar_malla(z: np.ndarray, r: np.ndarray, tol: float = 1e-9) -> tuple[int, float]:
    """Verifica que (z, r) sea una malla uniforme válida. Devuelve (n, Δz).

    Las fórmulas de Newton-Cotes compuestas que implementamos ABAJO suponen
    espaciamiento constante. Aplicarlas a datos no uniformes daría un número
    plausible pero incorrecto — el peor tipo de fallo. Por eso se verifica aquí,
    una sola vez, en lugar de confiar en que quien llame lo haya hecho bien.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if z.ndim != 1 or r.ndim != 1:
        raise CondicionNoSatisfecha("z y r deben ser arrays unidimensionales.")
    if len(z) != len(r):
        raise CondicionNoSatisfecha(
            f"z y r deben tener la misma longitud (z: {len(z)}, r: {len(r)})."
        )
    if len(z) < 2:
        raise CondicionNoSatisfecha(
            f"Se requieren al menos 2 puntos para integrar; se recibieron {len(z)}."
        )
    if not np.all(np.diff(z) > 0):
        raise CondicionNoSatisfecha(
            "Las alturas z deben ser estrictamente crecientes. "
            "Ordena los datos antes de integrar."
        )
    difs = np.diff(z)
    dz = float(difs[0])
    desviacion = float(np.max(np.abs(difs - dz)))
    if desviacion > tol * max(1.0, abs(dz)):
        raise CondicionNoSatisfecha(
            f"La malla NO es uniforme: el espaciamiento varía hasta {desviacion:.2e} cm "
            f"(Δz nominal = {dz:.6f} cm). Las fórmulas de Newton-Cotes compuestas "
            f"requieren espaciamiento constante. Solución: remuestrea el perfil a una "
            f"malla uniforme por interpolación (spline) antes de integrar, con n "
            f"múltiplo de 12 para que los cuatro métodos sean aplicables."
        )
    return len(z) - 1, dz


print("Infraestructura lista: ResultadoIntegracion, CondicionNoSatisfecha,")
print("                       _cronometrar, _validar_malla")''')

M(r"""**Cómo leer el resultado:** solo confirma que las definiciones se cargaron. La
utilidad de estas piezas se verá en las celdas siguientes.

Dos detalles de diseño que conviene notar porque son decisiones deliberadas:

- **`_cronometrar` devuelve el mínimo de 7 corridas, no el promedio.** El tiempo medido
  es el coste real *más* ruido del sistema operativo, y ese ruido solo puede ser
  positivo. El mínimo es por tanto el estimador menos contaminado del coste verdadero.
  Es exactamente el criterio del módulo `timeit` de la biblioteca estándar.
- **`_validar_malla` rechaza mallas no uniformes.** Podríamos haber dejado que las
  fórmulas se aplicaran igual: no habrían fallado, habrían devuelto un número. Un número
  **incorrecto**, sin ninguna señal de alarma. Fallar ruidosamente ante una precondición
  violada es siempre preferible a devolver silenciosamente basura verosímil —
  especialmente en un sistema de medición, donde nadie va a sospechar de un volumen que
  "se ve razonable".""")
