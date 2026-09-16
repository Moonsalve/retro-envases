# -*- coding: utf-8 -*-
"""Celdas: los cuatro metodos de integracion."""

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

# ----------------------------------------------------------------- TRAPECIO
M(r"""### 5.2 Regla del trapecio

#### La idea

Entre dos nodos consecutivos, sustituimos el integrando por la **recta** que une sus
valores. El área bajo esa recta es la de un trapecio:

$$\int_{z_i}^{z_{i+1}} f(z)\,dz \;\approx\; \frac{\Delta z}{2}\,\bigl[f(z_i) + f(z_{i+1})\bigr]$$

Sumando sobre los $n$ subintervalos, los nodos interiores aparecen **dos veces** (como
extremo derecho de un trapecio y como extremo izquierdo del siguiente), y los extremos
solo una:

$$\boxed{\;I \;\approx\; \frac{\Delta z}{2}\left[f_0 + 2\sum_{i=1}^{n-1} f_i + f_n\right]\;}$$

De ahí el patrón de pesos $[1, 2, 2, \dots, 2, 1]$.

#### Error

$$E \;=\; -\frac{(b-a)\,\Delta z^{2}}{12}\,f''(\xi), \qquad \xi \in (a,b)$$

Es **$O(\Delta z^2)$**: al **duplicar** el número de puntos, el error cae a **la cuarta
parte**. Lo verificaremos numéricamente en la Sección 8.

#### Grado de exactitud: 1

Integra exactamente rectas ($f$ de grado ≤ 1), y nada más. Nuestro integrando es
$f = r^2$, que en el cuerpo cilíndrico es **constante** (grado 0 — exacto ahí) pero en
la base, el hombro y la boca es **cuadrático** (porque $r$ es lineal y lo elevamos al
cuadrado). El trapecio falla en esas tres zonas, y ahí se concentra todo su error.

#### El signo del error: una predicción que vamos a ver fallar

El razonamiento habitual dice: como $f'' > 0$ en los tramos cónicos ($f = r^2$ con $r$
lineal es una parábola convexa), la cuerda queda **por encima** de la curva y el trapecio
**sobreestima**.

Ese razonamiento es correcto **dentro de cada tramo**, y el notebook lo verificará. Pero
el resultado global va a salir **al revés**: el trapecio **subestima**. La razón está en
los codos.

En un punto de quiebre, $f'$ salta de golpe. Eso significa que $f''$ contiene un
**impulso** (una delta de Dirac) justo ahí. En el hombro y en el pie, la pendiente cae
bruscamente — de creciente a constante en $z=1$, de constante a decreciente en $z=15$ —
así que esos impulsos son **negativos**. Y un impulso negativo de $f''$ hace que el
trapecio **subestime** en ese subintervalo.

Con malla ciega, los codos caen en mitad de un subintervalo y esas contribuciones
negativas **superan** a la sobreestimación convexa repartida por los tramos suaves. El
signo global se invierte.

> **La lección:** una cota de error del tipo $E = -\frac{(b-a)h^2}{12}f''(\xi)$ presupone
> que $f''$ **existe y es acotada**. Cuando el integrando tiene esquinas, $f''$ no es una
> función acotada, y la fórmula deja de predecir incluso el *signo*. Lo comprobaremos en
> los dos sentidos: globalmente sale negativo, y tramo a tramo sale positivo.""")

C(r'''def regla_trapecio(z: np.ndarray, r: np.ndarray) -> ResultadoIntegracion:
    """Regla del trapecio compuesta sobre la tabla (z, r).

    Integra f = r² con pesos [1, 2, 2, ..., 2, 1] · Δz/2.

    Condiciones: malla uniforme, n >= 1. No hay restricción de paridad, lo que la
    convierte en el método de respaldo universal cuando los demás no aplican.
    """
    n, dz = _validar_malla(z, r)
    f = np.asarray(r, dtype=float) ** 2          # el integrando: f(z) = r(z)²

    def _calcular() -> float:
        # Pesos explícitos en lugar de np.trapz: el objetivo del laboratorio es que
        # la fórmula sea visible y auditable, no que sea corta.
        pesos = np.full(n + 1, 2.0)
        pesos[0] = pesos[-1] = 1.0
        return float(dz / 2.0 * np.dot(pesos, f))

    integral, ms = _cronometrar(_calcular)
    return ResultadoIntegracion(
        metodo="Trapecio compuesto",
        familia="Newton-Cotes cerrada (grado 1)",
        n_subintervalos=n,
        n_evaluaciones=n + 1,
        a=float(z[0]), b=float(z[-1]),
        delta_z=dz,
        orden_error="O(Δz²)",
        grado_exactitud=1,
        integral_r2=integral,
        volumen_cm3=math.pi * integral,
        tiempo_ms=ms,
        notas="Sin restricción sobre n. Sobreestima si el integrando es convexo.",
    )


res_trap = regla_trapecio(Z_BASE, R_BASE)
print("=" * 74)
print("MÉTODO 1 — REGLA DEL TRAPECIO")
print("=" * 74)
print(res_trap.informe(V_EXACTO))
print("=" * 74)
_signo = "SOBREESTIMA" if res_trap.volumen_cm3 > V_EXACTO else "SUBESTIMA"
print(f"  Diferencia con signo: {res_trap.volumen_cm3 - V_EXACTO:+.6f} cm³  → {_signo}")''')

M(r"""**Cómo leer el resultado:**

- **Error relativo ≈ 1.0 × 10⁻² %** — unos 0.063 cm³ sobre 617. Es **0.2 % del margen
  del 5 %** que exige el reto. **El trapecio, el método más humilde de todos, ya basta
  sobradamente para este problema.** Conviene interiorizarlo desde ya: el cuello de
  botella de este proyecto no está aquí.
- **`Diferencia con signo` NEGATIVA → SUBESTIMA.** Es la predicción "fallida" de la
  sección anterior, y no es casualidad ni ruido: si reejecutas con $n = 12, 24, 120, 240,
  480$ el signo sigue siendo negativo en todos los casos. Los impulsos negativos de
  $f''$ en los codos dominan sistemáticamente sobre la convexidad de los tramos.
  **La cota de error clásica no predice ni siquiera el signo cuando el integrando tiene
  esquinas.** Lo verificaremos en la Sección 8 integrando tramo a tramo: ahí el signo
  sí sale positivo, como manda la teoría.
- **`Evaluaciones = 61`** — el trapecio usa todos los nodos y ninguno extra. Es el
  método más barato en evaluaciones por unidad de dato.
- **`Grado de exactitud = 1`** — en el cuerpo cilíndrico $f = r^2$ es constante (grado 0)
  y el trapecio es **exacto** ahí. Como ese tramo aporta el 82 % del volumen, el trapecio
  acierta de entrada la mayor parte del resultado. Todo su error proviene de los tramos
  cónicos y de los codos.""")

# ----------------------------------------------------------------- SIMPSON 1/3
M(r"""### 5.3 Regla de Simpson 1/3

#### La idea

En lugar de una recta por cada subintervalo, ajustamos una **parábola por cada *par* de
subintervalos**. Tres puntos definen una parábola; integrándola analíticamente sobre los
dos subintervalos que abarca:

$$\int_{z_i}^{z_{i+2}} f\,dz \;\approx\; \frac{\Delta z}{3}\bigl[f_i + 4f_{i+1} + f_{i+2}\bigr]$$

Compuesta sobre todo el intervalo:

$$\boxed{\;I \;\approx\; \frac{\Delta z}{3}\left[f_0 + 4\!\!\sum_{i\ \text{impar}}\!\! f_i \;+\; 2\!\!\sum_{\substack{i\ \text{par}\\ 0<i<n}}\!\! f_i \;+\; f_n\right]\;}$$

Patrón de pesos: $[1, 4, 2, 4, 2, \dots, 4, 1]$. Los índices **impares** son centros de
parábola (peso 4); los **pares interiores** son costuras compartidas entre dos parábolas
(peso 2 = 1+1).

#### La condición: n debe ser par

Cada parábola consume exactamente 2 subintervalos. Si $n$ fuese impar sobraría uno
suelto, sin pareja. **Esta es una de las condiciones cuyo incumplimiento el enunciado
(§8) obliga a reportar con un mensaje claro** — lo probaremos deliberadamente en la
Sección 6.

#### Error — y la sorpresa que lo hace tan bueno

$$E \;=\; -\frac{(b-a)\,\Delta z^{4}}{180}\,f^{(4)}(\xi)$$

**$O(\Delta z^4)$**: al duplicar los puntos, el error cae a **1/16**.

Lo notable es el **grado de exactitud: 3, no 2**. Construimos la fórmula con parábolas
(grado 2), pero resulta exacta también para **cúbicas**. La razón es una cancelación por
simetría: el error de la parábola sobre la mitad izquierda del intervalo es igual y de
signo opuesto al de la mitad derecha, y se anulan al sumar. Es un regalo geométrico: se
obtiene un grado gratis. Por eso Simpson 1/3 tiene una relación
**precisión/coste extraordinaria** y es la regla más usada en la práctica.

#### Implicación directa para nuestro envase — y una predicción incómoda

$f = r^2$ es a lo sumo **cuadrático por tramos** (grado 2 ≤ 3). Por tanto, **dentro de
cada tramo, Simpson es exacto salvo redondeo**. Todo el error que observemos vendrá
únicamente de las parábolas que queden **a caballo sobre un punto de quiebre**, donde el
integrando cambia de fórmula a mitad de camino.

Aquí conviene anticipar algo que sorprende: como cada parábola de Simpson abarca **dos**
subintervalos, mientras que cada trapecio abarca **uno**, las parábolas **cruzan codos
con el doble de frecuencia** que los trapecios. Y cuando una parábola se ajusta a tres
puntos que están sobre dos curvas distintas, el resultado no se parece a ninguna de las
dos.

Es perfectamente posible, entonces, que **Simpson salga peor que el trapecio** sobre este
envase. Suena a contradicción con todo lo que promete su cota $O(\Delta z^4)$, pero esa
cota exige $f^{(4)}$ acotada — y en un codo ni siquiera existe $f''$. Vamos a verlo.""")

C(r'''def regla_simpson_13(z: np.ndarray, r: np.ndarray) -> ResultadoIntegracion:
    """Regla de Simpson 1/3 compuesta.

    Pesos [1,4,2,4,...,4,1] · Δz/3.
    Condición: n PAR (cada parábola abarca 2 subintervalos).
    """
    n, dz = _validar_malla(z, r)
    if n % 2 != 0:
        raise CondicionNoSatisfecha(
            f"Simpson 1/3 requiere un número PAR de subintervalos y se recibió n = {n}.\n"
            f"  Motivo: cada parábola se ajusta sobre 2 subintervalos consecutivos; "
            f"con n impar queda uno sin pareja.\n"
            f"  Soluciones: usar n = {n - 1} o n = {n + 1} (remuestreando los datos), "
            f"o aplicar Simpson 3/8 al último tramo."
        )
    f = np.asarray(r, dtype=float) ** 2

    def _calcular() -> float:
        pesos = np.ones(n + 1)
        pesos[1:-1:2] = 4.0     # índices impares → centros de parábola
        pesos[2:-1:2] = 2.0     # índices pares interiores → costuras
        return float(dz / 3.0 * np.dot(pesos, f))

    integral, ms = _cronometrar(_calcular)
    return ResultadoIntegracion(
        metodo="Simpson 1/3 compuesto",
        familia="Newton-Cotes cerrada (grado 2)",
        n_subintervalos=n,
        n_evaluaciones=n + 1,
        a=float(z[0]), b=float(z[-1]),
        delta_z=dz,
        orden_error="O(Δz⁴)",
        grado_exactitud=3,
        integral_r2=integral,
        volumen_cm3=math.pi * integral,
        tiempo_ms=ms,
        notas="Requiere n par. Exacto hasta cúbicas por cancelación simétrica del error.",
    )


res_s13 = regla_simpson_13(Z_BASE, R_BASE)
print("=" * 74)
print("MÉTODO 2 — REGLA DE SIMPSON 1/3")
print("=" * 74)
print(res_s13.informe(V_EXACTO))
print("=" * 74)
_e_trap = res_trap.error_vs(V_EXACTO)[0]
_e_s13 = res_s13.error_vs(V_EXACTO)[0]
_cociente = _e_s13 / max(_e_trap, 1e-300)
print(f"  Error del trapecio    : {_e_trap:.6e} cm³   (orden teórico O(Δz²))")
print(f"  Error de Simpson 1/3  : {_e_s13:.6e} cm³   (orden teórico O(Δz⁴))")
if _cociente > 1:
    print(f"\n  ⚠ Simpson 1/3 es {_cociente:.2f}× PEOR que el trapecio, pese a ser de")
    print(f"    orden teórico muy superior. Causa: los puntos de quiebre del envase.")
    print(f"    Cada parábola abarca 2 subintervalos y cruza codos el doble de veces")
    print(f"    que un trapecio. Ver Sección 8 para la demostración cuantitativa.")
else:
    print(f"\n  Simpson 1/3 es {1/_cociente:.2f}× más preciso que el trapecio.")''')

M(r"""**Cómo leer el resultado — aquí está la primera sorpresa del notebook.**

- **Simpson 1/3 sale ≈ 4.2 veces PEOR que el trapecio** (≈ 0.263 cm³ frente a
  ≈ 0.063 cm³), con **exactamente los mismos 61 puntos**. Un método de orden teórico
  $O(\Delta z^4)$ derrotado por uno de orden $O(\Delta z^2)$.
- **Esto no es un error de implementación.** La verificación independiente de las
  fórmulas confirma que Simpson integra exactamente polinomios hasta grado 3 y falla en
  grado 4, justo como debe. El código está bien; **lo que no se cumple son las hipótesis
  de la cota de error**.
- **El mecanismo, en una frase:** cada parábola de Simpson abarca 2 subintervalos, así
  que cruza los codos del envase con el doble de frecuencia que un trapecio, y ajustar
  una parábola a tres puntos repartidos entre dos curvas distintas produce un error
  mayor que el de dos segmentos rectos.
- **El error no es cero pese a ser $f$ cuadrático por tramos**, exactamente por lo mismo:
  con $n = 60$ los quiebres ($z = 1.0,\ 15.0,\ 18.5,\ 20.0$) caen *dentro* de parábolas
  en vez de sobre nodos. En la Sección 8 alinearemos la malla y el error de Simpson caerá
  al **suelo de redondeo** — confirmando que la teoría es correcta y que el problema era
  dónde pusimos los nodos.
- **`Grado de exactitud = 3`** — el grado "regalado" por la cancelación simétrica.
- **El tiempo es prácticamente idéntico** al del trapecio: ambos son un producto escalar
  de 61 términos. El coste no distingue a los métodos; la estructura de los datos, sí.

> **Conclusión provisional, que la Sección 8 confirmará:** sobre datos con esquinas —y
> una foto de un envase real tiene esquinas *y* ruido— **el orden teórico de un método
> no predice su rendimiento**. Elegir la malla bien importa más que elegir la fórmula
> bien.""")

# ----------------------------------------------------------------- SIMPSON 3/8
M(r"""### 5.4 Regla de Simpson 3/8 — Newton-Cotes cerrada de grado 3

Esta es la fórmula que el PDF (§6) pide explícitamente: *"una fórmula cerrada de
Newton-Cotes, como 3/8 o una de orden superior"*.

#### La idea

Subimos un grado: ajustamos una **cúbica** por cada **tres** subintervalos (4 puntos
determinan una cúbica). Integrándola:

$$\int_{z_i}^{z_{i+3}} f\,dz \;\approx\; \frac{3\,\Delta z}{8}\bigl[f_i + 3f_{i+1} + 3f_{i+2} + f_{i+3}\bigr]$$

El nombre viene del coeficiente $3/8$. Patrón compuesto:
$[1, 3, 3, 2, 3, 3, 2, \dots, 3, 3, 1]$ — los pesos 2 son las costuras entre cúbicas
consecutivas (1+1).

#### Condición: n múltiplo de 3

#### La lección contraintuitiva: más grado NO significa más orden

$$E \;=\; -\frac{(b-a)\,\Delta z^{4}}{80}\,f^{(4)}(\xi)$$

Compárala con la de Simpson 1/3, que tiene $180$ en el denominador:

| | Grado del polinomio | Grado de exactitud | Orden del error | Constante |
|---|---|---|---|---|
| Simpson 1/3 | 2 (parábola) | 3 | $O(\Delta z^4)$ | $1/180$ |
| Simpson 3/8 | 3 (cúbica) | 3 | $O(\Delta z^4)$ | $1/80$ |

**Simpson 3/8 usa polinomios de mayor grado, consume más puntos por segmento, y tiene el
mismo orden y una constante de error 2.25 veces PEOR.**

¿Por qué? Porque la cancelación simétrica que le regalaba un grado extra a Simpson 1/3
**solo ocurre en las reglas de Newton-Cotes de grado par**. Simpson 1/3 (grado 2, par)
se beneficia; Simpson 3/8 (grado 3, impar) no. Es un patrón general:

> En Newton-Cotes cerradas, las reglas de **grado par** siempre ganan un grado de
> exactitud extra. Por eso 1/3 domina a 3/8, y por eso Boole (grado 4, par) es el
> siguiente salto real de orden.

#### Entonces, ¿para qué sirve 3/8?

Su utilidad real es de **compatibilidad**: cuando tienes un $n$ impar, Simpson 1/3 no
aplica. La solución estándar es aplicar 3/8 a los **tres últimos** subintervalos y 1/3 al
resto. Dado que $n$ impar $-3$ = par, el reparto siempre funciona. Es la herramienta que
te salva cuando el número de puntos te viene impuesto — que es exactamente lo que pasa
cuando los datos vienen de una foto y no puedes elegir cuántas filas de píxeles tiene.

Nuestra implementación hace justo eso: **compone automáticamente** ambas reglas si $n$
no es múltiplo de 3, en vez de rendirse.""")

C(r'''def regla_simpson_38(z: np.ndarray, r: np.ndarray,
                     permitir_composicion: bool = True) -> ResultadoIntegracion:
    """Regla de Simpson 3/8 compuesta (Newton-Cotes cerrada de grado 3).

    Pesos [1,3,3,2,3,3,2,...,3,3,1] · 3Δz/8.
    Condición nativa: n múltiplo de 3.

    Si n NO es múltiplo de 3 y permitir_composicion=True, se aplica 3/8 al mayor
    bloque múltiplo de 3 posible y Simpson 1/3 al resto (que queda par por
    construcción). Es la práctica estándar y evita rechazar datos utilizables.
    """
    n, dz = _validar_malla(z, r)
    f = np.asarray(r, dtype=float) ** 2

    resto = n % 3
    if resto != 0 and not permitir_composicion:
        raise CondicionNoSatisfecha(
            f"Simpson 3/8 requiere n múltiplo de 3 y se recibió n = {n} "
            f"(n % 3 = {resto}).\n"
            f"  Motivo: cada cúbica se ajusta sobre 3 subintervalos consecutivos.\n"
            f"  Soluciones: usar n = {n - resto} o n = {n + (3 - resto)}, o activar "
            f"permitir_composicion=True para combinar 3/8 con Simpson 1/3."
        )

    def _pesos_38(m: int) -> np.ndarray:
        """Pesos de 3/8 compuesta sobre m subintervalos (m múltiplo de 3)."""
        w = np.zeros(m + 1)
        for i in range(0, m, 3):        # bloques de 3 subintervalos
            w[i]     += 1.0
            w[i + 1] += 3.0
            w[i + 2] += 3.0
            w[i + 3] += 1.0
        return w * (3.0 / 8.0)

    def _pesos_13(m: int) -> np.ndarray:
        """Pesos de Simpson 1/3 compuesta sobre m subintervalos (m par)."""
        w = np.ones(m + 1)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * (1.0 / 3.0)

    if resto == 0:
        nota = "n múltiplo de 3: regla 3/8 pura."
        def _calcular() -> float:
            return float(dz * np.dot(_pesos_38(n), f))
    else:
        # El bloque de 3/8 debe ser múltiplo de 3; el sobrante va a Simpson 1/3
        # y por tanto debe quedar par.
        n_38 = (n // 3) * 3
        n_13 = n - n_38
        if n_13 % 2 != 0:                 # el sobrante debe ser par para 1/3
            n_38 -= 3
            n_13 = n - n_38
        nota = (f"n = {n} no es múltiplo de 3: composición automática → "
                f"3/8 sobre los primeros {n_38} subintervalos + "
                f"Simpson 1/3 sobre los últimos {n_13}.")
        def _calcular() -> float:
            total = 0.0
            if n_38 > 0:
                total += float(dz * np.dot(_pesos_38(n_38), f[: n_38 + 1]))
            if n_13 > 0:
                total += float(dz * np.dot(_pesos_13(n_13), f[n_38:]))
            return total

    integral, ms = _cronometrar(_calcular)
    return ResultadoIntegracion(
        metodo="Simpson 3/8 compuesto",
        familia="Newton-Cotes cerrada (grado 3)",
        n_subintervalos=n,
        n_evaluaciones=n + 1,
        a=float(z[0]), b=float(z[-1]),
        delta_z=dz,
        orden_error="O(Δz⁴)",
        grado_exactitud=3,
        integral_r2=integral,
        volumen_cm3=math.pi * integral,
        tiempo_ms=ms,
        notas=nota,
    )


res_s38 = regla_simpson_38(Z_BASE, R_BASE)
print("=" * 74)
print("MÉTODO 3 — REGLA DE SIMPSON 3/8  (Newton-Cotes cerrada)")
print("=" * 74)
print(res_s38.informe(V_EXACTO))
print("=" * 74)
_e_s38 = res_s38.error_vs(V_EXACTO)[0]
print(f"  Error de Simpson 1/3 : {_e_s13:.6e} cm³")
print(f"  Error de Simpson 3/8 : {_e_s38:.6e} cm³")
print(f"  Cociente 3/8 ÷ 1/3   : {_e_s38 / max(_e_s13, 1e-300):.2f}×"
      f"   (teoría para f suave: 180/80 = 2.25×)")''')

M(r"""**Cómo leer el resultado — la segunda sorpresa, y va en dirección contraria.**

- **La teoría dice que 3/8 debería ser 2.25× PEOR que 1/3** (constantes $1/80$ frente a
  $1/180$). **El cómputo da lo opuesto: 3/8 sale ≈ 3.7 veces MEJOR** (cociente ≈ 0.27).
  Con $n = 60$, Simpson 3/8 acumula ~0.071 cm³ de error frente a los ~0.263 cm³ de
  Simpson 1/3.
- **Por qué.** La comparación teórica $180$ vs $80$ solo vale si $f^{(4)}$ existe y es
  acotada. En nuestro envase no lo es. Lo que realmente decide el resultado es **cuántos
  bloques de la regla caen a caballo de un codo y cómo**, y eso depende de la aritmética
  concreta entre $n$, la posición de los quiebres y el ancho del bloque de cada regla —
  no de la constante de la cota.
- **La conclusión metodológica, que es lo que hay que llevarse:** cuando el error
  observado contradice la teoría, **lo que falla no es la teoría — es que sus hipótesis
  no se cumplen**. Distinguir "mi código tiene un bug" de "estoy aplicando una fórmula
  fuera de su dominio de validez" es una de las habilidades centrales del análisis
  numérico. Aquí es lo segundo, y lo demostramos: con malla alineada (Sección 8) ambas
  reglas caen al suelo de redondeo, tal como predice la teoría.
- **⚠️ No generalices este resultado.** Que 3/8 gane *aquí* con *este* $n$ es un accidente
  de alineación, no una propiedad del método. Con otro $n$ el orden puede invertirse otra
  vez. Es precisamente la marca de un régimen donde **el orden teórico ya no gobierna**:
  los resultados se vuelven erráticos e impredecibles. Lo verás con claridad en la
  gráfica de convergencia de la Sección 8, donde estas curvas suben y bajan en vez de
  descender limpiamente.
- **`Notas`** indica si se usó 3/8 puro o composición con 1/3. Con $n = 60$ (múltiplo de
  3) será pura.
- **Este método satisface el requisito explícito de la §6 del PDF** de implementar una
  Newton-Cotes cerrada tipo 3/8. El siguiente, Boole, cubre además la opción de *"orden
  superior"* mencionada en la misma línea.""")
