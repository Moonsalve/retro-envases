# -*- coding: utf-8 -*-
"""Celdas: validacion de condiciones, tabla comparativa, convergencia."""

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

M(r"""---
## 6. Validación de condiciones y mensajes claros — §8 del PDF

El enunciado exige literalmente:

> *"Mensajes claros cuando los datos no cumplen las condiciones requeridas por un
> método."*

**Qué hace esta celda:** provoca **a propósito** los tres fallos posibles y muestra el
mensaje que produce cada uno.

**Por qué probar los fallos es tan importante como probar los aciertos.** El día de la
prueba de campo, el número de puntos vendrá determinado por la altura en píxeles del
envase en la foto — un número arbitrario que nadie controla. Si sale $n = 437$, Simpson,
3/8 y Boole fallan los tres. Lo que decide si la aplicación sirve o no en ese momento no
es que calcule bien cuando todo va bien; es **qué hace cuando algo no encaja**.

Un buen mensaje de error tiene tres partes, y las tres son obligatorias:

1. **Qué falló** — el dato concreto y el requisito concreto.
2. **Por qué** — la razón matemática, no solo la regla.
3. **Cómo arreglarlo** — el valor válido más cercano, o la alternativa.

Un mensaje que solo dice *"Error: n inválido"* obliga al usuario a leer el código
fuente. Eso no es un mensaje de error, es una rendición.

Los tres escenarios que provocamos:

| # | Escenario | Método que debe rechazar |
|---|---|---|
| A | $n$ impar | Simpson 1/3 |
| B | $n$ no múltiplo de 4 | Boole |
| C | malla no uniforme | todas las Newton-Cotes |

El escenario C es el más relevante para la fase de cámara: si el contorno se extrae con
huecos — por un reflejo, una etiqueta o una sombra que rompen la segmentación — las
alturas dejan de estar equiespaciadas. **El sistema debe detectarlo y decirlo, nunca
integrar igual y devolver un número con aspecto razonable.**""")

C(r'''print("=" * 74)
print("PRUEBA DELIBERADA DE CONDICIONES NO SATISFECHAS")
print("=" * 74)

# --- Escenario A: n impar -> Simpson 1/3 debe rechazar -----------------
print("\n▸ ESCENARIO A — n impar (n = 61)")
print("-" * 74)
z_impar, r_impar = muestrear_perfil(ENVASE, 61)
try:
    regla_simpson_13(z_impar, r_impar)
    print("  ✗ FALLO DE LA PRUEBA: debió lanzar excepción y no lo hizo.")
except CondicionNoSatisfecha as e:
    print(f"  ✓ Rechazado correctamente:\n\n{e}")

# El trapecio, en cambio, acepta cualquier n.
res_impar = regla_trapecio(z_impar, r_impar)
print(f"\n  → El trapecio SÍ acepta n = 61: V = {res_impar.volumen_cm3:.6f} cm³")
print(f"    (el trapecio es el respaldo universal cuando n viene impuesto)")

# --- Escenario B: n no múltiplo de 4 -> Boole debe rechazar ------------
print("\n\n▸ ESCENARIO B — n = 30, no es múltiplo de 4")
print("-" * 74)
z_30, r_30 = muestrear_perfil(ENVASE, 30)
try:
    regla_boole(z_30, r_30)
    print("  ✗ FALLO DE LA PRUEBA: debió lanzar excepción.")
except CondicionNoSatisfecha as e:
    print(f"  ✓ Rechazado correctamente:\n\n{e}")

print(f"\n  → Simpson 1/3 sí acepta n = 30 (es par): "
      f"V = {regla_simpson_13(z_30, r_30).volumen_cm3:.6f} cm³")

# --- Escenario C: malla no uniforme -----------------------------------
print("\n\n▸ ESCENARIO C — malla NO uniforme (contorno con huecos)")
print("-" * 74)
# Simulamos la pérdida de tres filas del contorno, como ocurriría si un reflejo
# o una etiqueta rompieran la segmentación en esa franja del envase.
z_hueco = np.delete(Z_BASE, [20, 21, 22])
r_hueco = np.delete(R_BASE, [20, 21, 22])
print(f"  Puntos: {len(Z_BASE)} → {len(z_hueco)} (se perdieron 3 filas del contorno)")
print(f"  Espaciamientos únicos presentes: {np.unique(np.round(np.diff(z_hueco), 6))} cm")
try:
    regla_trapecio(z_hueco, r_hueco)
    print("  ✗ FALLO DE LA PRUEBA: debió lanzar excepción.")
except CondicionNoSatisfecha as e:
    print(f"  ✓ Rechazado correctamente:\n\n{e}")

print("\n" + "=" * 74)
print("Las tres condiciones se detectan y se comunican con un mensaje accionable.")
print("=" * 74)''')

M(r"""**Cómo leer el resultado:**

Las tres marcas ✓ confirman que el sistema **no calcula cuando no debe**. Fíjate en que
cada mensaje nombra el valor recibido, explica la razón geométrica y propone
alternativas concretas.

Observa también los dos **caminos de recuperación** que quedan demostrados:

- Con $n = 61$ (impar), el **trapecio sigue funcionando**. Es el respaldo universal: sin
  restricciones sobre $n$. En la aplicación final, si ningún método de orden superior
  aplica, el trapecio siempre da una respuesta válida.
- Con $n = 30$, Boole falla pero **Simpson 1/3 funciona**. La estrategia correcta no es
  "elegir un método" sino **probar los cuatro y reportar los que apliquen**, avisando de
  los que no. Eso es justo lo que hace la tabla de la sección siguiente.

El escenario C merece atención especial para la fase de cámara: la malla no uniforme
por pérdida de filas del contorno es un fallo **probable**, no hipotético. La solución
allí será **remuestrear por interpolación** a una malla uniforme con $n$ múltiplo de 12
antes de integrar — y declararlo explícitamente como parte del procedimiento, porque
introduce su propio error.""")

M(r"""---
## 7. Tabla comparativa de los cuatro métodos — §7 Etapa 5 y §8

**Qué hace esta celda:** ejecuta todos los métodos sobre **la misma tabla de datos** y
consolida los resultados.

La §7 Etapa 5 obliga a reportar, para cada método: número de subintervalos o puntos,
intervalo $[a,b]$, $\Delta z$, orden o número de nodos, integral estimada, volumen
estimado y tiempo de cálculo. La §7 Etapa 6 añade diferencia absoluta y error relativo
porcentual. **Todo eso está en las columnas de abajo.**

`ejecutar_todos` captura `CondicionNoSatisfecha` por método: si uno no aplica, se
registra con su motivo y **los demás siguen ejecutándose**. Un método incompatible no
debe tumbar el análisis completo.""")

C(r'''def ejecutar_todos(perfil: PerfilEnvase, n: int, m_gauss: int = 5,
                   verbose: bool = True) -> tuple[list[ResultadoIntegracion], list[tuple[str, str]]]:
    """Aplica todos los métodos a la misma tabla y recolecta resultados y fallos.

    Devuelve (resultados_exitosos, [(metodo, motivo_del_fallo), ...]).
    """
    z, r = muestrear_perfil(perfil, n)
    metodos: list[tuple[str, Callable[[], ResultadoIntegracion]]] = [
        ("Trapecio",            lambda: regla_trapecio(z, r)),
        ("Simpson 1/3",         lambda: regla_simpson_13(z, r)),
        ("Simpson 3/8",         lambda: regla_simpson_38(z, r)),
        ("Boole",               lambda: regla_boole(z, r)),
        ("Gauss (analítico)",   lambda: gauss_desde_funcion(perfil, m_gauss)),
        ("Gauss (spline)",      lambda: gauss_desde_datos(z, r, m_gauss)),
    ]
    exitosos: list[ResultadoIntegracion] = []
    fallidos: list[tuple[str, str]] = []
    for nombre, fn in metodos:
        try:
            exitosos.append(fn())
        except CondicionNoSatisfecha as e:
            fallidos.append((nombre, str(e).split("\n")[0]))
            if verbose:
                print(f"  ⚠ {nombre}: no aplicable — {str(e).splitlines()[0]}")
    return exitosos, fallidos


def tabla_comparativa(resultados: Sequence[ResultadoIntegracion],
                      v_referencia: float) -> pd.DataFrame:
    """Consolida los resultados con todos los campos que exige el enunciado."""
    filas = []
    for res in resultados:
        e_abs, e_rel = res.error_vs(v_referencia)
        filas.append({
            "Método": res.metodo,
            "Familia": res.familia,
            "n": res.n_subintervalos if res.n_subintervalos else "—",
            "Evals": res.n_evaluaciones,
            "a [cm]": res.a,
            "b [cm]": res.b,
            "Δz [cm]": res.delta_z if res.delta_z is not None else np.nan,
            "Orden / exactitud": res.orden_error,
            "I = ∫r²dz [cm³]": res.integral_r2,
            "V = πI [cm³]": res.volumen_cm3,
            "V [mL]": res.volumen_ml,
            "Err. abs. [cm³]": e_abs,
            "Err. rel. [%]": e_rel,
            "Tiempo [ms]": res.tiempo_ms,
        })
    return pd.DataFrame(filas)


print("=" * 74)
print(f"COMPARACIÓN DE MÉTODOS — n = {N_BASE}, Δz = {DZ_BASE:.6f} cm, "
      f"Gauss con {M_GAUSS} nodos")
print("=" * 74)
RESULTADOS, FALLIDOS = ejecutar_todos(ENVASE, N_BASE, M_GAUSS)
TABLA_COMP = tabla_comparativa(RESULTADOS, V_EXACTO)

print(f"\n  Valor de referencia (exacto): V = {V_EXACTO:.10f} cm³")
print(f"  Métodos aplicables: {len(RESULTADOS)}   |   No aplicables: {len(FALLIDOS)}\n")

display(TABLA_COMP[["Método", "n", "Evals", "Δz [cm]", "Orden / exactitud",
                    "I = ∫r²dz [cm³]", "V = πI [cm³]"]])
print()
display(TABLA_COMP[["Método", "V [mL]", "Err. abs. [cm³]", "Err. rel. [%]", "Tiempo [ms]"]]
        .sort_values("Err. abs. [cm³]")
        .reset_index(drop=True))

_mejor = TABLA_COMP.loc[TABLA_COMP["Err. abs. [cm³]"].idxmin()]
_peor = TABLA_COMP.loc[TABLA_COMP["Err. abs. [cm³]"].idxmax()]
print("\n" + "=" * 74)
print(f"  Más preciso : {_mejor['Método']}  →  {_mejor['Err. rel. [%]']:.4e} %")
print(f"  Menos preciso: {_peor['Método']}  →  {_peor['Err. rel. [%]']:.4e} %")
print(f"  Rango de volúmenes: [{TABLA_COMP['V [mL]'].min():.4f}, "
      f"{TABLA_COMP['V [mL]'].max():.4f}] mL  "
      f"(dispersión: {TABLA_COMP['V [mL]'].max() - TABLA_COMP['V [mL]'].min():.4f} mL)")
print(f"  Criterio del reto: 5 % de {V_EXACTO:.2f} cm³ = "
      f"±{0.05 * V_EXACTO:.2f} cm³")
print("=" * 74)''')

M(r"""**Cómo leer el resultado — esta es la tabla que va al informe.**

- **La primera tabla** documenta los *parámetros* de cada método: $n$, evaluaciones,
  $\Delta z$, orden y el par (integral, volumen). Es el registro auditable que pide la
  §7 Etapa 5. Nota que Gauss muestra `Δz = NaN`: **no tiene un Δz**, porque sus nodos no
  son equiespaciados. Es una propiedad del método, no un dato faltante, y conviene
  señalarlo así en el informe en lugar de inventar un valor.
- **La segunda tabla** está ordenada por error creciente: el mejor método arriba.
- **El ranking real, de mejor a peor, invierte la jerarquía teórica:**

  | Puesto | Método | Orden teórico | Error [cm³] | % del margen del 5 % |
  |---|---|---|---|---|
  | 1 | Trapecio | $O(\Delta z^2)$ | ≈ 0.063 | 0.21 % |
  | 2 | Simpson 3/8 | $O(\Delta z^4)$ | ≈ 0.071 | 0.23 % |
  | 3 | Simpson 1/3 | $O(\Delta z^4)$ | ≈ 0.263 | 0.85 % |
  | 4 | Boole | $O(\Delta z^6)$ | ≈ 0.301 | 0.97 % |
  | 5 | Gauss 5 nodos | grado 9 | ≈ 3.91 | **12.69 %** |

- **Los cuatro Newton-Cotes consumen menos del 1 % del margen del reto.** Para ellos, la
  elección del método es efectivamente irrelevante: la diferencia entre el mejor y el
  peor es de 0.24 cm³, sobre un margen permitido de 30.8 cm³.
- **Gauss con 5 nodos es la excepción y merece atención:** consume el **12.7 %** del
  margen él solo. No es despreciable, y con menos nodos ni siquiera pasaría el criterio
  (lo verás en la celda 5.7). **Es el único método del notebook cuya elección sí podría
  costarte la prueba de campo.**

> **Conclusión de ingeniería.** Cualquiera de los cuatro Newton-Cotes cumple el criterio
> del 5 % con enorme holgura, y entre ellos la elección es indiferente. Lo que decidirá
> el éxito o el fracaso en la prueba de campo es la **calidad del perfil $(z_i, r_i)$** —
> la calibración de la escala píxel→cm y la extracción del contorno — no la fórmula de
> cuadratura.
>
> Con una excepción que sí es una decisión de diseño: **no usar Gauss de bajo orden como
> método principal**. Se reporta porque el enunciado lo exige, pero el valor que se
> entrega debe venir de una regla compuesta sobre los datos crudos.
>
> Esto no invalida el laboratorio: lo *cuantifica*. Saber dónde **no** está el cuello de
> botella es tan valioso como saber dónde sí está, porque dice dónde invertir el esfuerzo
> restante del proyecto.""")

M(r"""---
## 8. Estudio de convergencia: el efecto de Δz y del número de puntos — §6

La §6 del PDF pide analizar, para el trapecio, *"efecto de Δz y del número de puntos"*, y
para Simpson, *"condiciones sobre n y comportamiento del perfil"*. Esta sección lo hace
de forma cuantitativa.

### Qué vamos a medir

Ejecutamos todos los métodos para muchos valores de $n$ y graficamos el error frente a
$\Delta z$ en **escala log-log**. Esa elección de ejes no es estética: si
$E \approx C\,\Delta z^{p}$, tomando logaritmos

$$\log E \;=\; \log C \;+\; p\,\log \Delta z$$

la relación se vuelve una **recta cuya pendiente es exactamente el orden $p$**. Medir la
pendiente por regresión lineal es, por tanto, **medir el orden de convergencia
empíricamente**. Podemos entonces contrastarlo con el orden teórico y ver si la teoría se
cumple.

### Las dos mallas que vamos a comparar — y por qué

Aquí está el experimento central del notebook. Ejecutamos el barrido dos veces:

- **Malla ciega** — $n$ múltiplos de 12, sin más. Los nodos caen donde caigan; los
  puntos de quiebre del envase quedan generalmente **entre** nodos.
- **Malla alineada** — los quiebres caen **exactamente sobre nodos**. Se logra integrando
  cada tramo por separado, con su propia malla, y sumando. Así ningún bloque de Simpson
  o Boole cruza jamás una esquina, y **dentro de cada tramo el integrando es un polinomio
  de grado 2, perfectamente suave**.

**La predicción:** con malla ciega, los métodos de orden alto quedarán **muy por debajo
de su orden teórico**, porque las hipótesis de suavidad se violan en las esquinas. Con
malla alineada, deberían alcanzar su orden teórico — o incluso el error de redondeo
directamente, ya que Simpson y Boole son *exactos* para cuadráticas.

Si esta predicción se cumple, queda demostrado algo aplicable a cualquier problema real:

> **No basta con elegir un método de orden alto. Hay que darle una malla que respete la
> estructura del problema.** Alinear la malla con las singularidades del integrando es
> una técnica mucho más rentable que subir el orden de la regla.

⏱️ Esta celda tarda unos segundos: ejecuta ~200 integraciones.""")

C(r'''def integrar_por_tramos(perfil: PerfilEnvase, metodo: Callable, n_por_tramo: int,
                        **kwargs) -> float:
    """Integra cada tramo con su propia malla y suma. MALLA ALINEADA.

    Al usar los límites de cada tramo como extremos de su malla, los puntos de
    quiebre caen SIEMPRE sobre nodos y ningún bloque de la regla los cruza. Dentro
    de cada tramo el integrando es un polinomio de grado 2 (r lineal, elevado al
    cuadrado), y por tanto suave.

    Es la estrategia estándar frente a integrandos definidos a trozos, y la misma
    idea que usa scipy.integrate.quad con su argumento `points`.
    """
    total = 0.0
    for tramo in perfil.tramos:
        z_t = np.linspace(tramo.z_ini, tramo.z_fin, n_por_tramo + 1)
        r_t = tramo.radio(z_t)
        total += metodo(z_t, r_t, **kwargs).integral_r2
    return total


# ----------------------------------------------------------------------
#  BARRIDO A — malla ciega (los quiebres caen entre nodos)
# ----------------------------------------------------------------------
NS = np.array([12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])
registros_ciega: list[dict] = []

for n in NS:
    z_n, r_n = muestrear_perfil(ENVASE, int(n))
    dz_n = (ENVASE.b - ENVASE.a) / n
    for nombre, fn in [("Trapecio", regla_trapecio),
                       ("Simpson 1/3", regla_simpson_13),
                       ("Simpson 3/8", regla_simpson_38),
                       ("Boole", regla_boole)]:
        try:
            res = fn(z_n, r_n)
            registros_ciega.append({
                "Método": nombre, "n": int(n), "Δz": dz_n,
                "V": res.volumen_cm3,
                "Error": abs(res.volumen_cm3 - V_EXACTO),
                "Err rel [%]": 100 * abs(res.volumen_cm3 - V_EXACTO) / V_EXACTO,
            })
        except CondicionNoSatisfecha:
            pass

CONV_CIEGA = pd.DataFrame(registros_ciega)

# ----------------------------------------------------------------------
#  BARRIDO B — malla alineada con los puntos de quiebre
# ----------------------------------------------------------------------
NS_TRAMO = np.array([12, 24, 48, 96, 192, 384, 768])
registros_alin: list[dict] = []

for n_t in NS_TRAMO:
    # Δz efectivo medio: altura total dividida entre el total de subintervalos
    n_total = int(n_t) * len(ENVASE.tramos)
    dz_eff = ENVASE.altura / n_total
    for nombre, fn in [("Trapecio", regla_trapecio),
                       ("Simpson 1/3", regla_simpson_13),
                       ("Simpson 3/8", regla_simpson_38),
                       ("Boole", regla_boole)]:
        try:
            I = integrar_por_tramos(ENVASE, fn, int(n_t))
            V = math.pi * I
            registros_alin.append({
                "Método": nombre, "n_por_tramo": int(n_t), "n_total": n_total,
                "Δz": dz_eff, "V": V,
                "Error": max(abs(V - V_EXACTO), 1e-16),   # piso: evitar log(0) al graficar
                "Err rel [%]": 100 * abs(V - V_EXACTO) / V_EXACTO,
            })
        except CondicionNoSatisfecha:
            pass

CONV_ALIN = pd.DataFrame(registros_alin)


# Umbral por debajo del cual un "error" ya no mide el método, sino el redondeo.
# Con V ≈ 617 y eps ≈ 2.2e-16, un solo redondeo vale ~1.4e-13; sumando miles de
# términos el ruido acumulado llega holgadamente a ~1e-11. Ajustar la regresión
# sobre esos puntos daría pendientes sin significado (incluso negativas).
UMBRAL_REDONDEO = 1e-10


def orden_observado(df: pd.DataFrame, metodo: str) -> float:
    """Pendiente de log(Error) vs log(Δz) = orden de convergencia empírico.

    Devuelve NaN si no quedan al menos 2 puntos por encima del suelo de redondeo:
    en ese caso el método fue exacto salvo aritmética finita y no hay ley de
    potencia que medir. NaN aquí significa "demasiado bueno para medirlo", no
    "falló".
    """
    sub = df[(df["Método"] == metodo) & (df["Error"] > UMBRAL_REDONDEO)]
    if len(sub) < 2:
        return float("nan")
    return float(np.polyfit(np.log(sub["Δz"]), np.log(sub["Error"]), 1)[0])


TEORICO = {"Trapecio": 2, "Simpson 1/3": 4, "Simpson 3/8": 4, "Boole": 6}

print("=" * 78)
print("ORDEN DE CONVERGENCIA OBSERVADO  (pendiente de log E vs log Δz)")
print("=" * 78)
print(f"{'Método':<16}{'Teórico':>9}{'Malla ciega':>15}{'Malla alineada':>17}   Diagnóstico")
print("-" * 78)
for met, p_teo in TEORICO.items():
    p_ciega = orden_observado(CONV_CIEGA, met)
    p_alin = orden_observado(CONV_ALIN, met)
    if math.isnan(p_alin):
        diag = "EXACTO (error en el suelo de redondeo)"
    elif abs(p_alin - p_teo) < 0.6:
        diag = "teoría confirmada"
    else:
        diag = "limitado por otra causa"
    print(f"{met:<16}{p_teo:>9}{p_ciega:>15.2f}{p_alin:>17.2f}   {diag}")
print("=" * 78)''')

M(r"""**Cómo leer el resultado — la tabla más importante del notebook.**

Compara las tres columnas numéricas para cada método:

- **Columna `Teórico`** — el orden que promete la cota de error de la fórmula.
- **Columna `Malla ciega`** — el orden realmente observado cuando los quiebres del
  envase caen entre nodos.
- **Columna `Malla alineada`** — el orden observado cuando los quiebres caen sobre nodos.

**Qué deberías ver — y es un resultado de libro de texto:**

| Método | Teórico | Malla ciega | Malla alineada |
|---|---|---|---|
| Trapecio | 2 | ≈ **1.97** | ≈ **2.00** |
| Simpson 1/3 | 4 | ≈ **1.98** | EXACTO (suelo) |
| Simpson 3/8 | 4 | ≈ **2.06** | EXACTO (suelo) |
| Boole | 6 | ≈ **1.81** | EXACTO (suelo) |

1. **Con malla ciega, los CUATRO métodos convergen a orden ≈ 2**, sin importar que su
   orden teórico sea 2, 4 o 6. **Todos colapsan al mismo orden.** No es una coincidencia
   de estos datos: es un resultado general del análisis numérico. Cuando el integrando es
   continuo pero no derivable ($C^0$ pero no $C^1$), **el orden de convergencia de
   cualquier regla de Newton-Cotes compuesta queda limitado a 2** por los subintervalos
   que contienen las singularidades. El eslabón más débil fija el ritmo de toda la
   cadena, y ese eslabón son los codos.
2. **La consecuencia es demoledora para la intuición habitual:** pagar por una regla de
   orden 6 y recibir convergencia de orden 2 significa que **todo el trabajo extra de
   Boole se desperdicia**. Y como además su constante de error es peor y cruza más codos,
   termina dando un resultado *peor* que el trapecio. Orden superior, coste superior,
   precisión inferior.
3. **Con malla alineada, Simpson 1/3, Simpson 3/8 y Boole caen directamente al suelo de
   redondeo** (~1e-13, reportado como `EXACTO`). No es que mejoren: es que **aciertan el
   valor exacto**. Tal como predijimos, dentro de cada tramo $f = r^2$ es un polinomio de
   grado 2, y los tres tienen grado de exactitud ≥ 3, así que lo integran **exactamente**.
   El error residual de 1e-13 es solo aritmética de punto flotante.
4. **El trapecio mantiene orden 2 en ambas columnas** — es el único que no mejora, porque
   su grado de exactitud (1) es menor que el grado del integrando (2), así que ni
   alineado puede ser exacto. Pero fíjate en el contraste: **es el único método al que
   las esquinas no le quitan nada**, porque no tenía orden alto que perder. *El método
   más humilde resultó el más robusto.*

**El resultado central de esta sección, en una frase:**

> Alinear la malla con los quiebres llevó a Simpson y Boole de **orden 2 efectivo a
> exactitud total**. Cambiar de fórmula no habría logrado nada de eso. **La estructura de
> la malla importó más que el orden del método** — por varios órdenes de magnitud.

**La conclusión, y por qué trasciende este laboratorio:**

> El orden teórico de un método es una **promesa condicionada** a que el integrando sea
> suficientemente suave. Cuando la función tiene esquinas — y los envases reales las
> tienen, y las fotos añaden ruido encima — esa promesa **no se cumple**, y un método de
> orden 6 puede rendir como uno de orden 2.
>
> Alinear la malla con la estructura del problema resultó **más eficaz que subir el
> orden de la regla**. En la fase de cámara esto se traduce en una recomendación
> concreta: detectar los quiebres del perfil (hombro, pie) e integrar por tramos, en
> lugar de buscar cuadraturas más sofisticadas.""")

M(r"""### 8.1 Gráfica de convergencia

**Qué hace esta celda:** dibuja el error frente a $\Delta z$ en log-log, para ambas
mallas, con rectas de referencia de pendiente 2, 4 y 6.

Las rectas punteadas grises son las **pendientes teóricas**. Una curva de error paralela
a la recta de pendiente 4 confirma orden 4; una curva más horizontal que su referencia
indica que el método **no está alcanzando su orden**.""")

C(r'''fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
colores = {"Trapecio": "#1f77b4", "Simpson 1/3": "#2ca02c",
           "Simpson 3/8": "#ff7f0e", "Boole": "#d62728"}

# Marcadores distintos: en el panel derecho tres curvas caen al mismo suelo y
# se superponen; sin marcadores diferenciados serían indistinguibles.
marcadores = {"Trapecio": "o", "Simpson 1/3": "s", "Simpson 3/8": "^", "Boole": "D"}

for ax, df, titulo in [
    (axes[0], CONV_CIEGA, "Malla CIEGA\n(quiebres entre nodos)"),
    (axes[1], CONV_ALIN, "Malla ALINEADA\n(quiebres sobre nodos)"),
]:
    for met, color in colores.items():
        sub = df[df["Método"] == met].sort_values("Δz")
        if len(sub):
            ax.loglog(sub["Δz"], sub["Error"], marcadores[met] + "-", color=color,
                      label=met, lw=1.5, ms=5, alpha=0.85, mfc="none", mew=1.4)

    # Rectas de referencia de pendiente p, ancladas al punto más grueso del trapecio
    ref = df[df["Método"] == "Trapecio"].sort_values("Δz")
    dz_ref = np.array([float(df["Δz"].min()), float(df["Δz"].max())])
    if len(ref):
        dz0, e0 = float(ref["Δz"].iloc[-1]), float(ref["Error"].iloc[-1])
        for p, estilo in [(2, ":"), (4, "--"), (6, "-.")]:
            ax.loglog(dz_ref, e0 * (dz_ref / dz0) ** p, estilo,
                      color="gray", lw=1.0, alpha=0.55)
            # Etiqueta al extremo izquierdo de cada recta, donde hay sitio libre
            ax.annotate(f"O(Δz^{p})", xy=(dz_ref[0], e0 * (dz_ref[0] / dz0) ** p),
                        xytext=(3, 2), textcoords="offset points",
                        fontsize=7.5, color="gray", va="bottom")

    ax.axhline(1e-13, color="crimson", ls="--", lw=1, alpha=0.55)
    ax.annotate("suelo de redondeo float64", xy=(dz_ref[0], 2e-13),
                fontsize=7.5, color="crimson", va="bottom")
    ax.set_xlabel("Δz  [cm]")
    ax.set_title(titulo, fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.grid(True, which="both", alpha=0.22)

# Señalamos explícitamente el hallazgo en cada panel
axes[0].annotate("las 4 curvas colapsan\na pendiente ≈ 2",
                 xy=(0.52, 0.40), xycoords="axes fraction", fontsize=9,
                 color="#444", ha="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                           ec="#bbb", alpha=0.95))
axes[1].annotate("Simpson y Boole: EXACTOS\n(pegados al suelo de redondeo)",
                 xy=(0.50, 0.12), xycoords="axes fraction", fontsize=9,
                 color="#444", ha="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow",
                           ec="#bbb", alpha=0.95))

axes[0].set_ylabel("Error absoluto en el volumen  |V − V_exacto|  [cm³]")
axes[0].set_ylim(1e-17, 1e3)
fig.suptitle("Convergencia de los métodos de Newton-Cotes — "
             f"{ENVASE.nombre}  (V_exacto = {V_EXACTO:.4f} cm³)", fontsize=12)
fig.tight_layout()
plt.show()

print("\nLectura de la gráfica:")
print("  • Izquierda (malla ciega): las curvas de Simpson y Boole son MÁS PLANAS que")
print("    sus rectas de referencia → no alcanzan su orden teórico. Las esquinas del")
print("    envase les impiden aprovechar su orden alto.")
print("  • Derecha (malla alineada): las curvas caen mucho más rápido, y varias chocan")
print("    contra la línea roja del suelo de redondeo → error exacto salvo aritmética.")
print("  • La línea roja (1e-13) es el límite de float64. Por debajo, el 'error' ya no")
print("    es del método: es ruido de punto flotante acumulado en las sumas.")''')

M(r"""**Cómo leer la gráfica:**

- **Eje X:** $\Delta z$ decreciente hacia la izquierda — más puntos, malla más fina.
- **Eje Y:** error absoluto en cm³, escala logarítmica.
- **Una curva más inclinada = método que mejora más rápido al refinar la malla.**
- **Panel izquierdo vs. derecho** — es el contraste que da sentido a toda la sección.
  A la izquierda, las curvas de Simpson 3/8 y Boole discurren casi paralelas a la del
  trapecio pese a ser teóricamente muy superiores. A la derecha se despegan y se hunden.
  **El mismo método, los mismos datos, y órdenes de magnitud de diferencia: lo único que
  cambió fue dónde se colocaron los nodos.**
- **La línea roja horizontal (1e-13)** es el suelo de redondeo. Cuando una curva llega
  ahí y se aplana, **ya no está midiendo el error del método**: está midiendo la
  acumulación de errores de redondeo de `float64` en la suma de miles de términos.
  Refinar más allá de ese punto **empeora** el resultado, porque se suman más términos y
  se acumula más redondeo, sin ganancia de aproximación. Es el equilibrio clásico entre
  **error de truncamiento** (baja al refinar) y **error de redondeo** (sube al refinar):
  existe un $\Delta z$ óptimo, y pasarse es contraproducente.""")
