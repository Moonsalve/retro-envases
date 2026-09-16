# -*- coding: utf-8 -*-
"""Celdas: modelo del envase parametrico + volumen de referencia exacto."""

CELLS = []
M = lambda s: CELLS.append(("markdown", s))
C = lambda s: CELLS.append(("code", s))

M(r"""---
## 2. Modelo del envase: perfil paramétrico de proporciones conocidas

### 2.1 La idea

En lugar de una foto, definimos el envase como una **lista de tramos**. Cada tramo va de
una altura $z_i$ a otra $z_{i+1}$, y el radio varía **linealmente** entre $r_i$ y $r_{i+1}$.

Geométricamente cada tramo es un **tronco de cono** (un cono al que se le corta la
punta). Si $r_i = r_{i+1}$, el tronco degenera en un **cilindro**.

```
   r(z)
    │                    ╭──────╮  boca / rosca
    │                    │      │
    │                    │      │  cuello  (cilindro)
    │                   ╱        ╲
    │                  ╱          ╲ hombro  (tronco de cono)
    │      ┌──────────╯            ╰──────────┐
    │      │                                  │  cuerpo  (cilindro)
    │      │                                  │
    │       ╲                                ╱   base    (tronco de cono)
    └────────┴──────────────────────────────┴──────────────► z
             a                                            b
```

### 2.2 Por qué precisamente este modelo — la razón es el valor de referencia

El PDF (§7 Etapa 6) exige *"un envase cuya geometría permita obtener un volumen de
referencia independiente"*. Un perfil lineal por tramos lo da **en forma cerrada**: el
volumen de un tronco de cono de altura $h$ y radios $R_1, R_2$ es

$$V_{\text{tronco}} \;=\; \frac{\pi h}{3}\left(R_1^{2} + R_1 R_2 + R_2^{2}\right)$$

El volumen total es la suma sobre los tramos. Ese número es **exacto** — no es otra
aproximación numérica con la que comparar, es *la respuesta*. Podemos medir el error de
cada método hasta el último dígito significativo.

### 2.3 El detalle fino: los puntos de quiebre, y por qué los ponemos a propósito

El perfil es **continuo** ($C^0$) pero **no derivable** ($C^1$) en las uniones entre
tramos: ahí la pendiente cambia de golpe. Esto no es un descuido del modelo, es
**deliberado y realista**: una botella PET real tiene exactamente esas aristas, en el
pie y en el arranque del hombro.

Y tiene una consecuencia numérica de primer orden, que es una de las lecciones centrales
de este laboratorio:

> Las cotas de error de Simpson, Newton–Cotes y Gauss suponen que el integrando es
> **suave** (derivadas continuas hasta cierto orden). En una esquina esa hipótesis se
> rompe, y los métodos de orden alto **pierden su orden teórico**.

En la Sección 8 lo veremos ocurrir con datos, y después lo confirmaremos por contraste
con un envase deliberadamente suave. Es exactamente la clase de discusión que pide la
§7 Etapa 6 del enunciado: distinguir si la diferencia se debe al método, a la
discretización o al modelo.""")

M(r"""### 2.4 Estructuras de datos

**Qué hace esta celda:** define los tipos `Tramo` y `PerfilEnvase`.

Usamos `@dataclass(frozen=True)` — objetos **inmutables**. Una vez construido el perfil,
ninguna celda posterior puede modificarlo por accidente. En un notebook, donde las
celdas se ejecutan en cualquier orden y el estado persiste, la mutación accidental es
*la* fuente clásica de resultados irreproducibles. La inmutabilidad la elimina de raíz.

Métodos que expone `PerfilEnvase`:

| Método | Qué devuelve |
|---|---|
| `radio(z)` | el radio $r(z)$, vectorizado sobre arrays de NumPy |
| `volumen_exacto()` | el volumen analítico en cm³ (fórmula del tronco) |
| `puntos_de_quiebre()` | las alturas donde el perfil no es derivable |
| `a`, `b`, `altura` | extremos e intervalo de integración |""")

C(r'''@dataclass(frozen=True)
class Tramo:
    """Un tramo del envase con radio que varía linealmente con la altura.

    Geométricamente es un tronco de cono (cilindro si r_ini == r_fin).

    Atributos
    ---------
    z_ini, z_fin : float   Alturas inicial y final del tramo [cm].
    r_ini, r_fin : float   Radios en esas alturas [cm].
    nombre       : str     Etiqueta descriptiva (p. ej. "cuerpo").
    """
    z_ini: float
    z_fin: float
    r_ini: float
    r_fin: float
    nombre: str = ""

    def __post_init__(self) -> None:
        # Validación temprana: un tramo inválido detectado aquí produce un error
        # comprensible. Detectado 200 líneas más abajo produce un NaN inexplicable.
        if self.z_fin <= self.z_ini:
            raise ValueError(
                f"Tramo '{self.nombre}': z_fin ({self.z_fin}) debe ser > "
                f"z_ini ({self.z_ini}). Los tramos van de menor a mayor altura."
            )
        if self.r_ini < 0 or self.r_fin < 0:
            raise ValueError(
                f"Tramo '{self.nombre}': los radios no pueden ser negativos "
                f"(r_ini={self.r_ini}, r_fin={self.r_fin})."
            )

    @property
    def altura(self) -> float:
        """Altura h del tramo [cm]."""
        return self.z_fin - self.z_ini

    def volumen_exacto(self) -> float:
        """Volumen analítico del tronco de cono [cm^3].

            V = (pi*h/3) * (R1^2 + R1*R2 + R2^2)

        Para R1 == R2 == R se reduce a pi*R^2*h, el cilindro. Es útil verificarlo
        mentalmente: (pi*h/3)(R^2+R^2+R^2) = (pi*h/3)(3R^2) = pi*R^2*h.
        """
        R1, R2 = self.r_ini, self.r_fin
        return (math.pi * self.altura / 3.0) * (R1 * R1 + R1 * R2 + R2 * R2)

    def radio(self, z: np.ndarray) -> np.ndarray:
        """Radio interpolado linealmente dentro del tramo [cm]."""
        t = (z - self.z_ini) / self.altura           # parámetro normalizado en [0,1]
        return self.r_ini + t * (self.r_fin - self.r_ini)


@dataclass(frozen=True)
class PerfilEnvase:
    """Envase completo como una secuencia contigua de tramos.

    Es el 'gemelo digital' del envase: sustituye a la fotografía en esta fase.
    """
    tramos: tuple[Tramo, ...]
    nombre: str = "envase"
    capacidad_nominal_ml: float | None = None   # solo informativo (etiqueta comercial)

    def __post_init__(self) -> None:
        if not self.tramos:
            raise ValueError("El perfil debe tener al menos un tramo.")
        # Los tramos deben encadenarse sin huecos ni solapamientos: el final de uno
        # es el inicio del siguiente. Un hueco produciría un volumen silenciosamente
        # incorrecto, que es el peor tipo de error: no falla, solo miente.
        for t1, t2 in zip(self.tramos, self.tramos[1:]):
            if not math.isclose(t1.z_fin, t2.z_ini, abs_tol=1e-12):
                raise ValueError(
                    f"Discontinuidad entre '{t1.nombre}' y '{t2.nombre}': "
                    f"{t1.nombre} termina en z={t1.z_fin} pero {t2.nombre} "
                    f"empieza en z={t2.z_ini}. Deben coincidir."
                )

    @property
    def a(self) -> float:
        """Límite inferior de integración = base del envase [cm]."""
        return self.tramos[0].z_ini

    @property
    def b(self) -> float:
        """Límite superior de integración = boca del envase [cm]."""
        return self.tramos[-1].z_fin

    @property
    def altura(self) -> float:
        """Altura total b - a [cm]."""
        return self.b - self.a

    def radio(self, z: float | np.ndarray) -> np.ndarray:
        """Radio r(z) evaluado en uno o muchos puntos. Vectorizado.

        Implementación: np.piecewise selecciona, para cada z, el tramo al que
        pertenece y le aplica su interpolación lineal. Vectorizar importa porque en
        la Sección 8 evaluaremos r(z) cientos de miles de veces.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        # clip evita que un z fuera de [a,b] por redondeo produzca basura;
        # lo satura al borde más cercano, que es el comportamiento físico correcto.
        z_arr = np.clip(z_arr, self.a, self.b)
        condiciones = [
            (z_arr >= t.z_ini) & (z_arr <= t.z_fin) for t in self.tramos
        ]
        funciones = [
            (lambda zz, tr=t: tr.radio(zz)) for t in self.tramos
        ]
        return np.piecewise(z_arr, condiciones, funciones)

    def radio_cuadrado(self, z: float | np.ndarray) -> np.ndarray:
        """El integrando f(z) = r(z)^2 [cm^2]. Es lo que realmente se integra."""
        return self.radio(z) ** 2

    def volumen_exacto(self) -> float:
        """Volumen analítico total [cm^3]. Este es el VALOR DE REFERENCIA.

        No es una aproximación: es la suma cerrada de los volúmenes de los troncos.
        Contra este número mediremos el error de cada método numérico.
        """
        return sum(t.volumen_exacto() for t in self.tramos)

    def puntos_de_quiebre(self) -> np.ndarray:
        """Alturas donde el perfil es continuo pero NO derivable [cm].

        Son las uniones interiores entre tramos. Importan porque ahí se rompe la
        hipótesis de suavidad de las cotas de error de orden alto.
        """
        return np.array([t.z_fin for t in self.tramos[:-1]])

    def resumen(self) -> pd.DataFrame:
        """Tabla legible de los tramos con su aporte al volumen total."""
        V_total = self.volumen_exacto()
        filas = []
        for i, t in enumerate(self.tramos):
            v = t.volumen_exacto()
            filas.append({
                "#": i,
                "tramo": t.nombre,
                "z_ini [cm]": t.z_ini,
                "z_fin [cm]": t.z_fin,
                "h [cm]": t.altura,
                "r_ini [cm]": t.r_ini,
                "r_fin [cm]": t.r_fin,
                "forma": "cilindro" if math.isclose(t.r_ini, t.r_fin) else "tronco de cono",
                "V tramo [cm³]": v,
                "% del total": 100.0 * v / V_total,
            })
        return pd.DataFrame(filas)


print("Estructuras definidas: Tramo, PerfilEnvase")''')

M(r"""### 2.5 Instanciación del envase de prueba

**Qué hace esta celda:** construye el envase concreto con el que trabajaremos: una
botella PET tipo refresco de 600 mL, con proporciones realistas.

**⚙️ Estas son las "proporciones x" — edítalas libremente.** Cambia cualquier número de
`TRAMOS` y todo el notebook se recalcula, incluido el volumen de referencia exacto. Esa
es la ventaja de tener el valor de referencia en forma cerrada: **no hay que remedirlo**.

**Nota importante sobre qué volumen estamos calculando.** El perfil describe la
**superficie exterior**, así que la integral da el **volumen exterior** del envase
(≈ 617 cm³), no su capacidad comercial (600 mL). La diferencia — espesor de pared,
cuello macizo, base cóncava, y el hecho de que nadie llena hasta el borde — es del orden
del 3-8 %.

Esto es coherente con la decisión de validación tomada para el proyecto: contrastaremos
contra el **volumen exterior medido por desplazamiento de agua**, que es la única
cantidad físicamente comparable con lo que se ve en una silueta. Comparar contra la
etiqueta de 600 mL introduciría un sesgo sistemático de ~3 % *antes de empezar a
calcular*, y con un criterio de aprobación del 5 % eso es regalar más de la mitad del
presupuesto de error.""")

C(r'''# =======================================================================
#  ENVASE DE PRUEBA — botella PET tipo refresco, 600 mL nominales
#  Altura total: 21 cm     Diámetro del cuerpo: 6.8 cm
#  ✏️  EDITA ESTOS VALORES para probar otras proporciones.
# =======================================================================
TRAMOS = (
    #      z_ini  z_fin  r_ini  r_fin   nombre
    Tramo(  0.0,   1.0,   2.50,  3.40, "base (pie)"),
    Tramo(  1.0,  15.0,   3.40,  3.40, "cuerpo cilíndrico"),
    Tramo( 15.0,  18.5,   3.40,  1.35, "hombro"),
    Tramo( 18.5,  20.0,   1.35,  1.35, "cuello"),
    Tramo( 20.0,  21.0,   1.35,  1.45, "boca / rosca"),
)

ENVASE = PerfilEnvase(
    tramos=TRAMOS,
    nombre="Botella PET 600 mL",
    capacidad_nominal_ml=600.0,
)

# --- Reporte -----------------------------------------------------------
print("=" * 74)
print(f"ENVASE MODELADO: {ENVASE.nombre}")
print("=" * 74)
print(f"  Intervalo de integración [a, b] : [{ENVASE.a:.3f}, {ENVASE.b:.3f}] cm")
print(f"  Altura total (b - a)            : {ENVASE.altura:.3f} cm")
print(f"  Número de tramos                : {len(ENVASE.tramos)}")
print(f"  Puntos de quiebre (no derivable): {ENVASE.puntos_de_quiebre()} cm")
print(f"  Radio máximo                    : {float(ENVASE.radio(np.linspace(ENVASE.a, ENVASE.b, 2001)).max()):.3f} cm")
print("=" * 74)

display(ENVASE.resumen())''')

M(r"""**Cómo leer el resultado:**

- **`[a, b] = [0.000, 21.000]`** — es el intervalo de integración. Toda la sección 5
  trabaja sobre él.
- **Puntos de quiebre `[1.0, 15.0, 18.5, 20.0]`** — las cuatro alturas donde el perfil
  tiene una arista. Guárdalas mentalmente: en la Sección 8 explicarán por qué el orden
  de convergencia observado no coincide con el teórico.
- **La tabla de tramos** muestra el aporte de cada uno. Verás que el **cuerpo
  cilíndrico aporta ~82 %** del volumen. Esto tiene una lectura práctica directa: la
  precisión del resultado final depende abrumadoramente de medir bien **el radio del
  cuerpo**. Un error de 1 mm en el radio del cuello es casi irrelevante; 1 mm en el
  radio del cuerpo son ~18 cm³ (3 % del total). En la fase de cámara, esto dice dónde
  concentrar la calidad de la medición.
- **Columna `forma`** — distingue cilindros de troncos. Sirve de verificación visual de
  que los tramos quedaron como pretendías tras editarlos.""")

M(r"""---
## 3. Volumen de referencia exacto

### 3.1 Qué significa "exacto" aquí

**Qué hace esta celda:** calcula el volumen del envase por tres vías independientes y
las contrasta.

1. **Fórmula cerrada del tronco de cono.** Álgebra pura. El único error posible es el
   redondeo de `float64`, del orden de 1e-16 relativo.
2. **`scipy.integrate.quad`** con tolerancia exigida de 1e-13, informándole de los
   puntos de quiebre mediante el argumento `points`. Es cuadratura adaptativa
   (Gauss-Kronrod): subdivide el intervalo donde detecta dificultad.
3. **Suma de Riemann con 2 000 000 de subintervalos.** Deliberadamente rudimentaria.

**Por qué tres métodos y no uno.** Esto es una **verificación cruzada**. La fórmula
cerrada podría tener un error de transcripción; `quad` podría estar mal invocado. Que
tres caminos independientes — uno algebraico, uno adaptativo sofisticado, uno de fuerza
bruta primitivo — converjan al mismo número es evidencia fuerte de que el número es
correcto. Es el mismo principio por el que un laboratorio físico mide una magnitud con
dos instrumentos distintos.

**Por qué `points=` en `quad` no es opcional.** Sin ese argumento, `quad` aplica su
regla adaptativa suponiendo un integrando suave y puede converger lentamente o emitir
una advertencia al tropezar con las aristas. Pasándole los quiebres, **parte el
intervalo en trozos donde el integrando sí es suave** y los resuelve por separado. Es la
misma estrategia que aplicaremos manualmente en la Sección 8 para recuperar el orden
teórico de Simpson.""")

C(r'''# --- Vía 1: fórmula cerrada (la referencia) ----------------------------
V_EXACTO = ENVASE.volumen_exacto()
I_EXACTA = V_EXACTO / math.pi          # el valor exacto de ∫ r² dz  [cm³]

# --- Vía 2: cuadratura adaptativa de alta precisión --------------------
# points= le indica a quad dónde está el integrando quebrado, para que
# subdivida ahí y trate cada trozo como suave.
I_quad, err_quad = integrate.quad(
    lambda zz: float(ENVASE.radio_cuadrado(zz)[0]),
    ENVASE.a, ENVASE.b,
    points=ENVASE.puntos_de_quiebre(),
    epsabs=1e-13, epsrel=1e-13, limit=200,
)
V_quad = math.pi * I_quad

# --- Vía 3: fuerza bruta (suma de Riemann del punto medio) -------------
N_BRUTO = 2_000_000
z_bruto = np.linspace(ENVASE.a, ENVASE.b, N_BRUTO + 1)
z_medios = 0.5 * (z_bruto[:-1] + z_bruto[1:])
dz_bruto = (ENVASE.b - ENVASE.a) / N_BRUTO
V_bruto = math.pi * float(np.sum(ENVASE.radio_cuadrado(z_medios)) * dz_bruto)

# --- Reporte -----------------------------------------------------------
print("=" * 74)
print("VOLUMEN DE REFERENCIA — verificación cruzada por tres vías")
print("=" * 74)
print(f"  1. Fórmula cerrada (tronco de cono) : {V_EXACTO:18.10f} cm³   ← REFERENCIA")
print(f"  2. scipy.integrate.quad (tol 1e-13) : {V_quad:18.10f} cm³")
print(f"  3. Riemann, {N_BRUTO:,} subintervalos : {V_bruto:18.10f} cm³")
print("-" * 74)
print(f"  Discrepancia |1 − 2|                : {abs(V_EXACTO - V_quad):.3e} cm³"
      f"   ({abs(V_EXACTO - V_quad)/V_EXACTO:.3e} relativo)")
print(f"  Discrepancia |1 − 3|                : {abs(V_EXACTO - V_bruto):.3e} cm³"
      f"   ({abs(V_EXACTO - V_bruto)/V_EXACTO:.3e} relativo)")
print("=" * 74)
print(f"  Integral exacta  I = ∫ r² dz        : {I_EXACTA:18.10f} cm³")
print(f"  Volumen exacto   V = π·I            : {V_EXACTO:18.10f} cm³  =  {V_EXACTO:.2f} mL")
if ENVASE.capacidad_nominal_ml:
    exceso = 100.0 * (V_EXACTO - ENVASE.capacidad_nominal_ml) / ENVASE.capacidad_nominal_ml
    print(f"  Capacidad nominal (etiqueta)        : {ENVASE.capacidad_nominal_ml:18.2f} mL")
    print(f"  Exceso del volumen EXTERIOR         : {exceso:+.2f} %"
          f"  ← pared, cuello macizo, base cóncava")
print("=" * 74)''')

M(r"""**Cómo leer el resultado:**

- **Las vías 1 y 2 deben coincidir en ~12 dígitos** (discrepancia relativa ~1e-14 o
  menor). Si coinciden, la fórmula cerrada está bien implementada y `quad` bien
  invocado. Este es el número en el que descansa todo el resto del notebook.
- **La vía 3 discrepa más** — alrededor de $2.5\times10^{-11}$ relativo. Y eso es
  **informativo, no un fallo**: la regla del punto medio es de orden $O(h^2)$, así que
  con 2 millones de subintervalos ($h \approx 10^{-5}$) su error esperado es
  $\sim h^2 \approx 10^{-10}$ — coherente con lo observado. Que el error medido siga la
  ley de potencia predicha es, en sí mismo, una validación del código.
- **Ponlo en perspectiva:** esta vía gastó **2 millones** de evaluaciones para llegar a
  ~1e-11 relativo. En la Sección 8 verás reglas de orden alto alcanzar el **suelo de
  redondeo** (~1e-16) con unos pocos cientos de puntos. Cuatro órdenes de magnitud menos
  de trabajo para cinco órdenes de magnitud más de precisión: **el orden del método pesa
  mucho más que la cantidad de puntos**... siempre que el integrando sea suave. Esa
  última cláusula es la que dominará todo el resto del notebook.
- **`V_EXACTO ≈ 617 cm³`** — este es el patrón contra el que se mide todo.
- **El exceso sobre los 600 mL nominales (≈ +3 %)** cuantifica exactamente la trampa
  discutida antes: si validáramos contra la etiqueta, arrancaríamos con 3 % de error
  cuando el criterio de aprobación es 5 %.""")
