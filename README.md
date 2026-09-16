# Reto: del píxel al volumen — Laboratorio de Análisis Numérico

Estimación del volumen de envases mediante visión por cámara, sólidos de revolución e
integración numérica.

**Estado: Fase 1 completa** — motor de integración numérica validado.
La fase de cámara (extracción del perfil desde una foto) aún no está implementada.

---

## Qué resuelve esta fase

El reto completo exige la cadena:

```
                          ┌──────── FASE 1 (esto) ────────┐
captura → imagen → escala → datos → modelo → integración → volumen → validación
└──── fase 2 (pendiente) ─┘
```

La Fase 1 sustituye la fotografía por un **envase definido matemáticamente**, cuyo
volumen exacto se conoce por fórmula cerrada. Eso permite medir el error del motor
numérico **aislado** de los errores de imagen, y por tanto saber cuánto del presupuesto
de error del 5 % queda disponible para la fase de cámara.

## Los dos notebooks

| Notebook | Celdas | Para qué |
|---|---|---|
| **`Lab_RetroEnvases.ipynb`** | **16** | **El entregable.** Los cuatro métodos del enunciado, directo al grano, y una celda final de texto con todos los resultados consolidados. |
| `Lab_Integracion_Numerica_RetroEnvases_Fase1.ipynb` | 58 | Versión extendida (incluye además Boole y las dos variantes de Gauss): estudio de convergencia con dos mallas, barrido del orden de Gauss, pruebas de condiciones no satisfechas. Material de apoyo. |

Ambos dan los mismos resultados; el largo los justifica con más experimentos.

## Cómo ejecutarlo

### Opción A — Google Colab (recomendada)

1. Abre [colab.research.google.com](https://colab.research.google.com)
2. `Archivo → Subir notebook` y selecciona `notebooks/Lab_RetroEnvases.ipynb`
3. `Entorno de ejecución → Ejecutar todo` (`Ctrl+F9`)

No requiere instalar nada: Colab ya trae `numpy`, `scipy`, `pandas` y `matplotlib`.
Tiempo de ejecución: ~5 segundos.

### Opción B — Local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib jupyter
jupyter notebook notebooks/Lab_RetroEnvases.ipynb
```

### Verificación independiente de las fórmulas

```bash
python3 tests/verificar_formulas.py
```

Reimplementa las cinco reglas de cuadratura en Python puro (sin NumPy) y comprueba que
los coeficientes son correctos, que cada regla integra exactamente polinomios hasta su
grado de exactitud declarado y falla un grado por encima, y que los hallazgos numéricos
que afirma el notebook se reproducen. No requiere dependencias.

## Personalizar el envase

En la celda 4 del notebook corto (sección 1), edita `TRAMOS`:

```python
# (z_inicio, z_fin, radio_inicio, radio_fin, nombre)   en cm
TRAMOS = [
    ( 0.0,  1.0, 2.50, 3.40, "base (pie)"),
    ( 1.0, 15.0, 3.40, 3.40, "cuerpo cilíndrico"),
    (15.0, 18.5, 3.40, 1.35, "hombro"),
    (18.5, 20.0, 1.35, 1.35, "cuello"),
    (20.0, 21.0, 1.35, 1.45, "boca / rosca"),
]
```

Cada tramo es un tronco de cono (cilindro si `r_ini == r_fin`). El volumen exacto de
referencia se recalcula solo: es la suma cerrada de los volúmenes de los troncos, así que
**no hay que volver a medir nada** al cambiar las proporciones.

## Métodos implementados

| Método | Familia | Condición sobre n | Orden del error | Grado de exactitud |
|---|---|---|---|---|
| **Regla del Trapecio** | Newton-Cotes grado 1 | ninguna | O(Δz²) | 1 |
| **Regla de Simpson** | Newton-Cotes grado 2 | n par | O(Δz⁴) | 3 |
| **Newton-Cotes 3/8** | Newton-Cotes grado 3 | n múltiplo de 3 | O(Δz⁴) | 3 |
| **Cuadratura de Gauss** | Gauss-Legendre | — (m nodos) | exacta grado ≤ 2m−1 | 2m−1 |

La función `newton_cotes(z, r, grado)` acepta también `grado=4` (regla de Boole,
O(Δz⁶)) si se quiere una fórmula de orden superior.

Gauss se aplica **por tramos** (una cuadratura en cada tramo del envase) sobre un spline
cúbico ajustado a los datos. El notebook muestra también qué pasa si se aplica de forma
**global**: un factor 150 peor.

## Resultados principales

Envase de prueba: botella PET de 600 mL, 21 cm de altura, V exacto = **616.608244 cm³**.

**1. Con malla ciega (n = 60), el ranking de precisión sale invertido respecto al orden
teórico:**

| Puesto | Método | Orden teórico | Evals | Error [cm³] | % del margen del 5 % |
|---|---|---|---|---|---|
| 1 | Gauss, 5 nodos por tramos | grado ≤ 9 | 25 | 0.020 | 0.07 % |
| 2 | Trapecio | O(Δz²) | 61 | 0.063 | 0.20 % |
| 3 | Newton-Cotes 3/8 | O(Δz⁴) | 61 | 0.071 | 0.23 % |
| 4 | Simpson | O(Δz⁴) | 61 | 0.263 | 0.85 % |

Entre los Newton-Cotes el ranking sale invertido: el trapecio (orden 2) es 4.2 veces más
preciso que Simpson (orden 4). Y Gauss gana con menos de la mitad de evaluaciones — pero
solo porque se aplica por tramos: de forma global da 0.49 % de error, 150 veces peor.

**2. Los Newton-Cotes convergen a orden ≈ 2, sin importar su orden teórico:**

| Método | Teórico | Malla ciega | Malla alineada |
|---|---|---|---|
| Trapecio | 2 | 1.97 | 2.00 |
| Simpson | 4 | 1.98 | exacto (suelo de redondeo) |
| Newton-Cotes 3/8 | 4 | 2.06 | exacto (suelo de redondeo) |

Los puntos de quiebre del envase (hombro, pie) rompen la hipótesis de suavidad de las
cotas de error de orden alto. Alinear la malla con esos quiebres los lleva de orden 2
efectivo a **exactitud total**: la estructura de la malla importa más que el orden de la
fórmula. *(Medido en la versión extendida.)*

**3. La misma cuadratura de Gauss cambia 150× según dónde se aplique.** Global (un solo
polinomio de grado 9 para todo el envase) da 0.49 % de error; con m = 2 o 3 nodos ni
siquiera cumple el criterio del 5 %. Aplicada **por tramos** da 0.003 % y es el método más
preciso, con 25 evaluaciones frente a 61. La corrección no está en cambiar de fórmula sino
en respetar la estructura del problema.

**4. El presupuesto de error lo domina la escala, no la cuadratura.** Como V ∝ r², un
error relativo ε en la escala píxel→cm produce 2ε en el volumen:

| Error de escala | Error en el volumen | ¿Pasa el 5 %? |
|---|---|---|
| 1 % | 2.01 % | ✓ |
| 2.5 % | 5.06 % | ✗ (frontera) |
| 5 % | 10.25 % | ✗ |

Un error de escala del 1 % pesa **47 veces más** que el peor método y **615 veces
más** que el mejor. Y el error del método se reduce gratis subiendo n (con n = 1200 el
trapecio baja a 2.6e-5 %), mientras que mejorar la escala cuesta trabajo físico real.

## Decisiones de diseño para la fase 2

| Aspecto | Decisión | Justificación |
|---|---|---|
| Método a producción | Trapecio o Simpson sobre los datos crudos | Robustos ante aristas y ruido; el trapecio no tiene restricciones sobre n |
| Estrategia de malla | Remuestrear a n múltiplo de 6; integrar por tramos si se detectan quiebres | Permite los tres Newton-Cotes y recupera el orden teórico |
| Gauss | Solo por tramos, nunca global | Global falla el 5 % con pocos nodos; por tramos exige detectar los quiebres del perfil |
| Validación | Volumen **exterior** por desplazamiento de agua | Única cantidad comparable con una silueta |
| Escala | ArUco + homografía, dos clics de respaldo | Error objetivo < 1 %, límite < 2.5 % |
| Segmentación | Otsu/Canny **y** GrabCut, comparados | Cubre el requisito de la Etapa 1 de investigar dos estrategias |
| Aplicación | Streamlit en Streamlit Cloud | `st.camera_input()` funciona en el celular; HTTPS gratis |

## Cobertura del enunciado

| Requisito del PDF | Dónde |
|---|---|
| §5 Contexto matemático | Sección 1 |
| §6 Trapecio, Simpson, Gauss, Newton-Cotes | Secciones 5.2 a 5.7 |
| §6 Efecto de Δz y del número de puntos | Sección 8 |
| §6 Influencia del orden de cuadratura | Sección 5.7 |
| §7 Etapa 4 — tabla (zᵢ, rᵢ, rᵢ², πrᵢ²) | Sección 4 |
| §7 Etapa 5 — parámetros de cada método | Sección 7 |
| §7 Etapa 6 — diferencia absoluta y error relativo | Sección 7 |
| §8 Reporte explícito de Δz | Secciones 4 y 7 |
| §8 Mensajes claros ante condiciones no cumplidas | Sección 6 |
| §8 Representación gráfica del perfil y de πr² | Sección 9 |
| §8 Exportación de datos y resultados | Sección 10 |
| §9 MVP — explicar las fuentes de error | Sección 11 |

Pendiente para la fase 2: §7 Etapas 2 y 3 (captura, escalamiento, extracción del perfil)
y los requisitos de interfaz de la §8.

## Estructura

```
retro-envases/
├── notebooks/
│   ├── Lab_RetroEnvases.ipynb                             ← el entregable (16 celdas)
│   └── Lab_Integracion_Numerica_RetroEnvases_Fase1.ipynb   versión extendida (58)
├── tests/
│   └── verificar_formulas.py      verificación independiente, sin dependencias
├── build/                         generadores (las celdas viven en .py, no en JSON)
│   ├── construir_corto.py         → Lab_RetroEnvases.ipynb
│   ├── cells_01_intro.py ... cells_07_graficas.py
│   └── construir_notebook.py      → versión extendida
└── README.md
```

El notebook se genera desde `build/`: las celdas viven en módulos Python normales, lo que
permite versionarlas y editarlas con herramientas de texto en vez de pelear con el JSON
de un `.ipynb`. Para reconstruirlo:

```bash
python3 build/construir_corto.py      # versión corta
python3 build/construir_notebook.py   # versión extendida
```

## Salida

Los notebooks **no escriben archivos ni descargan nada**: todo queda a la vista en las
celdas, listo para seleccionar y copiar.

El notebook corto termina con una **celda de texto** que consolida todo lo que hace falta
para el informe: datos del problema con Δz explícito, composición del envase por tramos,
tabla comparativa completa de los seis métodos (integral, volumen, error absoluto, error
relativo y porcentaje del margen), condiciones de aplicabilidad de cada método, tabla de
sensibilidad al error de escala, inventario de fuentes de error, presupuesto de error para
la fase 2 y el mapeo contra los artículos del enunciado.

*(Esa celda refleja el envase por defecto; si editas `TRAMOS` los números cambian y los
verás actualizados en las salidas de las celdas de cálculo.)*
