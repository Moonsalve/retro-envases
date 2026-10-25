# Reto: del píxel al volumen — Laboratorio de Análisis Numérico

Estimación del volumen de envases mediante visión por cámara, sólidos de revolución e
integración numérica.

**Estado:**
- **Fase 1 completa** — motor de integración numérica validado contra un volumen exacto.
- **Fase 2 en curso** — pipeline de visión (foto → perfil → volumen) con corrección de
  perspectiva, validado con fotos sintéticas renderizadas con una cámara pinhole real:
  ≤ 1 % de error entre 35 y 200 cm de distancia (sin corregir: −2 % a −20 %).
- **Escala con la hoja de referencia ArUco (opción B)** — la cámara se calibra y se ubica
  en 3D desde la propia foto: 48/48 escenas sintéticas dentro del 5 % sin conocer la focal.
- **App web para computador y celular** — `app/streamlit_app.py`, organizada según las
  etapas del enunciado, con cámara en vivo. Falta: desplegarla y validar con fotos reales.

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

**4. El presupuesto de error lo domina la escala, no la cuadratura.** La escala
píxel→cm convierte a la vez radios y alturas, así que V ∝ s³ y un error relativo ε en la
escala produce **≈ 3ε** en el volumen. (Un error solo en el borde del contorno afecta
únicamente al radio y produce 2ε.)

| Error de escala | Error en el volumen | ¿Pasa el 5 %? |
|---|---|---|
| 0.5 % | 1.50 % | ✓ |
| 1.0 % | 3.02 % | ✓ |
| 1.64 % | 5.00 % | frontera |
| 2.0 % | 6.11 % | ✗ |

Un error de escala del 1 % pesa **71 veces más** que el peor método. Y el error del método
se reduce gratis subiendo n (con n = 1200 el trapecio baja a 2.6e-5 %), mientras que
mejorar la escala cuesta trabajo físico real.

*Corrección respecto a una versión anterior de este README, que decía ×2 y un límite de
2.5 %: aquella simulación escalaba solo los radios. El factor correcto para un error de
escala es ×3, verificado con el pipeline completo.* Y el error del método se reduce gratis subiendo n (con n = 1200 el
trapecio baja a 2.6e-5 %), mientras que mejorar la escala cuesta trabajo físico real.

## Decisiones de diseño para la fase 2

| Aspecto | Decisión | Justificación |
|---|---|---|
| Método a producción | Trapecio o Simpson sobre los datos crudos | Robustos ante aristas y ruido; el trapecio no tiene restricciones sobre n |
| Estrategia de malla | Remuestrear a n múltiplo de 6; integrar por tramos si se detectan quiebres | Permite los tres Newton-Cotes y recupera el orden teórico |
| Gauss | Solo por tramos, nunca global | Global falla el 5 % con pocos nodos; por tramos exige detectar los quiebres del perfil |
| Validación | Fotos sintéticas de volumen exacto + un envase de geometría conocida (lata) medido con regla | Sin impresora ni medición con agua; el PDF admite "recipiente de geometría conocida" |
| Escala | **Hoja impresa con 4 marcadores ArUco bajo el envase** (antes: altura medida con regla) | Da la escala, la focal y la pose de la cámara desde la foto; ver «Opción B» |
| Segmentación | Otsu/Canny **y** GrabCut, comparados | Cubre el requisito de la Etapa 1 de investigar dos estrategias |
| Aplicación | Streamlit, web y celular, con cámara en vivo | El enunciado pide aplicación web y captura con la cámara |

---

# Fase 2 — De la foto al perfil

Paquete `retroenvases/`:

| Módulo | Qué hace |
|---|---|
| `integracion.py` | Los 4 métodos (Trapecio, Simpson, Newton-Cotes 3/8, Gauss), el remuestreo a malla uniforme y `calcular_todos()` |
| `vision.py` | Corrección de iluminación, segmentación **Otsu** y **GrabCut**, enderezado del eje, bordes subpíxel y escala por la altura |
| `pipeline.py` | `estimar_volumen(foto, altura_cm, segmentador, focal_px, pitch_grados, rect)`: la cadena completa |
| `foto.py` | Carga de fotos de celular: orientación EXIF, focal equivalente de 35 mm, resolución de trabajo |
| `sintetico.py` | Fotos sintéticas con volumen exacto: ortográficas (`generar`) y con cámara pinhole real (`generar_perspectiva`) |

## La cadena

```
foto ─► corrección de iluminación ─► segmentación (Otsu | GrabCut) ─► limpieza
     ─► enderezado del eje ─► bordes subpíxel ─► escala = altura_px / altura_cm
     ─► remuestreo a n múltiplo de 6 ─► 4 métodos de integración ─► volumen
```

**Decisiones que resultaron clave** (cada una corrigió un fallo medido en el banco):

1. **Corregir la iluminación antes de segmentar.** Se ajusta una superficie cuadrática por
   canal sobre el marco exterior de la foto (fondo puro, porque el envase no toca los
   bordes) y se resta. Sin esto, con un degradé de luz, ningún umbral global separa el
   envase: el error era +93 %.
2. **Umbralizar la distancia de color al fondo, no el gris.** No hay que adivinar si el
   envase es más claro u oscuro que el fondo. El umbral de Otsu se acota con el ruido del
   fondo (media + 6σ), porque Otsu falla cuando parte del envase se confunde con el fondo.
3. **Convexidad por filas.** La silueta de un sólido de revolución corta cada fila
   horizontal en un solo intervalo; rellenar cada fila entre sus extremos cierra cualquier
   hueco interior. Es el modelo físico, no un parche.
4. **Bordes subpíxel.** En RGB la distancia de un píxel de borde al fondo es lineal en su
   fracción de cobertura, así que el borde real está donde esa distancia cruza la mitad
   de su salto. Bajó el error del caso base de +2.3 % a −0.04 %.

## Banco de pruebas

```bash
python -m tests.banco_vision
```

3 envases (botella, lata, vaso cónico) × 10 escenas × 3 semillas × 2 segmentadores = 180
corridas contra el volumen exacto:

| | Otsu | GrabCut |
|---|---|---|
| Error medio \|ε\| | 0.14 % | 0.10 % |
| Peor caso | 1.67 % | 1.13 % |
| Dentro del 5 % | 100 % | 100 % |
| Tiempo por foto | ~95 ms | ~224 ms |

El peor caso en ambos es el "combo difícil" (envase claro sobre fondo claro, degradé,
inclinación y ruido juntos).

Estas fotos son proyección **ortográfica**. La perspectiva se trata en la sección
siguiente.

## Perspectiva: la fuente de error dominante

Renderizando con una cámara pinhole real (se lanza un rayo por píxel contra el sólido de
revolución), el pipeline sin corregir **subestima el volumen** de forma sistemática:

| Distancia | Botella 21 cm | Lata 12 cm | Vaso 12 cm |
|---|---|---|---|
| 25 cm | −16.8 % | −22.1 % | −26.4 % |
| 50 cm | −9.2 % | −12.1 % | −14.2 % |
| 100 cm | −4.8 % | −6.1 % | −7.1 % |
| 200 cm | −2.3 % | −2.9 % | −3.5 % |

**Por qué.** El tope y la base de la silueta no están sobre el eje: son los bordes de la
boca y de la base **más cercanos a la cámara**, r cm por delante, y se ven agrandados. La
altura en píxeles sale inflada, la escala también, y todo en cm queda chico (×3 en V).

### La corrección (`vision.corregir_perspectiva`)

Con la focal de la cámara en píxeles (del EXIF) se invierte el modelo pinhole:

1. **Distancia D.** Los rayos del tope y de la base, más la altura real H, dan dos
   ecuaciones: `H − hc = a_t·(D − r_t)` y `−hc = a_b·(D − r_b)` (a = pendiente del rayo).
   Restándolas se despeja D; la altura de la cámara hc se cancela y sale después.
2. **Perfil.** El rayo de cada borde de la silueta es tangente al envase: su punto de
   máximo acercamiento al eje da el radio y la altura reales.
3. **Arcos de boca y base.** Esos bordes son círculos que se ven como arcos de elipse,
   de alto ≈ |H − hc|·r/D. Las filas del arco (máximo acercamiento fuera del envase) no
   son pared. El radio del borde se toma del arco (cortándolo con el plano del borde) si
   el arco es grande, o de la pared adyacente si es pequeño: cada medición falla donde la
   otra acierta.

Con el celular inclinado, cada rayo se rota por la inclinación informada.

**Cosas que se probaron y se descartaron** (quedan documentadas en el código):
tangencia exacta usando la pendiente de la pared (exacta en teoría, peor en la práctica
por el ruido de la derivada numérica; el error de no usarla es < 0.002 cm) y medir los
arcos por columnas (deja el sistema indeterminado: el vértice del arco es el mismo punto
con el que se despeja D).

### Resultados (`python -m tests.banco_perspectiva`)

| Condición | Error corregido |
|---|---|
| Distancia 35–200 cm, cámara a media altura | ≤ 1.0 % |
| Distancia 25 cm | ≤ 1.6 % |
| Cámara entre la base y la boca (40 cm) | ≤ 1.5 % |
| Cámara 10 cm **sobre** la boca | hasta −4.7 % ⚠ |
| Celular inclinado ±10°, **inclinación informada** | ≤ 1.0 % |
| Celular inclinado ±10°, se asume vertical | hasta +6.5 % ⚠ |
| Focal equivocada ±5 % | ≤ 1.5 % |
| Focal equivocada ±10 % | hasta 2.4 % |

## Protocolo de captura de la versión sin hoja (histórico)

*Este protocolo correspondía a la escala por la altura medida. Lo reemplaza el de la
opción B, más abajo.*

1. **Distancia 40–100 cm**, sin zoom (así la focal del EXIF es la de la cámara principal).
2. **Cámara entre la base y la boca**, idealmente a media altura. Nunca por encima.
3. **Celular vertical**, usando el nivel de la cámara (±2°), o informar la inclinación.
4. Si la foto se pasa por WhatsApp, **enviarla como Documento** para conservar el EXIF
   con la focal; si no, se asume 26 mm.
5. **Medir la altura del envase al medio milímetro.** Es la única referencia de escala y
   su error se multiplica ×3.
6. Fondo liso que contraste con el envase, sin que el envase toque los bordes de la foto.

### Presupuesto de error resultante (siguiendo el protocolo)

| Fuente | Botella 21 cm | Lata 12 cm |
|---|---|---|
| Altura medida ±0.5 mm (×3) | 0.7 % | 1.2 % |
| Perspectiva residual | ≤ 1.0 % | ≤ 1.0 % |
| Inclinación ±2° no informada | ~0.5 % | ~0.5 % |
| Focal EXIF (±3 %) | ~0.6 % | ~0.9 % |
| Segmentación y bordes | ~0.2 % | ~0.2 % |
| **Peor caso (suma)** | **~3 %** | **~3.8 %** |

Cabe en el 5 %, pero con poco margen en envases pequeños: la altura medida con regla es
el término que más pesa y el único que depende del cuidado del usuario.

**Límites de la validación sintética:** no modela distorsión de lente (los celulares la
corrigen en la propia cámara), envases transparentes o con reflejos, ni etiquetas de
colores. Falta contrastar con fotos reales.

---

# Opción B — Hoja de referencia con marcadores ArUco

El envase se para en el centro de una hoja impresa (carta o A4) con cuatro marcadores
ArUco en las esquinas y círculos guía. La geometría de la hoja está definida una sola vez
en `retroenvases/plantilla.py`, que genera el PDF imprimible.

```
foto ─► marcadores (esquinas subpíxel) ─► focal y pose de la cámara (calibración en 1 vista)
     ─► GrabCut iniciado con la geometría ─► bordes subpíxel (mezcla de color local)
     ─► perfil 3D: cada borde es un rayo tangente al envase; la base está en z = 0
     ─► filtro de picos ─► remuestreo ─► 4 métodos
```

**Por qué resuelve los puntos débiles de la versión anterior:**

| Antes (altura medida) | Con la hoja |
|---|---|
| Altura medida con regla, su error ×3 en el volumen | La escala sale de la hoja; se mide la distancia ID 0 – ID 1 (~19 cm), ~3 veces menos sensible que medir el lado de un marcador |
| Focal del EXIF (WhatsApp la borra) | La focal se calibra desde los marcadores: < 1 % de error en todas las pruebas |
| Inclinación del celular desconocida | Sale de la pose de la cámara |
| Eje y altura de la cámara estimados | El eje es la vertical del círculo; la base está en z = 0 |

**Decisiones que salieron de fallos medidos en el banco:**
- El suelo *delante* del envase es fondo seguro para GrabCut: así aprende el negro de los
  marcadores como fondo (antes los incluía y el volumen salía +51 %).
- Tangencia exacta usando la pendiente de la pared: con la cámara mirando desde arriba,
  el rayo roza las paredes inclinadas lejos de su máximo acercamiento al eje.
- Bordes subpíxel con un modelo de mezcla local de color, que funciona delante de un
  marcador negro o de la hoja blanca (el modelo de fondo global fallaba ahí).
- Altura de la boca ajustando el arco de su elipse cuando la cámara está por encima.
- Filtro de Hampel sobre r(z) para quitar picos de segmentación.

**Banco** (`python -m tests.banco_hoja`), sin conocer la focal ni la pose: **48/48
escenas dentro del 5 %** (distancia 35–70 cm, cámara a 20–45 cm de altura, girada ±15°,
envase descentrado hasta 5 mm, focales 24–35 mm, hoja impresa al 97 % con su medida
corregida). Peor caso 4.4 %, típico 0.3–1.5 %. Sin corregir una hoja impresa al 97 % el
error es ~10 %: medir la hoja es obligatorio. Sesgo conocido: ~+1 % en envases con base
acampanada vista desde arriba.

## Protocolo de captura

1. Imprimir la hoja al 100 % y medir la distancia entre los bordes exteriores de ID 0 e
   ID 1 (191.9 mm en carta, 186.0 mm en A4).
2. Hoja plana sobre la mesa, con ID 0 e ID 1 hacia la cámara; envase de pie y centrado.
3. Cámara a 40–70 cm, un poco por encima del envase (20–35 cm sobre la mesa).
4. Que se vean completos al menos ID 0 e ID 1; iluminación pareja, sin reflejos.
5. En el celular, para máxima resolución, «Subir foto → Cámara» (cámara nativa).

# La aplicación web

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Funciona en el navegador del computador y del celular. Sus secciones siguen el enunciado:

| Sección de la app | Requisito |
|---|---|
| Antes de empezar: PDF de la hoja y protocolo de captura | Etapa 2 |
| 1. Foto: cámara en vivo, subir foto o ejemplo sintético | §8, Etapa 2, MVP |
| 2. Referencia de escala: medida de la hoja, detección dibujada sobre la foto y explicación de referencia, longitud, coordenadas en píxeles, conversión y supuestos | Etapa 2, §8 |
| 3. Perfil: tres estrategias (automática con la hoja, GrabCut con rectángulo, Otsu), comparables; imagen, contorno y perfil en cm con sus puntos | Etapas 1 y 3, §8 |
| 4. Intervalo [a, b], n, Δz y orden de Gauss; justificación del remuestreo | §5, §8 |
| 5. Tabla i, zᵢ, rᵢ, rᵢ², πrᵢ² | Etapa 4 |
| 6. Los 4 métodos con n, [a,b], Δz, orden, integral, volumen y tiempo; gráfica; avisos; análisis de cada método | §6, Etapa 5, §8 |
| 7. Validación contra un volumen de referencia y presupuesto de error por fuente | Etapa 6, MVP |
| 8. Datos para copiar al informe | §8 |

Con el ejemplo sintético (botella de 616.61 cm³) la app da **1.16 %** de error.
Verificada con `streamlit.testing` (AppTest) y en Chrome de escritorio.

## Prueba con una foto real (`data/foto_prueba_1.jpeg`)

Botella transparente de agua, fondo con ventana, laptop y objetos, tomada desde arriba y
reenviada por WhatsApp. Lo que mostró:

- **Otsu y GrabCut automático fallan** con un fondo desordenado: suponen que el marco de
  la foto es fondo liso. **GrabCut con un rectángulo sí extrae la botella**, incluso
  siendo transparente. Por eso el rectángulo es la opción recomendada en la app.
- **El refinamiento subpíxel no se activa** con un fondo desordenado (el borde queda con
  precisión de ~1 px, ~1 % en volumen). Mejora pendiente: usar el fondo local en vez del
  global.
- **La perspectiva sin corregir se nota a simple vista**: la botella aparece ~20 % más
  ancha arriba que abajo porque la foto se tomó desde arriba.
- **WhatsApp borra el EXIF** y recomprime a 1200×1600: sin focal no hay corrección de
  perspectiva confiable.
- Sin la altura medida el volumen no se puede fijar. En la app, suponiendo H = 26 cm
  (valor de prueba, no medido), da ~881 cm³ con asimetría 0.9 %; entre 769 y 887 cm³
  según la inclinación que se suponga, porque la real se desconoce.

Conclusión: la segmentación funciona en una foto real difícil; el volumen de esa foto no
es confiable por falta de datos de entrada, no por el algoritmo. Para validar hace falta
una foto que siga el protocolo.

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

Esa tabla corresponde al notebook. La cobertura de la app está en la sección «La
aplicación web».

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
