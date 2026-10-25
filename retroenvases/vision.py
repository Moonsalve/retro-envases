"""Visión: de la foto al perfil (z, r) en centímetros.

Etapa 3 del enunciado. Dos estrategias de segmentación (la Etapa 1 pide
investigar al menos dos):

  • Otsu   — umbral global elegido automáticamente sobre el histograma de grises.
             Rápido y sin parámetros; falla si envase y fondo tienen gris parecido.
  • GrabCut — modela el color del objeto y del fondo con mezclas gaussianas y
             separa por cortes de grafo. Más lento, pero usa color y vecindad.

La escala (Etapa 2) sale de la ALTURA REAL del envase, medida con regla por el
usuario: px_por_cm = altura_en_píxeles / altura_en_cm. La altura está sobre el
eje de revolución, así que la referencia queda en el plano correcto sin montar
nada en la escena.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


# ────────────────────────────────────────────────────────── segmentación ──
def _limpiar(mascara: np.ndarray) -> np.ndarray:
    """Limpia una máscara usando la geometría del envase.

    1. Conserva la componente más grande y las que se solapan con su caja
       (partes del mismo envase separadas por un brillo o una sombra), y
       descarta manchas lejanas.
    2. Rellena cada fila entre su píxel más a la izquierda y el más a la derecha.
       La silueta de un sólido de revolución es CONVEXA POR FILAS: cada fila
       horizontal la corta en un solo intervalo. Eso cierra cualquier hueco
       interior, aunque parte del envase tenga el mismo color que el fondo.
    """
    m = (mascara > 0).astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m * 255
    areas = stats[1:, cv2.CC_STAT_AREA]
    mayor = 1 + int(np.argmax(areas))
    x0, y0, w0, h0 = stats[mayor, :4]
    conservar = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        solapa = x < x0 + w0 and x + w > x0 and y < y0 + h0 and y + h > y0
        if i == mayor or (solapa and area >= 0.002 * areas.max()):
            conservar.append(i)
    m = np.isin(etiquetas, conservar).astype(np.uint8)
    salida = np.zeros_like(m)
    for f in np.where(m.any(axis=1))[0]:
        cols = np.flatnonzero(m[f])
        salida[f, cols[0]:cols[-1] + 1] = 1
    return salida * 255


def corregir_iluminacion(img: np.ndarray, borde: float = 0.06
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Aplana el fondo. Devuelve (imagen corregida, modelo de fondo), ambas float32.

    Supuesto: el envase no toca los bordes de la foto, así que un marco exterior
    del `borde` (6 %) es fondo puro. Sobre ese marco se ajusta, por mínimos
    cuadrados y por canal, una superficie cuadrática
        F(x, y) = c0 + c1·x + c2·y + c3·x² + c4·x·y + c5·y²
    que captura degradés de luz. Restarla deja el fondo casi constante, y así un
    solo umbral (o un solo modelo de color) vale para toda la foto.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = x / w, y / h
    bx, by = max(2, int(borde * w)), max(2, int(borde * h))
    marco = np.zeros((h, w), bool)
    marco[:by], marco[-by:], marco[:, :bx], marco[:, -bx:] = True, True, True, True
    X = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)
    img = img.astype(np.float32)
    fondo = np.empty_like(img)
    for c in range(img.shape[2]):
        coef, *_ = np.linalg.lstsq(X[marco], img[..., c][marco], rcond=None)
        fondo[..., c] = X @ coef
    media = fondo[marco].mean(axis=0)
    return np.clip(img - fondo + media, 0, 255), fondo


def _marco(h: int, w: int, borde: float = 0.06) -> np.ndarray:
    bx, by = max(2, int(borde * w)), max(2, int(borde * h))
    m = np.zeros((h, w), bool)
    m[:by], m[-by:], m[:, :bx], m[:, -bx:] = True, True, True, True
    return m


def mapa_distancia(img: np.ndarray) -> np.ndarray:
    """Distancia de color RGB de cada píxel al fondo corregido (float32).

    En RGB la mezcla de un píxel de borde es lineal: color = a·objeto + (1−a)·fondo,
    así que su distancia al fondo es a·|objeto − fondo|. Esa linealidad es la
    que permite ubicar el borde con precisión subpíxel (ver extraer_perfil).
    """
    corr, _ = corregir_iluminacion(img)
    marco = _marco(*corr.shape[:2])
    color_fondo = np.median(corr[marco], axis=0)
    return np.linalg.norm(corr - color_fondo, axis=-1).astype(np.float32)


def segmentar_otsu(img: np.ndarray) -> np.ndarray:
    """Máscara del envase (0/255) por umbral de Otsu sobre la distancia al fondo.

    Umbralizar la DISTANCIA de color al fondo (y no el gris) evita tener que
    adivinar si el envase es más claro o más oscuro que el fondo.

    Otsu supone un histograma bimodal. Si parte del envase tiene el mismo color
    que el fondo, el objeto deja de ser un solo modo y Otsu sube demasiado el
    umbral. Por eso se acota con el nivel de ruido del fondo (media + 6σ del
    marco): el contorno del envase queda cerrado y el relleno de huecos completa
    el interior.
    """
    d = cv2.GaussianBlur(mapa_distancia(img), (5, 5), 0)
    marco = _marco(*d.shape)
    cota = float(d[marco].mean() + 6 * d[marco].std())
    d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t_otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu = t_otsu / 255 * (d.max() - d.min()) + d.min()   # a unidades de distancia
    binaria = (d > min(t_otsu, cota)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, k)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k)
    return _limpiar(binaria)


def segmentar_grabcut(img: np.ndarray, rect: tuple[int, int, int, int] | None = None,
                      iteraciones: int = 5, lado_max: int = 500) -> np.ndarray:
    """Máscara del envase (0/255) por GrabCut sobre la imagen con la luz corregida.

    rect = (x, y, ancho, alto) que contiene al envase; por defecto, la imagen
    menos un margen del 4 %. GrabCut se ejecuta a resolución reducida
    (`lado_max`) por velocidad; la máscara se vuelve a la resolución original
    interpolando su versión suavizada, lo que da bordes con precisión subpíxel
    a escala reducida.
    """
    h, w = img.shape[:2]
    corr = corregir_iluminacion(img)[0].astype(np.uint8)
    f = min(1.0, lado_max / max(h, w))
    peq = cv2.resize(corr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else corr
    ph, pw = peq.shape[:2]
    if rect is None:
        mx, my = int(0.04 * pw), int(0.04 * ph)
        rect_p = (mx, my, pw - 2 * mx, ph - 2 * my)
    else:
        rect_p = tuple(int(round(v * f)) for v in rect)
    mascara = np.zeros((ph, pw), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    # GrabCut inicializa sus modelos de color con k-means aleatorio: sin fijar la
    # semilla, la misma foto da máscaras ligeramente distintas (±0.3 % de área).
    cv2.setRNGSeed(0)
    cv2.grabCut(peq, mascara, rect_p, bg, fg, iteraciones, cv2.GC_INIT_WITH_RECT)
    objeto = np.isin(mascara, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    if f < 1:
        objeto = cv2.resize(objeto, (w, h), interpolation=cv2.INTER_LINEAR)
    return _limpiar((objeto >= 0.5).astype(np.uint8) * 255)


SEGMENTADORES = {"otsu": segmentar_otsu, "grabcut": segmentar_grabcut}


# ─────────────────────────────────────────────────────────────── perfil ──
@dataclass(frozen=True)
class Perfil:
    """Perfil extraído de la foto, en píxeles y en centímetros."""
    z_px: np.ndarray            # altura sobre la base [px]
    r_px: np.ndarray            # radio (semiancho) [px]
    altura_px: float            # altura total de la silueta [px]
    inclinacion_grados: float   # inclinación del eje detectada y corregida
    asimetria: float            # |r_izq − r_der| / (r_izq + r_der), mediana [0..1]
    subpixel: float             # fracción de bordes refinados con precisión subpíxel
    mascara: np.ndarray         # máscara ya enderezada (0/255)
    px_por_cm: float
    z_cm: np.ndarray
    r_cm: np.ndarray
    # Geometría en píxeles (imagen ya enderezada), para corregir la perspectiva
    filas: np.ndarray           # índice de fila de cada muestra, de arriba abajo
    borde_izq: np.ndarray       # borde izquierdo subpíxel de cada fila [columna]
    borde_der: np.ndarray       # borde derecho subpíxel de cada fila [columna]
    tope_px: float              # borde superior subpíxel [fila]
    base_px: float              # borde inferior subpíxel [fila]
    rotacion: np.ndarray | None  # matriz 2×3 usada para enderezar (None si no se rotó)


def _bordes_por_fila(m: np.ndarray):
    filas = np.where(m.any(axis=1))[0]
    izq = np.array([np.argmax(m[f]) for f in filas], dtype=float)
    der = np.array([m.shape[1] - 1 - np.argmax(m[f, ::-1]) for f in filas], dtype=float)
    return filas, izq, der


def _inclinacion(m: np.ndarray) -> float:
    """Ángulo del eje [grados] por regresión robusta de los centros de fila."""
    filas, izq, der = _bordes_por_fila(m)
    centros = (izq + der) / 2
    usar = np.ones_like(centros, bool)
    for _ in range(3):                                   # descartar filas atípicas
        a, b = np.polyfit(filas[usar], centros[usar], 1)
        res = centros - (a * filas + b)
        mad = np.median(np.abs(res[usar])) + 1e-9
        usar = np.abs(res) < 3 * 1.4826 * mad + 0.5
    return math.degrees(math.atan(a))


def _cruce(perfil: np.ndarray, i0: int, hacia_afuera: int, nivel_fondo: float,
           min_salto: float, ventana: int = 12) -> float | None:
    """Posición subpíxel del borde cerca del índice i0 de un perfil 1-D.

    `hacia_afuera` = −1 si el fondo está en los índices menores, +1 si está en
    los mayores. `nivel_fondo` es el valor del fondo en el mapa de distancia
    (conocido: el mapa mide justamente la distancia al fondo).

    Se toma como nivel del objeto el máximo del perfil cerca del borde (del
    lado de adentro), y desde ahí se camina hacia afuera hasta que el perfil
    cae por debajo del punto medio entre ambos niveles: ahí la mezcla
    objeto/fondo del píxel es 50 %, es decir, el borde real. La ventana amplia
    tolera máscaras que sobresalen o se quedan cortas unos píxeles.
    """
    n = len(perfil)
    lo = i0 - hacia_afuera * ventana                      # extremo interior
    hi = i0 + hacia_afuera * ventana                      # extremo exterior
    if min(lo, hi) < 0 or max(lo, hi) >= n:
        return None
    paso = hacia_afuera
    interior = np.arange(lo, i0 + 4 * paso + paso, paso)  # de adentro hasta 4 px afuera
    pico = int(interior[np.argmax(perfil[interior])])
    nivel_obj = float(perfil[pico])
    if nivel_obj - nivel_fondo < min_salto:
        return None
    medio = (nivel_obj + nivel_fondo) / 2
    i = pico
    while i != hi:
        j = i + paso
        if perfil[i] >= medio > perfil[j]:
            t = (perfil[i] - medio) / (perfil[i] - perfil[j])
            return i + t * paso
        i = j
    return None


def extraer_perfil(mascara: np.ndarray, altura_cm: float,
                   distancia: np.ndarray | None = None) -> Perfil:
    """Máscara → perfil r(z) en cm, usando la altura real del envase como escala.

    Si se pasa `distancia` (salida de mapa_distancia), cada borde se refina con
    precisión subpíxel; si no, se usa el conteo de píxeles de la máscara.
    """
    m = (mascara > 0).astype(np.uint8)
    d = None if distancia is None else distancia.astype(np.float32)

    # 1) enderezar: rotar máscara (y mapa) para dejar el eje vertical
    ang = _inclinacion(m)
    R = None
    if abs(ang) > 0.05:
        ys, xs = np.nonzero(m)
        centro = (float(xs.mean()), float(ys.mean()))
        R = cv2.getRotationMatrix2D(centro, -ang, 1.0)
        tam = (m.shape[1], m.shape[0])
        m = (cv2.warpAffine(m.astype(np.float32), R, tam, flags=cv2.INTER_LINEAR) >= 0.5)
        m = (_limpiar(m.astype(np.uint8)) > 0).astype(np.uint8)
        if d is not None:
            d = cv2.warpAffine(d, R, tam, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

    filas, izq, der = _bordes_por_fila(m)
    eje = float(np.median((izq + der) / 2))

    # 2) bordes izquierdo y derecho de cada fila (subpíxel si hay mapa)
    #    Coordenadas: el índice i es el centro del píxel i; un borde en 211.3
    #    significa 0.3 px a la derecha del centro del píxel 211.
    bi, bd = izq - 0.5, der + 0.5                 # borde entero: a medio píxel
    refinados = 0
    if d is not None:
        dy = cv2.GaussianBlur(d, (1, 7), 0)       # suaviza el ruido solo en vertical
        marco = _marco(*d.shape)
        fondo = float(np.median(dy[marco]))
        # Salto mínimo para aceptar un borde: 5σ del ruido del perfil YA suavizado
        # (el mismo sobre el que se busca el cruce), no del mapa crudo.
        min_salto = float(5 * dy[marco].std() + 1e-6)
        for k, f in enumerate(filas):
            ci = _cruce(dy[f], int(izq[k]), -1, fondo, min_salto)
            cd = _cruce(dy[f], int(der[k]), +1, fondo, min_salto)
            if ci is not None:
                bi[k], refinados = ci, refinados + 1
            if cd is not None:
                bd[k], refinados = cd, refinados + 1
    r_px = (bd - bi) / 2
    r_izq, r_der = eje - bi, bd - eje
    asim = float(np.median(np.abs(r_izq - r_der) / np.maximum(r_izq + r_der, 1)))

    # 3) base y tope (subpíxel sobre las columnas centrales del eje)
    tope_e, base_e = filas.min() - 0.5, filas.max() + 0.5
    if d is not None:
        c = int(round(eje))
        ancho_col = max(1, int(0.3 * min(r_px[0], r_px[-1])))
        col = d[:, c - ancho_col:c + ancho_col + 1].mean(axis=1)
        t = _cruce(col, int(filas.min()), -1, fondo, min_salto)
        b = _cruce(col, int(filas.max()), +1, fondo, min_salto)
        tope_e = t if t is not None else tope_e
        base_e = b if b is not None else base_e
    altura_px = float(base_e - tope_e)

    geometria = (filas.copy(), bi.copy(), bd.copy(), float(tope_e), float(base_e), R)

    # 4) z desde la base; extender a los extremos 0 y H para integrar toda la altura
    z_px = (base_e - filas.astype(float))[::-1]
    r_px = r_px[::-1]
    dentro = (z_px > 0) & (z_px < altura_px)
    z_px, r_px = z_px[dentro], r_px[dentro]
    z_px = np.concatenate([[0.0], z_px, [altura_px]])
    r_px = np.concatenate([[r_px[0]], r_px, [r_px[-1]]])

    s = altura_px / altura_cm                     # 5) escala: px por cm
    frac = refinados / (2 * len(filas)) if d is not None else 0.0
    return Perfil(z_px, r_px, altura_px, ang, asim, frac, m * 255, s,
                  z_px / s, r_px / s, *geometria)


# ────────────────────────────────────────────────────────── perspectiva ──
@dataclass(frozen=True)
class CorreccionPerspectiva:
    z_cm: np.ndarray
    r_cm: np.ndarray
    distancia_cm: float          # distancia estimada de la cámara al eje del envase
    altura_camara_cm: float      # altura estimada de la cámara sobre la base


def focal_px_desde_35mm(focal_35mm: float, ancho_px: int, alto_px: int) -> float:
    """Focal en píxeles a partir de la focal equivalente de 35 mm (dato EXIF).

    La equivalencia de 35 mm se define por la diagonal (43.27 mm del negativo
    de 36×24 mm). Solo vale si la foto no fue recortada; redimensionarla sí vale.
    """
    return focal_35mm * math.hypot(ancho_px, alto_px) / 43.27


def corregir_perspectiva(perfil: Perfil, altura_cm: float, focal_px: float,
                         centro_px: tuple[float, float], pitch_grados: float = 0.0,
                         iteraciones: int = 5) -> CorreccionPerspectiva:
    """Reconstruye r(z) en cm invirtiendo la proyección de una cámara pinhole.

    Modelo: eje del envase vertical; centro óptico `centro_px` y focal
    `focal_px` conocidos; celular inclinado `pitch_grados` hacia abajo (0 =
    vertical; negativo = apunta hacia arriba). Cada rayo se rota por esa
    inclinación antes de hacer las cuentas.

    1. Distancia D. El tope y la base de la silueta son los bordes de la boca y
       de la base más CERCANOS a la cámara (o los más lejanos, si la cámara está
       por encima de la boca o por debajo de la base). Sus rayos y la altura real
       H dan dos ecuaciones:  H − hc = a_t·(D ± r_t)   y   −hc = a_b·(D ± r_b),
       con a = pendiente vertical del rayo. Restándolas se despeja D; la altura de
       la cámara hc se cancela y sale después.
    2. Perfil. El rayo de cada borde de la silueta es tangente al envase. Se
       toma su punto de MÁXIMO ACERCAMIENTO al eje: la distancia es el radio y la
       z de ese punto es la altura. Es exacto en paredes verticales; en una pared
       inclinada el punto de tangencia real se desplaza, pero comparado contra
       fuerza bruta el error es < 0.002 cm en r y < 0.03 cm en z (despreciable).
       Una versión con la pendiente de la pared resultó exacta en teoría pero
       peor en la práctica: la derivada numérica del perfil mete más ruido del
       que corrige.
    3. Arcos de los bordes. La boca y la base son círculos que en perspectiva se
       ven como arcos de elipse: si la cámara está por debajo de la boca, su punto
       más cercano aparece más arriba que sus extremos laterales. En esas filas la
       silueta no es la pared sino el círculo del borde. Se reconocen porque el
       punto de máximo acercamiento del rayo cae FUERA del envase (z > H o z < 0);
       para ellas se intersecta el rayo con el plano de la boca (z = H) o de la
       base (z = 0), lo que da directamente el radio del borde. Así esas filas,
       lejos de estorbar, miden los radios que necesita el paso 1.
    Los radios de boca y base del paso 1 salen del paso 3: se itera unas veces.
    """
    cx, cy = centro_px[0] - 0.5, centro_px[1] - 0.5     # a coordenadas de índice
    if perfil.rotacion is not None:
        cx, cy = perfil.rotacion @ np.array([cx, cy, 1.0])
    f = float(focal_px)
    th = math.radians(pitch_grados)
    ct, st = math.cos(th), math.sin(th)

    def rayo(u, v):
        """Pixel → rayo en el mundo, normalizado a avance unitario hacia el envase.

        Devuelve (x, a): desplazamiento lateral y subida por cm de profundidad.
        """
        xc = (np.asarray(u, float) - cx) / f
        zc = -(np.asarray(v, float) - cy) / f
        dy = ct + st * zc                                 # rotación por la inclinación
        dz = -st + ct * zc
        return xc / dy, dz / dy

    filas = perfil.filas.astype(float)
    u_eje = float(np.median((perfil.borde_izq + perfil.borde_der) / 2))
    x_eje, _ = rayo(u_eje, cy)
    _, a_t = rayo(u_eje, perfil.tope_px)
    _, a_b = rayo(u_eje, perfil.base_px)

    # arranque: radios de boca y base del perfil sin corregir (mediana del 2 %
    # extremo: la última fila suele estar cubierta solo en parte); paredes verticales
    zc, rc = perfil.z_cm, perfil.r_cm
    r_t = float(np.median(rc[zc > altura_cm * 0.98]))
    r_b = float(np.median(rc[zc < altura_cm * 0.02]))
    sig_t, sig_b = -1.0, -1.0                            # borde cercano (−) o lejano (+)
    D_prev = None
    for _ in range(iteraciones):
        D = (altura_cm - a_t * sig_t * r_t + a_b * sig_b * r_b) / (a_t - a_b)
        if not np.isfinite(D) or D <= max(r_t, r_b):    # salvaguarda: no divergir
            if D_prev is None:
                raise ValueError("No se pudo estimar la distancia de la cámara.")
            D = D_prev
        D_prev = D
        hc = -a_b * (D + sig_b * r_b)
        sig_t = -1.0 if hc < altura_cm else 1.0
        sig_b = -1.0 if hc > 0 else 1.0
        X0 = x_eje * D                                   # desplazamiento lateral del eje
        radios, alturas, en_boca, en_base = [], [], [], []
        for borde in (perfil.borde_izq, perfil.borde_der):
            x, a = rayo(borde, filas)
            t = (x * X0 + D) / (x * x + 1)               # máximo acercamiento al eje
            radios.append(np.hypot(t * x - X0, t - D))
            alturas.append(hc + a * t)
            en_boca.append(_corte_plano(x, a, altura_cm, hc, D, X0))
            en_base.append(_corte_plano(x, a, 0.0, hc, D, X0))
        r = (radios[0] + radios[1]) / 2
        z = (alturas[0] + alturas[1]) / 2
        # filas cuyo punto de máximo acercamiento cae fuera del envase: son arco
        arco_t, arco_b = z >= altura_cm, z <= 0
        pared = ~arco_t & ~arco_b
        r_t = _radio_borde(en_boca, arco_t, altura_cm, hc, D, z, r, pared, r_t)
        r_b = _radio_borde(en_base, arco_b, 0.0, hc, D, z, r, pared, r_b)

    # el perfil es la pared; se cierra con los radios de boca y base medidos
    z, r = z[pared], r[pared]
    orden = np.argsort(z)
    z, r = z[orden], r[orden]
    dentro = (z > 0) & (z < altura_cm)
    z, r = z[dentro], r[dentro]
    z = np.concatenate([[0.0], z, [altura_cm]])
    r = np.concatenate([[r_b], r, [r_t]])
    return CorreccionPerspectiva(z, r, float(D), float(hc))


def _corte_plano(x: np.ndarray, a: np.ndarray, z_plano: float, hc: float, D: float,
                 X0: float) -> np.ndarray:
    """Distancia al eje del punto donde cada rayo corta el plano z = z_plano.

    NaN si el corte cae fuera de la huella física del envase (p. ej. con la
    cámara a la altura de ese plano, donde el corte es degenerado: t ≈ 0).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plano - hc) / a
    rho = np.hypot(t * x - X0, t - D)
    return np.where(np.abs(t - D) < 0.5 * D, rho, np.nan)


def _radio_borde(cortes: list[np.ndarray], arco: np.ndarray, z_borde: float,
                 hc: float, D: float, z: np.ndarray, r: np.ndarray,
                 pared: np.ndarray, previo: float, umbral_arco_cm: float = 0.4) -> float:
    """Radio del borde (boca o base). Hay dos formas de medirlo y cada una falla
    donde la otra acierta; se elige por el alto del arco, ≈ |z_borde − hc|·r/D:

    • Arco grande (cámara cerca o lejos del plano del borde): la pared junto al
      borde queda oculta tras el arco o se ve muy oblicua (típico de un pie
      acampanado). Se usan las filas del arco, cortadas con el plano del borde.
    • Arco pequeño: la pared llega casi hasta el borde y se mide bien fila por
      fila; el arco, en cambio, son pocas filas casi horizontales cuyo ancho
      se sobrestima. Se usa la mediana de la pared en los 3 mm junto al borde.
    """
    alto_arco = abs(z_borde - hc) * previo / max(D - previo, 1e-6)
    if alto_arco > umbral_arco_cm and arco.sum() >= 3:
        valores = np.concatenate([c[arco] for c in cortes])
        valores = valores[np.isfinite(valores)]
        if valores.size >= 3:
            return float(np.median(valores))
    cerca = pared & (np.abs(z - z_borde) < 0.3)
    if cerca.sum() >= 3:
        return float(np.median(r[cerca]))
    return previo


def _cruce_local(fila: np.ndarray, i0: int, hacia_afuera: int) -> float | None:
    """Borde subpíxel con un modelo de mezcla LOCAL (sirve con cualquier fondo).

    `fila` es la fila de la imagen en color (ancho, 3). Se toma el color de afuera
    (4–10 px fuera del borde entero i0) y el de adentro (3–8 px dentro). Cada píxel
    cercano se proyecta sobre el segmento afuera→adentro: su posición α es la
    fracción del píxel cubierta por el objeto (0 fuera, 1 dentro). El borde real
    está donde α = 0.5. No depende de un color de fondo global, así que funciona
    delante de un marcador negro o de una hoja blanca por igual.
    """
    n = len(fila)
    afuera = [i0 + hacia_afuera * k for k in range(4, 11)]
    adentro = [i0 - hacia_afuera * k for k in range(3, 9)]
    if min(afuera + adentro) < 0 or max(afuera + adentro) >= n:
        return None
    c_f = np.median(fila[afuera], axis=0)
    c_o = np.median(fila[adentro], axis=0)
    eje = c_o - c_f
    norma2 = float(eje @ eje)
    if norma2 < 15.0 ** 2:                        # contraste insuficiente
        return None
    idx = np.arange(i0 - hacia_afuera * 4, i0 + hacia_afuera * 5, hacia_afuera)
    idx = idx[(idx >= 0) & (idx < n)]
    alfa = ((fila[idx] - c_f) @ eje) / norma2
    for a, b, va, vb in zip(idx, idx[1:], alfa, alfa[1:]):
        if va >= 0.5 > vb:
            return float(a + (va - 0.5) / (va - vb) * (b - a))
    return None


def bordes_por_fila(mascara: np.ndarray, distancia: np.ndarray | None = None,
                    img: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bordes izquierdo y derecho de cada fila de la silueta, SIN enderezar la imagen.

    Para la reconstrucción con la pose 3D de la cámara: cada borde es un rayo, así
    que no hace falta rotar nada. Con `distancia` (mapa_distancia) los bordes se
    refinan con precisión subpíxel respecto de un color de fondo global; con `img`
    (la foto en color) se usa el modelo de mezcla local, que sirve con cualquier
    fondo y tiene prioridad. Sin ninguno, los bordes quedan a medio píxel.
    Devuelve (filas, borde_izq, borde_der, fracción refinada).
    """
    m = (mascara > 0).astype(np.uint8)
    filas, izq, der = _bordes_por_fila(m)
    bi, bd = izq - 0.5, der + 0.5
    refinados = 0
    if img is not None:
        suave = cv2.GaussianBlur(img.astype(np.float32), (1, 5), 0)   # ruido, en vertical
        for k, f in enumerate(filas):
            ci = _cruce_local(suave[f], int(izq[k]), -1)
            cd = _cruce_local(suave[f], int(der[k]), +1)
            if ci is not None:
                bi[k], refinados = ci, refinados + 1
            if cd is not None:
                bd[k], refinados = cd, refinados + 1
    elif distancia is not None:
        dy = cv2.GaussianBlur(distancia.astype(np.float32), (1, 7), 0)
        marco = _marco(*dy.shape)
        fondo = float(np.median(dy[marco]))
        min_salto = float(5 * dy[marco].std() + 1e-6)
        for k, f in enumerate(filas):
            ci = _cruce(dy[f], int(izq[k]), -1, fondo, min_salto)
            cd = _cruce(dy[f], int(der[k]), +1, fondo, min_salto)
            if ci is not None:
                bi[k], refinados = ci, refinados + 1
            if cd is not None:
                bd[k], refinados = cd, refinados + 1
    return filas.astype(float), bi, bd, refinados / max(2 * len(filas), 1)
