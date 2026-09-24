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
                  z_px / s, r_px / s)
