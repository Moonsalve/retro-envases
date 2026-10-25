"""Fotos sintéticas de envases con volumen exacto conocido.

Sirven para validar el pipeline de visión (escala + contorno + integración)
contra una respuesta exacta, sin cámara ni mediciones físicas. Modelan una
proyección ortográfica: la silueta de un sólido de revolución visto de lado es
exactamente ±r(z).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# (z_inicio, z_fin, radio_inicio, radio_fin, nombre) en cm
Tramos = list[tuple[float, float, float, float, str]]

BOTELLA_600: Tramos = [
    (0.0, 1.0, 2.50, 3.40, "base (pie)"),
    (1.0, 15.0, 3.40, 3.40, "cuerpo cilíndrico"),
    (15.0, 18.5, 3.40, 1.35, "hombro"),
    (18.5, 20.0, 1.35, 1.35, "cuello"),
    (20.0, 21.0, 1.35, 1.45, "boca / rosca"),
]
LATA_330: Tramos = [
    (0.0, 0.6, 2.60, 3.30, "fondo"),
    (0.6, 11.0, 3.30, 3.30, "cuerpo"),
    (11.0, 11.6, 3.30, 2.70, "cuello"),
    (11.6, 12.2, 2.70, 2.70, "tapa"),
]
VASO_CONICO: Tramos = [
    (0.0, 12.0, 2.80, 4.20, "vaso"),
]


def volumen_exacto(tramos: Tramos) -> float:
    """Suma cerrada de troncos de cono [cm³]."""
    return sum(math.pi * (zf - zi) / 3 * (ri**2 + ri * rf + rf**2)
               for zi, zf, ri, rf, _ in tramos)


def radio(tramos: Tramos, z: np.ndarray) -> np.ndarray:
    """Radio r(z) [cm]; 0 fuera del envase."""
    z = np.asarray(z, dtype=float)
    r = np.zeros_like(z)
    for zi, zf, ri, rf, _ in tramos:
        m = (z >= zi) & (z <= zf)
        r[m] = ri + (rf - ri) * (z[m] - zi) / (zf - zi)
    return r


@dataclass(frozen=True)
class Escena:
    """Parámetros de la foto sintética."""
    px_por_cm: float = 45.0            # resolución: ~945 px para 21 cm
    margen_cm: float = 3.0
    color_envase: tuple[int, int, int] = (40, 90, 160)     # BGR
    color_fondo: tuple[int, int, int] = (225, 228, 232)    # BGR
    degrade_fondo: float = 0.25        # 0 = fondo plano
    textura_fondo: float = 6.0         # desviación del ruido de baja frecuencia
    ruido: float = 4.0                 # ruido de sensor (desv. estándar)
    desenfoque_px: float = 1.0         # sigma del desenfoque de la cámara
    inclinacion_grados: float = 0.0
    sombreado: float = 0.35            # oscurecimiento en los bordes (cilíndrico)
    semilla: int = 0


def generar(tramos: Tramos, escena: Escena = Escena()
            ) -> tuple[np.ndarray, dict]:
    """Renderiza la foto. Devuelve (imagen BGR uint8, verdad de referencia)."""
    rng = np.random.default_rng(escena.semilla)
    s = escena.px_por_cm
    altura_cm = tramos[-1][1] - tramos[0][0]
    rmax_cm = max(max(t[2], t[3]) for t in tramos)
    alto = int(round((altura_cm + 2 * escena.margen_cm) * s))
    ancho = int(round((2 * rmax_cm + 2 * escena.margen_cm) * s))
    ancho = max(ancho, int(alto * 0.6))            # proporción de foto razonable

    # --- silueta con antialiasing: supermuestreo 4× y promedio -------------
    k = 4
    ys = (np.arange(alto * k) + 0.5) / k           # centro de subpíxel, en px
    xs = (np.arange(ancho * k) + 0.5) / k
    base_px = alto - escena.margen_cm * s          # la base del envase, en px
    z_cm = (base_px - ys) / s                      # altura sobre la base, en cm
    r_px = radio(tramos, z_cm) * s
    cx = ancho / 2
    dx = xs[None, :] - cx
    dentro = np.abs(dx) <= r_px[:, None]
    alfa = dentro.reshape(alto, k, ancho, k).mean(axis=(1, 3)).astype(np.float32)

    img = _componer(alfa, escena, rng, rotar_centro=(cx, alto / 2))

    verdad = {
        "volumen_cm3": volumen_exacto(tramos),
        "altura_cm": altura_cm,
        "px_por_cm": s,
        "cortes_cm": [t[1] for t in tramos[:-1]],
    }
    return img, verdad


def _componer(alfa: np.ndarray, escena: Escena, rng: np.random.Generator,
              rotar_centro: tuple[float, float] | None = None) -> np.ndarray:
    """Pinta envase y fondo a partir del mapa de cobertura `alfa` (0..1)."""
    alto, ancho = alfa.shape
    # sombreado cilíndrico: 1 en el centro de cada fila, (1 − sombreado) en los bordes
    luz = np.ones_like(alfa)
    for f in np.where((alfa > 0.5).any(axis=1))[0]:
        cols = np.flatnonzero(alfa[f] > 0.5)
        c, sw = (cols[0] + cols[-1]) / 2, max((cols[-1] - cols[0]) / 2, 1.0)
        u = np.abs(np.arange(ancho) - c) / sw
        luz[f] = 1 - escena.sombreado * np.clip(u, 0, 1) ** 2
    ys1 = np.arange(alto) + 0.5
    grad = 1 - escena.degrade_fondo * (ys1[:, None] / alto) * np.ones((1, ancho))
    tex = cv2.GaussianBlur(rng.normal(0, 1, (alto, ancho)).astype(np.float32),
                           (0, 0), 25) * escena.textura_fondo * 25
    fondo = np.stack([np.clip(c * grad + tex, 0, 255) for c in escena.color_fondo], -1)
    envase = np.stack([c * luz for c in escena.color_envase], -1)
    img = alfa[..., None] * envase + (1 - alfa[..., None]) * fondo
    if escena.inclinacion_grados and rotar_centro is not None:
        M = cv2.getRotationMatrix2D(rotar_centro, escena.inclinacion_grados, 1.0)
        img = cv2.warpAffine(img.astype(np.float32), M, (ancho, alto),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if escena.desenfoque_px:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), escena.desenfoque_px)
    img = img + rng.normal(0, escena.ruido, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def generar_perspectiva(tramos: Tramos, escena: Escena = Escena(),
                        distancia_cm: float = 100.0,
                        altura_camara_cm: float | None = None,
                        pitch_grados: float = 0.0,
                        muestras: int = 160) -> tuple[np.ndarray, dict]:
    """Foto sintética con una cámara estenopeica (pinhole) REAL.

    Geometría (cm): eje del envase = eje Z, base en z = 0. Cámara en
    (0, −distancia_cm, altura_camara_cm) mirando hacia +Y, girada `pitch_grados`
    (positivo = apunta hacia abajo). Por defecto a media altura del envase.

    La focal se elige para que un objeto EN EL EJE mida escena.px_por_cm píxeles
    por cm: equivale a alejarse usando zoom, manteniendo la resolución.

    Para cada píxel se lanza un rayo y se calcula g = min_t [ρ(t) − r(z(t))], la
    distancia (con signo) del rayo a la superficie. g ≤ 0 → el rayo toca el envase.
    La cobertura del píxel se obtiene de g con una rampa del tamaño del píxel a
    esa profundidad, lo que da bordes con antialiasing subpíxel.
    """
    rng = np.random.default_rng(escena.semilla)
    H = tramos[-1][1] - tramos[0][0]
    rmax = max(max(t[2], t[3]) for t in tramos)
    D = float(distancia_cm)
    hc = H / 2 if altura_camara_cm is None else float(altura_camara_cm)
    f = escena.px_por_cm * D                              # focal [px]
    th = math.radians(pitch_grados)
    ct, st = math.cos(th), math.sin(th)

    # tamaño de la imagen: proyectar la caja del envase y dejar margen
    esquinas = [(x, y, z) for x in (-rmax, rmax) for y in (-rmax, rmax) for z in (0, H)]
    us, vs = [], []
    for x, y, z in esquinas:
        # mundo → cámara (deshacer el pitch)
        dy, dz = y + D, z - hc
        yc, zc = ct * dy - st * dz, st * dy + ct * dz
        us.append(f * x / yc)
        vs.append(-f * zc / yc)
    # Como en una foto real, el centro óptico es el centro de la imagen: el marco
    # se hace simétrico alrededor de él (si la cámara está a la altura de la base,
    # el envase queda en la mitad superior de la foto).
    m = escena.margen_cm * escena.px_por_cm
    semi_u = max(abs(u) for u in us) + m
    semi_v = max(abs(v) for v in vs) + m
    ancho = 2 * int(math.ceil(max(semi_u, semi_v * 0.6)))
    alto = 2 * int(math.ceil(semi_v))
    cu, cv = ancho / 2, alto / 2

    alfa = np.zeros((alto, ancho), np.float32)
    uu = (np.arange(ancho) + 0.5 - cu) / f
    tt = np.linspace(0.0, 1.0, muestras)
    for f0 in range(0, alto, 48):                          # por bloques de filas
        vv = (np.arange(f0, min(f0 + 48, alto)) + 0.5 - cv) / f
        X, Zc = np.meshgrid(uu, -vv)                      # dirección en cámara (x, 1, z)
        # rotar al mundo por el pitch
        dy = ct * 1.0 + st * Zc
        dz = -st * 1.0 + ct * Zc
        dx = X
        # recorrer solo el tramo del rayo donde |y| ≤ rmax
        ta = (D - rmax * 1.05) / dy
        tb = (D + rmax * 1.05) / dy
        T = ta[..., None] + (tb - ta)[..., None] * tt       # (filas, cols, muestras)
        px = T * dx[..., None]
        py = -D + T * dy[..., None]
        pz = hc + T * dz[..., None]
        rho = np.sqrt(px * px + py * py)
        # Distancia con signo al sólido = intersección de la pared lateral y las
        # dos tapas (base z=0 y boca z=H): el máximo de las tres restricciones.
        g = np.min(np.maximum(np.maximum(rho - radio(tramos, pz), pz - H), -pz),
                   axis=-1)
        prof = ta * 1.0 + rmax                              # ≈ profundidad del borde
        tam_px = prof / f                                   # tamaño del píxel en cm
        alfa[f0:f0 + vv.size] = np.clip(0.5 - g / tam_px, 0, 1)

    img = _componer(alfa, escena, rng)
    verdad = {
        "volumen_cm3": volumen_exacto(tramos),
        "altura_cm": H,
        "px_por_cm": escena.px_por_cm,
        "distancia_cm": D,
        "altura_camara_cm": hc,
        "pitch_grados": pitch_grados,
        "focal_px": f,
        "centro_px": (cu, cv),
    }
    return img, verdad


def generar_con_hoja(tramos: Tramos, hoja=None, *, distancia_cm: float = 50.0,
                     altura_camara_cm: float = 25.0, mirar_a_cm: float | None = None,
                     guinada_grados: float = 0.0, focal_35mm: float = 26.0,
                     tam: tuple[int, int] = (1200, 1600), escala_impresion: float = 1.0,
                     desplazamiento_mm: tuple[float, float] = (0.0, 0.0),
                     color_envase: tuple[int, int, int] = (40, 90, 160),
                     ruido: float = 3.0, semilla: int = 0,
                     muestras: int = 120) -> tuple[np.ndarray, dict]:
    """Foto sintética de la escena del protocolo: la hoja de referencia sobre la mesa
    y el envase parado en el centro del círculo, con una cámara pinhole real.

    Coordenadas de la hoja (cm): eje del envase vertical por (desplazamiento), base
    en z = 0. La cámara está a `distancia_cm` en horizontal, a `altura_camara_cm`
    sobre la mesa, girada `guinada_grados` alrededor del envase, y apunta a
    `mirar_a_cm` (por defecto, la mitad del envase). Se renderiza al doble de
    resolución y se reduce, para que los bordes tengan antialiasing realista.
    """
    from .plantilla import CARTA, textura
    hoja = hoja or CARTA
    rng = np.random.default_rng(semilla)
    H = tramos[-1][1] - tramos[0][0]
    rmax = max(max(t[2], t[3]) for t in tramos)
    ox, oy = desplazamiento_mm[0] / 10, desplazamiento_mm[1] / 10      # eje real [cm]
    W, Hh = tam
    ss = 2                                                   # supermuestreo
    Ws, Hs = W * ss, Hh * ss
    f = focal_35mm * math.hypot(W, Hh) / 43.27 * ss

    # ---- cámara (look-at) -----------------------------------------------
    g = math.radians(guinada_grados)
    C = np.array([distancia_cm * math.sin(g), -distancia_cm * math.cos(g), altura_camara_cm])
    T = np.array([0.0, 0.0, H / 2 if mirar_a_cm is None else mirar_a_cm])
    fw = (T - C) / np.linalg.norm(T - C)
    xc = np.cross(fw, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(fw, xc)
    Rm = np.vstack([xc, yc, fw])                             # mundo → cámara
    cu, cv = Ws / 2, Hs / 2

    # ---- rayo de cada píxel ------------------------------------------------
    uu, vv = np.meshgrid((np.arange(Ws) + 0.5 - cu) / f, (np.arange(Hs) + 0.5 - cv) / f)
    d = np.stack([uu, vv, np.ones_like(uu)], -1) @ Rm         # (Hs, Ws, 3) en el mundo

    # ---- fondo: mesa y hoja (plano z = 0) o pared ---------------------------
    ppm = 6.0                                                # textura: px por mm
    tex = textura(hoja, ppm).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        tz = -C[2] / d[..., 2]
    X = (C[0] + tz * d[..., 0]) * 10 / escala_impresion       # mm de la hoja
    Y = (C[1] + tz * d[..., 1]) * 10 / escala_impresion
    mapx = ((X + hoja.ancho_mm / 2) * ppm - 0.5).astype(np.float32)
    mapy = ((hoja.alto_mm / 2 - Y) * ppm - 0.5).astype(np.float32)
    suelo = (d[..., 2] < 0) & np.isfinite(tz)
    mapx[~suelo] = -1e6
    mesa = 150.0
    gris = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=mesa)
    pared = 205 + 25 * vv / vv.max()
    gris = np.where(suelo, gris, pared).astype(np.float32)
    fondo = np.stack([gris * 0.98, gris, gris * 1.02], -1)

    # ---- envase: distancia con signo a lo largo del rayo ---------------------
    alfa = np.zeros((Hs, Ws), np.float32)
    px, py = C[0] - ox, C[1] - oy
    a_ = d[..., 0] ** 2 + d[..., 1] ** 2
    b_ = 2 * (px * d[..., 0] + py * d[..., 1])
    disc = b_ * b_ - 4 * a_ * (px * px + py * py - (rmax * 1.05) ** 2)
    candidatos = disc > 0
    filas_c = np.where(candidatos.any(axis=1))[0]
    tt = np.linspace(0, 1, muestras)
    for f0 in range(filas_c.min(), filas_c.max() + 1, 64):
        bl = slice(f0, min(f0 + 64, filas_c.max() + 1))
        ok = candidatos[bl]
        if not ok.any():
            continue
        dd = d[bl][ok]; aa = a_[bl][ok]; bb = b_[bl][ok]; sq = np.sqrt(disc[bl][ok])
        t0, t1 = (-bb - sq) / (2 * aa), (-bb + sq) / (2 * aa)
        Tt = t0[:, None] + (t1 - t0)[:, None] * tt
        PX = px + Tt * dd[:, 0:1]; PY = py + Tt * dd[:, 1:2]; PZ = C[2] + Tt * dd[:, 2:3]
        rho = np.hypot(PX, PY)
        gsd = np.min(np.maximum(np.maximum(rho - radio(tramos, PZ), PZ - H), -PZ), axis=1)
        prof = t0 * np.linalg.norm(dd, axis=1)
        a_bl = np.zeros(ok.shape, np.float32)
        a_bl[ok] = np.clip(0.5 - gsd / (prof / f), 0, 1)
        alfa[bl] = a_bl

    luz = np.ones_like(alfa)
    for fila in np.where((alfa > 0.5).any(axis=1))[0]:
        cols = np.flatnonzero(alfa[fila] > 0.5)
        c0, sw = (cols[0] + cols[-1]) / 2, max((cols[-1] - cols[0]) / 2, 1.0)
        luz[fila] = 1 - 0.35 * np.clip(np.abs(np.arange(Ws) - c0) / sw, 0, 1) ** 2
    envase = np.stack([c * luz for c in color_envase], -1)
    img = alfa[..., None] * envase + (1 - alfa[..., None]) * fondo
    img = cv2.resize(img.astype(np.float32), (W, Hh), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 0.6) + rng.normal(0, ruido, (Hh, W, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)

    verdad = {"volumen_cm3": volumen_exacto(tramos), "altura_cm": H,
              "focal_px": f / ss, "camara_cm": C, "eje_cm": (ox, oy),
              "inclinacion_grados": math.degrees(math.asin(-fw[2]))}
    return img, verdad
