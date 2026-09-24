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

    # sombreado cilíndrico: 1 en el centro, (1 - sombreado) en los bordes
    xs1 = np.arange(ancho) + 0.5
    ys1 = np.arange(alto) + 0.5
    r1 = radio(tramos, (base_px - ys1) / s) * s
    u = np.abs(xs1[None, :] - cx) / np.maximum(r1[:, None], 1e-6)
    luz = 1 - escena.sombreado * np.clip(u, 0, 1) ** 2

    # --- fondo: degradé + textura de baja frecuencia ------------------------
    grad = 1 - escena.degrade_fondo * (ys1[:, None] / alto) * np.ones((1, ancho))
    tex = cv2.GaussianBlur(rng.normal(0, 1, (alto, ancho)).astype(np.float32),
                           (0, 0), 25) * escena.textura_fondo * 25
    fondo = np.stack([np.clip(c * grad + tex, 0, 255) for c in escena.color_fondo], -1)
    envase = np.stack([c * luz for c in escena.color_envase], -1)
    img = alfa[..., None] * envase + (1 - alfa[..., None]) * fondo

    # --- inclinación de la cámara respecto al eje --------------------------
    if escena.inclinacion_grados:
        M = cv2.getRotationMatrix2D((cx, alto / 2), escena.inclinacion_grados, 1.0)
        img = cv2.warpAffine(img.astype(np.float32), M, (ancho, alto),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- óptica y sensor -----------------------------------------------------
    if escena.desenfoque_px:
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), escena.desenfoque_px)
    img = img + rng.normal(0, escena.ruido, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    verdad = {
        "volumen_cm3": volumen_exacto(tramos),
        "altura_cm": altura_cm,
        "px_por_cm": s,
        "cortes_cm": [t[1] for t in tramos[:-1]],
    }
    return img, verdad
