"""La hoja de referencia: marcadores ArUco y un círculo donde se para el envase.

Es la ÚNICA definición de la geometría de la hoja. La usan:
  • el PDF que se imprime (pdf_plantilla),
  • la detección y la estimación de la pose de la cámara (vision_aruco),
  • el renderizador de fotos sintéticas (sintetico).
Así no pueden desincronizarse.

Sistema de coordenadas de la hoja (mm): origen en el centro del círculo, X hacia
la derecha, Y hacia el fondo (alejándose de la cámara), Z hacia arriba. El eje
del envase es la vertical que pasa por el origen.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np

DICCIONARIO = cv2.aruco.DICT_4X4_50
LADO_MARCADOR_MM = 60.0          # lado nominal del cuadrado negro de cada marcador


@dataclass(frozen=True)
class Hoja:
    nombre: str
    ancho_mm: float
    alto_mm: float
    margen_mm: float = 12.0
    lado_mm: float = LADO_MARCADOR_MM
    ids: tuple[int, int, int, int] = (0, 1, 2, 3)   # cerca-izq, cerca-der, fondo-der, fondo-izq
    circulos_mm: tuple[float, ...] = (40.0, 60.0, 80.0, 100.0)   # diámetros guía

    def centros(self) -> dict[int, tuple[float, float]]:
        """Centro de cada marcador en coordenadas de la hoja [mm]."""
        cx = self.ancho_mm / 2 - self.margen_mm - self.lado_mm / 2
        cy = self.alto_mm / 2 - self.margen_mm - self.lado_mm / 2
        cerca_izq, cerca_der, fondo_der, fondo_izq = self.ids
        return {cerca_izq: (-cx, -cy), cerca_der: (cx, -cy),
                fondo_der: (cx, cy), fondo_izq: (-cx, cy)}

    def esquinas(self, escala: float = 1.0) -> dict[int, np.ndarray]:
        """Las 4 esquinas de cada marcador, (4, 3) en mm, en el orden de OpenCV
        (sup-izq, sup-der, inf-der, inf-izq de la imagen del marcador).

        La imagen del marcador se imprime con su "arriba" hacia el fondo de la hoja
        (+Y). `escala` corrige una impresión agrandada o achicada: lado medido / 60.
        """
        s = self.lado_mm / 2
        local = np.array([[-s, s], [s, s], [s, -s], [-s, -s]])
        salida = {}
        for i, (x, y) in self.centros().items():
            xy = (local + [x, y]) * escala
            salida[i] = np.column_stack([xy, np.zeros(4)])
        return salida


CARTA = Hoja("Carta", 215.9, 279.4)
A4 = Hoja("A4", 210.0, 297.0)
HOJAS = {"Carta": CARTA, "A4": A4}


def celdas_marcador(i: int) -> np.ndarray:
    """Matriz binaria (6×6, con el borde negro) del marcador ArUco `i`. 1 = negro."""
    d = cv2.aruco.getPredefinedDictionary(DICCIONARIO)
    img = cv2.aruco.generateImageMarker(d, i, 6)      # 4×4 bits + borde = 6 celdas
    return (img < 128).astype(np.uint8)


def textura(hoja: Hoja, px_por_mm: float = 4.0) -> np.ndarray:
    """La hoja como imagen en escala de grises (para el renderizador sintético).

    La fila 0 es el fondo de la hoja (+Y): la imagen se ve como la hoja impresa
    mirada desde arriba con el borde cercano a la cámara abajo.
    """
    w, h = int(round(hoja.ancho_mm * px_por_mm)), int(round(hoja.alto_mm * px_por_mm))
    img = np.full((h, w), 245, np.uint8)
    a_px = lambda x, y: (int(round((x + hoja.ancho_mm / 2) * px_por_mm)),
                         int(round((hoja.alto_mm / 2 - y) * px_por_mm)))
    for i, (cx, cy) in hoja.centros().items():
        cel = celdas_marcador(i)
        lado = hoja.lado_mm / 6
        for fila in range(6):
            for col in range(6):
                if cel[fila, col]:
                    x0 = cx - hoja.lado_mm / 2 + col * lado
                    y0 = cy + hoja.lado_mm / 2 - fila * lado
                    p0, p1 = a_px(x0, y0), a_px(x0 + lado, y0 - lado)
                    cv2.rectangle(img, p0, (p1[0] - 1, p1[1] - 1), 0, -1)
    c = a_px(0, 0)
    for dmm in hoja.circulos_mm:
        cv2.circle(img, c, int(round(dmm / 2 * px_por_mm)), 90, max(1, int(px_por_mm * 0.6)),
                   cv2.LINE_AA)
    largo = int(hoja.circulos_mm[-1] / 2 * px_por_mm)
    cv2.line(img, (c[0] - largo, c[1]), (c[0] + largo, c[1]), 90, 1, cv2.LINE_AA)
    cv2.line(img, (c[0], c[1] - largo), (c[0], c[1] + largo), 90, 1, cv2.LINE_AA)
    return img


def pdf_plantilla(hoja: Hoja = CARTA) -> bytes:
    """PDF imprimible a tamaño real. Los marcadores se dibujan como vectores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    pulg = 25.4
    fig = plt.figure(figsize=(hoja.ancho_mm / pulg, hoja.alto_mm / pulg))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-hoja.ancho_mm / 2, hoja.ancho_mm / 2)
    ax.set_ylim(-hoja.alto_mm / 2, hoja.alto_mm / 2)
    ax.set_aspect("equal")
    ax.axis("off")

    lado = hoja.lado_mm / 6
    for i, (cx, cy) in hoja.centros().items():
        cel = celdas_marcador(i)
        for fila in range(6):
            for col in range(6):
                if cel[fila, col]:
                    ax.add_patch(Rectangle((cx - hoja.lado_mm / 2 + col * lado,
                                            cy + hoja.lado_mm / 2 - (fila + 1) * lado),
                                           lado, lado, color="black", lw=0))
        # etiqueta del lado interior del marcador, lejos de los bordes de la hoja
        hacia_centro = -1 if cy > 0 else 1
        ax.text(cx, cy + hacia_centro * (hoja.lado_mm / 2 + 2.5), f"ID {i}", ha="center",
                va="bottom" if hacia_centro > 0 else "top", fontsize=6, color="0.4")
    for dmm in hoja.circulos_mm:
        ax.add_patch(Circle((0, 0), dmm / 2, fill=False, lw=0.6, color="0.35"))
        ax.text(dmm / 2 * 0.71 + 1, dmm / 2 * 0.71 + 1, f"⌀{dmm:.0f}", fontsize=5, color="0.45")
    r = hoja.circulos_mm[-1] / 2
    ax.plot([-r, r], [0, 0], lw=0.4, color="0.35")
    ax.plot([0, 0], [-r, r], lw=0.4, color="0.35")

    # regla de control de impresión: 100 mm con marcas cada cm
    y_r = -hoja.alto_mm / 2 + 6
    ax.plot([-50, 50], [y_r, y_r], color="black", lw=0.8)
    for k in range(11):
        alto = 2.5 if k % 5 == 0 else 1.5
        ax.plot([-50 + 10 * k, -50 + 10 * k], [y_r, y_r + alto], color="black", lw=0.6)
    ax.text(0, y_r + 3.5, "Esta regla debe medir 100 mm  ·  cada cuadrado negro, 60 mm",
            ha="center", fontsize=6.5)
    ax.text(0, hoja.alto_mm / 2 - 4,
            "Del píxel al volumen  ·  hoja de referencia  ·  imprimir al 100 % (sin «ajustar a la página»)",
            ha="center", va="top", fontsize=6.5)
    ax.text(0, -hoja.circulos_mm[-1] / 2 - 4,
            "Para el envase en el centro del círculo  ·  este borde hacia la cámara",
            ha="center", va="top", fontsize=6.5, color="0.3")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()
