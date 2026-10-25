"""De la foto al volumen con la hoja de referencia (marcadores ArUco).

    foto → marcadores → focal y pose de la cámara → rectángulo del envase
         → GrabCut → bordes por fila → perfil 3D (r, z) en cm
         → remuestreo → los cuatro métodos de integración
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .integracion import Resultado, calcular_todos, n_compatible, remuestrear
from .plantilla import CARTA, Hoja
from .vision import bordes_por_fila, mapa_distancia, segmentar_grabcut
from .vision_aruco import (Camara, Perfil3D, detectar, estimar_camara, rectangulo_envase,
                           reconstruir, segmentar_con_hoja)


@dataclass(frozen=True)
class EstimacionHoja:
    camara: Camara
    rect: tuple[int, int, int, int]
    mascara: np.ndarray
    perfil: Perfil3D
    subpixel: float
    z_cm: np.ndarray
    r_cm: np.ndarray
    resultados: list[Resultado]

    def volumen(self, metodo: str = "Trapecio") -> float:
        return next(r.volumen for r in self.resultados if r.metodo.startswith(metodo))


def estimar_volumen_hoja(img: np.ndarray, hoja: Hoja = CARTA,
                         lado_medido_mm: float | None = None,
                         altura_cm: float | None = None,
                         focal_px_inicial: float | None = None,
                         rect: tuple[int, int, int, int] | None = None,
                         n: int | None = None, m_gauss: int = 5,
                         paneles_gauss: int = 6) -> EstimacionHoja:
    """Volumen del envase parado sobre la hoja de referencia.

    lado_medido_mm  lado real del cuadrado negro en la hoja impresa (corrige la escala
                    de la impresora). Es la referencia de escala de toda la medición.
    altura_cm       opcional: si se da, cierra el perfil en esa altura; si no, la
                    altura se estima de la foto. No fija la escala.
    rect            opcional: si no se da, sale de proyectar un cilindro sobre el eje.
    """
    cam = estimar_camara(img.shape, detectar(img), hoja, lado_medido_mm, focal_px_inicial)
    if rect is None:      # lo normal: la geometría inicia la segmentación
        rect = rectangulo_envase(cam, img.shape, altura_cm=altura_cm)
        mascara = segmentar_con_hoja(img, cam, altura_cm=altura_cm)
    else:                 # el usuario dibujó un rectángulo a mano
        mascara = segmentar_grabcut(img, rect=rect)
    filas, bi, bd, frac = bordes_por_fila(mascara, img=img)
    perfil = reconstruir(cam, filas, bi, bd, altura_cm)
    n = n or n_compatible(len(perfil.z_cm) - 1)
    z, r = remuestrear(perfil.z_cm, perfil.r_cm, n)
    cortes = list(np.linspace(z[0], z[-1], paneles_gauss + 1)[1:-1])
    return EstimacionHoja(cam, rect, mascara, perfil, frac, z, r,
                          calcular_todos(z, r, m_gauss, cortes))
