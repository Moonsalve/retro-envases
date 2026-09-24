"""De la foto al volumen: la cadena completa en una función.

    foto → corrección de iluminación → segmentación (Otsu o GrabCut)
         → enderezado del eje → bordes subpíxel → escala por la altura real
         → remuestreo a malla uniforme → los cuatro métodos de integración
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .integracion import Resultado, calcular_todos, n_compatible, remuestrear
from .vision import SEGMENTADORES, Perfil, extraer_perfil, mapa_distancia


@dataclass(frozen=True)
class Estimacion:
    perfil: Perfil
    z_cm: np.ndarray            # malla uniforme usada para integrar
    r_cm: np.ndarray
    resultados: list[Resultado]
    segmentador: str

    def volumen(self, metodo: str = "Trapecio") -> float:
        return next(r.volumen for r in self.resultados if r.metodo.startswith(metodo))


def estimar_volumen(img: np.ndarray, altura_cm: float, segmentador: str = "otsu",
                    n: int | None = None, m_gauss: int = 5,
                    paneles_gauss: int = 6) -> Estimacion:
    """Estima el volumen del envase de la foto.

    altura_cm     altura real del envase, medida con regla (fija la escala).
    n             subintervalos de la malla; por defecto, uno por fila de píxeles,
                  redondeado a múltiplo de 6 (compatible con Simpson y 3/8).
    paneles_gauss Gauss se aplica por paneles iguales: sin detección de aristas,
                  partir en paneles evita el fallo de la regla global.
    """
    if segmentador not in SEGMENTADORES:
        raise ValueError(f"segmentador debe ser uno de {list(SEGMENTADORES)}")
    mascara = SEGMENTADORES[segmentador](img)
    perfil = extraer_perfil(mascara, altura_cm, mapa_distancia(img))
    n = n or n_compatible(len(perfil.z_cm) - 1)
    z, r = remuestrear(perfil.z_cm, perfil.r_cm, n)
    cortes = list(np.linspace(z[0], z[-1], paneles_gauss + 1)[1:-1])
    return Estimacion(perfil, z, r, calcular_todos(z, r, m_gauss, cortes), segmentador)
