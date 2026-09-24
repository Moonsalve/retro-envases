"""Los cuatro métodos de integración numérica del enunciado.

Todas las funciones reciben el perfil como tabla (z, r) en unidades físicas y
devuelven I = ∫ r(z)² dz. El volumen es V = π·I.

Las fórmulas son las mismas del notebook de la Fase 1, donde se validaron contra
un volumen exacto (ver tests/verificar_formulas.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import interpolate


class CondicionNoSatisfecha(Exception):
    """Los datos no cumplen los requisitos del método. Error esperado, no un bug."""


def _preparar(z: np.ndarray, r: np.ndarray, requiere: int, nombre: str
              ) -> tuple[int, float, np.ndarray]:
    """Valida malla uniforme y la condición sobre n. Devuelve (n, Δz, f = r²)."""
    z = np.asarray(z, dtype=float)
    dz = np.diff(z)
    if len(z) < 2 or not np.allclose(dz, dz[0]):
        raise CondicionNoSatisfecha(
            f"{nombre}: la malla no es uniforme. Remuestrea el perfil con "
            f"remuestrear() antes de integrar.")
    n = len(z) - 1
    if n % requiere:
        raise CondicionNoSatisfecha(
            f"{nombre} requiere n múltiplo de {requiere}, y n = {n}. "
            f"Usa n = {n - n % requiere} o n = {n + requiere - n % requiere}.")
    return n, float(dz[0]), np.asarray(r, dtype=float) ** 2


def trapecio(z: np.ndarray, r: np.ndarray) -> float:
    """Regla del trapecio compuesta. Pesos [1, 2, ..., 2, 1] · Δz/2. Cualquier n."""
    n, dz, f = _preparar(z, r, 1, "Trapecio")
    w = np.full(n + 1, 2.0)
    w[0] = w[-1] = 1.0
    return float(dz / 2 * w @ f)


def simpson(z: np.ndarray, r: np.ndarray) -> float:
    """Regla de Simpson 1/3 compuesta. Pesos [1, 4, 2, 4, ..., 4, 1] · Δz/3. n par."""
    n, dz, f = _preparar(z, r, 2, "Simpson")
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(dz / 3 * w @ f)


def newton_cotes(z: np.ndarray, r: np.ndarray, grado: int = 3) -> float:
    """Newton-Cotes cerrada: grado=3 → regla 3/8 (n múltiplo de 3); grado=4 → Boole."""
    bloques = {
        3: (np.array([1., 3., 3., 1.]), 3 / 8, "Newton-Cotes 3/8"),
        4: (np.array([7., 32., 12., 32., 7.]), 2 / 45, "Newton-Cotes (Boole)"),
    }
    if grado not in bloques:
        raise CondicionNoSatisfecha(f"grado={grado} no soportado; usa 3 o 4.")
    pesos, coef, nombre = bloques[grado]
    n, dz, f = _preparar(z, r, grado, nombre)
    w = np.zeros(n + 1)
    for i in range(0, n, grado):
        w[i:i + grado + 1] += pesos
    return float(coef * dz * w @ f)


def gauss(z: np.ndarray, r: np.ndarray, m: int = 5,
          cortes: list[float] | None = None) -> float:
    """Gauss-Legendre de m nodos sobre un spline cúbico de los datos.

    cortes=None → una sola cuadratura en [a, b]; cortes=[...] → una por tramo.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if len(z) < 4:
        raise CondicionNoSatisfecha("Gauss (spline cúbico) requiere al menos 4 puntos.")
    spl = interpolate.InterpolatedUnivariateSpline(z, r, k=3)
    x, w = np.polynomial.legendre.leggauss(m)
    bordes = [z[0], z[-1]] if not cortes else [z[0], *cortes, z[-1]]
    total = 0.0
    for izq, der in zip(bordes, bordes[1:]):
        semi, centro = (der - izq) / 2, (der + izq) / 2
        total += semi * w @ spl(semi * x + centro) ** 2
    return float(total)


def remuestrear(z: np.ndarray, r: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Lleva un perfil cualquiera a una malla uniforme de n subintervalos.

    Los datos de una foto no vienen con n múltiplo de 6 (n lo fija la altura en
    píxeles). Se interpola LINEALMENTE: no inventa oscilaciones cerca de las
    aristas del envase, a diferencia de un spline.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    zu = np.linspace(z[0], z[-1], n + 1)
    return zu, np.interp(zu, z, r)


def n_compatible(n_deseado: int, multiplo: int = 6) -> int:
    """Múltiplo de `multiplo` más cercano a n_deseado (mínimo `multiplo`)."""
    return max(multiplo, int(round(n_deseado / multiplo)) * multiplo)


@dataclass(frozen=True)
class Resultado:
    metodo: str
    n: int
    evaluaciones: int
    delta_z: float | None
    integral: float          # I = ∫ r² dz   [cm³]
    volumen: float           # V = π·I       [cm³]


def calcular_todos(z: np.ndarray, r: np.ndarray, m_gauss: int = 5,
                   cortes: list[float] | None = None) -> list[Resultado]:
    """Aplica los cuatro métodos a la misma tabla. Omite los que no apliquen."""
    n = len(z) - 1
    dz = float(z[1] - z[0])
    tramos_gauss = (len(cortes) + 1) if cortes else 1
    metodos = [
        ("Trapecio", lambda: trapecio(z, r), n + 1, dz),
        ("Simpson", lambda: simpson(z, r), n + 1, dz),
        ("Newton-Cotes 3/8", lambda: newton_cotes(z, r, 3), n + 1, dz),
        (f"Gauss m={m_gauss}", lambda: gauss(z, r, m_gauss, cortes),
         m_gauss * tramos_gauss, None),
    ]
    salida = []
    for nombre, fn, evals, d in metodos:
        try:
            integral = fn()
        except CondicionNoSatisfecha:
            continue
        salida.append(Resultado(nombre, n, evals, d, integral, math.pi * integral))
    return salida
