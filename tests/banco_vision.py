"""Banco de pruebas del pipeline de visión con fotos sintéticas de volumen exacto.

Ejecutar desde la raíz del repo:   python -m tests.banco_vision
"""
import time
from dataclasses import replace

import numpy as np

from retroenvases.pipeline import estimar_volumen
from retroenvases.sintetico import BOTELLA_600, LATA_330, VASO_CONICO, Escena, generar

ENVASES = {"botella 600": BOTELLA_600, "lata 330": LATA_330, "vaso cónico": VASO_CONICO}
BASE = Escena()
ESCENAS = {
    "oscuro sobre claro":   BASE,
    "claro sobre claro":    replace(BASE, color_envase=(200, 205, 210)),
    "degradé fuerte":       replace(BASE, degrade_fondo=0.45),
    "inclinado 3°":         replace(BASE, inclinacion_grados=3.0),
    "inclinado −6°":        replace(BASE, inclinacion_grados=-6.0),
    "ruido alto":           replace(BASE, ruido=10.0),
    "desenfoque fuerte":    replace(BASE, desenfoque_px=2.5),
    "baja resolución":      replace(BASE, px_por_cm=20.0),
    "oscuro sobre oscuro":  replace(BASE, color_envase=(35, 40, 45), color_fondo=(75, 78, 80)),
    "combo difícil":        replace(BASE, color_envase=(200, 205, 210), degrade_fondo=0.4,
                                    inclinacion_grados=3.0, ruido=6.0),
}
SEMILLAS = (0, 1, 2)


def main() -> None:
    filas = []
    for n_env, tramos in ENVASES.items():
        for n_esc, esc in ESCENAS.items():
            for sem in SEMILLAS:
                img, verdad = generar(tramos, replace(esc, semilla=sem))
                for seg in ("otsu", "grabcut"):
                    t0 = time.perf_counter()
                    try:
                        est = estimar_volumen(img, verdad["altura_cm"], seg)
                        err = {r.metodo.split()[0]: 100 * (r.volumen - verdad["volumen_cm3"])
                               / verdad["volumen_cm3"] for r in est.resultados}
                    except Exception as e:           # un fallo cuenta como error total
                        err = {"Trapecio": float("nan")}
                        print(f"  FALLO {n_env} / {n_esc} / {seg}: {e}")
                    filas.append((n_env, n_esc, sem, seg, err,
                                  1000 * (time.perf_counter() - t0)))

    print(f"\n{'escena':<22}{'Otsu (trapecio)':>22}{'GrabCut (trapecio)':>24}")
    print(f"{'':<22}{'medio  |peor|':>22}{'medio  |peor|':>24}")
    print("-" * 68)
    for n_esc in ESCENAS:
        cel = []
        for seg in ("otsu", "grabcut"):
            e = np.array([f[4]["Trapecio"] for f in filas if f[1] == n_esc and f[3] == seg])
            cel.append(f"{np.nanmean(e):+7.2f} % {np.nanmax(np.abs(e)):6.2f} %")
        print(f"{n_esc:<22}{cel[0]:>22}{cel[1]:>24}")

    print("\nPor envase (todas las escenas):")
    for n_env in ENVASES:
        for seg in ("otsu", "grabcut"):
            e = np.array([f[4]["Trapecio"] for f in filas if f[0] == n_env and f[3] == seg])
            print(f"  {n_env:<12} {seg:<8} medio {np.nanmean(e):+6.2f} %   "
                  f"|peor| {np.nanmax(np.abs(e)):5.2f} %")

    print("\nResumen por segmentador y método (todas las corridas):")
    for seg in ("otsu", "grabcut"):
        sub = [f for f in filas if f[3] == seg]
        t = np.mean([f[5] for f in sub])
        for met in ("Trapecio", "Simpson", "Newton-Cotes", "Gauss"):
            e = np.array([f[4].get(met, np.nan) for f in sub])
            dentro = np.mean(np.abs(e) < 5) * 100
            print(f"  {seg:<8} {met:<13} |error| medio {np.nanmean(np.abs(e)):5.2f} %   "
                  f"peor {np.nanmax(np.abs(e)):6.2f} %   dentro del 5 %: {dentro:5.1f} %")
        print(f"  {seg:<8} tiempo medio por foto: {t:.0f} ms\n")


if __name__ == "__main__":
    main()
