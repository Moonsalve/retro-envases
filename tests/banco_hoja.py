"""Banco de la opción B: hoja con marcadores ArUco bajo el envase.

Fotos sintéticas con la escena completa (mesa, hoja, envase, cámara pinhole). El
pipeline NO conoce la focal ni la pose: las estima de los marcadores.
Ejecutar desde la raíz del repo:   python -m tests.banco_hoja
"""
import time

import numpy as np

from retroenvases.pipeline_hoja import estimar_volumen_hoja
from retroenvases.plantilla import CARTA
from retroenvases.sintetico import BOTELLA_600, LATA_330, VASO_CONICO, generar_con_hoja

ENVASES = {"botella 21 cm": BOTELLA_600, "lata 12 cm": LATA_330, "vaso 12 cm": VASO_CONICO}


def caso(tramos, medido=None, altura=False, **kw):
    img, v = generar_con_hoja(tramos, **kw)
    t0 = time.perf_counter()
    try:
        e = estimar_volumen_hoja(img, CARTA, lado_medido_mm=medido,
                                 altura_cm=v["altura_cm"] if altura else None)
    except Exception as ex:
        return {"err": np.nan, "fallo": str(ex)[:60]}
    V = v["volumen_cm3"]
    return {"err": 100 * (e.volumen() - V) / V,
            "focal": 100 * (e.camara.focal_px / v["focal_px"] - 1),
            "H": e.perfil.altura_estimada_cm - v["altura_cm"],
            "marc": len(e.camara.marcadores), "ms": 1000 * (time.perf_counter() - t0)}


def tabla(titulo, casos):
    print(f"\n{titulo}")
    print(f"   {'':<30}" + "".join(f"{n:>20}" for n in ENVASES))
    todos = []
    for et, kw in casos:
        celdas = []
        for t in ENVASES.values():
            r = caso(t, **kw)
            todos.append(r["err"])
            celdas.append(f"FALLO" if np.isnan(r["err"]) else
                          f"{r['err']:+6.2f}% (f{r['focal']:+.1f}%,{r['marc']}m)")
        print(f"   {et:<30}" + "".join(f"{c:>20}" for c in celdas))
    return todos


def main():
    errores = []
    errores += tabla("1) DISTANCIA (cámara a 30 cm de altura)",
                     [(f"{D} cm", dict(distancia_cm=D, altura_camara_cm=30)) for D in (35, 50, 70)])
    errores += tabla("2) ALTURA DE LA CÁMARA (a 50 cm)",
                     [(f"{h} cm", dict(distancia_cm=50, altura_camara_cm=h)) for h in (20, 30, 45)])
    errores += tabla("3) CÁMARA GIRADA ALREDEDOR DEL ENVASE",
                     [(f"{g:+d}°", dict(distancia_cm=50, altura_camara_cm=30, guinada_grados=g))
                      for g in (-15, 15)])
    errores += tabla("4) ENVASE MAL CENTRADO EN EL CÍRCULO",
                     [(f"{dx:+d}, {dy:+d} mm", dict(distancia_cm=50, altura_camara_cm=30,
                                                     desplazamiento_mm=(dx, dy)))
                      for dx, dy in ((5, 0), (0, 5), (-4, -4))])
    errores += tabla("5) FOCAL DE LA CÁMARA (el pipeline no la conoce)",
                     [(f"{f} mm", dict(distancia_cm=50, altura_camara_cm=30, focal_35mm=f))
                      for f in (24, 28, 35)])
    errores += tabla("6) HOJA IMPRESA AL 97 %",
                     [("sin corregir el lado", dict(distancia_cm=50, altura_camara_cm=30,
                                                    escala_impresion=0.97)),
                      ("lado medido = 58.2 mm", dict(distancia_cm=50, altura_camara_cm=30,
                                                     escala_impresion=0.97, medido=58.2))])
    errores += tabla("7) ALTURA MEDIDA DADA (en vez de estimada)",
                     [("con altura medida", dict(distancia_cm=50, altura_camara_cm=30, altura=True))])
    e = np.array([x for x in errores if np.isfinite(x)])
    print(f"\nRESUMEN: {len(e)}/{len(errores)} casos sin fallo · |error| medio {np.abs(e).mean():.2f} % · "
          f"peor {np.abs(e).max():.2f} % · dentro del 5 %: {100*np.mean(np.abs(e) < 5):.0f} %")


if __name__ == "__main__":
    main()
