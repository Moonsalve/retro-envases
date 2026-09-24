"""Efecto de la perspectiva de la cámara, y de su corrección, sobre el volumen.

Fotos renderizadas con una cámara pinhole real (sintetico.generar_perspectiva).
Ejecutar desde la raíz del repo:   python -m tests.banco_perspectiva
"""
from retroenvases.pipeline import estimar_volumen
from retroenvases.sintetico import BOTELLA_600, LATA_330, VASO_CONICO, Escena, generar_perspectiva

ENVASES = {"botella 21 cm": BOTELLA_600, "lata 12 cm": LATA_330, "vaso 12 cm": VASO_CONICO}


def _err(tramos, *, corregir=True, factor_focal=1.0, pitch_informado=None, **kw) -> float:
    img, v = generar_perspectiva(tramos, Escena(), **kw)
    extra = {}
    if corregir:
        extra = dict(focal_px=v["focal_px"] * factor_focal, centro_px=v["centro_px"],
                     pitch_grados=kw.get("pitch_grados", 0.0)
                     if pitch_informado is None else pitch_informado)
    V = estimar_volumen(img, v["altura_cm"], "grabcut", **extra).volumen()
    return 100 * (V - v["volumen_cm3"]) / v["volumen_cm3"]


def tabla(titulo: str, casos: list[tuple[str, dict]], columnas=(("corregido", {}),)) -> None:
    print(f"\n{titulo}")
    enc = "".join(f"{n:>{14 * len(columnas)}}" for n in ENVASES)
    print(f"   {'':<24}{enc}")
    if len(columnas) > 1:
        print(f"   {'':<24}" + "".join("".join(f"{c:>14}" for c, _ in columnas) for _ in ENVASES))
    for etiqueta, kw in casos:
        celdas = []
        for t in ENVASES.values():
            H = t[-1][1]
            k = {c: (v(H) if callable(v) else v) for c, v in kw.items()}
            for _, extra in columnas:
                celdas.append(f"{_err(t, **k, **extra):+13.2f}%")
        print(f"   {etiqueta:<24}" + "".join(celdas))


def main() -> None:
    tabla("1) DISTANCIA  (cámara a media altura, celular vertical)",
          [(f"{D} cm", dict(distancia_cm=D)) for D in (25, 35, 50, 75, 100, 200)],
          (("sin corregir", dict(corregir=False)), ("corregido", {})))
    tabla("2) ALTURA DE LA CÁMARA  (a 40 cm)",
          [("a la base", dict(distancia_cm=40, altura_camara_cm=lambda H: 0.0)),
           ("a 1/4 de la altura", dict(distancia_cm=40, altura_camara_cm=lambda H: H / 4)),
           ("a la boca", dict(distancia_cm=40, altura_camara_cm=lambda H: H)),
           ("10 cm sobre la boca ⚠", dict(distancia_cm=40, altura_camara_cm=lambda H: H + 10))])
    tabla("3) CELULAR INCLINADO  (a 50 cm; + = apunta hacia abajo)",
          [(f"{p:+d}°", dict(distancia_cm=50, pitch_grados=p)) for p in (-10, -5, 5, 10)],
          (("informa 0°", dict(pitch_informado=0.0)), ("informa real", {})))
    tabla("4) FOCAL EQUIVOCADA  (a 50 cm)",
          [(f"focal {100 * (f - 1):+.0f} %", dict(distancia_cm=50, factor_focal=f))
           for f in (0.9, 0.95, 1.05, 1.1)])


if __name__ == "__main__":
    main()
