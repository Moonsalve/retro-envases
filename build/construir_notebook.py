# -*- coding: utf-8 -*-
"""Ensambla el notebook .ipynb a partir de los modulos de celdas."""
import json
import importlib.util
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parent
MODULOS = [
    "cells_01_intro",
    "cells_02_envase",
    "cells_03_datos",
    "cells_04_metodos",
    "cells_05_gauss",
    "cells_06_comparacion",
    "cells_07_graficas",
]


def cargar(nombre: str):
    ruta = BASE / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CELLS


def a_fuente(texto: str) -> list[str]:
    """nbformat guarda 'source' como lista de lineas con \\n final (salvo la ultima)."""
    lineas = texto.split("\n")
    return [l + "\n" for l in lineas[:-1]] + [lineas[-1]]


def main() -> int:
    celdas_crudas = []
    for m in MODULOS:
        celdas_crudas.extend(cargar(m))

    celdas = []
    n_md = n_code = 0
    errores = []
    for tipo, texto in celdas_crudas:
        if tipo == "markdown":
            n_md += 1
            celdas.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": a_fuente(texto),
            })
        else:
            n_code += 1
            # Verificacion de sintaxis antes de escribir nada.
            try:
                compile(texto, f"<celda {n_code}>", "exec")
            except SyntaxError as e:
                errores.append(f"  Celda de codigo #{n_code}: {e}")
            celdas.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": a_fuente(texto),
            })

    if errores:
        print("ERRORES DE SINTAXIS:", file=sys.stderr)
        for e in errores:
            print(e, file=sys.stderr)
        return 1

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True,
                "name": "Lab_Integracion_Numerica_RetroEnvases_Fase1.ipynb",
            },
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": celdas,
    }

    salida = BASE.parent / "notebooks" / "Lab_Integracion_Numerica_RetroEnvases_Fase1.ipynb"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"Notebook escrito: {salida}")
    print(f"  Celdas markdown : {n_md}")
    print(f"  Celdas de codigo: {n_code}")
    print(f"  Total           : {len(celdas)}")
    print(f"  Tamano          : {salida.stat().st_size:,} bytes")
    print("  Sintaxis de todas las celdas de codigo: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
