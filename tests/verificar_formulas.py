# -*- coding: utf-8 -*-
"""Verificacion independiente de las formulas de cuadratura del notebook.

Reimplementa las reglas en Python puro (sin numpy) y comprueba propiedades que
deben cumplirse EXACTAMENTE por construccion. El objetivo es detectar errores de
transcripcion en los coeficientes, que es el fallo mas probable y mas silencioso.

Criterio: cada regla de Newton-Cotes de grado de exactitud d debe integrar
polinomios de grado <= d con error de solo redondeo.
"""
import math
import sys

TOL = 1e-11
fallos = []


def chk(nombre, obtenido, esperado, tol=TOL, rel=True):
    denom = max(abs(esperado), 1.0) if rel else 1.0
    err = abs(obtenido - esperado) / denom
    ok = err < tol
    estado = "PASA" if ok else "FALLA"
    print(f"  [{estado}] {nombre:<56} err={err:.3e}")
    if not ok:
        fallos.append(f"{nombre}: obtenido={obtenido!r} esperado={esperado!r} err={err:.3e}")
    return ok


def malla(a, b, n):
    h = (b - a) / n
    return [a + i * h for i in range(n + 1)], h


# ---------------------------------------------------------------- REGLAS
def trapecio(f, a, b, n):
    z, h = malla(a, b, n)
    s = 0.5 * f(z[0]) + 0.5 * f(z[-1])
    for i in range(1, n):
        s += f(z[i])
    return h * s


def simpson13(f, a, b, n):
    assert n % 2 == 0
    z, h = malla(a, b, n)
    s = f(z[0]) + f(z[-1])
    for i in range(1, n):
        s += (4.0 if i % 2 == 1 else 2.0) * f(z[i])
    return h / 3.0 * s


def simpson38(f, a, b, n):
    assert n % 3 == 0
    z, h = malla(a, b, n)
    w = [0.0] * (n + 1)
    for i in range(0, n, 3):
        w[i] += 1.0; w[i+1] += 3.0; w[i+2] += 3.0; w[i+3] += 1.0
    return h * 3.0 / 8.0 * sum(wi * f(zi) for wi, zi in zip(w, z))


def boole(f, a, b, n):
    assert n % 4 == 0
    z, h = malla(a, b, n)
    w = [0.0] * (n + 1)
    for i in range(0, n, 4):
        w[i] += 7.0; w[i+1] += 32.0; w[i+2] += 12.0; w[i+3] += 32.0; w[i+4] += 7.0
    return 2.0 * h / 45.0 * sum(wi * f(zi) for wi, zi in zip(w, z))


# Nodos y pesos de Gauss-Legendre (tablas estandar)
GAUSS = {
    2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
    3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
        [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
    5: ([-0.9061798459386640, -0.5384693101056831, 0.0,
         0.5384693101056831, 0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
         0.4786286704993665, 0.2369268850561891]),
}


def gauss(f, a, b, m):
    x, w = GAUSS[m]
    semi = (b - a) / 2.0
    centro = (a + b) / 2.0
    return semi * sum(wk * f(semi * xk + centro) for xk, wk in zip(x, w))


# ------------------------------------------------------- PRUEBA 1: grados
print("=" * 78)
print("PRUEBA 1 — Grado de exactitud de cada regla (Newton-Cotes)")
print("=" * 78)
print("Cada regla debe integrar EXACTAMENTE polinomios hasta su grado de exactitud.")
A, B = 0.0, 21.0   # mismo intervalo que el envase del notebook

def poly(k):
    return lambda x: x ** k

def poly_exacta(k, a, b):
    return (b ** (k + 1) - a ** (k + 1)) / (k + 1)

reglas = [
    ("Trapecio",    lambda f: trapecio(f, A, B, 12),  1, 2),
    ("Simpson 1/3", lambda f: simpson13(f, A, B, 12), 3, 4),
    ("Simpson 3/8", lambda f: simpson38(f, A, B, 12), 3, 4),
    ("Boole",       lambda f: boole(f, A, B, 12),     5, 6),
]
for nombre, regla, grado_ex, grado_falla in reglas:
    print(f"\n  {nombre} (grado de exactitud esperado: {grado_ex})")
    for k in range(0, grado_ex + 1):
        chk(f"    exacta para x^{k}", regla(poly(k)), poly_exacta(k, A, B))
    # Un grado por encima debe FALLAR (si no, el grado declarado esta mal)
    k = grado_falla
    obtenido = regla(poly(k))
    esperado = poly_exacta(k, A, B)
    err = abs(obtenido - esperado) / abs(esperado)
    ok = err > 1e-9
    print(f"  [{'PASA' if ok else 'FALLA'}]     NO exacta para x^{k} (debe fallar)"
          f"{'':<13} err={err:.3e}")
    if not ok:
        fallos.append(f"{nombre} resulto exacta para x^{k}; el grado declarado es incorrecto")

# ------------------------------------------------------- PRUEBA 2: Gauss
print("\n" + "=" * 78)
print("PRUEBA 2 — Cuadratura de Gauss-Legendre")
print("=" * 78)
for m in (2, 3, 5):
    x, w = GAUSS[m]
    chk(f"  m={m}: suma de pesos = 2 (longitud de [-1,1])", sum(w), 2.0)
    chk(f"  m={m}: nodos simetricos respecto de 0", sum(x), 0.0, rel=False)
    d = 2 * m - 1
    print(f"  m={m}: grado de exactitud esperado = 2m-1 = {d}")
    for k in range(0, d + 1):
        chk(f"    exacta para x^{k}", gauss(poly(k), A, B, m), poly_exacta(k, A, B))
    obtenido = gauss(poly(d + 1), A, B, m)
    esperado = poly_exacta(d + 1, A, B)
    err = abs(obtenido - esperado) / abs(esperado)
    ok = err > 1e-9
    print(f"  [{'PASA' if ok else 'FALLA'}]     NO exacta para x^{d+1} (debe fallar)"
          f"{'':<12} err={err:.3e}")
    if not ok:
        fallos.append(f"Gauss m={m} resulto exacta para x^{d+1}")

# --------------------------------------------- PRUEBA 3: tronco de cono
print("\n" + "=" * 78)
print("PRUEBA 3 — Formula del volumen del tronco de cono")
print("=" * 78)

def v_tronco(h, r1, r2):
    return math.pi * h / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2)

chk("  Degenera en cilindro: V(h=2, r=3, r=3) = pi*9*2", v_tronco(2, 3, 3), math.pi * 9 * 2)
chk("  Degenera en cono:     V(h=3, r=0, r=4) = pi*16*3/3", v_tronco(3, 0, 4), math.pi * 16 * 3 / 3)
chk("  Cono invertido igual al cono", v_tronco(3, 4, 0), v_tronco(3, 0, 4))
# Contraste con integracion directa de pi*r(z)^2 para r lineal
h, r1, r2 = 3.5, 3.40, 1.35
f_tronco = lambda z: math.pi * (r1 + (r2 - r1) * z / h) ** 2
chk("  Coincide con Simpson sobre pi*r(z)^2 (r lineal)",
    simpson13(f_tronco, 0.0, h, 120), v_tronco(h, r1, r2))
chk("  Coincide con Boole sobre pi*r(z)^2 (r lineal)",
    boole(f_tronco, 0.0, h, 120), v_tronco(h, r1, r2))

# ---------------------------------------- PRUEBA 4: el envase del notebook
print("\n" + "=" * 78)
print("PRUEBA 4 — El envase concreto del notebook")
print("=" * 78)
TRAMOS = [
    (0.0,  1.0,  2.50, 3.40, "base (pie)"),
    (1.0,  15.0, 3.40, 3.40, "cuerpo cilindrico"),
    (15.0, 18.5, 3.40, 1.35, "hombro"),
    (18.5, 20.0, 1.35, 1.35, "cuello"),
    (20.0, 21.0, 1.35, 1.45, "boca / rosca"),
]
# Continuidad de los tramos
for (z1i, z1f, r1i, r1f, n1), (z2i, z2f, r2i, r2f, n2) in zip(TRAMOS, TRAMOS[1:]):
    chk(f"  Continuidad {n1} -> {n2}", z1f, z2i, rel=False)
    chk(f"  Radio continuo {n1} -> {n2}", r1f, r2i, rel=False)

V_exacto = sum(v_tronco(zf - zi, ri, rf) for zi, zf, ri, rf, _ in TRAMOS)
print(f"\n  Volumen exacto por formula cerrada: {V_exacto:.10f} cm^3")

def radio(z):
    for zi, zf, ri, rf, _ in TRAMOS:
        if zi <= z <= zf:
            return ri + (rf - ri) * (z - zi) / (zf - zi)
    return 0.0

f_env = lambda z: math.pi * radio(z) ** 2

# Integracion por tramos (malla alineada) -> debe dar el valor exacto
V_tramos = sum(simpson13(f_env, zi, zf, 60) for zi, zf, _, _, _ in TRAMOS)
chk("  Simpson por tramos (malla alineada) == formula cerrada", V_tramos, V_exacto)
V_tramos_b = sum(boole(f_env, zi, zf, 60) for zi, zf, _, _, _ in TRAMOS)
chk("  Boole por tramos (malla alineada)  == formula cerrada", V_tramos_b, V_exacto)

# Malla ciega -> debe ser aproximado pero NO exacto
V_ciega_t = trapecio(f_env, 0.0, 21.0, 60)
V_ciega_s = simpson13(f_env, 0.0, 21.0, 60)
V_ciega_b = boole(f_env, 0.0, 21.0, 60)
print(f"\n  Con malla ciega (n=60, sin alinear con los quiebres):")
for nom, v in (("Trapecio", V_ciega_t), ("Simpson 1/3", V_ciega_s), ("Boole", V_ciega_b)):
    e = abs(v - V_exacto)
    print(f"     {nom:<14} V = {v:12.6f} cm^3   error = {e:.3e} cm^3 "
          f"({100*e/V_exacto:.3e} %)")

# --- Signo del error del trapecio ---
# TRAMO A TRAMO (malla alineada) debe SOBREESTIMAR: f = r^2 es convexa en los
# tramos conicos y exacta en los cilindricos.
print("\n  Signo del error del trapecio, tramo a tramo (malla alineada):")
todos_no_negativos = True
for zi, zf, ri, rf, nom in TRAMOS:
    d = trapecio(f_env, zi, zf, 12) - v_tronco(zf - zi, ri, rf)
    if d < -1e-12:
        todos_no_negativos = False
    print(f"    [{zi:5.1f},{zf:5.1f}] {nom:<20} dif = {d:+.6e}")
print(f"  [{'PASA' if todos_no_negativos else 'FALLA'}] Tramo a tramo el trapecio "
      f"sobreestima o es exacto (f convexa)")
if not todos_no_negativos:
    fallos.append("Algun tramo dio error negativo con malla alineada")

# GLOBALMENTE, con malla ciega, el signo se INVIERTE: los codos aportan impulsos
# negativos de f'' que dominan. Debe subestimar para todo n probado.
print("\n  Signo del error del trapecio con malla CIEGA (debe SUBESTIMAR siempre):")
todos_negativos = True
for n in (12, 24, 60, 120, 240, 480):
    d = trapecio(f_env, 0.0, 21.0, n) - V_exacto
    if d >= 0:
        todos_negativos = False
    print(f"    n={n:>4}  dif = {d:+.6e}  {'SUB' if d < 0 else 'SOBRE'}")
print(f"  [{'PASA' if todos_negativos else 'FALLA'}] El trapecio subestima "
      f"sistematicamente con malla ciega (los codos dominan)")
if not todos_negativos:
    fallos.append("El trapecio no subestimo consistentemente con malla ciega")

# --- El ranking invertido: hallazgo central del notebook ---
print("\n  Ranking por error con malla ciega, n=60 (debe estar INVERTIDO respecto")
print("  al orden teorico: el metodo de orden 2 gana, el de orden 6 pierde):")
V_ciega_38 = simpson38(f_env, 0.0, 21.0, 60)
ranking = sorted([
    ("Trapecio", 2, abs(V_ciega_t - V_exacto)),
    ("Simpson 3/8", 4, abs(V_ciega_38 - V_exacto)),
    ("Simpson 1/3", 4, abs(V_ciega_s - V_exacto)),
    ("Boole", 6, abs(V_ciega_b - V_exacto)),
], key=lambda t: t[2])
for i, (nom, orden, e) in enumerate(ranking, 1):
    print(f"    {i}. {nom:<14} orden teorico {orden}   error = {e:.4e} cm^3")
ok = ranking[0][0] == "Trapecio" and ranking[-1][0] == "Boole"
print(f"  [{'PASA' if ok else 'FALLA'}] Gana el Trapecio (orden 2) y pierde Boole (orden 6)")
if not ok:
    fallos.append(f"El ranking no salio invertido: gano {ranking[0][0]}, perdio {ranking[-1][0]}")

# Simpson 1/3 debe ser PEOR que el trapecio (contraintuitivo pero real)
cociente = abs(V_ciega_s - V_exacto) / abs(V_ciega_t - V_exacto)
ok = cociente > 1
print(f"  [{'PASA' if ok else 'FALLA'}] Simpson 1/3 es {cociente:.2f}x PEOR que el trapecio")
if not ok:
    fallos.append("Simpson 1/3 no resulto peor que el trapecio")

# --- Malla alineada: Simpson y Boole deben ser EXACTOS ---
print("\n  Malla alineada: Simpson y Boole deben alcanzar el suelo de redondeo:")
for nom, fn in (("Simpson 1/3", simpson13), ("Simpson 3/8", simpson38), ("Boole", boole)):
    V_a = sum(fn(f_env, zi, zf, 60) for zi, zf, _, _, _ in TRAMOS)
    e = abs(V_a - V_exacto)
    ok = e < 1e-10
    print(f"  [{'PASA' if ok else 'FALLA'}] {nom:<14} error = {e:.3e} cm^3 "
          f"(exacto: f es de grado 2 en cada tramo)")
    if not ok:
        fallos.append(f"{nom} no fue exacto con malla alineada (error {e:.3e})")

# --- Gauss global de bajo orden debe FALLAR el criterio del 5% ---
print("\n  Gauss-Legendre global sobre el envase quebrado:")
for m in (2, 3, 5):
    Vg = math.pi * gauss(lambda z: radio(z) ** 2, 0.0, 21.0, m)
    rel = 100 * abs(Vg - V_exacto) / V_exacto
    marca = "FALLA EL 5%" if rel > 5 else "pasa el 5%"
    print(f"    m={m}  V={Vg:10.4f}  error rel = {rel:6.3f} %   -> {marca}")
Vg2 = math.pi * gauss(lambda z: radio(z) ** 2, 0.0, 21.0, 2)
Vg3 = math.pi * gauss(lambda z: radio(z) ** 2, 0.0, 21.0, 3)
ok = (100 * abs(Vg2 - V_exacto) / V_exacto > 5) and (100 * abs(Vg3 - V_exacto) / V_exacto > 5)
print(f"  [{'PASA' if ok else 'FALLA'}] Gauss con m=2 y m=3 NO cumple el criterio del 5%")
if not ok:
    fallos.append("Gauss de bajo orden no fallo el 5% como se afirma en el notebook")

# Todos los metodos deben cumplir el criterio del 5% con muchisima holgura
margen = 0.05 * V_exacto
for nom, v in (("Trapecio", V_ciega_t), ("Simpson 1/3", V_ciega_s), ("Boole", V_ciega_b)):
    e = abs(v - V_exacto)
    ok = e < margen / 100      # al menos 100x mejor que el criterio
    print(f"  [{'PASA' if ok else 'FALLA'}] {nom} esta >100x por debajo del margen del 5%")
    if not ok:
        fallos.append(f"{nom} no cumple el margen holgado")

# ------------------------------------------ PRUEBA 5: sensibilidad a escala
print("\n" + "=" * 78)
print("PRUEBA 5 — Propagacion del error de escala (dV/V = 2 dr/r)")
print("=" * 78)
for s in (0.001, 0.01, 0.025, 0.05):
    f_s = lambda z: math.pi * ((1 + s) * radio(z)) ** 2
    V_s = sum(simpson13(f_s, zi, zf, 60) for zi, zf, _, _, _ in TRAMOS)
    rel = (V_s - V_exacto) / V_exacto
    esperado = (1 + s) ** 2 - 1
    chk(f"  escala +{100*s:5.2f}% -> volumen {100*rel:+7.3f}% (teoria {100*esperado:+.3f}%)",
        rel, esperado)
print(f"\n  Frontera del criterio: error de escala del 2.5% -> "
      f"{100*((1.025)**2 - 1):.3f}% en volumen (limite: 5%)")

# --------------------------------------------------------------- RESUMEN
print("\n" + "=" * 78)
if fallos:
    print(f"RESULTADO: {len(fallos)} FALLO(S)")
    for f_ in fallos:
        print(f"  - {f_}")
    sys.exit(1)
print("RESULTADO: TODAS LAS VERIFICACIONES PASAN")
print("=" * 78)
print("Verificado:")
print("  - Coeficientes de trapecio, Simpson 1/3, Simpson 3/8, Boole y Gauss-Legendre.")
print("  - Los grados de exactitud declarados se cumplen y fallan un grado por encima.")
print("  - Formula del tronco de cono y volumen exacto del envase del notebook.")
print("  - El trapecio subestima con malla ciega y sobreestima tramo a tramo.")
print("  - El ranking de errores sale INVERTIDO respecto al orden teorico.")
print("  - Con malla alineada, Simpson y Boole son exactos salvo redondeo.")
print("  - Gauss global con m=2,3 no cumple el criterio del 5% del reto.")
print("")
print("Todas las afirmaciones numericas del notebook estan respaldadas por estas")
print("comprobaciones, hechas con una implementacion INDEPENDIENTE en Python puro.")
