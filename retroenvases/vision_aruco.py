"""Escala y perspectiva a partir de la hoja de referencia con marcadores ArUco.

Con la hoja sobre la mesa y el envase parado en su círculo:

1. Se detectan los marcadores y sus esquinas (precisión subpíxel).
2. Las esquinas están en un plano de medidas conocidas visto en perspectiva: eso
   alcanza para CALIBRAR la cámara (su focal, suponiendo el centro óptico en el
   centro de la imagen) y obtener su POSE: dónde está y hacia dónde mira, en
   centímetros, respecto de la hoja. No hace falta el EXIF de la foto.
3. El eje del envase es la vertical que pasa por el centro del círculo y la base
   está en z = 0. Cada borde de la silueta define un rayo en 3D; su punto de
   máximo acercamiento al eje da el radio y la altura reales.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .plantilla import DICCIONARIO, Hoja


class SinMarcadores(Exception):
    """No se detectaron suficientes marcadores para fijar la escala."""


# ───────────────────────────────────────────────────────────── detección ──
def detectar(img: np.ndarray) -> dict[int, np.ndarray]:
    """Marcadores detectados: id → esquinas (4, 2) en píxeles, orden de OpenCV."""
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICCIONARIO), params)
    esquinas, ids, _ = det.detectMarkers(gris)
    if ids is None:
        return {}
    return {int(i): e.reshape(4, 2).astype(np.float64) for e, i in zip(esquinas, ids.ravel())}


# ──────────────────────────────────────────────────────────────── cámara ──
@dataclass(frozen=True)
class Camara:
    K: np.ndarray              # matriz intrínseca 3×3
    R: np.ndarray              # rotación hoja → cámara
    t: np.ndarray              # traslación hoja → cámara [cm]
    error_px: float            # error de reproyección de las esquinas (RMS) [px]
    marcadores: tuple[int, ...]
    focal_estimada: bool       # True: la focal salió de la foto; False: se supuso

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    @property
    def centro(self) -> np.ndarray:
        """Posición de la cámara en coordenadas de la hoja [cm]."""
        return (-self.R.T @ self.t).ravel()

    @property
    def inclinacion_grados(self) -> float:
        """Ángulo del eje óptico bajo la horizontal (+ = mira hacia abajo)."""
        eje = self.R.T @ np.array([0.0, 0.0, 1.0])
        return float(math.degrees(math.asin(-eje[2] / np.linalg.norm(eje))))

    def rayos(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Dirección (en la hoja) del rayo de cada píxel (u, v). Forma (N, 3)."""
        uv1 = np.column_stack([np.asarray(u, float), np.asarray(v, float), np.ones(np.size(u))])
        d_cam = uv1 @ np.linalg.inv(self.K).T
        return d_cam @ self.R            # equivale a (R.T @ d_cam.T).T

    def proyectar(self, pts: np.ndarray) -> np.ndarray:
        """Puntos 3D de la hoja [cm] → píxeles (N, 2)."""
        p, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 1, 3),
                                 cv2.Rodrigues(self.R)[0], self.t, self.K, None)
        return p.reshape(-1, 2)


def estimar_camara(img_shape: tuple[int, ...], marcadores: dict[int, np.ndarray],
                   hoja: Hoja, lado_medido_mm: float | None = None,
                   focal_px_inicial: float | None = None) -> Camara:
    """Calibra la focal y estima la pose de la cámara a partir de los marcadores.

    lado_medido_mm   lado real del cuadrado negro medido en la hoja impresa: corrige
                     una impresora que agrande o achique (por defecto, el nominal).
    focal_px_inicial punto de partida (del EXIF, si lo hay). Con 2 o más marcadores
                     la focal se ajusta desde la foto; con 1 solo marcador no hay
                     información suficiente y se usa este valor (o el típico).
    """
    validos = {i: c for i, c in marcadores.items() if i in hoja.ids}
    if not validos:
        raise SinMarcadores(
            "No se ve ningún marcador de la hoja. Asegúrate de que la hoja esté en la "
            "foto, bien iluminada y sin reflejos, y de que al menos dos marcadores se "
            "vean completos.")
    escala = (lado_medido_mm or hoja.lado_mm) / hoja.lado_mm
    geo = hoja.esquinas(escala)
    obj = np.concatenate([geo[i] for i in validos]).astype(np.float32) / 10.0    # → cm
    img = np.concatenate([validos[i] for i in validos]).astype(np.float32)
    h, w = img_shape[:2]
    f0 = focal_px_inicial or 0.75 * max(w, h)        # ≈ cámara principal de celular
    K0 = np.array([[f0, 0, w / 2 - 0.5], [0, f0, h / 2 - 0.5], [0, 0, 1]], np.float64)

    if len(validos) >= 2:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                 cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
                 cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        rms, K, _, rvecs, tvecs = cv2.calibrateCamera(
            [obj], [img], (w, h), K0, np.zeros(5), flags=flags)
        rvec, tvec, estimada = rvecs[0], tvecs[0], True
        # salvaguarda: una focal absurda indica una vista casi frontal de la hoja
        # (la perspectiva no alcanza para medirla); se vuelve a la inicial
        if not (0.25 * max(w, h) < K[0, 0] < 4 * max(w, h)):
            K, estimada = K0, False
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    else:
        K, estimada = K0, False
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    proy, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proy.reshape(-1, 2) - img) ** 2, axis=1))))
    return Camara(K, cv2.Rodrigues(rvec)[0], tvec.reshape(3, 1), rms,
                  tuple(sorted(validos)), estimada)


def rectangulo_envase(cam: Camara, img_shape, radio_max_cm: float = 6.0,
                      altura_cm: float | None = None) -> tuple[int, int, int, int]:
    """Rectángulo (x, y, ancho, alto) que contiene al envase, para iniciar GrabCut.

    Proyecta un cilindro alrededor del eje (desde la base hasta `altura_cm`, o
    hasta 40 cm si no se conoce) y toma su caja en la imagen.
    """
    h, w = img_shape[:2]
    alto = (altura_cm or 40.0) * 1.08
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = [(radio_max_cm * np.cos(a), radio_max_cm * np.sin(a), z)
           for a in ang for z in np.linspace(0, alto, 9)]
    uv = cam.proyectar(np.array(pts))
    x0, y0 = np.clip(uv.min(axis=0), 0, [w - 1, h - 1])
    x1, y1 = np.clip(uv.max(axis=0), 0, [w - 1, h - 1])
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


# ────────────────────────────────────────────────────────── perfil en 3D ──
@dataclass(frozen=True)
class Perfil3D:
    z_cm: np.ndarray
    r_cm: np.ndarray
    altura_cm: float             # usada para cerrar el perfil (medida o estimada)
    altura_estimada_cm: float    # la que sale de la geometría de la foto
    asimetria: float             # |r_izq − r_der|/(r_izq + r_der), mediana
    eje_cm: tuple[float, float]  # posición del eje sobre la hoja


def reconstruir(cam: Camara, filas: np.ndarray, borde_izq: np.ndarray,
                borde_der: np.ndarray, altura_cm: float | None = None,
                eje_cm: tuple[float, float] = (0.0, 0.0),
                refinar_eje: bool = True, iteraciones: int = 6) -> Perfil3D:
    """Perfil r(z) en cm a partir de los bordes de la silueta y la pose conocida.

    filas, borde_izq, borde_der: coordenadas (en píxeles de la foto original) de
    los bordes izquierdo y derecho de cada fila de la silueta.

    • Pared: el rayo de cada borde es tangente al envase; su punto de máximo
      acercamiento al eje da (r, z).
    • Base: está en z = 0 (el envase se apoya en la hoja). Las filas cuyo máximo
      acercamiento cae bajo la mesa son el arco de la base vista desde arriba: su
      rayo se corta con el plano z = 0 y da el radio de la base.
    • Boca: si no se da `altura_cm`, se estima con el punto más alto de la silueta,
      que es el borde de la boca más lejano (cámara por encima) o el más cercano
      (cámara por debajo).
    • Eje: la vertical por el centro del círculo; si `refinar_eje`, su posición
      lateral se ajusta para que la silueta quede simétrica (por si el envase no
      quedó perfectamente centrado).
    """
    C = cam.centro
    filas = np.asarray(filas, float)
    d_izq = cam.rayos(borde_izq, filas)
    d_der = cam.rayos(borde_der, filas)
    X0, Y0 = eje_cm

    def acercamiento(d, X0, Y0):
        """Máximo acercamiento horizontal de cada rayo al eje: (ρ, z, t)."""
        px, py = C[0] - X0, C[1] - Y0
        t = -(px * d[:, 0] + py * d[:, 1]) / (d[:, 0] ** 2 + d[:, 1] ** 2)
        rho = np.hypot(px + t * d[:, 0], py + t * d[:, 1])
        return rho, C[2] + t * d[:, 2], t

    if refinar_eje:
        # mover el eje en la dirección perpendicular a la línea de visión hasta que
        # los radios izquierdo y derecho coincidan (una búsqueda 1-D basta)
        vis = np.array([-C[0], -C[1]]) / np.hypot(C[0], C[1])
        perp = np.array([-vis[1], vis[0]])
        def desbalance(s):
            x, y = X0 + s * perp[0], Y0 + s * perp[1]
            ri, zi, _ = acercamiento(d_izq, x, y)
            rd, zd, _ = acercamiento(d_der, x, y)
            ok = (zi > 0) & (zd > 0)
            return float(np.median(ri[ok] - rd[ok])) if ok.any() else 0.0
        a, b = -2.0, 2.0
        fa, fb = desbalance(a), desbalance(b)
        if fa * fb < 0:
            for _ in range(40):
                m = (a + b) / 2
                fm = desbalance(m)
                a, b, fa, fb = (m, b, fm, fb) if fa * fm > 0 else (a, m, fa, fm)
            s = (a + b) / 2
            X0, Y0 = X0 + s * perp[0], Y0 + s * perp[1]

    def tangencia(d, k):
        """Punto donde el rayo es TANGENTE a una pared de pendiente m (k = m·dz).

        Con ρ(t)² = A t² − 2B t + C (distancia horizontal al eje), la tangencia
        exige ρ'(t) = m·dz. Solución cerrada (k = 0 → máximo acercamiento al eje):
            w = k·√(Δ/(A − k²)),  Δ = A·C − B²,  t = (w + B)/A,  r = √((w² + Δ)/A)
        Verificada contra búsqueda por fuerza bruta.
        """
        px, py = C[0] - X0, C[1] - Y0
        A = d[:, 0] ** 2 + d[:, 1] ** 2
        B = -(px * d[:, 0] + py * d[:, 1])
        delta = np.maximum(A * (px * px + py * py) - B * B, 0.0)
        # cota: la fórmula es singular en k² = A (rayo paralelo a la pared); una
        # pendiente que acerque k a ese valor solo puede venir de ruido
        lim = 0.7 * np.sqrt(A)
        k = np.clip(k, -lim, lim)
        w = k * np.sqrt(delta / np.maximum(A - k * k, 1e-12))
        t = (w + B) / A
        return np.sqrt((w * w + delta) / A), C[2] + t * d[:, 2]

    pendiente = np.zeros(len(filas))
    for _ in range(4):
        r_i, z_i = tangencia(d_izq, pendiente * d_izq[:, 2])
        r_d, z_d = tangencia(d_der, pendiente * d_der[:, 2])
        r, z = (r_i + r_d) / 2, (z_i + z_d) / 2
        # la pendiente se estima SOLO con la pared: las filas de los arcos de la boca
        # y de la base no son pared y su "pendiente" no significa nada
        H_prov = altura_cm or float(np.percentile(z, 99))
        pared_prov = (z > 0.02 * H_prov) & (z < 0.97 * H_prov)
        pendiente = np.where(pared_prov, _pendiente_suave(np.where(pared_prov, z, np.nan),
                                                          r), 0.0)
    asim = float(np.median(np.abs(r_i - r_d) / np.maximum(r_i + r_d, 1e-9)))

    # base: arco (z < 0) cortado con el plano de la mesa
    def corte_plano(d, h):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (h - C[2]) / d[:, 2]
        return np.hypot(C[0] + t * d[:, 0] - X0, C[1] + t * d[:, 1] - Y0), t
    arco_b = z <= 0
    r_b = None
    if arco_b.sum() >= 3:
        rb_i, _ = corte_plano(d_izq[arco_b], 0.0)
        rb_d, _ = corte_plano(d_der[arco_b], 0.0)
        cand = np.concatenate([rb_i, rb_d])
        cand = cand[np.isfinite(cand)]
        if cand.size >= 3:
            r_b = float(np.median(cand))

    # boca: altura estimada con la fila más alta de la silueta (eje del envase)
    tope = int(np.argmin(filas))
    u_eje = (borde_izq[tope] + borde_der[tope]) / 2
    d_tope = cam.rayos(np.array([u_eje]), np.array([filas[tope]]))[0]
    H = altura_cm or float(np.max(z))
    H_est = H
    for _ in range(iteraciones):
        pared = (z > 0) & (z < H)
        cerca = pared & (z > H - 0.4)
        r_t = float(np.median(r[cerca])) if cerca.sum() >= 3 else float(r[np.argmax(z)])
        # punto del rayo del tope que está a distancia r_t del eje: lado lejano si la
        # cámara está por encima de la boca, lado cercano si está por debajo
        px, py = C[0] - X0, C[1] - Y0
        a_ = d_tope[0] ** 2 + d_tope[1] ** 2
        b_ = 2 * (px * d_tope[0] + py * d_tope[1])
        c_ = px * px + py * py - r_t * r_t
        disc = b_ * b_ - 4 * a_ * c_
        if disc < 0:
            break
        raices = ((-b_ - math.sqrt(disc)) / (2 * a_), (-b_ + math.sqrt(disc)) / (2 * a_))
        t_borde = raices[1] if C[2] > H else raices[0]
        candidata = float(C[2] + t_borde * d_tope[2])
        z_pared = z[(z > 0) & np.isfinite(z)]
        # salvaguarda: la altura no puede apartarse mucho de lo que ya muestra la pared
        if not np.isfinite(candidata) or z_pared.size == 0 or \
                not (0.8 * z_pared.max() < candidata < 1.25 * z_pared.max()):
            break
        H_est = candidata
        if altura_cm is None:
            H = H_est

    # Con la cámara claramente por encima de la boca, la boca se ve como una elipse
    # abierta y conviene usarla: sus filas (arco) son puntos del círculo de la boca, a
    # una altura H desconocida. Solo con la H correcta, sus rayos cortados con el plano
    # z = H forman un círculo de radio constante e igual al de la pared justo debajo.
    if altura_cm is None and C[2] > H + 2.0:
        H_arco = _altura_por_arco(z, r, d_izq, d_der, C, X0, Y0, H)
        if H_arco is not None:
            H = H_est = H_arco

    pared = (z > 0) & (z < H)
    zz, rr = z[pared], r[pared]
    orden = np.argsort(zz)
    zz, rr = zz[orden], rr[orden]
    r0 = r_b if r_b is not None else float(rr[0])
    r1 = float(np.median(rr[zz > H - 0.4])) if (zz > H - 0.4).sum() >= 3 else float(rr[-1])
    zz = np.concatenate([[0.0], zz, [H]])
    rr = np.concatenate([[r0], rr, [r1]])
    return Perfil3D(zz, rr, float(H), float(H_est), asim, (float(X0), float(Y0)))


# ──────────────────────────────────────────── segmentación con la geometría ──
def segmentar_con_hoja(img: np.ndarray, cam: Camara, radio_max_cm: float = 6.0,
                       altura_cm: float | None = None, lado_max: int = 800) -> np.ndarray:
    """Máscara del envase (0/255) por GrabCut, iniciado con lo que la geometría ya sabe.

    • FONDO SEGURO: todo lo que queda fuera del cilindro de radio `radio_max_cm`
      alrededor del eje, y el suelo DELANTE del envase (entre la cámara y el
      envase: el envase nunca puede taparlo). Así el modelo de color del fondo
      aprende el blanco de la hoja y el negro de los marcadores.
    • OBJETO SEGURO: un núcleo de 0.5 cm alrededor del eje, en la parte baja.
    • PROBABLE: el resto del cilindro; GrabCut decide.
    """
    h, w = img.shape[:2]
    f = min(1.0, lado_max / max(h, w))
    hs, ws = int(round(h * f)), int(round(w * f))
    peq = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA) if f < 1 else img.copy()

    # rayo de cada píxel (a la resolución reducida) y su corte con el suelo
    uu, vv = np.meshgrid((np.arange(ws) + 0.5) / f - 0.5, (np.arange(hs) + 0.5) / f - 0.5)
    d = cam.rayos(uu.ravel(), vv.ravel()).reshape(hs, ws, 3)
    C = cam.centro
    with np.errstate(divide="ignore", invalid="ignore"):
        t_suelo = -C[2] / d[..., 2]
    y_suelo = C[1] + t_suelo * d[..., 1]
    # máximo acercamiento horizontal de cada rayo al eje, y su altura
    t_ax = -(C[0] * d[..., 0] + C[1] * d[..., 1]) / (d[..., 0] ** 2 + d[..., 1] ** 2)
    rho = np.hypot(C[0] + t_ax * d[..., 0], C[1] + t_ax * d[..., 1])
    z_ax = C[2] + t_ax * d[..., 2]
    tope = (altura_cm or 40.0) * 1.08
    nucleo_z = min(altura_cm or 8.0, 8.0)

    m = np.full((hs, ws), cv2.GC_BGD, np.uint8)
    cilindro = (rho < radio_max_cm) & (z_ax > -radio_max_cm) & (z_ax < tope)
    m[cilindro] = cv2.GC_PR_BGD
    m[(rho < 1.5) & (z_ax > 0.3) & (z_ax < tope)] = cv2.GC_PR_FGD
    m[(rho < 0.5) & (z_ax > 1.0) & (z_ax < nucleo_z)] = cv2.GC_FGD
    delante = (d[..., 2] < 0) & (y_suelo < -radio_max_cm)      # suelo entre cámara y envase
    m[delante] = cv2.GC_BGD

    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(peq, m, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    obj = np.isin(m, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
    if f < 1:
        obj = cv2.resize(obj, (w, h), interpolation=cv2.INTER_LINEAR)
    obj = (obj >= 0.5).astype(np.uint8)

    # quedarse con la componente que contiene el eje y rellenar cada fila
    n, etq, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    eje_uv = cam.proyectar(np.array([[0.0, 0.0, 2.0]]))[0]
    x_e, y_e = int(np.clip(eje_uv[0], 0, w - 1)), int(np.clip(eje_uv[1], 0, h - 1))
    etiqueta = etq[y_e, x_e]
    if etiqueta == 0 and n > 1:
        etiqueta = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    obj = (etq == etiqueta).astype(np.uint8)
    salida = np.zeros_like(obj)
    for fila in np.where(obj.any(axis=1))[0]:
        cols = np.flatnonzero(obj[fila])
        salida[fila, cols[0]:cols[-1] + 1] = 1
    return salida * 255


def _pendiente_suave(z: np.ndarray, r: np.ndarray, paso_cm: float = 0.05,
                     ventana_cm: float = 0.5, limite: float = 2.0) -> np.ndarray:
    """Pendiente dr/dz de la pared en cada fila, estimada sin amplificar el ruido.

    Se remuestrea el perfil a una malla uniforme en z, se suaviza con una media
    móvil de `ventana_cm` y se deriva ahí; después se vuelve a cada fila. Derivar
    fila por fila (≈ 0.02 cm entre filas) multiplicaría el ruido del borde.
    Fuera de la pared (z < 0 o z > máx.) la pendiente es 0.
    """
    ok = np.isfinite(z) & np.isfinite(r) & (z > 0)
    if ok.sum() < 10:
        return np.zeros_like(z)
    orden = np.argsort(z[ok])
    zs, rs = z[ok][orden], r[ok][orden]
    malla = np.arange(zs[0], zs[-1], paso_cm)
    if malla.size < 5:
        return np.zeros_like(z)
    rm = np.interp(malla, zs, rs)
    k = max(1, int(round(ventana_cm / paso_cm)))
    rm = np.convolve(np.pad(rm, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    dm = np.clip(np.gradient(rm, paso_cm), -limite, limite)
    salida = np.interp(z, malla, dm, left=0.0, right=0.0)
    salida[~np.isfinite(salida)] = 0.0
    return salida


def _altura_por_arco(z, r, d_izq, d_der, C, X0, Y0, H0) -> float | None:
    """Altura de la boca ajustando el arco de la elipse de la boca (cámara por encima).

    Búsqueda en una malla de alturas candidatas H alrededor de H0. Para cada H:
      • arco: filas con z > H; sus rayos se cortan con el plano z = H (puntos del
        borde de la boca) y se mide su distancia al eje;
      • pared: filas con H − 0.5 < z < H.
    Costo = dispersión de esas distancias + |su mediana − radio de la pared|.
    Devuelve None si no hay arco suficiente para decidir.
    """
    mejores = []
    for H in np.arange(0.85 * H0, 1.15 * H0, 0.01):
        arco = z > H
        pared = (z > H - 0.5) & (z < H)
        if arco.sum() < 8 or pared.sum() < 5:
            continue
        rads = []
        for d in (d_izq[arco], d_der[arco]):
            t = (H - C[2]) / d[:, 2]
            rads.append(np.hypot(C[0] + t * d[:, 0] - X0, C[1] + t * d[:, 1] - Y0))
        rads = np.concatenate(rads)
        rads = rads[np.isfinite(rads)]
        if rads.size < 8:
            continue
        r_pared = float(np.median(r[pared]))
        costo = float(np.std(rads)) + abs(float(np.median(rads)) - r_pared)
        mejores.append((costo, float(H)))
    if not mejores:
        return None
    return min(mejores)[1]
