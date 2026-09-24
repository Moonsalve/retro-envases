"""Carga de fotos de celular: orientación, focal (EXIF) y resolución de trabajo."""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .vision import focal_px_desde_35mm

_EXIF_IFD = 0x8769
_FOCAL_35MM = 0xA405          # FocalLengthIn35mmFilm
_FOCAL_MM = 0x920A            # FocalLength (mm reales del lente)
_MODELO = 0x0110


@dataclass(frozen=True)
class Foto:
    bgr: np.ndarray                 # imagen de trabajo (BGR, ya orientada y reducida)
    focal_35mm: float | None        # focal equivalente de 35 mm leída del EXIF
    modelo: str | None              # modelo de la cámara, si viene en el EXIF
    tamano_original: tuple[int, int]
    escala: float                   # tamaño de trabajo / tamaño original

    def focal_px(self, focal_35mm: float | None = None) -> float | None:
        """Focal en píxeles de la imagen de trabajo (usa la del EXIF si no se da)."""
        f35 = focal_35mm or self.focal_35mm
        if not f35:
            return None
        h, w = self.bgr.shape[:2]
        return focal_px_desde_35mm(f35, w, h)


def cargar(datos: bytes, lado_max: int = 1600) -> Foto:
    """Abre una foto desde bytes (JPEG, PNG, WEBP).

    • Aplica la orientación EXIF: los celulares guardan la foto girada y anotan
      la rotación; sin esto, una foto vertical llega acostada.
    • Lee la focal equivalente de 35 mm. Las fotos reenviadas por WhatsApp u
      otras redes pierden el EXIF: hay que subir el archivo original.
    • Reduce a `lado_max` píxeles. La focal en píxeles se calcula sobre la
      imagen ya reducida (con su diagonal), así que sigue siendo consistente.
    """
    im = Image.open(io.BytesIO(datos))
    exif = im.getexif()
    sub = exif.get_ifd(_EXIF_IFD) if exif else {}
    f35 = sub.get(_FOCAL_35MM) or exif.get(_FOCAL_35MM)
    modelo = exif.get(_MODELO)
    im = ImageOps.exif_transpose(im).convert("RGB")
    w0, h0 = im.size
    esc = min(1.0, lado_max / max(w0, h0))
    if esc < 1.0:
        im = im.resize((round(w0 * esc), round(h0 * esc)), Image.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return Foto(bgr, float(f35) if f35 else None, str(modelo) if modelo else None,
                (w0, h0), esc)
