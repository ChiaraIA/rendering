"""Spazio colore: riconoscimento ICC e conversioni lineare <-> codificato.

Il compositing avviene in luce lineare. Le funzioni qui sono pure e deterministiche.
"""

from __future__ import annotations

import io
from functools import cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms

from app.core.models import ColorSpace

Trc = Literal["srgb", "adobe"]

_ADOBE_GAMMA = 563.0 / 256.0  # 2.19921875, dalla specifica Adobe RGB (1998)

SRGB_LIKE = frozenset({ColorSpace.SRGB, ColorSpace.ASSUMED_SRGB})


def describe_icc(icc: bytes) -> tuple[str | None, str | None]:
    """Ritorna (descrizione, spazio colore ICC es. 'RGB ' o 'GRAY'). None se illeggibile."""
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = ImageCms.getProfileDescription(prof).strip()
        return desc, prof.profile.xcolor_space
    except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
        return None, None


def classify_icc(icc: bytes | None) -> tuple[ColorSpace, str | None]:
    if icc is None:
        return ColorSpace.ASSUMED_SRGB, None
    desc, _ = describe_icc(icc)
    d = (desc or "").lower()
    if "srgb" in d:
        return ColorSpace.SRGB, desc
    if "p3" in d:
        return ColorSpace.DISPLAY_P3, desc
    if "adobe rgb" in d or "adobergb" in d:
        return ColorSpace.ADOBE_RGB, desc
    return ColorSpace.UNKNOWN, desc


def trc_for(cs: ColorSpace) -> Trc:
    return "adobe" if cs is ColorSpace.ADOBE_RGB else "srgb"


def _decode(v: NDArray[np.float64], trc: Trc) -> NDArray[np.float64]:
    if trc == "adobe":
        return np.power(v, _ADOBE_GAMMA)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def _encode(lin: NDArray[np.float64], trc: Trc) -> NDArray[np.float64]:
    lin = np.clip(lin, 0.0, 1.0)
    if trc == "adobe":
        return np.power(lin, 1.0 / _ADOBE_GAMMA)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


_LUTS: dict[Trc, NDArray[np.float64]] = {
    t: _decode(np.arange(256, dtype=np.float64) / 255.0, t) for t in ("srgb", "adobe")
}


def to_linear(values: NDArray[np.uint8], trc: Trc) -> NDArray[np.float64]:
    """uint8 codificato -> float64 lineare in [0, 1] (tramite LUT esatta)."""
    if values.dtype != np.uint8:
        raise TypeError(f"atteso uint8, trovato {values.dtype}")
    return _LUTS[trc][values]


def from_linear(lin: NDArray[np.float64], trc: Trc) -> NDArray[np.uint8]:
    """float lineare -> uint8 codificato (arrotondamento al più vicino, clip)."""
    enc = _encode(np.asarray(lin, dtype=np.float64), trc)
    return np.clip(np.rint(enc * 255.0), 0, 255).astype(np.uint8)


@cache
def srgb_profile_bytes() -> bytes:
    """Una volta per processo: l'header ICC generato contiene data e ora di creazione."""
    return bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


def convert_srgb_to_profile(rgb: NDArray[np.uint8], icc: bytes) -> NDArray[np.uint8]:
    """Converte un'immagine RGB sRGB nel profilo della foto (intento colorimetrico relativo)."""
    src = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    dst = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    img = Image.fromarray(np.ascontiguousarray(rgb))
    out = ImageCms.profileToProfile(
        img,
        src,
        dst,
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
        outputMode="RGB",
    )
    assert out is not None
    return np.asarray(out, dtype=np.uint8)
