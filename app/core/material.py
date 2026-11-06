"""Materiale: dal catalogo a un layer in luce lineare nello spazio colore della foto.

M1 implementa solo il modo "flat": il colore medio (in luce lineare) della texture di
catalogo. Tiling, fughe, prospettiva e anti-aliasing arrivano in M2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from app.core.colorspace import SRGB_LIKE, convert_srgb_to_profile, to_linear, trc_for
from app.core.models import BBox, MaterialLayer, MaterialSpec, SourceImage


def texture_in_photo_space(
    texture_srgb: NDArray[np.uint8], source: SourceImage
) -> tuple[NDArray[np.uint8], list[str]]:
    """Porta la texture (sRGB) nello spazio colore della foto, se diverso."""
    cs = source.meta.color_space
    if cs in SRGB_LIKE or source.icc is None:
        return texture_srgb, []
    try:
        return convert_srgb_to_profile(texture_srgb, source.icc), []
    except Exception as exc:  # profilo non utilizzabile da LittleCMS
        return texture_srgb, [f"conversione ICC della texture fallita ({exc}): usata sRGB"]


def flat_color_linear(texture: NDArray[np.uint8], source: SourceImage) -> NDArray[np.float64]:
    """Colore medio fisicamente corretto: media in luce lineare, non sui valori codificati."""
    lin = to_linear(texture.reshape(-1, 3), trc_for(source.meta.color_space))
    return np.asarray(lin.mean(axis=0), dtype=np.float64)


def flat_layer(
    spec: MaterialSpec, texture_srgb: NDArray[np.uint8], source: SourceImage, bbox: BBox
) -> MaterialLayer:
    tex, notes = texture_in_photo_space(texture_srgb, source)
    color = flat_color_linear(tex, source).astype(np.float32)
    x0, y0, x1, y1 = bbox
    rgb = np.broadcast_to(color, (y1 - y0, x1 - x0, 3))
    return MaterialLayer(
        rgb_linear=rgb,
        bbox=bbox,
        material_id=spec.id,
        render_mode="flat",
        notes=tuple(notes),
    )
