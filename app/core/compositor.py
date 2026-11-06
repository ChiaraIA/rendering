"""Compositor: l'UNICO modulo che può creare l'immagine finale (regola 4).

Integrità per costruzione:
    final = original.copy()            # i pixel esterni non passano mai per float
    final[alpha > 0, :3] = blend(...)  # solo dentro la maschera, in luce lineare

con  blend = a * materiale * luce + (1 - a) * originale,  a = alpha / 255.
Il canale alpha della foto (se presente) non viene mai toccato.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.core.colorspace import from_linear, to_linear, trc_for
from app.core.errors import IntegrityError
from app.core.ingest import sha256_array
from app.core.masking import bbox_of, require_approved
from app.core.models import BBox, LightMap, MaskRecord, MaterialLayer, SourceImage

_SEAL = object()


@dataclass(frozen=True, eq=False)
class CompositeResult:
    """Immagine finale. Costruibile solo da `composite()`.

    `alpha` è la maschera effettivamente usata, uint8 0..255 (alpha == 0 <=> fuori maschera).
    """

    pixels: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    bbox: BBox | None
    final_sha256: str
    _seal: object = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if self._seal is not _SEAL:
            raise TypeError("CompositeResult può essere creato solo da compositor.composite()")
        if self.pixels.flags.writeable or self.alpha.flags.writeable:
            raise TypeError("CompositeResult deve contenere array in sola lettura")


def _inside(inner: BBox, outer: BBox) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def composite(
    source: SourceImage,
    mask: NDArray[np.uint8],
    mask_record: MaskRecord,
    layer: MaterialLayer,
    light: LightMap | None = None,
) -> CompositeResult:
    require_approved(mask_record, mask, source.meta.normalized_sha256)
    original = source.pixels
    if mask.shape != original.shape[:2]:
        raise IntegrityError(f"maschera {mask.shape} e immagine {original.shape[:2]} diverse")

    final = original.copy()  # copia esatta: base dell'integrità esterna
    bbox = bbox_of(mask)
    if bbox is not None:
        if not _inside(bbox, layer.bbox):
            raise IntegrityError("il layer materiale non copre tutta la maschera")
        x0, y0, x1, y1 = bbox
        sub_a = mask[y0:y1, x0:x1]
        idx = sub_a > 0

        lx0, ly0 = layer.bbox[0], layer.bbox[1]
        lin_m = np.asarray(
            layer.rgb_linear[y0 - ly0 : y1 - ly0, x0 - lx0 : x1 - lx0][idx], dtype=np.float64
        )
        if light is not None:
            if not _inside(bbox, light.bbox):
                raise IntegrityError("la mappa di luce non copre tutta la maschera")
            gx0, gy0 = light.bbox[0], light.bbox[1]
            gain = np.asarray(
                light.gain[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0][idx], dtype=np.float64
            )
            lin_m = lin_m * gain[:, None]

        trc = trc_for(source.meta.color_space)
        rgb_view = final[y0:y1, x0:x1, :3]
        lin_o = to_linear(rgb_view[idx], trc)
        a = (sub_a[idx].astype(np.float64) / 255.0)[:, None]
        rgb_view[idx] = from_linear(a * lin_m + (1.0 - a) * lin_o, trc)

    final.setflags(write=False)
    alpha = mask.copy()
    alpha.setflags(write=False)
    return CompositeResult(
        pixels=final, alpha=alpha, bbox=bbox, final_sha256=sha256_array(final), _seal=_SEAL
    )
