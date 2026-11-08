"""Compositor: l'UNICO modulo che può creare l'immagine finale (regola 4).

Integrità per costruzione:
    final = original.copy()            # i pixel esterni non passano mai per float
    final[alpha > 0, :3] = blend(...)  # solo dentro la maschera, in luce lineare

con  blend = originale + a * (materiale * luce - vecchio_pavimento),  a = alpha / 255,
dove vecchio_pavimento è il pixel stesso dove a = 1 (=> materiale * luce) e, sui bordi
morbidi, il colore del vecchio pavimento lì accanto (=> si sostituisce solo la sua parte).
Il canale alpha della foto (se presente) non viene mai toccato.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.colorspace import Trc, from_linear, to_linear, trc_for
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


EDGE_SIGMA_FRACTION = 0.002  # raggio della stima del vecchio pavimento: ~3 px su 1600


def _old_floor_near_edges(
    rgb_view: NDArray[np.uint8],
    sub_a: NDArray[np.uint8],
    trc: Trc,
    lin_o: NDArray[np.float64],
    idx: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Colore (lineare) del vecchio pavimento per ogni pixel della maschera.

    Dove alpha = 255 è il pixel stesso. Sui bordi morbidi è la media dei pixel pienamente
    di pavimento lì vicino; se vicino non ce ne sono, il pixel stesso (= blend classico).
    """
    out = lin_o.copy()
    soft = (sub_a > 0) & (sub_a < 255)
    full = sub_a == 255
    if not soft.any() or not full.any():
        return out
    sigma = max(1.5, EDGE_SIGMA_FRACTION * max(sub_a.shape))
    lin = to_linear(rgb_view, trc).astype(np.float32)
    w = full.astype(np.float32)
    num = cv2.GaussianBlur(lin * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    ok = soft & (den > 0.05)
    est = num[ok] / den[ok][:, None]
    pos = np.flatnonzero(idx.ravel())  # posizione di ogni pixel di idx nell'array compresso
    lookup = np.full(idx.size, -1, dtype=np.int64)
    lookup[pos] = np.arange(pos.size)
    out[lookup[np.flatnonzero(ok.ravel())]] = est
    return out


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
            lin_m = lin_m * gain[:, None] * np.asarray(light.tint, dtype=np.float64)
            if light.add is not None:
                lin_m = lin_m + np.asarray(
                    light.add[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0][idx], dtype=np.float64
                )

        trc = trc_for(source.meta.color_space)
        rgb_view = final[y0:y1, x0:x1, :3]
        lin_o = to_linear(rgb_view[idx], trc)
        a = (sub_a[idx].astype(np.float64) / 255.0)[:, None]
        # Un pixel di bordo (0 < alpha < 1) è in parte pavimento e in parte oggetto (gamba,
        # mobile): si sostituisce solo la parte di pavimento, togliendo il colore del vecchio
        # pavimento lì accanto invece del pixel intero. Così non resta l'alone chiaro del
        # vecchio pavimento intorno alle gambe. Con alpha = 1 il risultato è il materiale.
        old_floor = _old_floor_near_edges(rgb_view, sub_a, trc, lin_o, idx)
        rgb_view[idx] = from_linear(np.clip(lin_o + a * (lin_m - old_floor), 0.0, 1.0), trc)

    final.setflags(write=False)
    alpha = mask.copy()
    alpha.setflags(write=False)
    return CompositeResult(
        pixels=final, alpha=alpha, bbox=bbox, final_sha256=sha256_array(final), _seal=_SEAL
    )
