"""Luce: ombre e sfumature a bassa frequenza estratte dalla foto, senza il vecchio pattern.

Si misura la luminanza (lineare) del pavimento originale dentro la maschera, la si
"spalma" con un filtro molto largo (mediana + gaussiana normalizzata dalla maschera, così
i bordi non prendono luce da muri e mobili) e la si normalizza sulla mediana.
Il risultato moltiplica il materiale: le zone in ombra restano in ombra, le fughe e le
venature del vecchio pavimento no.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.colorspace import to_linear, trc_for
from app.core.models import BBox, LightMap, SourceImage

GAIN_MIN, GAIN_MAX = 0.35, 1.6
WORK_SIDE = 256  # lato di lavoro: la luce è a bassa frequenza, non serve di più


def light_map(
    source: SourceImage, mask: NDArray[np.uint8], bbox: BBox, strength: float = 1.0
) -> LightMap:
    x0, y0, x1, y1 = bbox
    rgb = source.pixels[y0:y1, x0:x1, :3]
    lin = to_linear(rgb, trc_for(source.meta.color_space))
    lum = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)
    m = (mask[y0:y1, x0:x1] >= 128).astype(np.float32)
    h, w = lum.shape
    if m.sum() < 16:
        return LightMap(gain=np.ones((h, w), np.float32), bbox=bbox)

    scale = min(1.0, WORK_SIDE / max(h, w))
    sw, sh = max(4, round(w * scale)), max(4, round(h * scale))
    small_l = cv2.resize(lum * m, (sw, sh), interpolation=cv2.INTER_AREA)
    small_m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    filled = np.where(small_m > 0.05, small_l / np.maximum(small_m, 1e-6), 0).astype(np.float32)
    # mediana: elimina fughe, venature e piccoli oggetti
    med = np.where(small_m > 0.05, filled, np.median(filled[small_m > 0.05])).astype(np.float32)
    for _ in range(max(1, min(sw, sh) // 40)):  # mediane ripetute = mediana larga
        med = cv2.medianBlur(med, 5)
    sigma = max(sw, sh) / 10
    num = cv2.GaussianBlur(med * small_m, (0, 0), sigma)
    den = cv2.GaussianBlur(small_m, (0, 0), sigma)
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0)
    ref = float(np.median(smooth[small_m > 0.5])) if np.any(small_m > 0.5) else 1.0
    gain_small = np.clip(smooth / max(ref, 1e-6), GAIN_MIN, GAIN_MAX).astype(np.float32)
    gain = cv2.resize(gain_small, (w, h), interpolation=cv2.INTER_CUBIC)
    gain = np.clip(1.0 + (gain - 1.0) * strength, GAIN_MIN, GAIN_MAX).astype(np.float32)
    return LightMap(gain=gain, bbox=bbox)


def contact_shadow(
    mask: NDArray[np.uint8],
    bbox: BBox,
    px_per_mm: NDArray[np.float32] | float,
    radius_mm: float = 260.0,
    depth: float = 0.42,
) -> NDArray[np.float32]:
    """Occlusione ambientale vicino a ciò che sta sul pavimento o lo delimita (mobili, muri).

    La distanza dal bordo della maschera viene misurata in millimetri reali: l'ombra ha la
    stessa larghezza fisica vicino e lontano dalla camera. I bordi dell'immagine non contano.
    """
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    # fuori dall'immagine = pavimento (il bordo della foto non è un oggetto)
    padded = np.pad((mask >= 128).astype(np.uint8), 1, constant_values=1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1][y0:y1, x0:x1]
    dist_mm = dist / np.maximum(px_per_mm, 1e-6)
    ao = 1.0 - depth * np.exp(-((dist_mm / radius_mm) ** 1.3))
    return ao.astype(np.float32)


def combined_light(
    source: SourceImage,
    mask: NDArray[np.uint8],
    bbox: BBox,
    px_per_mm: NDArray[np.float32] | float,
) -> LightMap:
    base = light_map(source, mask, bbox)
    gain = base.gain * contact_shadow(mask, bbox, px_per_mm)
    return LightMap(gain=np.clip(gain, GAIN_MIN, GAIN_MAX).astype(np.float32), bbox=bbox)
