"""Import nel catalogo: raddrizza la foto di una piastrella scattata in prospettiva e toglie
la luce dello studio fotografico (riflessi, vignettatura), che altrimenti si ripeterebbe
uguale su ogni piastrella posata."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.core import ingest
from app.core.colorspace import from_linear, to_linear


def flatten_lighting(tex: NDArray[np.uint8], strength: float = 1.0) -> NDArray[np.uint8]:
    """Toglie la variazione di luce a bassa frequenza, mantenendo il dettaglio.

    Si lavora in luce lineare, canale per canale: ogni canale viene sfocato molto (~1/6 del
    lato) e riportato alla sua media. Così spariscono sia lo schiarimento sia lo sbiadimento
    del colore causati da riflessi e vignettatura; il dettaglio fine resta intatto.
    Adatto a piastrelle a tinta unita o maculate; per decori grandi usare --mantieni-luce.
    """
    lin = to_linear(np.ascontiguousarray(tex[..., :3]), "srgb").astype(np.float32)
    sigma = max(tex.shape[:2]) / 6
    low = cv2.GaussianBlur(lin, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    mean = low.reshape(-1, 3).mean(axis=0)
    gain = mean / np.maximum(low, 1e-6)
    gain = 1.0 + (gain - 1.0) * strength
    out = np.clip(lin * gain, 0.0, 1.0)
    return from_linear(out.astype(np.float64), "srgb")


def rectify_tile(
    src: Path, corners: str, w_cm: float, h_cm: float, dst: Path, flatten: bool = True
) -> None:
    """Raddrizza la faccia della piastrella fotografata in prospettiva (vista frontale)."""
    pts = np.array(
        [[float(v) for v in p.split(",")] for p in corners.replace(";", " ").split()],
        dtype=np.float32,
    )
    if pts.shape != (4, 2):
        raise ValueError("--angoli: servono 4 punti x,y")
    img = ingest.load_image(src).pixels[..., :3]
    scale = min(1024 / w_cm, 1024 / h_cm, 30.0)  # px per cm, lato massimo 1024 px
    out_w, out_h = round(w_cm * scale), round(h_cm * scale)
    target = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, target)
    flat = cv2.warpPerspective(
        np.ascontiguousarray(img),
        m,
        (out_w, out_h),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inset = max(1, round(min(out_w, out_h) * 0.03))  # via smusso e riflessi del bordo
    flat = flat[inset:-inset, inset:-inset]
    if flatten:
        flat = flatten_lighting(flat)
    Image.fromarray(np.ascontiguousarray(flat)).resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    ).save(dst, format="PNG")
