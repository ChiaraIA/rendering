"""Import nel catalogo: raddrizza la foto di una piastrella scattata in prospettiva."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core import ingest


def rectify_tile(src: Path, corners: str, w_cm: float, h_cm: float, dst: Path) -> None:
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
    inset = max(1, round(min(out_w, out_h) * 0.01))  # via il filo del bordo/smusso
    flat = flat[inset:-inset, inset:-inset]
    Image.fromarray(np.ascontiguousarray(flat)).resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    ).save(dst, format="PNG")
