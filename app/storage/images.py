"""Codifica di immagini in memoria (per mostrarle agli agenti o servirle via web)."""

from __future__ import annotations

import io

from PIL import Image


def jpeg_bytes(img: Image.Image, quality: int = 88) -> bytes:
    """JPEG ricodificato da zero: nessun metadato o dato nascosto del file originale."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
