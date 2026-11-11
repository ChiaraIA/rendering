"""Codifica di immagini in memoria (per mostrarle agli agenti o servirle via web)."""

from __future__ import annotations

import io

from PIL import Image


def jpeg_bytes(img: Image.Image, quality: int = 88) -> bytes:
    """JPEG ricodificato da zero: nessun metadato o dato nascosto del file originale.

    Pillow, salvando, ricopia da `info` alcuni campi dell'immagine d'origine (il commento
    JPEG, per esempio): si salva quindi un'immagine nuova che ha solo i pixel."""
    rgb = img.convert("RGB")
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
