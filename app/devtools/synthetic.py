"""Generatori di dati di prova: foto, maschere, EXIF, ICC, catalogo.

Tutto è deterministico (seed espliciti) e scritto via codice: nei test non ci sono binari.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageDraw

from app.core.ingest import sha256_file

MaskKind = Literal["empty", "full", "pixel", "polygon", "soft", "holes"]


def make_photo(w: int, h: int, channels: int = 3, seed: int = 0) -> NDArray[np.uint8]:
    """Rumore + gradienti + bordi netti: qualunque pixel alterato si nota."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            (xx * 255 // max(w - 1, 1)),
            (yy * 255 // max(h - 1, 1)),
            ((xx + yy) * 127 // max(w + h - 2, 1)),
        ],
        axis=-1,
    ).astype(np.int16)
    noise = rng.integers(-40, 41, size=(h, w, 3), dtype=np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    img[h // 3 : h // 3 + max(1, h // 20), :, :] = (250, 20, 20)  # bordo netto
    if channels == 4:
        a = rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)
        a[: h // 4] = 255
        a[h // 4 : h // 4 + 2] = 0  # completamente trasparente
        img = np.concatenate([img, a], axis=-1)
    return img


def make_mask(h: int, w: int, kind: MaskKind = "polygon", seed: int = 0) -> NDArray[np.uint8]:
    m = np.zeros((h, w), dtype=np.uint8)
    if kind == "empty":
        return m
    if kind == "full":
        m[:] = 255
        return m
    if kind == "pixel":
        rng = np.random.default_rng(seed)
        m[rng.integers(0, h), rng.integers(0, w)] = 255
        return m
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    floor = [(0, h - 1), (w * 0.2, h * 0.55), (w * 0.8, h * 0.55), (w - 1, h - 1)]
    draw.polygon(floor, fill=255)
    m = np.asarray(img, dtype=np.uint8).copy()
    if kind == "holes":
        m[int(h * 0.75) : int(h * 0.85), int(w * 0.4) : int(w * 0.6)] = 0  # "mobile" sul pavimento
    if kind == "soft":
        # bordo morbido solo verso l'interno: erosione progressiva
        inner = m > 0
        for v in (64, 128, 192):
            eroded = inner.copy()
            eroded[1:, :] &= inner[:-1, :]
            eroded[:-1, :] &= inner[1:, :]
            eroded[:, 1:] &= inner[:, :-1]
            eroded[:, :-1] &= inner[:, 1:]
            m[inner & ~eroded] = np.where(m[inner & ~eroded] == 255, v, m[inner & ~eroded])
            inner = eroded
    return m


@cache
def srgb_icc() -> bytes:
    """Calcolato una volta: LittleCMS scrive data e ora di creazione nell'header ICC."""
    return bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


def save_image(
    arr: NDArray[np.uint8],
    path: Path,
    fmt: str = "PNG",
    icc: bytes | None = None,
    exif_orientation: int | None = None,
    quality: int = 95,
) -> Path:
    img = Image.fromarray(np.ascontiguousarray(arr))
    kwargs: dict[str, object] = {}
    if icc is not None:
        kwargs["icc_profile"] = icc
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    if fmt == "JPEG":
        kwargs["quality"] = quality
    img.save(path, format=fmt, **kwargs)
    return path


# Mappatura EXIF -> come appare l'immagine memorizzata (dalla specifica EXIF, non da Pillow).
DISPLAY_FROM_STORED: dict[int, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]] = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1, :],
    5: lambda a: a.transpose(1, 0, 2),
    6: lambda a: np.rot90(a, -1),
    7: lambda a: np.rot90(a, 2).transpose(1, 0, 2),
    8: lambda a: np.rot90(a, 1),
}


def make_tile_texture(
    size: int = 128, base: tuple[int, int, int] = (128, 124, 118), seed: int = 0
) -> NDArray[np.uint8]:
    """Gres "maculato" procedurale con marker d'angolo (per verificare l'orientamento in M2)."""
    rng = np.random.default_rng(seed)
    tex = np.empty((size, size, 3), dtype=np.int16)
    tex[:] = base
    tex += rng.integers(-18, 19, size=(size, size, 1), dtype=np.int16)
    tex[:6, :6] = (200, 40, 40)
    return np.clip(tex, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class SampleMaterial:
    id: str
    name: str
    base: tuple[int, int, int]
    tile_mm: tuple[float, float]


SAMPLE_MATERIALS = (
    SampleMaterial("gres-grigio-30x30", "Gres grigio 30x30", (128, 124, 118), (300, 300)),
    SampleMaterial("cotto-20x40", "Cotto 20x40", (176, 92, 60), (200, 400)),
    SampleMaterial("marmo-bianco-60x60", "Marmo bianco 60x60", (232, 230, 226), (600, 600)),
)


def write_catalog(root: Path, materials: tuple[SampleMaterial, ...] = SAMPLE_MATERIALS) -> Path:
    """Scrive catalog.json + textures/ con hash calcolati. Ritorna il path del catalogo."""
    root = Path(root)
    (root / "textures").mkdir(parents=True, exist_ok=True)
    out: list[dict[str, object]] = []
    for i, m in enumerate(materials):
        tex_name = f"{m.id}.png"
        tex_path = root / "textures" / tex_name
        save_image(make_tile_texture(base=m.base, seed=i), tex_path)
        out.append(
            {
                "id": m.id,
                "name": m.name,
                "version": "1",
                "texture_file": tex_name,
                "texture_sha256": sha256_file(tex_path),
                "texture_color_space": "srgb",
                "tile_w_mm": m.tile_mm[0],
                "tile_h_mm": m.tile_mm[1],
                "joint_default_mm": 3,
                "joint_range_mm": [2, 5],
                "joint_color": "#b8b4ac",
                "allowed_patterns": ["grid", "running_bond_50"],
            }
        )
    path = root / "catalog.json"
    path.write_text(json.dumps({"schema_version": "1.0", "materials": out}, indent=2) + "\n")
    return path


if __name__ == "__main__":  # rigenera il catalogo di esempio distribuito col pacchetto
    print(write_catalog(Path(__file__).resolve().parents[1] / "catalog" / "data"))
