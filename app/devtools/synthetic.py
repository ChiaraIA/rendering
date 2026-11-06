"""Generatori di dati di prova: foto, maschere, EXIF, ICC, catalogo.

Tutto è deterministico (seed espliciti) e scritto via codice: nei test non ci sono binari.
"""

from __future__ import annotations

import io
import json
import struct
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


def make_exif(
    orientation: int | None = None,
    focal_35mm: int | None = None,
    make: str | None = None,
    model: str | None = None,
) -> Image.Exif:
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if focal_35mm is not None:
        exif.get_ifd(0x8769)[0xA405] = focal_35mm
    return exif


def save_image(
    arr: NDArray[np.uint8],
    path: Path,
    fmt: str = "PNG",
    icc: bytes | None = None,
    exif_orientation: int | None = None,
    quality: int = 95,
    focal_35mm: int | None = None,
    camera: tuple[str, str] | None = None,
) -> Path:
    img = Image.fromarray(np.ascontiguousarray(arr))
    kwargs: dict[str, object] = {}
    if icc is not None:
        kwargs["icc_profile"] = icc
    if exif_orientation is not None or focal_35mm is not None or camera is not None:
        make, model = camera or (None, None)
        kwargs["exif"] = make_exif(exif_orientation, focal_35mm, make, model)
    if fmt == "JPEG":
        kwargs["quality"] = quality
    img.save(path, format=fmt, **kwargs)
    return path


def save_heic(
    arr: NDArray[np.uint8],
    path: Path,
    exif_orientation: int | None = None,
    primaries: int = 1,
    transfer: int = 13,
    icc: bytes | None = None,
    focal_35mm: int | None = None,
    camera: tuple[str, str] | None = None,
) -> Path:
    """HEIC come da iPhone. Lossless (RGB 4:4:4, matrice identità) per confronti esatti."""
    img = Image.fromarray(np.ascontiguousarray(arr))
    if icc is None:
        img.info["nclx_profile"] = {
            "color_primaries": primaries,
            "transfer_characteristics": transfer,
            "matrix_coefficients": 0,
            "full_range_flag": 1,
        }
    make, model = camera or (None, None)
    exif = make_exif(exif_orientation, focal_35mm, make, model)
    kwargs: dict[str, object] = {"quality": -1, "chroma": 444, "exif": exif.tobytes()}
    if icc is not None:
        kwargs["icc_profile"] = icc
    img.save(path, format="HEIF", **kwargs)
    return path


def save_mpo(frames: list[NDArray[np.uint8]], path: Path, focal_35mm: int | None = None) -> Path:
    """JPEG multi-immagine (come Ultra HDR / Samsung): Pillow lo apre come MPO.

    Pillow perde il segmento MPF se gli si passa l'EXIF in un salvataggio MPO: scriviamo
    l'MPO senza EXIF e inseriamo l'APP1 Exif subito dopo l'APP0. Gli offset MPF sono
    relativi al proprio header, quindi restano validi.
    """
    imgs = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, format="MPO", save_all=True, append_images=imgs[1:], quality=95)
    data = buf.getvalue()
    if focal_35mm is not None:
        payload = make_exif(focal_35mm=focal_35mm).tobytes()
        if not payload.startswith(b"Exif\x00\x00"):
            payload = b"Exif\x00\x00" + payload
        app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        app0_end = 4 + struct.unpack(">H", data[4:6])[0]
        assert data[2:4] == b"\xff\xe0"
        data = data[:app0_end] + app1 + data[app0_end:]
    path.write_bytes(data)
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


def _clouds(h: int, w: int, rng: np.random.Generator, cells: int = 6) -> NDArray[np.float64]:
    """Rumore a bassa frequenza in [-1, 1] (griglia casuale ingrandita bicubica)."""
    small = rng.uniform(0, 255, size=(cells, max(2, cells * w // max(h, 1)))).astype(np.uint8)
    big = Image.fromarray(small).resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(big, dtype=np.float64) / 127.5 - 1.0


TextureKind = Literal["speckle", "wood", "marble"]


def make_tile_texture(
    size: int | tuple[int, int] = 128,
    base: tuple[int, int, int] = (128, 124, 118),
    seed: int = 0,
    kind: TextureKind = "speckle",
    marker: bool = False,
) -> NDArray[np.uint8]:
    """Texture procedurale di UNA piastrella (w, h in pixel). `marker` = quadratino rosso
    nell'angolo in alto a sinistra, utile per verificare rotazioni e orientamento."""
    w, h = (size, size) if isinstance(size, int) else size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    tone = rng.uniform(-10, 10)
    if kind == "wood":
        warp = _clouds(h, w, rng, cells=4) * 6
        grain = np.sin((yy + warp) / h * np.pi * rng.uniform(5, 9) + rng.uniform(0, 6))
        lum = tone + 14 * grain + 6 * _clouds(h, w, rng, cells=3)
    elif kind == "marble":
        turb = _clouds(h, w, rng, cells=5) * 3.0 + _clouds(h, w, rng, cells=9) * 1.2
        veins = np.abs(np.sin((xx + yy) / w * np.pi * 2.0 + turb))
        lum = tone - 38 * np.power(1 - veins, 8) + 4 * _clouds(h, w, rng, cells=4)
    else:
        lum = tone + 9 * _clouds(h, w, rng, cells=5)
    tex = np.asarray(base, dtype=np.float64) + lum[..., None]
    tex += rng.normal(0, 7, size=(h, w, 1))
    out = np.clip(np.rint(tex), 0, 255).astype(np.uint8)
    if marker:
        m = max(2, min(w, h) // 10)
        out[:m, :m] = (220, 30, 30)
    return out


@dataclass(frozen=True)
class SampleMaterial:
    id: str
    name: str
    base: tuple[int, int, int]
    kind: TextureKind
    tile_mm: tuple[float, float]
    joint_mm: float
    joint_color: str | None
    pattern: dict[str, object]
    n_variants: int
    variant_rotations: tuple[int, ...]


SAMPLE_MATERIALS = (
    SampleMaterial(
        "gres-grigio-30x30",
        "Gres grigio 30x30",
        (128, 124, 118),
        "speckle",
        (300, 300),
        3,
        "#b8b4ac",
        {"type": "grid"},
        4,
        (0, 90, 180, 270),
    ),
    SampleMaterial(
        "laminato-rovere-120x20",
        "Laminato rovere a incastro 120x20",
        (164, 120, 78),
        "wood",
        (1200, 200),
        0,
        None,
        {"type": "running_bond", "bond_offset": 0.33},
        3,
        (0, 180),
    ),
    SampleMaterial(
        "marmo-bianco-60x60",
        "Marmo bianco 60x60 in diagonale",
        (226, 224, 219),
        "marble",
        (600, 600),
        2,
        "#d9d6cf",
        {"type": "grid", "rotation_deg": 45},
        3,
        (0, 90, 180, 270),
    ),
)

TEXTURE_PX_PER_MM = 0.8
TEXTURE_MAX_SIDE = 768


def texture_size_px(tile_mm: tuple[float, float]) -> tuple[int, int]:
    scale = min(TEXTURE_PX_PER_MM, TEXTURE_MAX_SIDE / max(tile_mm))
    return (max(8, round(tile_mm[0] * scale)), max(8, round(tile_mm[1] * scale)))


def write_catalog(
    root: Path, materials: tuple[SampleMaterial, ...] = SAMPLE_MATERIALS, marker: bool = False
) -> Path:
    """Scrive catalog.json (schema 2.0) + textures/ con hash calcolati."""
    root = Path(root)
    (root / "textures").mkdir(parents=True, exist_ok=True)
    out: list[dict[str, object]] = []
    for i, m in enumerate(materials):
        variants = []
        for v in range(m.n_variants):
            tex_name = f"{m.id}-v{v + 1}.png"
            tex_path = root / "textures" / tex_name
            tex = make_tile_texture(
                texture_size_px(m.tile_mm), m.base, seed=100 * i + v, kind=m.kind, marker=marker
            )
            save_image(tex, tex_path)
            variants.append({"texture_file": tex_name, "texture_sha256": sha256_file(tex_path)})
        out.append(
            {
                "id": m.id,
                "name": m.name,
                "version": "1",
                "tile_w_mm": m.tile_mm[0],
                "tile_h_mm": m.tile_mm[1],
                "joint_mm": m.joint_mm,
                "joint_color": m.joint_color,
                "pattern": m.pattern,
                "variants": variants,
                "variant_rotations": list(m.variant_rotations),
                "texture_color_space": "srgb",
            }
        )
    path = root / "catalog.json"
    path.write_text(json.dumps({"schema_version": "2.0", "materials": out}, indent=2) + "\n")
    return path


if __name__ == "__main__":  # rigenera il catalogo di esempio distribuito col pacchetto
    print(write_catalog(Path(__file__).resolve().parents[1] / "catalog" / "data"))
