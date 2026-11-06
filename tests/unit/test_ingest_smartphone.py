"""Foto da smartphone: HEIC (iPhone), nclx P3, MPO/Ultra HDR, EXIF della fotocamera."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core import icc, ingest
from app.core.models import ColorSpace
from app.devtools import synthetic


def _asym(h: int = 12, w: int = 20) -> np.ndarray:
    return (np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3) * 7 % 251).astype(np.uint8)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_heic_orientation_is_applied_once(tmp_path: Path, orientation: int) -> None:
    stored = _asym()
    p = synthetic.save_heic(stored, tmp_path / "IMG_0001.HEIC", exif_orientation=orientation)
    src = ingest.load_image(p)
    expected = synthetic.DISPLAY_FROM_STORED[orientation](stored)
    assert src.meta.source_format == "HEIF"
    assert src.meta.exif_orientation == orientation
    assert np.array_equal(src.pixels, expected)  # nessuna doppia rotazione


def test_heic_nclx_p3_gets_equivalent_icc(tmp_path: Path) -> None:
    p = synthetic.save_heic(_asym(), tmp_path / "p3.heic", primaries=12)
    src = ingest.load_image(p)
    assert src.meta.color_space is ColorSpace.DISPLAY_P3
    assert src.meta.color_origin == "nclx"
    assert src.icc == icc.display_p3()
    assert any("nclx" in w for w in ingest.colorspace_warnings(src.meta))


@pytest.mark.parametrize(
    ("primaries", "transfer", "cs"),
    [(1, 13, ColorSpace.SRGB), (9, 13, ColorSpace.UNKNOWN), (12, 16, ColorSpace.UNKNOWN)],
)
def test_heic_nclx_other_spaces(
    tmp_path: Path, primaries: int, transfer: int, cs: ColorSpace
) -> None:
    p = synthetic.save_heic(_asym(), tmp_path / "x.heic", primaries=primaries, transfer=transfer)
    src = ingest.load_image(p)
    assert src.meta.color_space is cs and src.icc is None


def test_heic_with_embedded_icc_keeps_it(tmp_path: Path) -> None:
    p = synthetic.save_heic(_asym(), tmp_path / "icc.heic", icc=icc.display_p3())
    src = ingest.load_image(p)
    assert src.icc == icc.display_p3() and src.meta.color_origin == "icc"


@pytest.mark.parametrize("fmt", ["JPEG", "HEIF"])
def test_camera_exif_is_read(tmp_path: Path, fmt: str) -> None:
    arr = _asym()
    cam = ("Apple", "iPhone 15")
    if fmt == "HEIF":
        p = synthetic.save_heic(arr, tmp_path / "c.heic", focal_35mm=26, camera=cam)
    else:
        p = synthetic.save_image(arr, tmp_path / "c.jpg", fmt="JPEG", focal_35mm=26, camera=cam)
    meta = ingest.load_image(p).meta
    assert meta.focal_35mm == 26 and meta.camera == "Apple iPhone 15"


def test_no_exif_means_no_focal(tmp_path: Path) -> None:
    meta = ingest.load_image(synthetic.save_image(_asym(), tmp_path / "w.jpg", fmt="JPEG")).meta
    assert meta.focal_35mm is None and meta.camera is None


def test_mpo_uses_primary_frame(tmp_path: Path) -> None:
    base = synthetic.make_photo(64, 48, seed=1)
    gain = np.full((24, 32, 3), 200, dtype=np.uint8)
    p = synthetic.save_mpo([base, gain], tmp_path / "ultrahdr.jpg", focal_35mm=24)
    src = ingest.load_image(p)
    assert src.meta.source_format == "MPO"
    assert (src.meta.width, src.meta.height) == (64, 48)
    assert src.meta.extra_images_ignored and src.meta.focal_35mm == 24
    first = np.asarray(Image.open(p).convert("RGB"))
    assert np.array_equal(src.pixels, first)
    assert any("gain map" in w for w in ingest.colorspace_warnings(src.meta))
