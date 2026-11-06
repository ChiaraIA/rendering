from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from app import config
from app.core import ingest
from app.core.errors import UnsupportedImageError
from app.core.models import ColorSpace
from app.devtools import synthetic


def _asym(h: int = 6, w: int = 10) -> np.ndarray:
    return (np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3) % 251).astype(np.uint8)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_png(tmp_path: Path, orientation: int) -> None:
    stored = _asym()
    p = synthetic.save_image(stored, tmp_path / "o.png", exif_orientation=orientation)
    src = ingest.load_image(p)
    expected = synthetic.DISPLAY_FROM_STORED[orientation](stored)
    assert src.meta.exif_orientation == orientation
    assert (src.meta.height, src.meta.width) == expected.shape[:2]
    assert np.array_equal(src.pixels, expected)


def test_exif_orientation_jpeg_dimensions(tmp_path: Path) -> None:
    p = synthetic.save_image(_asym(40, 64), tmp_path / "o.jpg", fmt="JPEG", exif_orientation=6)
    src = ingest.load_image(p)
    assert (src.meta.width, src.meta.height) == (40, 64)
    assert src.meta.source_format == "JPEG"


def test_hashes_and_readonly(tmp_path: Path) -> None:
    p = synthetic.save_image(synthetic.make_photo(20, 10), tmp_path / "a.png")
    src = ingest.load_image(p)
    assert src.meta.source_sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    assert src.meta.normalized_sha256 == ingest.sha256_array(src.pixels)
    assert not src.pixels.flags.writeable
    with pytest.raises(ValueError):
        src.pixels[0, 0, 0] = 0
    # stessa sequenza di byte, forma diversa => hash diverso
    flat = np.zeros((2, 6), dtype=np.uint8)
    assert ingest.sha256_array(flat) != ingest.sha256_array(flat.reshape(3, 4))


def test_rgba_png_keeps_alpha(tmp_path: Path) -> None:
    arr = synthetic.make_photo(16, 12, channels=4)
    src = ingest.load_image(synthetic.save_image(arr, tmp_path / "a.png"))
    assert src.meta.channels == 4
    assert np.array_equal(src.pixels, arr)


def test_srgb_icc_is_kept_byte_for_byte(tmp_path: Path) -> None:
    icc = synthetic.srgb_icc()
    src = ingest.load_image(
        synthetic.save_image(synthetic.make_photo(8, 8), tmp_path / "a.png", icc=icc)
    )
    assert src.icc == icc
    assert src.meta.color_space is ColorSpace.SRGB
    assert src.meta.icc_sha256 == hashlib.sha256(icc).hexdigest()


def test_no_icc_is_assumed_srgb_with_warning(tmp_path: Path) -> None:
    src = ingest.load_image(synthetic.save_image(synthetic.make_photo(8, 8), tmp_path / "a.png"))
    assert src.meta.color_space is ColorSpace.ASSUMED_SRGB
    assert any("assunto sRGB" in w for w in ingest.colorspace_warnings(src.meta))


def test_palette_and_gray_are_normalized(tmp_path: Path) -> None:
    pal = Image.fromarray(synthetic.make_photo(8, 8)).convert("P")
    pal.info["transparency"] = 0
    p1 = tmp_path / "p.png"
    pal.save(p1, transparency=0)
    assert ingest.load_image(p1).meta.channels == 4

    gray = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8))
    p2 = tmp_path / "g.png"
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    gray.save(p2, icc_profile=lab)  # profilo non RGB su immagine L
    src = ingest.load_image(p2)
    assert src.meta.channels == 3 and src.icc is None and src.meta.icc_dropped
    assert np.array_equal(src.pixels[..., 0], src.pixels[..., 2])
    assert any("scartato" in w for w in ingest.colorspace_warnings(src.meta))


@pytest.mark.parametrize(
    ("mode", "fmt", "match"),
    [("CMYK", "TIFF", "non supportata"), ("I;16", "PNG", "16 bit"), ("RGB", "BMP", "formato")],
)
def test_rejected_inputs(tmp_path: Path, mode: str, fmt: str, match: str) -> None:
    p = tmp_path / f"x.{fmt.lower()}"
    Image.new(mode, (4, 4)).save(p, format=fmt)
    with pytest.raises(UnsupportedImageError, match=match):
        ingest.load_image(p)


def test_garbage_missing_and_too_big(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(UnsupportedImageError, match="decodificare"):
        ingest.load_image(bad)
    with pytest.raises(UnsupportedImageError, match="non trovato"):
        ingest.load_image(tmp_path / "missing.png")
    monkeypatch.setattr(config, "MAX_PIXELS", 10)
    with pytest.raises(UnsupportedImageError, match="troppo grande"):
        ingest.load_image(synthetic.save_image(synthetic.make_photo(8, 8), tmp_path / "a.png"))


def test_multiframe_rejected(tmp_path: Path) -> None:
    p = tmp_path / "m.tiff"
    frames = [Image.new("RGB", (4, 4), c) for c in ("red", "blue")]
    frames[0].save(p, format="TIFF", save_all=True, append_images=frames[1:])
    with pytest.raises(UnsupportedImageError, match="multi-frame"):
        ingest.load_image(p)


def test_from_array_validation() -> None:
    with pytest.raises(UnsupportedImageError):
        ingest.from_array(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(UnsupportedImageError):
        ingest.from_array(np.zeros((4, 4, 3), dtype=np.uint16))  # type: ignore[arg-type]


def test_decode_from_bytes_matches_file(tmp_path: Path) -> None:
    p = synthetic.save_image(synthetic.make_photo(8, 8), tmp_path / "a.png")
    a = ingest.load_image(p)
    b = ingest.decode_image(io.BytesIO(p.read_bytes()).getvalue(), "a.png")
    assert a.meta == b.meta
