from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core import segmentation
from app.core.errors import MaskInvalidError, MaskNotConfirmedError
from app.core.models import MaskStatus
from app.devtools import synthetic


def test_load_valid_masks(tmp_path: Path) -> None:
    m = synthetic.make_mask(20, 30, "soft")
    p = synthetic.save_image(m, tmp_path / "m.png")
    loaded = segmentation.load_mask_file(p, (20, 30))
    assert np.array_equal(loaded, m) and not loaded.flags.writeable

    rgb = np.repeat(m[..., None], 3, axis=-1)
    assert np.array_equal(
        segmentation.load_mask_file(synthetic.save_image(rgb, tmp_path / "rgb.png"), (20, 30)), m
    )

    bw = tmp_path / "bw.png"
    Image.fromarray(m > 0).save(bw)
    assert set(np.unique(segmentation.load_mask_file(bw, (20, 30)))) == {0, 255}


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("size", "dimensioni"),
        ("empty", "vuota"),
        ("color", "scala di grigi"),
        ("rgba", "modalità"),
        ("exif", "orientamento"),
        ("garbage", "illeggibile"),
    ],
)
def test_reject_bad_masks(tmp_path: Path, kind: str, match: str) -> None:
    p = tmp_path / "m.png"
    m = synthetic.make_mask(20, 30, "polygon")
    if kind == "size":
        synthetic.save_image(m[:-1], p)
    elif kind == "empty":
        synthetic.save_image(np.zeros_like(m), p)
    elif kind == "color":
        rgb = np.repeat(m[..., None], 3, axis=-1)
        rgb[0, 0] = (1, 2, 3)
        synthetic.save_image(rgb, p)
    elif kind == "rgba":
        synthetic.save_image(np.zeros((20, 30, 4), dtype=np.uint8), p)
    elif kind == "exif":
        synthetic.save_image(m, p, exif_orientation=6)
    else:
        p.write_bytes(b"xx")
    with pytest.raises(MaskInvalidError, match=match):
        segmentation.load_mask_file(p, (20, 30))


def test_record_and_confirmation_flow() -> None:
    m = synthetic.make_mask(20, 30, "soft")
    rec = segmentation.make_record(m, "img")
    assert rec.status is MaskStatus.PROPOSED
    assert rec.area_px == int(np.count_nonzero(m)) and rec.soft_edge_px > 0
    with pytest.raises(MaskNotConfirmedError, match="almeno"):
        segmentation.confirm(rec, rec.mask_sha256[:4])
    with pytest.raises(MaskNotConfirmedError, match="non corrisponde"):
        segmentation.confirm(rec, "f" * 12 if not rec.mask_sha256.startswith("f") else "0" * 12)
    ok = segmentation.confirm(rec, rec.mask_sha256[:12].upper())
    assert ok.status is MaskStatus.CONFIRMED and ok.confirmed_at

    segmentation.require_confirmed(ok, m, "img")
    with pytest.raises(MaskNotConfirmedError, match="assente"):
        segmentation.require_confirmed(None, m, "img")
    with pytest.raises(MaskNotConfirmedError, match="non confermata"):
        segmentation.require_confirmed(rec, m, "img")
    changed = m.copy()
    changed[0, 0] ^= 1
    with pytest.raises(MaskNotConfirmedError, match="cambiata"):
        segmentation.require_confirmed(ok, changed, "img")
    with pytest.raises(MaskNotConfirmedError, match="altra immagine"):
        segmentation.require_confirmed(ok, m, "other")


def test_make_record_rejects_bad_arrays() -> None:
    with pytest.raises(MaskInvalidError):
        segmentation.make_record(np.zeros((2, 2, 1), dtype=np.uint8), "x")
    assert segmentation.bbox_of(np.zeros((3, 3), dtype=np.uint8)) is None
    m = np.zeros((5, 7), dtype=np.uint8)
    m[1:3, 2:6] = 9
    assert segmentation.bbox_of(m) == (2, 1, 6, 3)
