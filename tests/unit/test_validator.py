"""Iniezione di guasti: ogni manomissione deve rendere il risultato non valido."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core import compositor, ingest, masking, validator
from app.core.models import MaterialLayer, ValidationReport
from app.devtools import synthetic
from tests.conftest import confirmed, forge


def _setup(channels: int = 3):  # type: ignore[no-untyped-def]
    src = ingest.from_array(synthetic.make_photo(40, 30, channels=channels, seed=2))
    mask = synthetic.make_mask(30, 40, "polygon")
    rec = confirmed(mask, src.meta.normalized_sha256)
    layer = MaterialLayer(
        rgb_linear=np.broadcast_to(np.float32([0.5, 0.5, 0.5]), (30, 40, 3)),
        bbox=(0, 0, 40, 30),
        material_id="t",
        render_mode="flat",
    )
    res = compositor.composite(src, mask, rec, layer)
    return src, mask, rec, res


def _failed(rep: ValidationReport) -> set[str]:
    return {c.name for c in rep.checks if not c.passed}


def test_valid_result_passes() -> None:
    src, mask, rec, res = _setup(channels=4)
    rep = validator.validate(src, res, mask, rec)
    assert rep.valid and _failed(rep) == set()
    assert "photo_alpha_preserved" in {c.name for c in rep.checks}


def test_one_lsb_outside_mask_is_caught() -> None:
    src, mask, rec, res = _setup()
    bad = res.pixels.copy()
    ys, xs = np.nonzero(mask == 0)
    bad[ys[0], xs[0], 1] ^= 1
    rep = validator.validate(src, forge(bad, res.alpha), mask, rec)
    assert not rep.valid
    assert _failed(rep) == {"outside_mask_identical"}
    assert next(c for c in rep.checks if c.name == "outside_mask_identical").value == 1


def test_resolution_change_is_caught() -> None:
    src, mask, rec, res = _setup()
    rep = validator.validate(src, forge(res.pixels[:-1], res.alpha[:-1]), mask, rec)
    assert {
        "outside_mask_identical",
        "resolution_match",
        "pixel_format_match",
        "alpha_within_mask",
    } <= _failed(rep)


def test_alpha_leaking_outside_mask_is_caught() -> None:
    src, mask, rec, res = _setup()
    alpha = res.alpha.copy()
    ys, xs = np.nonzero(mask == 0)
    alpha[ys[0], xs[0]] = 255
    rep = validator.validate(src, forge(res.pixels, alpha), mask, rec)
    assert "alpha_within_mask" in _failed(rep)


def test_photo_alpha_change_is_caught() -> None:
    src, mask, rec, res = _setup(channels=4)
    bad = res.pixels.copy()
    ys, xs = np.nonzero(mask)
    bad[ys[0], xs[0], 3] ^= 0xFF
    rep = validator.validate(src, forge(bad, res.alpha), mask, rec)
    assert "photo_alpha_preserved" in _failed(rep)


def test_hash_tampering_is_caught() -> None:
    src, mask, rec, res = _setup()
    rep = validator.validate(src, forge(res.pixels, res.alpha, sha="0" * 64), mask, rec)
    assert _failed(rep) == {"final_hash_consistent"}


def test_unconfirmed_or_changed_mask_is_caught() -> None:
    src, mask, rec, res = _setup()
    proposed = masking.make_record(mask, src.meta.normalized_sha256)
    assert "mask_approved" in _failed(validator.validate(src, res, mask, proposed))
    other = mask.copy()
    other[0, 0] = 255
    assert "mask_approved" in _failed(validator.validate(src, res, other, rec))


def test_modified_source_copy_is_caught(tmp_path: Path) -> None:
    src, mask, rec, res = _setup()
    f = tmp_path / "copy.bin"
    f.write_bytes(b"not the original")
    rep = validator.validate(src, res, mask, rec, source_copy=f)
    assert "original_unchanged" in _failed(rep)
