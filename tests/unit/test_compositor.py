from __future__ import annotations

import numpy as np
import pytest

from app.core import compositor, ingest, masking
from app.core.colorspace import from_linear
from app.core.errors import IntegrityError, MaskNotConfirmedError
from app.core.models import LightMap, MaterialLayer
from app.devtools import synthetic
from tests.conftest import confirmed


def _layer(color: tuple[float, float, float], h: int, w: int) -> MaterialLayer:
    rgb = np.broadcast_to(np.asarray(color, dtype=np.float32), (h, w, 3))
    return MaterialLayer(rgb_linear=rgb, bbox=(0, 0, w, h), material_id="t", render_mode="flat")


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("kind", ["full", "pixel", "polygon", "soft", "holes"])
def test_outside_mask_identical_and_same_resolution(channels: int, kind: str) -> None:
    src = ingest.from_array(synthetic.make_photo(80, 60, channels=channels, seed=3))
    mask = synthetic.make_mask(60, 80, kind)  # type: ignore[arg-type]
    res = compositor.composite(
        src, mask, confirmed(mask, src.meta.normalized_sha256), _layer((0.2, 0.4, 0.6), 60, 80)
    )
    assert res.pixels.shape == src.pixels.shape
    assert res.pixels.dtype == src.pixels.dtype
    assert np.array_equal(res.pixels[res.alpha == 0], src.pixels[res.alpha == 0])
    if channels == 4:
        assert np.array_equal(res.pixels[..., 3], src.pixels[..., 3])


def test_full_alpha_gets_exact_material_color() -> None:
    src = ingest.from_array(synthetic.make_photo(32, 16))
    mask = synthetic.make_mask(16, 32, "full")
    color = (0.1, 0.5, 0.9)
    res = compositor.composite(
        src, mask, confirmed(mask, src.meta.normalized_sha256), _layer(color, 16, 32)
    )
    expected = from_linear(np.asarray([color]), "srgb")[0]
    assert np.all(res.pixels == expected)


def test_soft_edge_is_between_original_and_material() -> None:
    src = ingest.from_array(np.full((4, 4, 3), 10, dtype=np.uint8))
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 128
    mask[2, 2] = 255
    res = compositor.composite(
        src, mask, confirmed(mask, src.meta.normalized_sha256), _layer((1.0, 1.0, 1.0), 4, 4)
    )
    assert np.all(res.pixels[2, 2] == 255)
    assert 10 < int(res.pixels[1, 1, 0]) < 255
    # blend in luce lineare: 50% di bianco su quasi-nero è più chiaro del 50% codificato
    assert int(res.pixels[1, 1, 0]) > (10 + 255) // 2


def test_empty_mask_returns_original() -> None:
    src = ingest.from_array(synthetic.make_photo(20, 10))
    mask = np.zeros((10, 20), dtype=np.uint8)
    res = compositor.composite(
        src, mask, confirmed(mask, src.meta.normalized_sha256), _layer((0, 0, 0), 1, 1)
    )
    assert res.bbox is None
    assert np.array_equal(res.pixels, src.pixels)


def test_light_map_multiplies_material() -> None:
    src = ingest.from_array(synthetic.make_photo(8, 8))
    mask = synthetic.make_mask(8, 8, "full")
    gain = np.full((8, 8), 0.5, dtype=np.float32)
    res = compositor.composite(
        src,
        mask,
        confirmed(mask, src.meta.normalized_sha256),
        _layer((0.8, 0.8, 0.8), 8, 8),
        LightMap(gain=gain, bbox=(0, 0, 8, 8)),
    )
    assert np.all(res.pixels == from_linear(np.asarray([[0.4, 0.4, 0.4]]), "srgb")[0])
    with pytest.raises(IntegrityError, match="luce"):
        compositor.composite(
            src,
            mask,
            confirmed(mask, src.meta.normalized_sha256),
            _layer((0.8, 0.8, 0.8), 8, 8),
            LightMap(gain=gain[:4], bbox=(0, 0, 8, 4)),
        )


def test_requires_confirmed_mask() -> None:
    src = ingest.from_array(synthetic.make_photo(8, 8))
    mask = synthetic.make_mask(8, 8, "full")
    rec = masking.make_record(mask, src.meta.normalized_sha256)
    with pytest.raises(MaskNotConfirmedError):
        compositor.composite(src, mask, rec, _layer((0, 0, 0), 8, 8))


def test_rejects_mismatched_mask_and_uncovered_layer() -> None:
    src = ingest.from_array(synthetic.make_photo(8, 8))
    small = synthetic.make_mask(4, 8, "full")
    with pytest.raises(IntegrityError, match="maschera"):
        compositor.composite(
            src, small, confirmed(small, src.meta.normalized_sha256), _layer((0, 0, 0), 4, 8)
        )
    mask = synthetic.make_mask(8, 8, "full")
    with pytest.raises(IntegrityError, match="layer"):
        compositor.composite(
            src, mask, confirmed(mask, src.meta.normalized_sha256), _layer((0, 0, 0), 4, 4)
        )


def test_result_is_sealed_and_readonly() -> None:
    arr = ingest.readonly(np.zeros((2, 2, 3), dtype=np.uint8))
    a = ingest.readonly(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(TypeError, match="solo da"):
        compositor.CompositeResult(pixels=arr, alpha=a, bbox=None, final_sha256="x")
    with pytest.raises(TypeError, match="sola lettura"):
        compositor.CompositeResult(
            pixels=np.zeros((2, 2, 3), dtype=np.uint8),
            alpha=a,
            bbox=None,
            final_sha256="x",
            _seal=compositor._SEAL,
        )
    src = ingest.from_array(synthetic.make_photo(8, 8))
    mask = synthetic.make_mask(8, 8, "full")
    res = compositor.composite(
        src, mask, confirmed(mask, src.meta.normalized_sha256), _layer((0, 0, 0), 8, 8)
    )
    with pytest.raises(ValueError):
        res.pixels[0, 0, 0] = 1
    assert res.final_sha256 == ingest.sha256_array(res.pixels)


def test_deterministic() -> None:
    src = ingest.from_array(synthetic.make_photo(64, 48, channels=4, seed=1))
    mask = synthetic.make_mask(48, 64, "soft")
    rec = confirmed(mask, src.meta.normalized_sha256)
    a = compositor.composite(src, mask, rec, _layer((0.3, 0.2, 0.1), 48, 64))
    b = compositor.composite(src, mask, rec, _layer((0.3, 0.2, 0.1), 48, 64))
    assert a.final_sha256 == b.final_sha256
