"""Proprietà: per QUALSIASI immagine e maschera, i pixel fuori maschera restano identici."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.core import compositor, ingest, validator
from app.core.models import MaterialLayer
from tests.conftest import confirmed


@st.composite
def scene(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    h = draw(st.integers(1, 24))
    w = draw(st.integers(1, 24))
    c = draw(st.sampled_from([3, 4]))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
    style = draw(st.sampled_from(["binary", "soft", "empty", "full", "pixel"]))
    if style == "binary":
        mask = (rng.random((h, w)) < 0.5).astype(np.uint8) * 255
    elif style == "soft":
        mask = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        mask[rng.random((h, w)) < 0.3] = 0
    elif style == "empty":
        mask = np.zeros((h, w), dtype=np.uint8)
    elif style == "full":
        mask = np.full((h, w), 255, dtype=np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[rng.integers(0, h), rng.integers(0, w)] = draw(st.integers(1, 255))
    color = tuple(draw(st.floats(0.0, 1.0)) for _ in range(3))
    return img, mask, color  # type: ignore[return-value]


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scene())
def test_outside_mask_always_identical(
    data: tuple[np.ndarray, np.ndarray, tuple[float, float, float]],
) -> None:
    img, mask, color = data
    src = ingest.from_array(img)
    h, w = mask.shape
    layer = MaterialLayer(
        rgb_linear=np.broadcast_to(np.asarray(color, dtype=np.float32), (h, w, 3)),
        bbox=(0, 0, w, h),
        material_id="p",
        render_mode="flat",
    )
    rec = confirmed(mask, src.meta.normalized_sha256)
    res = compositor.composite(src, mask, rec, layer)

    alpha = res.alpha
    final, original = res.pixels, src.pixels
    assert np.array_equal(final[alpha == 0], original[alpha == 0])
    assert final.shape == original.shape
    assert not np.any((alpha > 0) & (mask == 0))
    assert validator.validate(src, res, mask, rec).valid
