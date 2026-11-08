"""Segmentazione semantica: logica delle maschere con probabilità finte (niente modello)."""

from __future__ import annotations

import numpy as np

from app.adapters import semantic


def _fake(h: int = 60, w: int = 80) -> semantic.Segmentation:
    labels = {"wall": 0, "floor": 1, "chair": 2}
    probs = np.zeros((3, h, w), np.float32)
    probs[0, : h // 2] = 1.0  # parete in alto
    probs[1, h // 2 :] = 1.0  # pavimento in basso
    # gamba di sedia sul pavimento: il pavimento è ancora "vincente" ma la sedia è probabile
    probs[1, h // 2 :, 40:44] = 0.6
    probs[2, h // 2 :, 40:44] = 0.4
    return semantic.Segmentation(probs=probs, labels=labels)


def test_floor_mask_and_objects_win() -> None:
    img = np.full((60, 80, 3), 128, np.uint8)
    m = semantic.surface_mask(img, "pavimento", seg=_fake(), refine=False)
    assert m.shape == (60, 80)
    assert m[50, 10] == 255 and m[10, 10] == 0  # pavimento sì, parete no
    assert m[50, 42] == 0  # la gamba della sedia esce dalla maschera anche se "floor" è 0.6


def test_wall_and_countertop_are_supported_surfaces() -> None:
    img = np.full((60, 80, 3), 128, np.uint8)
    wall = semantic.surface_mask(img, "parete", seg=_fake(), refine=False)
    assert wall[10, 10] == 255 and wall[50, 10] == 0
    assert "piano_di_lavoro" in semantic.SURFACES
