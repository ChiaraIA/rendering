"""Import di foto di piastrelle: raddrizzamento prospettico e rimozione della luce di studio."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.devtools import synthetic
from app.storage.catalog_import import flatten_lighting, rectify_tile

CORNERS = np.float32([[128, 63], [718, 60], [800, 640], [45, 640]])


def _perspective_photo(tex: np.ndarray, path: Path) -> Path:
    bg = np.full((735, 861, 3), 235, np.uint8)
    h, w = tex.shape[:2]
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), CORNERS)
    warp = cv2.warpPerspective(tex, m, (861, 735))
    inside = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, (861, 735)) > 0
    bg[inside] = warp[inside]
    Image.fromarray(bg).save(path)
    return path


def test_rectify_restores_frontal_tile(tmp_path: Path) -> None:
    tex = synthetic.make_tile_texture(600, (40, 70, 160), seed=3, marker=True)
    src = _perspective_photo(tex, tmp_path / "foto.png")
    dst = tmp_path / "tex.png"
    corners = " ".join(f"{x:g},{y:g}" for x, y in CORNERS)
    rectify_tile(src, corners, 20, 20, dst, flatten=False)
    out = np.asarray(Image.open(dst))
    assert out.shape[0] == out.shape[1]  # 20x20 cm -> quadrata
    assert out[5, 5, 0] > 150 and out[5, 5, 2] < 80  # marcatore rosso ancora in alto a sinistra
    centre = out[out.shape[0] // 2, out.shape[1] // 2].astype(int)
    assert abs(centre[2] - 160) < 40 and centre[2] > centre[0]  # resta blu


def test_flatten_removes_glare_but_keeps_detail() -> None:
    rng = np.random.default_rng(0)
    base = np.full((256, 256, 3), (40, 70, 160), np.float64)
    detail = rng.normal(0, 12, (256, 256, 1))
    yy, xx = np.mgrid[0:256, 0:256]
    glare = 90 * np.exp(-(((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 60.0**2)))[..., None]
    tex = np.clip(base + detail + glare, 0, 255).astype(np.uint8)
    flat = flatten_lighting(tex).astype(float)
    corner, far = flat[20:60, 20:60].mean(), flat[190:230, 190:230].mean()
    assert abs(corner - far) < 0.25 * abs(tex[20:60, 20:60].mean() - tex[190:230, 190:230].mean())
    assert flat[100:150, 100:150, 2].std() > 5  # il dettaglio fine resta
