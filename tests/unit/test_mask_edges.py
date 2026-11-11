"""Contorni della maschera: bordi portati sulla foto, oggetti nei riquadri, zone a rischio."""

from __future__ import annotations

import cv2
import numpy as np

from app.adapters import semantic
from app.core import masking, overlays


def _floor_with_leg(h: int = 240, w: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Pavimento chiaro con venature, muro in alto, una gamba scura e la sua ombra."""
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), (186, 170, 150), np.float32)
    img += rng.normal(0, 4, img.shape)
    img[:80] = (232, 228, 220)  # muro
    yy, xx = np.mgrid[0:h, 0:w]
    shadow = np.exp(-(((xx - 190) / 26.0) ** 2 + ((yy - 200) / 12.0) ** 2))
    img *= 1 - 0.45 * shadow[..., None]  # ombra: più scura, stesso colore
    img[90:205, 158:170] = (52, 40, 34)  # gamba di una sedia
    truth = np.zeros((h, w), bool)
    truth[80:] = True
    truth[90:205, 158:170] = False
    return np.clip(img, 0, 255).astype(np.uint8), truth


def test_snap_moves_a_sloppy_border_onto_the_real_edge() -> None:
    img, truth = _floor_with_leg()
    # il "modello" sbaglia di 5 px il bordo col muro e allarga la gamba di 4 px per lato
    sloppy = np.zeros(truth.shape, np.float32)
    sloppy[75:] = 1.0
    sloppy[86:209, 154:174] = 0.0
    sloppy = cv2.GaussianBlur(sloppy, (0, 0), 2.0)
    before = np.mean((sloppy >= 0.5) != truth)
    snapped = masking.snap_to_edges(img, sloppy, radius=14)
    after = np.mean(snapped != truth)
    assert after < before * 0.1  # il bordo torna sul contorno vero
    assert snapped[200, 190]  # l'ombra della gamba resta pavimento (non è un oggetto)
    assert not snapped[150, 164]  # la gamba resta fuori
    assert snapped[150, 100] and not snapped[40, 100]  # lontano dal bordo non cambia nulla


def test_edge_alpha_is_sharp_at_full_resolution() -> None:
    img, truth = _floor_with_leg(480, 640)
    small = cv2.resize(truth.astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST) > 0
    alpha = masking.edge_alpha(img, small)
    assert alpha.shape == truth.shape and alpha.dtype == np.uint8
    assert alpha[300, 200] == 255 and alpha[60, 200] == 0  # interno pieno, esterno zero
    soft = np.count_nonzero((alpha > 0) & (alpha < 255))
    edge_len = np.count_nonzero(truth ^ cv2.erode(truth.astype(np.uint8), None).astype(bool))
    assert soft < 4 * edge_len  # sfumatura di un paio di pixel, non un alone
    assert np.mean((alpha >= 128) != truth) < 0.004  # segue il contorno vero in grande
    assert masking.edge_alpha(img, np.zeros((240, 320), bool)).max() == 0


def test_object_in_box_finds_the_contour_inside_a_loose_box() -> None:
    rng = np.random.default_rng(1)
    img = np.full((200, 300, 3), (180, 165, 140), np.float32) + rng.normal(0, 3, (200, 300, 3))
    yy, xx = np.mgrid[0:200, 0:300]
    disc = (xx - 150) ** 2 + (yy - 100) ** 2 <= 22**2  # un vaso visto dall'alto
    img[disc] = (40, 110, 70)
    img = np.clip(img, 0, 255).astype(np.uint8)
    obj = masking.object_in_box(img, (110, 60, 190, 140))  # riquadro largo il doppio
    assert obj[100, 150] and not obj[65, 115]  # il vaso sì, l'angolo del riquadro no
    inter = np.count_nonzero(obj & disc) / np.count_nonzero(obj | disc)
    assert inter > 0.8
    assert np.count_nonzero(disc & ~obj) == 0  # nessun pezzo di vaso resta scoperto
    # nessuna figura distinta nel riquadro: vale il riquadro intero (lo ha chiesto l'agente)
    flat = masking.object_in_box(img, (10, 10, 60, 50))
    assert flat[10:50, 10:60].all() and flat.sum() == 40 * 50
    assert not masking.object_in_box(img, (500, 500, 600, 600)).any()  # fuori dalla foto


def test_finalize_snaps_only_where_the_agent_drew() -> None:
    img, truth = _floor_with_leg()
    drawn = truth.copy()
    drawn[84:212, 150:178] = False  # l'agente ha escluso la gamba con un poligono largo
    zone = masking.outline_zone(truth.shape, [[(150, 84), (178, 84), (178, 212), (150, 212)]], 18)
    out = masking.finalize(img, drawn, zone, reach_px=8)
    assert not out.flags.writeable
    assert out[150, 164] == 0 and out[150, 152] == 255  # gamba fuori, pavimento accanto dentro
    assert np.mean((out >= 128) != truth) < np.mean(drawn != truth) * 0.5
    same = masking.finalize(img, drawn)  # senza zona: resta il bordo disegnato
    assert same[150, 152] == 0


def test_risk_map_flags_objects_left_inside_but_not_shadows() -> None:
    img, truth = _floor_with_leg()
    wrong = np.where(truth, 255, 0).astype(np.uint8)
    wrong[90:205, 158:170] = 255  # la gamba è rimasta dentro la maschera
    risk = masking.risk_map(img, wrong)
    assert risk.shape == wrong.shape
    assert risk[150, 164] > 0.5  # oggetto dentro: rischio alto
    assert risk[200, 215] < 0.2  # ombra: non è un oggetto
    assert risk[150, 60] < 0.1  # pavimento libero
    # gli ingrandimenti per il controllore vanno dove c'è il rischio, non sul bordo col muro
    wins = overlays.detail_windows(wrong, 1, (80, 80), risk)
    x0, y0, cw, ch = wins[0]
    assert x0 <= 164 < x0 + cw and y0 <= 150 < y0 + ch


def test_details_are_cropped_from_the_full_resolution_photo() -> None:
    img, truth = _floor_with_leg(960, 1280)
    mask = np.where(truth, 255, 0).astype(np.uint8)
    sheet = overlays.boundary_details(img, mask, n=2, crop=(120, 90), scale=0.25)
    assert sheet.shape[0] >= 180 and sheet.shape[2] == 3
    pair = overlays.light_details(img, img, mask, n=2, crop=(120, 90), scale=0.25)
    assert pair.shape[1] > 2 * 240  # PRIMA | DOPO affiancati, ingranditi 2x


def test_assign_planes_splits_the_mask_at_the_corner() -> None:
    from app.core import geometry

    size = (400, 300)
    left = geometry.room_geometry(
        [(0, 50), (200, 80), (200, 220), (0, 250)], 2000, 1000, size, kind="parete"
    )
    right = geometry.room_geometry(
        [(200, 80), (400, 50), (400, 250), (200, 220)], 2000, 1000, size, kind="parete"
    )
    labels = geometry.assign_planes([left, right], (300, 400))
    assert labels[150, 50] == 0 and labels[150, 350] == 1
    assert labels[10, 50] == 0 and labels[290, 390] == 1  # fuori dai rettangoli: il più vicino
    assert geometry.assign_planes([left], (300, 400)).max() == 0
    ppm = geometry.px_per_mm_map_multi([left, right], labels, (0, 0, 400, 300))
    assert ppm.shape == (300, 400) and ppm.min() > 0


def test_objects_win_over_a_doubtful_surface() -> None:
    labels = {"wall": 0, "floor": 1, "vase": 2, "ceiling": 3}
    probs = np.zeros((4, 60, 80), np.float32)
    probs[0, :30] = 1.0
    probs[1, 30:] = 1.0
    probs[1, 30:, 40:44], probs[2, 30:, 40:44] = 0.65, 0.35  # dubbio fondato: è un oggetto
    probs[1, 30:, 60:64], probs[0, 30:, 60:64] = 0.65, 0.35  # dubbio pavimento/muro: resta
    seg = semantic.Segmentation(probs=probs, labels=labels)
    img = np.full((60, 80, 3), 128, np.uint8)
    m = semantic.surface_mask(img, "pavimento", seg=seg, refine=False)
    assert m[50, 42] == 0 and m[50, 62] == 255 and m[50, 10] == 255
    assert seg.object_prob((80, 60))[50, 42] > 0.3 > seg.object_prob((80, 60))[50, 62]
