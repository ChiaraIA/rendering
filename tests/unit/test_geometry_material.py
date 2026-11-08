"""Geometria della stanza, posa in prospettiva, luce e maschere su scena con verità nota."""

from __future__ import annotations

import numpy as np
import pytest

from app.catalog.loader import Catalog
from app.core import geometry, ingest, lighting, masking, material, overlays
from app.core.errors import GeometryError
from app.core.models import TILE_NONE, LightParams
from app.devtools import synthetic

S = synthetic.make_room_scene(width=640, height=480)
ROOM = S.room_mm


def _geom(room: tuple[float, float] = ROOM, focal: float | None = 26.0):  # type: ignore[no-untyped-def]
    return geometry.room_geometry([tuple(c) for c in S.corners_px], *room, (640, 480), focal)


def test_homography_matches_ground_truth() -> None:
    g = _geom()
    h = np.asarray(g.H)
    assert np.allclose(h / h[2, 2], S.H / S.H[2, 2], rtol=1e-8, atol=1e-8)
    assert g.reprojection_error_px < 1e-6
    # due angoli vicini sono fuori dall'inquadratura: va bene
    assert any(not (0 <= x <= 640 and 0 <= y <= 480) for x, y in g.corners_px)


def test_focal_check() -> None:
    ok = _geom().focal
    assert ok.consistent and abs(ok.aspect_error) < 0.05 and abs(ok.axes_angle_deg - 90) < 2
    assert ok.f_px_estimated == pytest.approx(ok.f_px_exif, rel=0.05)
    wrong = _geom(room=(ROOM[0], ROOM[1] * 0.8)).focal
    assert wrong.consistent is False
    assert _geom(focal=None).focal.consistent is None


@pytest.mark.parametrize(
    "corners",
    [
        [(0, 0), (100, 0), (200, 0), (300, 1)],  # allineati
        [(0, 0), (100, 100), (100, 0), (0, 100)],
    ],  # intrecciati
)
def test_degenerate_quads(corners) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(GeometryError):
        geometry.room_geometry(corners, 4000, 3000, (640, 480))


def test_mask_beyond_horizon_is_rejected() -> None:
    low = synthetic.make_room_scene(
        width=640, height=480, pitch_deg=8.0, cam=(2300.0, -1500.0, 1550.0)
    )
    g = geometry.room_geometry([tuple(c) for c in low.corners_px], *ROOM, (640, 480))
    geometry.check_mask_in_front(g, low.floor_mask)
    with pytest.raises(GeometryError, match="orizzonte"):
        geometry.check_mask_in_front(g, np.full((480, 640), 255, np.uint8))


@pytest.mark.parametrize(
    "mid", ["gres-grigio-30x30", "laminato-rovere-120x20", "marmo-bianco-60x60"]
)
def test_tile_pitch_on_the_floor(catalog: Catalog, mid: str) -> None:
    """Le piastrelle rese sul pavimento hanno il passo del catalogo (misurato sul piano)."""
    spec = catalog.get(mid)
    g = _geom()
    src = ingest.from_array(S.photo)
    layer = material.tiled_layer(spec, catalog.load_variants(spec), src, g, S.floor_mask, 7, 2)
    tij = layer.tile_ij
    assert tij is not None
    ok = tij[..., 0] != TILE_NONE
    ys, xs = np.nonzero(ok)
    x0, y0 = layer.bbox[:2]
    u, v, _ = geometry.plane_coords(np.asarray(g.H_inv), xs + x0 + 0.5, ys + y0 + 0.5)
    lay = material.Laying.of(spec, *ROOM)
    a, b = lay.local(u, v)
    i, j = tij[ys, xs, 0], tij[ys, xs, 1]
    key = i.astype(np.int64) * 100000 + j
    # solo piastrelle intere: nessun pixel vicino a mobili, muri o bordo della foto
    import cv2

    blocked = np.pad(S.floor_mask == 0, 0)
    blocked[[0, -1], :] = True
    blocked[:, [0, -1]] = True
    near = cv2.dilate(blocked.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    touching = set(key[near[ys + y0, xs + x0]].tolist())
    keep = ~np.isin(key, list(touching))
    # centroidi pesati per l'area reale (mm²) coperta da ogni pixel
    w = np.abs(
        geometry.jacobian_det_image_to_plane(np.asarray(g.H_inv), xs + x0 + 0.5, ys + y0 + 0.5)
    )
    uniq, inv = np.unique(key[keep], return_inverse=True)
    wk = w[keep]
    ca = np.bincount(inv, a[keep] * wk) / np.bincount(inv, wk)
    cells = {int(k): ca[n] for n, k in enumerate(uniq)}
    steps = [cells[k + 100000] - cells[k] for k in cells if k + 100000 in cells]
    assert steps, "nessuna coppia di piastrelle intere misurabile"
    assert np.median(steps) == pytest.approx(spec.pitch_mm[0], rel=0.01)


def test_tile_counts_match_brute_force(catalog: Catalog) -> None:
    for spec in catalog.materials:
        whole, cut = material.count_tiles(spec, *ROOM)
        lay = material.Laying.of(spec, *ROOM)
        step = 10.0
        u, v = np.meshgrid(
            np.arange(-600, ROOM[0] + 600, step) + step / 2,
            np.arange(-600, ROOM[1] + 600, step) + step / 2,
        )
        a, b = lay.local(u, v)
        i, j, s, t = lay.cell(a, b)
        tile = (s < lay.tile_w) & (t < lay.tile_h)
        inside = (u >= 0) & (u <= ROOM[0]) & (v >= 0) & (v <= ROOM[1])
        key = (i * 100000 + j)[tile]
        ins = set(key[inside[tile]].tolist())
        outs = set(key[~inside[tile]].tolist())
        assert len(ins - outs) == pytest.approx(whole, abs=2)
        assert len(ins & outs) == pytest.approx(cut, abs=2)


def test_variants_are_deterministic_and_not_repeated(catalog: Catalog) -> None:
    spec = catalog.get("gres-grigio-30x30")
    lay = material.Laying.of(spec, *ROOM)
    a = material.VariantMap.build(spec, lay, *ROOM, seed=3)
    b = material.VariantMap.build(spec, lay, *ROOM, seed=3)
    assert np.array_equal(a.variant, b.variant)
    v = a.variant
    assert not np.any(v[:, 1:] == v[:, :-1]) and not np.any(v[1:] == v[:-1])
    assert set(np.unique(v)) == set(range(len(spec.variants)))
    far_i, far_j = np.array([10_000]), np.array([-10_000])
    assert np.array_equal(a.lookup(far_i, far_j)[0], b.lookup(far_i, far_j)[0])


def test_light_and_contact_shadow() -> None:
    src = ingest.from_array(S.photo)
    bb = masking.bbox_of(S.floor_mask)
    assert bb is not None
    ppm = geometry.px_per_mm_map(_geom(), bb)
    lm = lighting.combined_light(src, S.floor_mask, bb, ppm)
    assert lm.gain.shape == (bb[3] - bb[1], bb[2] - bb[0])
    assert lm.gain.min() >= lighting.GAIN_MIN and lm.gain.max() <= lighting.GAIN_MAX
    ao = lighting.contact_shadow(S.floor_mask, bb, ppm)
    # più scuro vicino al divano che al centro del pavimento in primo piano
    sofa_edge = np.nonzero(S.floor_mask[:, 320])[0]
    gap = sofa_edge[np.argmax(np.diff(sofa_edge))] + 1 if np.any(np.diff(sofa_edge) > 1) else None
    y_near = (gap if gap is not None else sofa_edge.min()) + 2
    assert ao[y_near - bb[1], 320 - bb[0]] < ao[470 - bb[1], 320 - bb[0]]


def test_rasterize_and_refine_stay_close() -> None:
    poly = [(10.5, 10.5), (90, 10.5), (90, 60), (10.5, 60)]
    m = masking.rasterize((80, 100), [poly], [[(40, 30), (60, 30), (60, 50), (40, 50)]])
    assert m[35, 20] == 255 and m[40, 50] == 0 and m[5, 5] == 0
    assert 0 < m[10, 50] < 255  # bordo anti-aliasing (mezzo pixel coperto)
    img = np.zeros((80, 100, 3), np.uint8)
    img[10:60, 10:90] = 200
    img[30:50, 40:60] = 30
    r = masking.refine(img, m, band_px=6)
    assert r[35, 20] == 255 and r[2, 2] == 0


def test_preview_of_masks_keeps_geometry() -> None:
    """Regressione: le maschere 2D non devono perdere colonne quando vengono ridotte."""
    m = np.zeros((1200, 1600), np.uint8)
    m[600:, 800:] = 255
    p, s = overlays.resize_to_width(m, 1280)
    assert p.shape == (960, 1280) and s == 0.8
    assert p[700, 100] == 0 and p[700, 1000] == 255


def test_boundary_details_focus_on_objects() -> None:
    img = np.full((480, 640, 3), 200, np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    mask[240:, :] = 255  # pavimento fino al bordo della foto
    for x in range(300, 340, 8):  # "gambe di sedia": strisce sottili fuori dalla maschera
        mask[300:420, x : x + 3] = 0
    sheet = overlays.boundary_details(img, mask, n=2, crop=(200, 150))
    assert sheet.shape[2] == 3 and sheet.shape[0] >= 300  # almeno un dettaglio ingrandito 2x
    empty = overlays.boundary_details(img, np.zeros((480, 640), np.uint8))
    assert empty.shape == img.shape  # nessun bordo: restituisce l'overlay intero


def test_scene_tint_follows_warm_light_and_ignores_the_floor() -> None:
    # stanza "neutra": nessuna dominante
    gray = np.full((120, 160, 3), 128, np.uint8)
    gray[:20] = 235
    assert np.allclose(lighting.scene_tint(ingest.from_array(gray)), 1.0, atol=1e-6)
    # luce calda: le superfici bianche appaiono calde -> tinta R > G > B, luminanza 1
    warm = np.full((120, 160, 3), (110, 90, 70), np.uint8)
    warm[:20] = (240, 222, 190)  # soffitto bianco sotto luce calda
    floor = np.zeros((120, 160), np.uint8)
    warm[80:] = (245, 200, 120)  # vecchio pavimento beige e chiarissimo: non deve contare
    floor[80:] = 255
    r, g, b = lighting.scene_tint(ingest.from_array(warm), floor)
    assert r > g > b
    assert abs(0.2126 * r + 0.7152 * g + 0.0722 * b - 1.0) < 1e-6
    assert b > 0.7  # dominante del soffitto, non del pavimento giallo


def test_far_joints_are_continuous_lines_not_dashes() -> None:
    lay = material.Laying(
        tile_w=200.0, tile_h=200.0, pitch_w=203.0, pitch_h=203.0, shift=0.0,
        cos=1.0, sin=0.0, cx=0.0, cy=0.0,
    )  # fmt: skip
    s = np.arange(0.0, 203.0, 0.05)  # un campione ogni 0.05 mm lungo un passo
    t = np.full_like(s, 100.0)
    # impronta di 10 mm (fuga di 3 mm più stretta del campione): mai 0 o 1 netto vicino alla fuga
    jf = lay.joint_fraction(s, t, np.full_like(s, 10.0), np.full_like(s, 10.0))
    assert jf.max() == pytest.approx(0.3, abs=1e-6)
    # integrale conservato: la fuga "pesa" quanto la sua larghezza vera
    assert jf.mean() * 203 == pytest.approx(3.0, rel=1e-3)
    # impronta minuscola: torna il test dentro/fuori
    sharp = lay.joint_fraction(s, t, np.full_like(s, 1e-4), np.full_like(s, 1e-4))
    assert np.count_nonzero((sharp > 1e-3) & (sharp < 1 - 1e-3)) <= 2  # solo sul bordo esatto


def _old_floor(h: int = 240, w: int = 320) -> np.ndarray:
    """Vecchio pavimento uniforme con fughe sottili, un'ombra sotto un 'tavolo' e un riflesso."""
    img = np.full((h, w, 3), 180, np.uint8)
    img[::30, :] = 120  # fughe orizzontali (1 px)
    img[:, ::30] = 120  # fughe verticali
    yy, xx = np.mgrid[0:h, 0:w]
    shadow = np.exp(-(((xx - 220) / 30.0) ** 2 + ((yy - 140) / 22.0) ** 2))
    img = (img * (1 - 0.6 * shadow[..., None])).astype(np.uint8)
    streak = (np.abs(xx - 70) < 6) & (yy > 60) & (yy < 200)  # riflesso di una tenda
    img[streak] = np.minimum(255, img[streak].astype(int) + 60).astype(np.uint8)
    return img


def test_photo_shading_keeps_shadows_and_reflections_not_the_old_joints() -> None:
    img = _old_floor()
    src = ingest.from_array(img)
    mask = np.full(img.shape[:2], 255, np.uint8)
    bbox = (0, 0, img.shape[1], img.shape[0])
    ratio, excess = lighting.photo_shading(src, mask, bbox, 0.3)
    assert ratio[140, 220] < 0.6  # l'ombra resta
    open_floor = ratio[30:50, 120:170]  # zona libera con fughe
    assert open_floor.std() < 0.03  # le vecchie fughe non passano
    assert abs(float(np.median(open_floor)) - 1.0) < 0.12
    assert excess[130, 70].sum() > 0.02 and excess[130, 150].sum() < 0.005  # riflesso


def test_scene_light_parameters() -> None:
    img = _old_floor()
    src = ingest.from_array(img)
    mask = np.full(img.shape[:2], 255, np.uint8)
    bbox = (0, 0, img.shape[1], img.shape[0])
    none = lighting.scene_light(
        src, mask, bbox, 0.3, LightParams(ombre=0, riflessi=0, contatto=0, grana=0)
    )
    full = lighting.scene_light(src, mask, bbox, 0.3, LightParams(ombre=1.0, riflessi=0.7))
    assert full.gain[140, 220] < none.gain[140, 220] * 0.8  # ombre più marcate
    assert none.add is None and full.add is not None and full.add[130, 70].sum() > 0
    assert lighting.default_params("lucida").riflessi > lighting.default_params("opaca").riflessi
