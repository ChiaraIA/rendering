"""Posa esagonale, collezioni miste (pesi e rotazioni per variante) e import di esagoni."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from app.core import geometry, ingest, material
from app.core.models import SQRT3, MaterialSpec, TextureVariant
from app.devtools import synthetic
from app.storage.catalog_import import detect_hexagon, hex_vertices, rectify_hexagon

ROOM = (4200.0, 3600.0)
SHA = "0" * 64


def _spec(**kw: object) -> MaterialSpec:
    base: dict[str, object] = {
        "id": "esagona-test",
        "name": "t",
        "version": "1",
        "tile_w_mm": 250.0,
        "tile_h_mm": 250.0 * SQRT3 / 2,
        "joint_mm": 3.0,
        "joint_color": "#cccccc",
        "pattern": {"type": "hex"},
        "variants": [
            {"texture_file": "a.png", "texture_sha256": SHA, "weight": 6},
            {"texture_file": "b.png", "texture_sha256": SHA, "weight": 1, "rotations": [0, 180]},
            {"texture_file": "c.png", "texture_sha256": SHA, "weight": 1},
        ],
        "variant_rotations": [0, 60, 120, 180, 240, 300],
    }
    base.update(kw)
    return MaterialSpec.model_validate(base)


def test_hex_spec_validation() -> None:
    spec = _spec()
    assert spec.is_hex and spec.rotations_of(1) == [0, 180] and spec.rotations_of(0)[1] == 60
    with pytest.raises(ValidationError, match="√3"):
        _spec(tile_h_mm=250.0)
    with pytest.raises(ValidationError, match="60"):
        _spec(variant_rotations=[0, 90])
    with pytest.raises(ValidationError, match="90"):
        _spec(pattern={"type": "grid"}, tile_h_mm=250.0, variant_rotations=[0, 60])
    assert TextureVariant(texture_file="x.png", texture_sha256=SHA).weight == 1.0


def test_hex_lattice_geometry() -> None:
    """Centri a distanza 2·apotema+fuga; ogni punto cade nell'esagono più vicino."""
    spec = _spec()
    lay = material.Laying.of(spec, *ROOM)
    ca, cb = lay.hex_center(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    d1 = np.hypot(ca[1] - ca[0], cb[1] - cb[0])
    d2 = np.hypot(ca[2] - ca[0], cb[2] - cb[0])
    apothem = spec.tile_h_mm / 2
    assert d1 == pytest.approx(2 * apothem + spec.joint_mm)
    assert d2 == pytest.approx(2 * apothem + spec.joint_mm)

    rng = np.random.default_rng(0)
    a = rng.uniform(-2000, 2000, 20000)
    b = rng.uniform(-2000, 2000, 20000)
    q, r, dx, dy, inside = lay.locate(a, b)
    centres = np.stack(lay.hex_center(q.astype(float), r.astype(float)), 1)
    # nessun altro centro vicino è più vicino di quello scelto
    for dq, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)):
        other = np.stack(lay.hex_center((q + dq).astype(float), (r + dr).astype(float)), 1)
        pts = np.stack([a, b], 1)
        assert np.all(
            np.linalg.norm(pts - centres, axis=1) <= np.linalg.norm(pts - other, axis=1) + 1e-6
        )
    # frazione di superficie coperta da piastrelle = area piastrella / area cella
    cell = 1.5 * SQRT3 * lay.r_lat**2
    assert inside.mean() == pytest.approx(lay.tile_area / cell, abs=0.01)


def test_hex_tile_counts_match_brute_force() -> None:
    spec = _spec()
    whole, cut = material.count_tiles(spec, *ROOM)
    lay = material.Laying.of(spec, *ROOM)
    step = 8.0
    u, v = np.meshgrid(
        np.arange(-400, ROOM[0] + 400, step) + step / 2,
        np.arange(-400, ROOM[1] + 400, step) + step / 2,
    )
    a, b = lay.local(u, v)
    q, r, _, _, tile = lay.locate(a, b)
    inside = (u >= 0) & (u <= ROOM[0]) & (v >= 0) & (v <= ROOM[1])
    key = (q * 100000 + r)[tile]
    ins = set(key[inside[tile]].tolist())
    outs = set(key[~inside[tile]].tolist())
    assert len(ins - outs) == pytest.approx(whole, abs=3)
    assert len(ins & outs) == pytest.approx(cut, abs=3)


def test_weighted_variants_and_rotations() -> None:
    spec = _spec()
    lay = material.Laying.of(spec, *ROOM)
    vm = material.VariantMap.build(spec, lay, *ROOM, seed=5)
    v, rot = vm.variant.ravel(), vm.rotation.ravel()
    share = np.bincount(v, minlength=3) / v.size
    assert share[0] > 0.5 and share[1] > 0.05 and share[2] > 0.05  # peso 6:1:1, con vincoli
    assert set(np.unique(rot[v == 1])) <= {0, 180}  # rotazioni proprie della variante
    assert len(set(np.unique(rot[v == 0]))) > 2
    far = vm.lookup(np.array([50_000, 50_001]), np.array([-50_000, 7]))
    again = vm.lookup(np.array([50_000, 50_001]), np.array([-50_000, 7]))
    assert np.array_equal(far[0], again[0]) and np.array_equal(far[1], again[1])
    for k, deg in zip(*far, strict=True):
        assert deg in spec.rotations_of(int(k))


def test_hex_rotation_maps_texture_correctly() -> None:
    lay = material.Laying.of(_spec(), *ROOM)
    r = lay.r_tile
    s, t = np.array([r * 0.9]), np.array([0.0])  # vicino alla punta destra
    sn, tn = lay.texcoords(s, t, np.array([0]))
    assert sn[0] == pytest.approx(0.95) and tn[0] == pytest.approx(0.5)
    sn, tn = lay.texcoords(s, t, np.array([180]))  # ruotata di 180°: punta sinistra
    assert sn[0] == pytest.approx(0.05) and tn[0] == pytest.approx(0.5)


def test_hex_render_keeps_integrity(tmp_path: Path) -> None:
    from app.catalog.loader import Catalog
    from app.core import compositor, masking, validator

    scene = synthetic.make_room_scene(width=640, height=480)
    tex = [
        synthetic.make_tile_texture((256, 222), c, seed=k)
        for k, c in enumerate([(230, 220, 200), (150, 90, 60), (200, 170, 130)])
    ]
    spec = _spec()
    src = ingest.from_array(scene.photo)
    g = geometry.room_geometry([tuple(c) for c in scene.corners_px], *ROOM, (640, 480), 26)
    layer = material.tiled_layer(spec, tex, src, g, scene.floor_mask, seed=1, supersampling=2)
    rec = masking.approve(
        masking.make_record(scene.floor_mask, src.meta.normalized_sha256), scene.floor_mask, "test"
    )
    res = compositor.composite(src, scene.floor_mask, rec, layer)
    assert validator.validate(src, res, scene.floor_mask, rec).valid
    assert Catalog  # import usato solo per coerenza dei moduli


def test_detect_and_rectify_hexagon(tmp_path: Path) -> None:
    w, h = 900, 760
    img = Image.new("RGB", (w, h), (249, 249, 249))
    verts = hex_vertices(380, w / 2, h / 2) * np.array([1.0, 0.97])  # leggermente schiacciato
    verts[:, 1] += h / 2 * 0.03
    ImageDraw.Draw(img).polygon([tuple(p) for p in verts], fill=(214, 190, 160))
    ImageDraw.Draw(img).ellipse(
        [w / 2 + 250, h / 2 - 20, w / 2 + 300, h / 2 + 30], fill=(120, 60, 40)
    )
    src = tmp_path / "esagono.png"
    img.save(src)
    found = detect_hexagon(ingest.load_image(src).pixels)
    assert np.abs(found - verts).max() < 6
    dst = tmp_path / "tex.png"
    rectify_hexagon(src, dst, radius_px=128)
    out = np.asarray(Image.open(dst))
    assert out.shape == (round(SQRT3 * 128), 256, 3)
    assert np.abs(out.reshape(-1, 3).mean(0) - (214, 190, 160)).max() < 25  # niente sfondo bianco
    assert out[out.shape[0] // 2, 220, 0] < 170  # il pallino vicino alla punta destra resta lì


def test_hex_import_cli(tmp_path: Path) -> None:
    import json

    from typer.testing import CliRunner

    from app.catalog.loader import Catalog
    from app.cli import app

    root = tmp_path / "cat"
    path = synthetic.write_catalog(root)
    photos = []
    for k, col in enumerate([(214, 190, 160), (150, 90, 60)]):
        img = Image.new("RGB", (500, 440), (249, 249, 249))
        ImageDraw.Draw(img).polygon([tuple(p) for p in hex_vertices(200, 250, 220)], fill=col)
        photos.append(tmp_path / f"h{k}.png")
        img.save(photos[-1])
    r = CliRunner().invoke(
        app,
        [
            "aggiungi-materiale",
            *map(str, photos),
            "--codice",
            "esagona-prova",
            "--nome",
            "Prova",
            "--formato",
            "25",
            "--posa",
            "hex",
            "--fuga",
            "3",
            "--colore-fuga",
            "#c9c3b8",
            "--pesi",
            "3,1",
            "--rotazioni",
            "0,60,120,180,240,300",
            "--rotazioni",
            "0,180",
            "--correzione",
            "no",
            "--correzione",
            "1.1,1.0,0.9",
            "--catalogo",
            str(path),
        ],
    )
    assert r.exit_code == 0, r.output
    spec = Catalog.load(path).get("esagona-prova")
    assert spec.is_hex and spec.tile_h_mm == pytest.approx(250 * SQRT3 / 2, rel=1e-3)
    assert [v.weight for v in spec.variants] == [3, 1]
    assert spec.rotations_of(1) == [0, 180]
    assert len(Catalog.load(path).load_variants(spec)) == 2
    assert json.loads(path.read_text())["materials"][-1]["pattern"]["type"] == "hex"
