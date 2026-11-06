from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageCms

from app.catalog.loader import Catalog
from app.core import colorspace, icc, ingest, material
from app.core.errors import CatalogError, JobStateError
from app.core.models import ColorSpace
from app.devtools import synthetic
from app.pipeline.job import JobDocument, JobState


@pytest.mark.parametrize("trc", ["srgb", "adobe"])
def test_trc_roundtrip_is_exact(trc: colorspace.Trc) -> None:
    v = np.arange(256, dtype=np.uint8)
    assert np.array_equal(colorspace.from_linear(colorspace.to_linear(v, trc), trc), v)
    with pytest.raises(TypeError):
        colorspace.to_linear(v.astype(np.uint16), trc)  # type: ignore[arg-type]


def test_classify_icc() -> None:
    assert colorspace.classify_icc(None)[0] is ColorSpace.ASSUMED_SRGB
    assert colorspace.classify_icc(colorspace.srgb_profile_bytes())[0] is ColorSpace.SRGB
    assert colorspace.classify_icc(b"garbage")[0] is ColorSpace.UNKNOWN
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    assert colorspace.classify_icc(lab)[0] is ColorSpace.UNKNOWN
    assert colorspace.trc_for(ColorSpace.ADOBE_RGB) == "adobe"
    assert colorspace.trc_for(ColorSpace.DISPLAY_P3) == "srgb"


def test_convert_to_srgb_profile_is_near_identity() -> None:
    tex = synthetic.make_tile_texture(16)
    out = colorspace.convert_srgb_to_profile(tex, colorspace.srgb_profile_bytes())
    assert np.abs(out.astype(int) - tex.astype(int)).max() <= 1


def test_catalog_load_and_variants(catalog: Catalog) -> None:
    spec = catalog.get("gres-grigio-30x30")
    variants = catalog.load_variants(spec)
    assert len(variants) == 4
    for tex in variants:
        assert tex.dtype == np.uint8 and tex.shape[2] == 3 and not tex.flags.writeable
    assert spec.pitch_mm == (303, 303)
    plank = catalog.get("laminato-rovere-120x20")
    assert plank.joint_mm == 0 and plank.joint_color is None
    assert plank.pattern.type == "running_bond"
    assert catalog.get("marmo-bianco-60x60").pattern.rotation_deg == 45
    with pytest.raises(CatalogError, match="non presente"):
        catalog.get("legno-inventato")


def test_default_catalog_is_consistent() -> None:
    cat = Catalog.load()
    assert len(cat.materials) >= 3
    for spec in cat.materials:
        cat.load_variants(spec)


def test_catalog_integrity_errors(tmp_path: Path) -> None:
    path = synthetic.write_catalog(tmp_path / "c")
    cat = Catalog.load(path)
    spec = cat.get("laminato-rovere-120x20")
    tex = tmp_path / "c" / "textures" / spec.variants[1].texture_file
    tex.write_bytes(b"tampered")
    with pytest.raises(CatalogError, match="hash"):
        cat.load_variants(spec)
    tex.unlink()
    with pytest.raises(CatalogError, match="mancante"):
        cat.load_variants(spec)

    doc = json.loads(path.read_text())

    def set_(i: int, **kw: object):  # type: ignore[no-untyped-def]
        return lambda d: d["materials"][i].update(kw)

    for mutate, match in [
        (lambda d: d["materials"].append(dict(d["materials"][0])), "duplicati"),
        (lambda d: d.update(schema_version="1.0"), "non valido"),
        (set_(0, joint_color=None), "non valido"),  # fuga senza colore
        (set_(1, variant_rotations=[0, 90]), "non valido"),  # 90° su piastrella non quadrata
        (set_(0, variant_rotations=[0, 0]), "non valido"),
        (set_(0, variants=[]), "non valido"),
        (set_(0, joint_mm=-1), "non valido"),
        (set_(0, pattern={"type": "herringbone"}), "non valido"),
        (
            lambda d: d["materials"][0]["variants"].append(d["materials"][0]["variants"][0]),
            "non valido",
        ),
        (lambda d: d["materials"][0]["variants"][0].update(texture_file="../x.png"), "non valido"),
    ]:
        d = json.loads(json.dumps(doc))
        mutate(d)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(d))
        with pytest.raises(CatalogError, match=match):
            Catalog.load(bad)
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(CatalogError, match="non valido"):
        Catalog.load(tmp_path / "broken.json")
    with pytest.raises(CatalogError, match="non trovato"):
        Catalog.load(tmp_path / "none.json")


def test_flat_layer_is_linear_mean_over_variants(catalog: Catalog) -> None:
    spec = catalog.get("gres-grigio-30x30")
    variants = catalog.load_variants(spec)
    src = ingest.from_array(synthetic.make_photo(10, 10))
    layer = material.flat_layer(spec, variants, src, (2, 3, 7, 9))
    assert layer.rgb_linear.shape == (6, 5, 3)
    expected = np.mean(
        [colorspace.to_linear(t.reshape(-1, 3), "srgb").mean(axis=0) for t in variants], axis=0
    )
    assert np.allclose(layer.rgb_linear[0, 0], expected, atol=1e-6)
    # media lineare != media dei valori codificati
    tex = variants[0]
    lin = colorspace.to_linear(tex.reshape(-1, 3), "srgb").mean(axis=0)
    naive = colorspace.to_linear(np.rint(tex.reshape(-1, 3).mean(axis=0)).astype(np.uint8), "srgb")
    assert not np.allclose(lin, naive, atol=1e-6)


def test_texture_conversion_fallback_is_reported(catalog: Catalog) -> None:
    spec = catalog.get("gres-grigio-30x30")
    variants = catalog.load_variants(spec)
    lab = bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes())
    src = ingest.from_array(synthetic.make_photo(4, 4), icc=lab)
    layer = material.flat_layer(spec, variants, src, (0, 0, 4, 4))
    assert any("fallita" in n for n in layer.notes)


def test_texture_is_converted_into_p3_photo_space() -> None:
    src = ingest.from_array(synthetic.make_photo(4, 4), icc=icc.display_p3())
    assert src.meta.color_space is ColorSpace.DISPLAY_P3
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[..., 1] = 255
    conv, notes = material.texture_in_photo_space(green, src)
    assert notes == []
    r, g, b = (int(c) for c in conv[0, 0])
    # il verde sRGB puro sta dentro il gamut P3: in valori P3 è meno saturo
    assert 100 <= r <= 130 and g >= 245 and 60 <= b <= 90


@pytest.mark.parametrize(
    ("factory", "cs"),
    [
        (icc.display_p3, ColorSpace.DISPLAY_P3),
        (icc.adobe_rgb, ColorSpace.ADOBE_RGB),
        (icc.srgb, ColorSpace.SRGB),
    ],
)
def test_generated_icc_profiles_are_valid(factory, cs) -> None:  # type: ignore[no-untyped-def]
    data = factory()
    assert factory() == data  # deterministico (data fissa nell'header)
    desc, xcs = colorspace.describe_icc(data)
    assert desc and xcs and xcs.strip() == "RGB"
    assert colorspace.classify_icc(data)[0] is cs


def test_job_state_machine() -> None:
    meta = ingest.from_array(synthetic.make_photo(2, 2)).meta
    doc = JobDocument(
        schema_version="1",
        job_id="j",
        created_at="t",
        state=JobState.INGESTED,
        source_file="s",
        original_path="o",
        image=meta,
    )
    assert not doc.renderable
    with pytest.raises(JobStateError):
        doc.advance(JobState.EXPORTED)
    doc = doc.advance(JobState.MASK_PROPOSED).advance(JobState.MASK_CONFIRMED)
    assert doc.renderable
    assert doc.advance(JobState.REJECTED).advance(JobState.EXPORTED).renderable
