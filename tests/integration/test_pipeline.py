from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.catalog.loader import Catalog
from app.core import compositor, ingest
from app.core.errors import IntegrityError, MaskNotConfirmedError
from app.core.models import LayoutParams, RenderRequest
from app.devtools import synthetic
from app.pipeline import render as pipeline
from app.storage import jobstore
from tests.conftest import MATERIAL, forge


def _read(path: Path) -> tuple[np.ndarray, bytes | None]:
    img = Image.open(path)
    img.load()
    return np.asarray(img), img.info.get("icc_profile")


def test_full_flow_outputs_and_report(
    make_job: Callable[..., Path], catalog: Catalog, request_flat: RenderRequest
) -> None:
    job = make_job(channels=4, icc=synthetic.srgb_icc())
    out = pipeline.render(job, request_flat, catalog)
    assert out.valid
    files = {p.name for p in out.render_dir.iterdir()}
    assert files == {"final.png", "mask.png", "preview.jpg", "report.json"}

    j = jobstore.load(job)
    final, icc = _read(out.render_dir / "final.png")
    alpha, _ = _read(out.render_dir / "mask.png")
    original = j.source.pixels
    # Regola 8 sul file effettivamente esportato.
    assert np.array_equal(final[alpha == 0], original[alpha == 0])
    assert final.shape == original.shape
    assert icc is not None and icc == j.source.icc
    assert ingest.sha256_array(final) == out.report.final_sha256

    report = json.loads((out.render_dir / "report.json").read_text())
    assert report["valid"] is True
    assert report["input"]["source_sha256"] == j.doc.image.source_sha256
    assert {c["name"] for c in report["checks"]} >= {
        "outside_mask_identical",
        "resolution_match",
        "original_unchanged",
        "mask_confirmed",
        "master_readback_identical",
        "icc_preserved",
    }
    assert j.doc.state == "exported" and j.doc.renders[0].valid


def test_original_untouched(tmp_path: Path, catalog: Catalog, request_flat: RenderRequest) -> None:
    photo = synthetic.save_image(synthetic.make_photo(64, 48), tmp_path / "foto.png")
    mask = synthetic.save_image(synthetic.make_mask(48, 64), tmp_path / "m.png")
    before_sha, before_stat = ingest.sha256_file(photo), photo.stat()

    job = tmp_path / "job"
    pipeline.ingest_job(photo, job)
    rec = pipeline.import_mask(job, mask)
    pipeline.confirm_mask(job, rec.mask_sha256[:12])
    pipeline.render(job, request_flat, catalog)

    assert ingest.sha256_file(photo) == before_sha
    assert photo.stat().st_mtime_ns == before_stat.st_mtime_ns
    copy = job / "source" / "foto.png"
    assert stat.S_IMODE(copy.stat().st_mode) == 0o444
    assert ingest.sha256_file(copy) == before_sha
    assert not jobstore.load(job).source.pixels.flags.writeable


def test_blocked_without_mask_or_confirmation(
    tmp_path: Path, make_job: Callable[..., Path], catalog: Catalog, request_flat: RenderRequest
) -> None:
    photo = synthetic.save_image(synthetic.make_photo(32, 24), tmp_path / "f.png")
    bare = tmp_path / "bare"
    pipeline.ingest_job(photo, bare)
    with pytest.raises(MaskNotConfirmedError, match="assente"):
        pipeline.render(bare, request_flat, catalog)

    pending = make_job(confirm=False)
    with pytest.raises(MaskNotConfirmedError, match="non confermata"):
        pipeline.render(pending, request_flat, catalog)
    assert not any((pending / "renders").iterdir())


def test_blocked_if_mask_changes_after_confirmation(
    make_job: Callable[..., Path], catalog: Catalog, request_flat: RenderRequest
) -> None:
    job = make_job()
    j = jobstore.load(job)
    mask_png = job / "masks" / f"{j.doc.active_mask}.png"
    m = np.asarray(Image.open(mask_png)).copy()
    m[0, 0] = 255 if m[0, 0] == 0 else 0
    synthetic.save_image(m, mask_png)
    with pytest.raises(MaskNotConfirmedError, match="cambiata"):
        pipeline.render(job, request_flat, catalog)


def test_reimport_requires_new_confirmation(
    tmp_path: Path, make_job: Callable[..., Path], catalog: Catalog, request_flat: RenderRequest
) -> None:
    job = make_job(w=96, h=64)
    other = synthetic.save_image(synthetic.make_mask(64, 96, "full"), tmp_path / "full.png")
    pipeline.import_mask(job, other)
    with pytest.raises(MaskNotConfirmedError):
        pipeline.render(job, request_flat, catalog)


def test_tampered_job_is_rejected(make_job: Callable[..., Path]) -> None:
    job = make_job()
    norm = job / "normalized.png"
    arr = np.asarray(Image.open(norm)).copy()
    arr[0, 0, 0] ^= 1
    synthetic.save_image(arr, norm)
    with pytest.raises(IntegrityError, match="normalized"):
        jobstore.load(job)

    job2 = make_job()
    copy = next((job2 / "source").iterdir())
    os.chmod(copy, 0o644)
    copy.write_bytes(copy.read_bytes() + b"x")
    with pytest.raises(IntegrityError, match="sorgente"):
        jobstore.load(job2)


def test_fault_injection_blocks_export(
    make_job: Callable[..., Path],
    catalog: Catalog,
    request_flat: RenderRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = compositor.composite

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = real(*args, **kwargs)
        bad = res.pixels.copy()
        ys, xs = np.nonzero(res.alpha == 0)
        bad[ys[-1], xs[-1], 0] ^= 1  # 1 LSB fuori maschera
        return forge(bad, res.alpha)

    monkeypatch.setattr(compositor, "composite", broken)
    job = make_job()
    out = pipeline.render(job, request_flat, catalog)
    assert not out.valid
    assert {p.name for p in out.render_dir.iterdir()} == {"report.json"}
    assert not list(out.render_dir.glob("final.*"))
    assert jobstore.load(job).doc.state == "rejected"

    out2 = pipeline.render(job, request_flat, catalog, keep_invalid=True)
    assert not out2.valid
    assert [o.kind for o in out2.report.outputs] == ["quarantine"]
    assert (job / "quarantine" / f"{out2.render_id}.INVALID.png").is_file()
    assert not list(out2.render_dir.glob("final.*"))


def test_tiff_master_preserves_pixels_and_icc(
    make_job: Callable[..., Path], catalog: Catalog
) -> None:
    job = make_job(channels=4, icc=synthetic.srgb_icc())
    req = RenderRequest(layout=LayoutParams(material_id=MATERIAL), output_format="tiff")
    out = pipeline.render(job, req, catalog)
    assert out.valid
    final, icc = _read(out.render_dir / "final.tiff")
    assert np.array_equal(final, np.asarray(Image.open(out.render_dir / "final.tiff")))
    assert ingest.sha256_array(final) == out.report.final_sha256
    assert icc is not None and icc == jobstore.load(job).source.icc


def test_determinism(
    make_job: Callable[..., Path], catalog: Catalog, request_flat: RenderRequest
) -> None:
    a = pipeline.render(make_job(seed=5, icc=synthetic.srgb_icc()), request_flat, catalog)
    b = pipeline.render(make_job(seed=5, icc=synthetic.srgb_icc()), request_flat, catalog)
    assert a.report.final_sha256 == b.report.final_sha256
    assert a.report.determinism_key == b.report.determinism_key
    assert (a.render_dir / "final.png").read_bytes() == (b.render_dir / "final.png").read_bytes()
    assert (a.render_dir / "mask.png").read_bytes() == (b.render_dir / "mask.png").read_bytes()
