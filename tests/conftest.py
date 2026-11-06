from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from app.catalog.loader import Catalog
from app.core import compositor, ingest, segmentation
from app.core.compositor import CompositeResult
from app.core.models import LayoutParams, MaskRecord, RenderRequest
from app.devtools import synthetic
from app.pipeline import render as pipeline

MATERIAL = "gres-grigio-30x30"


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog.load(synthetic.write_catalog(tmp_path / "catalog"))


@pytest.fixture
def request_flat() -> RenderRequest:
    return RenderRequest(layout=LayoutParams(material_id=MATERIAL))


@pytest.fixture
def make_job(tmp_path: Path) -> Callable[..., Path]:
    """Crea un job pronto: foto + maschera importata (+ confermata se richiesto)."""
    counter = {"n": 0}

    def _make(
        w: int = 96,
        h: int = 64,
        channels: int = 3,
        mask_kind: synthetic.MaskKind = "holes",
        confirm: bool = True,
        icc: bytes | None = None,
        seed: int = 0,
        name: str | None = None,
    ) -> Path:
        counter["n"] += 1
        base = tmp_path / (name or f"job{counter['n']}")
        base.mkdir()
        photo = synthetic.save_image(
            synthetic.make_photo(w, h, channels=channels, seed=seed), base / "foto.png", icc=icc
        )
        mask = synthetic.save_image(synthetic.make_mask(h, w, mask_kind), base / "mask.png")
        job = base / "job"
        pipeline.ingest_job(photo, job)
        rec = pipeline.import_mask(job, mask)
        if confirm:
            pipeline.confirm_mask(job, rec.mask_sha256[:12])
        return job

    return _make


def confirmed(mask: NDArray[np.uint8], image_sha: str) -> MaskRecord:
    rec = segmentation.make_record(mask, image_sha)
    return segmentation.confirm(rec, rec.mask_sha256[:12])


def forge(
    pixels: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    sha: str | None = None,
) -> CompositeResult:
    """Solo per i test: crea un CompositeResult arbitrario (simula un compositor difettoso)."""
    p = ingest.readonly(pixels)
    a = ingest.readonly(alpha)
    return CompositeResult(
        pixels=p,
        alpha=a,
        bbox=segmentation.bbox_of(a),
        final_sha256=sha or ingest.sha256_array(p),
        _seal=compositor._SEAL,
    )
