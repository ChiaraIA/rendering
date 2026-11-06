"""Orchestrazione: ingest -> maschera -> conferma -> render -> validazione -> export."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from app import __version__, config
from app.catalog.loader import Catalog
from app.core import compositor, material, segmentation, validator
from app.core.errors import IntegrityError, MaskNotConfirmedError
from app.core.ingest import colorspace_warnings, sha256_bytes
from app.core.models import (
    CheckResult,
    MaskRecord,
    MaterialInfo,
    OutputFile,
    RenderReport,
    RenderRequest,
)
from app.pipeline.job import JobState, RenderSummary
from app.storage import exporter, jobstore


def ingest_job(photo: Path, job_dir: Path) -> jobstore.Job:
    return jobstore.create(photo, job_dir)


def import_mask(job_dir: Path, mask_path: Path) -> MaskRecord:
    job = jobstore.load(job_dir)
    mask = segmentation.load_mask_file(mask_path, job.source.shape_hw)
    record = segmentation.make_record(mask, job.source.meta.normalized_sha256)
    job = jobstore.save_mask(job, mask, record)
    jobstore.with_state(job, JobState.MASK_PROPOSED)
    return record


def confirm_mask(job_dir: Path, sha_prefix: str) -> MaskRecord:
    job = jobstore.load(job_dir)
    record, mask = jobstore.load_active_mask(job)
    # La maschera su disco deve essere ancora quella proposta.
    if segmentation.make_record(mask, record.image_sha256).mask_sha256 != record.mask_sha256:
        raise MaskNotConfirmedError("la maschera su disco non corrisponde a quella proposta")
    confirmed = segmentation.confirm(record, sha_prefix)
    job = jobstore.save_mask_record(job, confirmed)
    jobstore.with_state(job, JobState.MASK_CONFIRMED)
    return confirmed


def _versions() -> dict[str, str]:
    out = {"posa": __version__, "python": platform.python_version()}
    for pkg in ("numpy", "pillow", "pydantic", "opencv-python-headless"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "n/a"
    return out


@dataclass(frozen=True)
class RenderOutcome:
    valid: bool
    render_id: str
    render_dir: Path
    report: RenderReport


def render(
    job_dir: Path,
    request: RenderRequest,
    catalog: Catalog | None = None,
    keep_invalid: bool = False,
) -> RenderOutcome:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    job = jobstore.load(job_dir)
    record, mask = jobstore.load_active_mask(job)
    # Regola 7: niente render senza conferma, anche se job.json dicesse il contrario.
    segmentation.require_confirmed(record, mask, job.source.meta.normalized_sha256)
    if not job.doc.renderable:
        raise MaskNotConfirmedError(f"stato del job non renderizzabile: {job.doc.state}")
    assert record.bbox is not None

    catalog = catalog or Catalog.load()
    spec = catalog.get(request.layout.material_id)
    variants = catalog.load_variants(spec)
    timings["load"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    layer = material.flat_layer(spec, variants, job.source, record.bbox)
    result = compositor.composite(job.source, mask, record, layer)
    timings["composite"] = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    validation = validator.validate(job.source, result, mask, record, job.source_copy)
    timings["validate"] = (time.perf_counter() - t2) * 1000

    render_id, out_dir = jobstore.new_render_dir(job)
    outputs: list[OutputFile] = []
    extra_checks: list[CheckResult] = []

    t3 = time.perf_counter()
    if validation.valid:
        master, extra_checks = exporter.export_master(
            result, validation, job.source.icc, out_dir, request.output_format
        )
        validation = validation.with_checks(extra_checks)
        if master is not None:
            outputs.append(master)
            outputs.extend(exporter.export_extras(result, job.source.icc, out_dir))
    if not validation.valid and keep_invalid:
        outputs.append(exporter.export_quarantine(result, job.dir, render_id))
    timings["export"] = (time.perf_counter() - t3) * 1000

    versions = _versions()
    determinism_key = sha256_bytes(
        jobstore.jsonable(
            {
                "image": job.source.meta.normalized_sha256,
                "icc": job.source.meta.icc_sha256,
                "mask": record.mask_sha256,
                "request": request.model_dump(mode="json"),
                "variants": [v.texture_sha256 for v in spec.variants],
                "catalog": catalog.sha256,
                "versions": versions,
            }
        ).encode()
    )

    flat = material.flat_color_linear(
        material.variants_in_photo_space(variants, job.source)[0], job.source
    )
    report = RenderReport(
        schema_version=config.REPORT_SCHEMA_VERSION,
        render_id=render_id,
        job_id=job.doc.job_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        valid=validation.valid,
        input=job.source.meta,
        mask=record,
        request=request,
        material=MaterialInfo(
            id=spec.id,
            name=spec.name,
            version=spec.version,
            variant_sha256=[v.texture_sha256 for v in spec.variants],
            catalog_sha256=catalog.sha256,
            flat_color_linear=(float(flat[0]), float(flat[1]), float(flat[2])),
        ),
        checks=validation.checks,
        warnings=[*colorspace_warnings(job.source.meta), *layer.notes],
        outputs=outputs,
        final_sha256=result.final_sha256,
        determinism_key=determinism_key,
        versions=versions,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
    )
    jobstore.write_json(out_dir / "report.json", report.model_dump_json(indent=2))

    summary = RenderSummary(
        render_id=render_id,
        valid=report.valid,
        final_sha256=result.final_sha256,
        report=f"renders/{render_id}/report.json",
    )
    jobstore.with_state(
        job,
        JobState.EXPORTED if report.valid else JobState.REJECTED,
        renders=[*job.doc.renders, summary],
    )

    # Ultima difesa: l'originale non deve essere cambiato durante l'elaborazione.
    reloaded = jobstore.load(job_dir)
    if not np.array_equal(reloaded.source.pixels, job.source.pixels):
        raise IntegrityError("l'originale normalizzato è cambiato durante il render")

    return RenderOutcome(report.valid, render_id, out_dir, report)
