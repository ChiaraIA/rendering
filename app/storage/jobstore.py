"""Archivio dei job su disco.

runs/<job>/
  source/<file originale>   copia byte per byte, permessi 0o444
  source.icc                profilo colore originale (se presente)
  normalized.png            originale normalizzato (riferimento per i controlli)
  job.json
  masks/<sha>.png|.json
  renders/<render_id>/...
"""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app import config
from app.core import ingest
from app.core.errors import IntegrityError, JobStateError, MaskNotConfirmedError
from app.core.ingest import readonly, sha256_array, sha256_file
from app.core.models import MaskRecord, SourceImage
from app.pipeline.job import JobDocument, JobState


@dataclass(frozen=True)
class Job:
    dir: Path
    doc: JobDocument
    source: SourceImage

    @property
    def source_copy(self) -> Path:
        return self.dir / self.doc.source_file


def _write_json(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def save_doc(job_dir: Path, doc: JobDocument) -> None:
    _write_json(job_dir / "job.json", doc.model_dump_json(indent=2))


def _write_png(arr: NDArray[np.uint8], path: Path) -> None:
    Image.fromarray(np.ascontiguousarray(arr)).save(
        path, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL
    )


def _read_png(path: Path) -> NDArray[np.uint8]:
    img = Image.open(io.BytesIO(path.read_bytes()))
    img.load()
    return np.asarray(img, dtype=np.uint8)


def create(photo: Path, job_dir: Path) -> Job:
    photo = Path(photo)
    job_dir = Path(job_dir)
    if job_dir.exists() and any(job_dir.iterdir()):
        raise JobStateError(f"la cartella del job esiste e non è vuota: {job_dir}")

    source = ingest.load_image(photo)  # sola lettura sull'originale dell'utente

    (job_dir / "source").mkdir(parents=True, exist_ok=True)
    (job_dir / "masks").mkdir(exist_ok=True)
    (job_dir / "renders").mkdir(exist_ok=True)

    copy = job_dir / "source" / photo.name
    shutil.copyfile(photo, copy)
    os.chmod(copy, 0o444)
    if sha256_file(copy) != source.meta.source_sha256:
        raise IntegrityError("la copia del sorgente non corrisponde all'originale")

    _write_png(source.pixels, job_dir / "normalized.png")
    if source.icc is not None:
        (job_dir / "source.icc").write_bytes(source.icc)

    doc = JobDocument(
        schema_version=config.JOB_SCHEMA_VERSION,
        job_id=uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        state=JobState.INGESTED,
        source_file=f"source/{photo.name}",
        original_path=str(photo.resolve()),
        image=source.meta,
    )
    save_doc(job_dir, doc)
    return Job(job_dir, doc, source)


def load(job_dir: Path) -> Job:
    job_dir = Path(job_dir)
    doc_path = job_dir / "job.json"
    if not doc_path.is_file():
        raise JobStateError(f"job non trovato: {job_dir}")
    doc = JobDocument.model_validate_json(doc_path.read_text(encoding="utf-8"))

    pixels = readonly(_read_png(job_dir / "normalized.png"))
    if sha256_array(pixels) != doc.image.normalized_sha256:
        raise IntegrityError("normalized.png non corrisponde all'hash registrato nel job")
    icc_path = job_dir / "source.icc"
    icc = icc_path.read_bytes() if icc_path.is_file() else None
    if (ingest.sha256_bytes(icc) if icc else None) != doc.image.icc_sha256:
        raise IntegrityError("profilo ICC del job non corrisponde all'hash registrato")

    source = SourceImage(pixels=pixels, icc=icc, meta=doc.image)
    job = Job(job_dir, doc, source)
    if sha256_file(job.source_copy) != doc.image.source_sha256:
        raise IntegrityError("la copia del file sorgente è stata modificata")
    return job


def save_mask(job: Job, mask: NDArray[np.uint8], record: MaskRecord) -> Job:
    _write_png(mask, job.dir / "masks" / f"{record.mask_sha256}.png")
    return save_mask_record(job, record, activate=True)


def save_mask_record(job: Job, record: MaskRecord, activate: bool = False) -> Job:
    _write_json(job.dir / "masks" / f"{record.mask_sha256}.json", record.model_dump_json(indent=2))
    masks = {**job.doc.masks, record.mask_sha256: record}
    update: dict[str, object] = {"masks": masks}
    if activate:
        update["active_mask"] = record.mask_sha256
    doc = job.doc.model_copy(update=update)
    save_doc(job.dir, doc)
    return Job(job.dir, doc, job.source)


def load_active_mask(job: Job) -> tuple[MaskRecord, NDArray[np.uint8]]:
    sha = job.doc.active_mask
    if sha is None or sha not in job.doc.masks:
        raise MaskNotConfirmedError("maschera assente: importala con `posa mask import`")
    path = job.dir / "masks" / f"{sha}.png"
    if not path.is_file():
        raise MaskNotConfirmedError("file della maschera attiva mancante")
    return job.doc.masks[sha], readonly(_read_png(path))


def with_state(job: Job, state: JobState, **update: object) -> Job:
    doc = job.doc.advance(state)
    if update:
        doc = doc.model_copy(update=update)
    save_doc(job.dir, doc)
    return Job(job.dir, doc, job.source)


def new_render_dir(job: Job) -> tuple[str, Path]:
    render_id = f"render-{len(job.doc.renders) + 1:03d}"
    path = job.dir / "renders" / render_id
    path.mkdir(parents=True, exist_ok=False)
    return render_id, path


def write_json(path: Path, data: str) -> None:
    _write_json(path, data)


def jsonable(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
