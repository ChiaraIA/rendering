"""Export: l'unico punto che scrive su disco l'immagine finale.

Il compositor *crea* l'immagine, l'exporter la *scrive*, e solo se la validazione è passata.
Il master viene scritto su un file temporaneo, riletto e confrontato pixel per pixel e
ICC per ICC prima di essere rinominato: se qualcosa non torna, non esiste nessun master.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app import config
from app.core.compositor import CompositeResult
from app.core.errors import IntegrityError
from app.core.ingest import sha256_file
from app.core.models import CheckResult, OutputFile, ValidationReport


def _image(arr: NDArray[np.uint8]) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(arr))


def _save_master(img: Image.Image, path: Path, fmt: str, icc: bytes | None) -> None:
    kwargs: dict[str, object] = {}
    if icc is not None:
        kwargs["icc_profile"] = icc
    if fmt == "png":
        img.save(path, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL, **kwargs)
    else:
        img.save(path, format="TIFF", compression="tiff_deflate", **kwargs)


def _readback(path: Path) -> tuple[NDArray[np.uint8], bytes | None]:
    img = Image.open(io.BytesIO(path.read_bytes()))
    img.load()
    return np.asarray(img, dtype=np.uint8), (img.info.get("icc_profile") or None)


def export_master(
    result: CompositeResult,
    validation: ValidationReport,
    icc: bytes | None,
    out_dir: Path,
    fmt: Literal["png", "tiff"],
) -> tuple[OutputFile | None, list[CheckResult]]:
    """Scrive il master solo per un risultato valido e lo verifica rileggendolo."""
    if not validation.valid or validation.final_sha256 != result.final_sha256:
        raise IntegrityError("export del master rifiutato: risultato non validato")

    ext = "png" if fmt == "png" else "tiff"
    final_path = out_dir / f"final.{ext}"
    tmp = out_dir / f".final.tmp.{ext}"
    _save_master(_image(result.pixels), tmp, fmt, icc)

    back, back_icc = _readback(tmp)
    same_pixels = back.shape == result.pixels.shape and bool(np.array_equal(back, result.pixels))
    same_icc = back_icc == icc
    checks = [
        CheckResult(name="master_readback_identical", blocking=True, passed=same_pixels),
        CheckResult(name="icc_preserved", blocking=True, passed=same_icc),
    ]
    if not (same_pixels and same_icc):
        tmp.unlink(missing_ok=True)
        return None, checks
    os.replace(tmp, final_path)
    return OutputFile(kind="master", path=final_path.name, sha256=sha256_file(final_path)), checks


def export_extras(result: CompositeResult, icc: bytes | None, out_dir: Path) -> list[OutputFile]:
    """Maschera usata e anteprima JPEG (non master, esclusa dal controllo esatto)."""
    mask_path = out_dir / "mask.png"
    _image(result.alpha).save(mask_path, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL)

    preview = _image(result.pixels).convert("RGB")
    scale = config.PREVIEW_MAX_SIDE / max(preview.size)
    if scale < 1:
        size = (max(1, round(preview.width * scale)), max(1, round(preview.height * scale)))
        preview = preview.resize(size, Image.Resampling.LANCZOS)
    preview_path = out_dir / "preview.jpg"
    kwargs: dict[str, object] = {"icc_profile": icc} if icc is not None else {}
    preview.save(preview_path, format="JPEG", quality=config.PREVIEW_JPEG_QUALITY, **kwargs)

    return [
        OutputFile(kind="mask", path=mask_path.name, sha256=sha256_file(mask_path)),
        OutputFile(kind="preview", path=preview_path.name, sha256=sha256_file(preview_path)),
    ]


def export_quarantine(result: CompositeResult, job_dir: Path, render_id: str) -> OutputFile:
    """Solo con --keep-invalid: candidato non valido, mai come master."""
    qdir = job_dir / "quarantine"
    qdir.mkdir(exist_ok=True)
    path = qdir / f"{render_id}.INVALID.png"
    _image(result.pixels).save(path, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL)
    return OutputFile(
        kind="quarantine", path=str(path.relative_to(job_dir)), sha256=sha256_file(path)
    )
