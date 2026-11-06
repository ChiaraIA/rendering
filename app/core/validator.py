"""Validazione: controlli bloccanti e warning (regole 8 e 9).

M1: integrità. M2 aggiunge scala/geometria, M3 pattern, colore e bordi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from app.core.compositor import CompositeResult
from app.core.ingest import sha256_array, sha256_file
from app.core.models import CheckResult, MaskRecord, MaskStatus, SourceImage, ValidationReport


def check_outside_mask_identical(
    final: NDArray[np.uint8], original: NDArray[np.uint8], alpha: NDArray[np.uint8]
) -> CheckResult:
    name = "outside_mask_identical"
    if final.shape != original.shape or alpha.shape != original.shape[:2]:
        return CheckResult(name=name, blocking=True, passed=False, message="forme diverse")
    ok = bool(np.array_equal(final[alpha == 0], original[alpha == 0]))
    diff = 0 if ok else int(np.count_nonzero(np.any(final != original, axis=-1) & (alpha == 0)))
    return CheckResult(
        name=name,
        blocking=True,
        passed=ok,
        value=diff,
        threshold=0,
        message="pixel diversi fuori maschera",
    )


def validate(
    source: SourceImage,
    result: CompositeResult,
    mask: NDArray[np.uint8],
    mask_record: MaskRecord,
    source_copy: Path | None = None,
) -> ValidationReport:
    original = source.pixels
    final = result.pixels
    alpha = result.alpha
    checks: list[CheckResult] = []

    # 1. Regola 8: uguaglianza esatta fuori maschera.
    checks.append(check_outside_mask_identical(final, original, alpha))

    # 2. Stessa risoluzione e stesso formato pixel.
    checks.append(
        CheckResult(
            name="resolution_match",
            blocking=True,
            passed=final.shape[:2] == original.shape[:2],
            value=f"{final.shape[1]}x{final.shape[0]}",
            threshold=f"{original.shape[1]}x{original.shape[0]}",
        )
    )
    checks.append(
        CheckResult(
            name="pixel_format_match",
            blocking=True,
            passed=final.shape == original.shape and final.dtype == original.dtype,
            value=f"{final.dtype}{list(final.shape[2:])}",
            threshold=f"{original.dtype}{list(original.shape[2:])}",
        )
    )

    # 3. Originale invariato (array normalizzato e copia del file sorgente).
    norm_ok = sha256_array(original) == source.meta.normalized_sha256
    src_ok = source_copy is None or sha256_file(source_copy) == source.meta.source_sha256
    checks.append(
        CheckResult(
            name="original_unchanged",
            blocking=True,
            passed=norm_ok and src_ok and not original.flags.writeable,
            message="" if norm_ok and src_ok else "hash dell'originale cambiato",
        )
    )

    # 4. Maschera confermata, invariata e legata a questa immagine.
    mask_ok = (
        mask_record.status is MaskStatus.CONFIRMED
        and sha256_array(mask) == mask_record.mask_sha256
        and mask_record.image_sha256 == source.meta.normalized_sha256
    )
    checks.append(CheckResult(name="mask_confirmed", blocking=True, passed=mask_ok))

    # 5. Il supporto di alpha non esce dalla maschera confermata.
    same_shape = alpha.shape == mask.shape
    leak = int(np.count_nonzero((alpha > 0) & (mask == 0))) if same_shape else -1
    checks.append(
        CheckResult(
            name="alpha_within_mask", blocking=True, passed=leak == 0, value=leak, threshold=0
        )
    )

    # 6. Canale alpha della foto invariato ovunque.
    if original.ndim == 3 and original.shape[2] == 4:
        same = final.shape == original.shape and bool(
            np.array_equal(final[..., 3], original[..., 3])
        )
        checks.append(CheckResult(name="photo_alpha_preserved", blocking=True, passed=same))

    # 7. L'hash dichiarato corrisponde ai pixel (nessuna manomissione successiva).
    checks.append(
        CheckResult(
            name="final_hash_consistent",
            blocking=True,
            passed=sha256_array(final) == result.final_sha256,
        )
    )

    return ValidationReport(final_sha256=result.final_sha256, checks=checks)
