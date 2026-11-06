"""Segmentazione (M1: import della maschera) e flusso di conferma.

Convenzioni:
- maschera uint8 0..255, stessa risoluzione dell'originale normalizzato;
- coordinate già orientate (post-EXIF): nessuna trasformazione implicita;
- fuori maschera <=> mask == 0.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app import config
from app.core.errors import MaskInvalidError, MaskNotConfirmedError
from app.core.ingest import readonly, sha256_array
from app.core.models import BBox, MaskOrigin, MaskRecord, MaskStatus


def bbox_of(mask: NDArray[np.uint8]) -> BBox | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def load_mask_file(path: Path, expected_hw: tuple[int, int]) -> NDArray[np.uint8]:
    """Legge una maschera da file. Rifiuta tutto ciò che è ambiguo invece di indovinare."""
    try:
        img = Image.open(io.BytesIO(Path(path).read_bytes()))
        img.load()
    except OSError as exc:
        raise MaskInvalidError(f"maschera illeggibile: {exc}") from exc

    orientation = int(img.getexif().get(0x0112, 1) or 1)
    if orientation != 1:
        raise MaskInvalidError(
            "la maschera ha un orientamento EXIF: deve essere già nelle coordinate "
            "dell'originale normalizzato"
        )

    if img.mode == "L":
        arr = np.asarray(img, dtype=np.uint8)
    elif img.mode == "1":
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    elif img.mode == "RGB":
        rgb = np.asarray(img, dtype=np.uint8)
        if not (
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
        ):
            raise MaskInvalidError("maschera RGB non in scala di grigi")
        arr = rgb[..., 0]
    else:
        raise MaskInvalidError(f"modalità maschera non supportata: {img.mode!r} (usa L 8 bit)")

    if arr.shape != expected_hw:
        raise MaskInvalidError(
            f"dimensioni maschera {arr.shape[1]}x{arr.shape[0]} diverse dall'originale "
            f"{expected_hw[1]}x{expected_hw[0]}: nessun ridimensionamento implicito"
        )
    if not np.any(arr):
        raise MaskInvalidError("maschera vuota: nessun pixel selezionato")
    return readonly(arr)


def make_record(
    mask: NDArray[np.uint8], image_sha256: str, origin: MaskOrigin = MaskOrigin.IMPORTED
) -> MaskRecord:
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise MaskInvalidError(f"attesa maschera uint8 2D, trovato {mask.dtype} {mask.shape}")
    return MaskRecord(
        mask_sha256=sha256_array(mask),
        image_sha256=image_sha256,
        origin=origin,
        status=MaskStatus.PROPOSED,
        width=int(mask.shape[1]),
        height=int(mask.shape[0]),
        area_px=int(np.count_nonzero(mask)),
        soft_edge_px=int(np.count_nonzero((mask > 0) & (mask < 255))),
        bbox=bbox_of(mask),
    )


def confirm(record: MaskRecord, sha_prefix: str) -> MaskRecord:
    """Conferma esplicita: l'utente deve ripetere almeno i primi N caratteri dello sha."""
    prefix = sha_prefix.strip().lower()
    if len(prefix) < config.CONFIRM_PREFIX_MIN_LEN:
        raise MaskNotConfirmedError(
            f"servono almeno {config.CONFIRM_PREFIX_MIN_LEN} caratteri dello sha della maschera"
        )
    if not record.mask_sha256.startswith(prefix):
        raise MaskNotConfirmedError("lo sha indicato non corrisponde alla maschera attiva")
    return record.model_copy(
        update={
            "status": MaskStatus.CONFIRMED,
            "confirmed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )


def require_confirmed(
    record: MaskRecord | None, mask: NDArray[np.uint8] | None, image_sha256: str
) -> None:
    """Regola 7: il render parte solo con una maschera confermata e non modificata."""
    if record is None or mask is None:
        raise MaskNotConfirmedError("maschera assente")
    if record.status is not MaskStatus.CONFIRMED:
        raise MaskNotConfirmedError("maschera non confermata")
    if sha256_array(mask) != record.mask_sha256:
        raise MaskNotConfirmedError("la maschera è cambiata dopo la conferma")
    if record.image_sha256 != image_sha256:
        raise MaskNotConfirmedError("la maschera è stata confermata per un'altra immagine")
