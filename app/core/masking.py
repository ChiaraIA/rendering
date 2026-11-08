"""Maschera della superficie: rasterizzazione, rifinitura, approvazione.

Convenzioni:
- maschera uint8 0..255, stessa risoluzione dell'originale normalizzato;
- coordinate già orientate (post-EXIF);
- fuori maschera <=> mask == 0.

Regola: il render parte solo con una maschera APPROVATA da un controllore indipendente,
e l'approvazione vale solo per quei byte esatti (sha) su quell'immagine (sha).
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from app.core.errors import MaskInvalidError, MaskNotConfirmedError
from app.core.ingest import readonly, sha256_array
from app.core.models import BBox, MaskOrigin, MaskRecord, MaskStatus

Polygon = list[tuple[float, float]]


def bbox_of(mask: NDArray[np.uint8]) -> BBox | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def rasterize(
    shape_hw: tuple[int, int],
    include: list[Polygon],
    exclude: list[Polygon] | None = None,
    supersample: int = 4,
) -> NDArray[np.uint8]:
    """Poligoni (coordinate pixel dell'immagine piena) -> maschera con bordi anti-aliasing."""
    h, w = shape_hw
    ss = supersample
    img = Image.new("L", (w * ss, h * ss), 0)
    draw = ImageDraw.Draw(img)
    for poly in include:
        if len(poly) >= 3:
            draw.polygon([(x * ss, y * ss) for x, y in poly], fill=255)
    for poly in exclude or []:
        if len(poly) >= 3:
            draw.polygon([(x * ss, y * ss) for x, y in poly], fill=0)
    small = img.resize((w, h), Image.Resampling.BOX)
    return readonly(np.asarray(small, dtype=np.uint8))


def refine(
    image_rgb: NDArray[np.uint8], coarse: NDArray[np.uint8], band_px: int = 12, seed: int = 0
) -> NDArray[np.uint8]:
    """Aggancia i bordi del poligono ai bordi reali della foto (GrabCut, deterministico).

    Solo una fascia di `band_px` attorno al bordo proposto può cambiare: il cuore della
    maschera resta pavimento, ciò che è lontano resta fuori.
    """
    fg = coarse >= 128
    if not np.any(fg) or np.all(fg):
        return readonly(coarse.copy())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    sure_fg = cv2.erode(fg.astype(np.uint8), k).astype(bool)
    maybe = cv2.dilate(fg.astype(np.uint8), k).astype(bool)
    gc = np.full(coarse.shape, cv2.GC_BGD, dtype=np.uint8)
    gc[maybe] = cv2.GC_PR_BGD
    gc[fg] = cv2.GC_PR_FGD
    gc[sure_fg] = cv2.GC_FGD
    bgr = np.ascontiguousarray(image_rgb[..., :3][..., ::-1])
    cv2.setRNGSeed(seed)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    hard = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    # tieni solo componenti collegate al cuore della maschera (niente isole spurie)
    n, labels = cv2.connectedComponents(hard)
    keep = np.unique(labels[sure_fg & (hard > 0)])
    hard = np.isin(labels, keep[keep > 0]).astype(np.uint8)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(hard.astype(np.float32) * 255.0, (0, 0), 0.8)
    soft[hard == 0] = np.minimum(soft[hard == 0], 254)  # bordo morbido
    return readonly(np.clip(np.rint(soft), 0, 255).astype(np.uint8))


def readonly_mask(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return readonly(np.ascontiguousarray(mask, dtype=np.uint8))


def load_mask_file(path: Path, expected_hw: tuple[int, int]) -> NDArray[np.uint8]:
    """Legge una maschera da file. Rifiuta tutto ciò che è ambiguo invece di indovinare."""
    try:
        img = Image.open(io.BytesIO(Path(path).read_bytes()))
        img.load()
    except OSError as exc:
        raise MaskInvalidError(f"maschera illeggibile: {exc}") from exc
    if img.mode == "1":
        img = img.convert("L")
    if img.mode != "L":
        raise MaskInvalidError(f"modalità maschera non supportata: {img.mode!r} (usa L 8 bit)")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != expected_hw:
        raise MaskInvalidError(
            f"dimensioni maschera {arr.shape[1]}x{arr.shape[0]} diverse dalla foto "
            f"{expected_hw[1]}x{expected_hw[0]}"
        )
    if not np.any(arr):
        raise MaskInvalidError("maschera vuota")
    return readonly(arr)


def make_record(
    mask: NDArray[np.uint8], image_sha256: str, origin: MaskOrigin = MaskOrigin.POLYGON
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


def approve(record: MaskRecord, mask: NDArray[np.uint8], approver: str) -> MaskRecord:
    if sha256_array(mask) != record.mask_sha256:
        raise MaskNotConfirmedError("la maschera da approvare non è quella proposta")
    return record.model_copy(
        update={
            "status": MaskStatus.APPROVED,
            "approved_by": approver,
            "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )


def require_approved(
    record: MaskRecord | None, mask: NDArray[np.uint8] | None, image_sha256: str
) -> None:
    """Il render parte solo con una maschera approvata e non modificata."""
    if record is None or mask is None:
        raise MaskNotConfirmedError("maschera assente")
    if record.status is not MaskStatus.APPROVED:
        raise MaskNotConfirmedError("maschera non approvata")
    if sha256_array(mask) != record.mask_sha256:
        raise MaskNotConfirmedError("la maschera è cambiata dopo l'approvazione")
    if record.image_sha256 != image_sha256:
        raise MaskNotConfirmedError("la maschera è stata approvata per un'altra immagine")
