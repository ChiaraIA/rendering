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


# ------------------------------------------------------------- bordi sulla foto vera
def _box(x: NDArray[np.float32], r: int) -> NDArray[np.float32]:
    k = (2 * r + 1, 2 * r + 1)
    return np.asarray(cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT), dtype=np.float32)


def _guided_coefficients(
    guide: NDArray[np.float32], p: NDArray[np.float32], r: int, eps: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Coefficienti (a, b) del filtro guidato a colori (He, Sun, Tang 2010): in ogni
    finestra p ≈ a·I + b. `a` ha forma (h, w, 3), `b` (h, w); entrambi già mediati."""
    mean_i = _box(guide, r)
    mean_p = _box(p, r)
    cov_ip = _box(guide * p[..., None], r) - mean_i * mean_p[..., None]
    i0, i1, i2 = guide[..., 0], guide[..., 1], guide[..., 2]
    m0, m1, m2 = mean_i[..., 0], mean_i[..., 1], mean_i[..., 2]
    rr = _box(i0 * i0, r) - m0 * m0 + eps
    rg = _box(i0 * i1, r) - m0 * m1
    rb = _box(i0 * i2, r) - m0 * m2
    gg = _box(i1 * i1, r) - m1 * m1 + eps
    gb = _box(i1 * i2, r) - m1 * m2
    bb = _box(i2 * i2, r) - m2 * m2 + eps
    # inversa della matrice di covarianza 3x3 (simmetrica), in forma chiusa
    c_rr, c_rg, c_rb = gg * bb - gb * gb, gb * rb - rg * bb, rg * gb - gg * rb
    c_gg, c_gb, c_bb = rr * bb - rb * rb, rb * rg - rr * gb, rr * gg - rg * rg
    det = rr * c_rr + rg * c_rg + rb * c_rb
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    c0, c1, c2 = cov_ip[..., 0], cov_ip[..., 1], cov_ip[..., 2]
    a = np.stack(
        [
            (c_rr * c0 + c_rg * c1 + c_rb * c2) / det,
            (c_rg * c0 + c_gg * c1 + c_gb * c2) / det,
            (c_rb * c0 + c_gb * c1 + c_bb * c2) / det,
        ],
        axis=-1,
    ).astype(np.float32)
    b = (mean_p - (a * mean_i).sum(axis=-1)).astype(np.float32)
    return _box(a, r), _box(b, r)


def snap_to_edges(
    image_rgb: NDArray[np.uint8],
    soft: NDArray[np.float32],
    radius: int,
    eps: float = 1e-3,
) -> NDArray[np.bool_]:
    """Porta il bordo di una maschera (probabilità 0..1) sui bordi veri della foto.

    Filtro guidato: dentro ogni finestra la maschera diventa una funzione lineare del
    colore, quindi il passaggio superficie/oggetto si sposta dove cambia il colore. A
    differenza di GrabCut non usa un modello di colore unico per tutta la foto: l'ombra
    sotto un tavolo resta superficie, un oggetto dello stesso colore lontano non conta.
    Cambia solo una fascia di `radius` px attorno al bordo proposto.
    """
    hard = soft >= 0.5
    if not hard.any() or hard.all():
        return np.asarray(hard)
    guide = np.ascontiguousarray(image_rgb[..., :3]).astype(np.float32) / 255.0
    a, b = _guided_coefficients(guide, soft.astype(np.float32), radius, eps)
    q = (a * guide).sum(axis=-1) + b
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    h8 = hard.astype(np.uint8)
    band = cv2.dilate(h8, k).astype(bool) & ~cv2.erode(h8, k).astype(bool)
    snapped = np.where(band, q >= 0.5, hard)
    # dove la foto non ha un contorno netto (tappeto chiaro su pavimento chiaro, ombre) il
    # bordo trovato è frastagliato: lì si liscia. Dove il contorno è netto (gambe sottili,
    # spigoli) resta com'è, così una gamba di due pixel non sparisce.
    gray = cv2.cvtColor(np.ascontiguousarray(image_rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    crisp = cv2.dilate(grad, np.ones((5, 5), np.uint8)) >= CRISP_EDGE
    smooth = cv2.medianBlur(snapped.astype(np.uint8), 7).astype(bool)
    return np.asarray(np.where(band & ~crisp, smooth, snapped))


def edge_alpha(
    image_rgb: NDArray[np.uint8],
    hard_small: NDArray[np.bool_],
    radius: int = 2,
    eps: float = 4e-4,
    contrast: float = 3.0,
) -> NDArray[np.uint8]:
    """Maschera 0..255 alla risoluzione piena da una maschera netta a risoluzione ridotta.

    I coefficienti del filtro guidato si calcolano in piccolo e si applicano alla foto
    intera ("fast guided filter"): il bordo segue i contorni a piena risoluzione invece di
    essere l'ingrandimento sgranato di quello piccolo. L'interno resta 255, l'esterno 0, e
    la sfumatura è larga circa un pixel (antialiasing), non un alone.
    """
    h, w = image_rgb.shape[:2]
    sh, sw = hard_small.shape
    if not hard_small.any():
        return np.zeros((h, w), np.uint8)
    full_rgb = np.ascontiguousarray(image_rgb[..., :3])
    small_rgb = cv2.resize(full_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    guide = small_rgb.astype(np.float32) / 255.0
    a, b = _guided_coefficients(guide, hard_small.astype(np.float32), radius, eps)
    scale = max(h / sh, w / sw)
    band_r = max(2, round(radius * scale))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_r + 1, 2 * band_r + 1))
    near = cv2.resize(hard_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    inside = cv2.erode(near, k).astype(bool)
    outside = ~cv2.dilate(near, k).astype(bool)
    out = np.where(inside, 255, 0).astype(np.uint8)
    band = ~(inside | outside)
    ys, xs = np.nonzero(band)
    if ys.size:
        # solo sulla fascia del bordo: una foto da 12 MP non passa tutta per i float
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        a_full = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        b_full = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        g = full_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
        q = (a_full * g).sum(axis=-1) + b_full
        q = np.clip((q - 0.5) * contrast + 0.5, 0.0, 1.0)
        sub = band[y0:y1, x0:x1]
        out[y0:y1, x0:x1][sub] = np.rint(q[sub] * 255).astype(np.uint8)
    return out


REFINE_SIDE = 1600  # lato lungo a cui si decidono i bordi (poi ricalcolati in grande)
CRISP_EDGE = 60.0  # gradiente (Sobel su grigio 0..255) oltre il quale un contorno è "netto"
SNAP_REACH = 2.5  # raggio del filtro guidato = 2,5 volte lo spostamento che deve recuperare

Box = tuple[float, float, float, float]
"""(x0, y0, x1, y1) in pixel dell'immagine piena."""


def object_in_box(
    image_rgb: NDArray[np.uint8],
    box: Box,
    surface: NDArray[np.bool_] | None = None,
    max_side: int = 512,
) -> NDArray[np.bool_]:
    """Contorno esatto dell'oggetto dentro un riquadro approssimativo.

    L'agente indica solo DOVE sta l'oggetto (un rettangolo largo attorno a un bollitore, una
    presa, un giocattolo): il contorno lo trova il codice sulla foto. Con `surface` (la
    maschera attuale) la domanda posta a GrabCut è precisa: "dentro il riquadro, che cosa
    somiglia alla superficie che c'è attorno?". La superficie attorno al riquadro è lo
    sfondo certo; tutto ciò che attorno NON è superficie (piano, mobili) e l'interno del
    riquadro sono figura, certa o probabile. Resta superficie solo ciò che le somiglia.
    Se nel riquadro non emerge nulla di distinto (una presa bianca su un muro bianco) si
    esclude l'intero riquadro. Ritorna una maschera grande come la foto (True = oggetto).
    """
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((min(max(x0, 0.0), w), min(max(x1, 0.0), w)))
    y0, y1 = sorted((min(max(y0, 0.0), h), min(max(y1, 0.0), h)))
    out = np.zeros((h, w), bool)
    bx0, by0, bx1, by1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return out
    # contesto attorno al riquadro: lì GrabCut impara com'è la superficie
    mx, my = max(8, (bx1 - bx0) // 2), max(8, (by1 - by0) // 2)
    cx0, cy0, cx1, cy1 = max(0, bx0 - mx), max(0, by0 - my), min(w, bx1 + mx), min(h, by1 + my)
    crop = np.ascontiguousarray(image_rgb[cy0:cy1, cx0:cx1, :3])
    ch, cw = crop.shape[:2]
    sc = min(1.0, max_side / max(ch, cw))
    sw, sh = max(8, round(cw * sc)), max(8, round(ch * sc))
    small = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
    rx, ry = round((bx0 - cx0) * sc), round((by0 - cy0) * sc)
    rw, rh = max(2, round((bx1 - bx0) * sc)), max(2, round((by1 - by0) * sc))
    in_box = np.zeros((sh, sw), bool)
    in_box[ry : ry + rh, rx : rx + rw] = True
    if surface is None:
        surf = ~in_box  # senza maschera: tutto ciò che sta fuori dal riquadro è sfondo
        gc = np.where(in_box, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    else:
        part = surface[cy0:cy1, cx0:cx1].astype(np.uint8)
        surf = cv2.resize(part, (sw, sh), interpolation=cv2.INTER_AREA) > 0
        gc = np.full((sh, sw), cv2.GC_FGD, np.uint8)  # ciò che non è superficie resta fuori
        ring = surf & ~in_box
        gc[ring] = cv2.GC_BGD
        gc[surf & in_box] = cv2.GC_PR_FGD
        if ring.sum() >= 64:
            # attorno al riquadro può esserci un altro oggetto sfuggito alla maschera (un
            # cavo, una spina): non deve insegnare a GrabCut che "la superficie è anche
            # nera". I pixel dal colore anomalo rispetto alla superficie tipica sono dubbi.
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
            dist = np.linalg.norm(lab - np.median(lab[ring], axis=0), axis=-1)
            spread = float(np.median(dist[ring]))
            odd = ring & (dist > 3.0 * spread + 12.0)
            gc[odd] = cv2.GC_PR_BGD
    asked = surf & in_box if surface is not None else in_box  # i pixel su cui si decide
    fg_small = in_box.copy()
    if (gc == cv2.GC_BGD).sum() >= 64 and asked.sum() >= 16:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.setRNGSeed(0)
        try:
            cv2.grabCut(small[..., ::-1].copy(), gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
            fg_small = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)) & in_box
        except cv2.error:
            fg_small = in_box.copy()
    # briciole staccate dall'oggetto (un pezzo di fuga vecchia, un riflesso) non sono oggetto
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(fg_small.astype(np.uint8), 8)
    if n > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= 0.05 * areas.max()
        fg_small = keep[lab_cc]
    share = float((fg_small & asked).sum()) / max(1, int(asked.sum()))
    if share < 0.005 or share > 0.97:
        out[by0:by1, bx0:bx1] = True  # nessuna figura distinta: vale il riquadro
        return out
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg8 = cv2.dilate(fg_small.astype(np.uint8), k)  # un pixel di margine attorno all'oggetto
    fg = cv2.resize(fg8, (cw, ch), interpolation=cv2.INTER_LINEAR) > 0
    out[cy0:cy1, cx0:cx1] = fg
    out[:by0], out[by1:], out[:, :bx0], out[:, bx1:] = False, False, False, False
    return out


def finalize(
    image_rgb: NDArray[np.uint8],
    binary: NDArray[np.bool_],
    snap_zone: NDArray[np.bool_] | None = None,
    reach_px: float = 8.0,
) -> NDArray[np.uint8]:
    """Maschera definitiva 0..255 da una maschera netta a piena risoluzione.

    `snap_zone`: dove il bordo è stato disegnato a mano (poligoni degli agenti) e va
    portato sui contorni veri della foto; altrove il bordo resta dov'è. `reach_px`: di
    quanti pixel (della foto piena) il bordo disegnato può essere sbagliato. In ogni caso
    il bordo finale è ricalcolato sulla foto intera (`edge_alpha`).
    """
    h, w = binary.shape
    s = min(1.0, REFINE_SIDE / max(h, w))
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    area = cv2.INTER_AREA
    small = cv2.resize(binary.astype(np.float32), (sw, sh), interpolation=area)
    hard = small >= 0.5
    if snap_zone is not None and snap_zone.any() and hard.any() and not hard.all():
        zone = cv2.resize(snap_zone.astype(np.uint8), (sw, sh), interpolation=area) > 0
        rgb_small = cv2.resize(
            np.ascontiguousarray(image_rgb[..., :3]), (sw, sh), interpolation=area
        )
        # il filtro guidato sposta il bordo di circa metà del suo raggio: serve un raggio
        # di due volte e mezza l'errore atteso
        radius = max(3, round(SNAP_REACH * reach_px * s))
        soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), max(1.0, radius / 6))
        snapped = snap_to_edges(rgb_small, soft, radius=radius)
        hard = np.where(zone, snapped, hard)
    return readonly(edge_alpha(image_rgb, np.asarray(hard, dtype=bool)))


def outline_zone(
    shape_hw: tuple[int, int], polygons: list[Polygon], width_px: int
) -> NDArray[np.bool_]:
    """Fascia attorno al contorno dei poligoni (dove il bordo disegnato a mano va rifinito)."""
    img = Image.new("L", (shape_hw[1], shape_hw[0]), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        if len(poly) >= 3:
            draw.line([*poly, poly[0]], fill=255, width=max(1, int(width_px)))
    return np.asarray(img, dtype=np.uint8) > 0


def risk_map(image_rgb: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Dove la maschera è più probabilmente sbagliata (0..1), alla risoluzione di `mask`.

    Due segnali, entrambi misurati sulla foto:
    - dentro la maschera, zone il cui colore si stacca dalla superficie attorno (un oggetto
      rimasto dentro: verrebbe coperto dal rivestimento). Un'ombra normale non conta
      (cambia la luminosità ma non il colore); una zona molto più scura sì, perché può
      essere un oggetto nero come un'ombra profonda: la guarda il controllore;
    - tratti di bordo della maschera che non coincidono con nessun contorno della foto (il
      bordo taglia una zona uniforme: di solito superficie vecchia lasciata fuori).
    Serve a scegliere gli ingrandimenti da mostrare ai controllori, non a decidere.
    """
    h, w = mask.shape
    s = min(1.0, 800 / max(h, w))
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    rgb = cv2.resize(
        np.ascontiguousarray(image_rgb[..., :3]), (sw, sh), interpolation=cv2.INTER_AREA
    )
    m = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA) >= 128
    risk = np.zeros((sh, sw), np.float32)
    if m.sum() < 64:
        return np.asarray(cv2.resize(risk, (w, h)), dtype=np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    wgt = m.astype(np.float32)
    sigma = max(sw, sh) / 16
    den = cv2.GaussianBlur(wgt, (0, 0), sigma)
    local = cv2.GaussianBlur(lab * wgt[..., None], (0, 0), sigma) / np.maximum(den, 1e-4)[..., None]
    fine = cv2.GaussianBlur(lab, (0, 0), 1.5)
    d_color = np.hypot(fine[..., 1] - local[..., 1], fine[..., 2] - local[..., 2])
    d_light = np.abs(fine[..., 0] - local[..., 0])
    # scarto tipico della superficie stessa (venature, decori): si misura, non si suppone
    typical = float(np.quantile(d_color[m], 0.9)) + 4.0
    odd = np.clip((d_color - typical) / 18.0, 0, 1) + np.clip((d_light - 65.0) / 50.0, 0, 1)
    risk[m] = np.clip(odd, 0, 1)[m]
    # bordo senza contorno: gradiente della foto debole lungo il bordo della maschera
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.dilate(np.hypot(gx, gy), np.ones((7, 7), np.uint8))
    edge = m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge[:2], edge[-2:], edge[:, :2], edge[:, -2:] = False, False, False, False
    weak = edge & (grad < 40.0)
    risk = np.maximum(risk, cv2.dilate(weak.astype(np.float32) * 0.6, np.ones((5, 5), np.uint8)))
    risk = cv2.GaussianBlur(risk, (0, 0), 2.0)
    return np.asarray(cv2.resize(risk, (w, h), interpolation=cv2.INTER_LINEAR), dtype=np.float32)


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
