"""Import nel catalogo: raddrizza la foto di una piastrella scattata in prospettiva e toglie
la luce dello studio fotografico (riflessi, vignettatura), che altrimenti si ripeterebbe
uguale su ogni piastrella posata."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.core import ingest
from app.core.colorspace import from_linear, to_linear


def flatten_lighting(tex: NDArray[np.uint8], strength: float = 1.0) -> NDArray[np.uint8]:
    """Toglie la variazione di luce a bassa frequenza, mantenendo il dettaglio.

    Si lavora in luce lineare, canale per canale: ogni canale viene sfocato molto (~1/6 del
    lato) e riportato alla sua media. Così spariscono sia lo schiarimento sia lo sbiadimento
    del colore causati da riflessi e vignettatura; il dettaglio fine resta intatto.
    Adatto a piastrelle a tinta unita o maculate; per decori grandi usare --mantieni-luce.
    """
    lin = to_linear(np.ascontiguousarray(tex[..., :3]), "srgb").astype(np.float32)
    sigma = max(tex.shape[:2]) / 6
    low = cv2.GaussianBlur(lin, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    mean = low.reshape(-1, 3).mean(axis=0)
    gain = mean / np.maximum(low, 1e-6)
    gain = 1.0 + (gain - 1.0) * strength
    out = np.clip(lin * gain, 0.0, 1.0)
    return from_linear(out.astype(np.float64), "srgb")


def rectify_tile(
    src: Path, corners: str, w_cm: float, h_cm: float, dst: Path, flatten: bool = True
) -> None:
    """Raddrizza la faccia della piastrella fotografata in prospettiva (vista frontale)."""
    pts = parse_points(corners, 4).astype(np.float32)
    img = ingest.load_image(src).pixels[..., :3]
    scale = min(1024 / w_cm, 1024 / h_cm, 30.0)  # px per cm, lato massimo 1024 px
    out_w, out_h = round(w_cm * scale), round(h_cm * scale)
    target = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, target)
    flat = cv2.warpPerspective(
        np.ascontiguousarray(img),
        m,
        (out_w, out_h),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inset = max(1, round(min(out_w, out_h) * 0.03))  # via smusso e riflessi del bordo
    flat = flat[inset:-inset, inset:-inset]
    if flatten:
        flat = flatten_lighting(flat)
    Image.fromarray(np.ascontiguousarray(flat)).resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    ).save(dst, format="PNG")


# ------------------------------------------------------------------- esagoni

SQRT3 = float(np.sqrt(3.0))


def hex_vertices(radius: float, cx: float = 0.0, cy: float = 0.0) -> NDArray[np.float64]:
    """Vertici di un esagono "flat-top" (lati piatti in alto e in basso, punte a dx e sx),
    in senso orario a partire dalla punta destra (coordinate immagine, y verso il basso)."""
    ang = np.deg2rad(np.arange(6) * 60.0)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)


def detect_hexagon(img: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Trova i 6 vertici della piastrella esagonale su sfondo chiaro uniforme (foto prodotto).

    Ritorna i vertici nell'ordine di `hex_vertices` (punta destra, poi in senso orario).
    """
    rgb = img[..., :3].astype(np.int32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.abs(rgb - bg).sum(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    tile = ((diff > 30) | (sat > 14)).astype(np.uint8) * 255
    tile = cv2.morphologyEx(tile, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    tile = cv2.morphologyEx(tile, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("nessuna piastrella trovata nella foto")
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    poly = None
    for eps in np.linspace(0.005, 0.08, 60):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 6:
            poly = approx[:, 0, :].astype(np.float64)
            break
    if poly is None:
        raise ValueError("la sagoma trovata non è un esagono: indica gli angoli a mano")
    c = poly.mean(axis=0)
    ang = np.mod(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]), 2 * np.pi)
    poly = poly[np.argsort(ang)]
    # la punta destra (angolo più vicino a 0) diventa il primo vertice
    start = int(np.argmin(np.minimum(np.sort(ang), 2 * np.pi - np.sort(ang))))
    poly = np.roll(poly, -start, axis=0)
    return _refine_corners(poly, hull[:, 0, :].astype(np.float64))


def _refine_corners(poly: NDArray[np.float64], outline: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vertici precisi: retta ai minimi quadrati su ogni lato (solo la parte centrale, lontana
    da angoli smussati) e intersezione dei lati adiacenti."""
    dense = []
    for k in range(len(outline)):  # il guscio convesso ha pochi punti: lo infittiamo
        a, b = outline[k], outline[(k + 1) % len(outline)]
        n = max(2, int(np.hypot(*(b - a))))
        dense.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    pts = np.concatenate(dense)
    lines = []
    for k in range(6):
        a, b = poly[k], poly[(k + 1) % 6]
        d = b - a
        length = float(np.hypot(*d))
        t = ((pts - a) @ d) / (length**2)
        rel = pts - a
        dist = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / length
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.05 * length)]
        if len(sel) < 5:
            return poly
        vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)[:, 0]
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = np.empty_like(poly)
    for k in range(6):  # il vertice k sta tra il lato k-1 e il lato k
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        m = np.array([d1, -d2]).T
        if abs(np.linalg.det(m)) < 1e-9:
            return poly
        s = np.linalg.solve(m, p2 - p1)[0]
        out[k] = p1 + s * d1
    return out if np.abs(out - poly).max() < 0.1 * np.ptp(poly[:, 0]) else poly


def _fill_outside(img: NDArray[np.uint8], inside: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """Riempie i pixel fuori dalla sagoma con il colore del pixel interno più vicino
    (evita aloni di sfondo lungo i bordi quando la texture viene filtrata)."""
    _, labels = cv2.distanceTransformWithLabels(
        (~inside).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(~inside == 0)
    lut = np.zeros((labels.max() + 1, 3), np.uint8)
    lut[labels[ys, xs]] = img[ys, xs]
    out = img.copy()
    out[~inside] = lut[labels[~inside]]
    return out


def apply_gains(tex: NDArray[np.uint8], gains: tuple[float, float, float]) -> NDArray[np.uint8]:
    """Correzione di colore/esposizione in luce lineare (es. bilanciamento del bianco di una
    foto diversa da quelle prodotto)."""
    lin = to_linear(np.ascontiguousarray(tex[..., :3]), "srgb") * np.asarray(gains)
    return from_linear(np.clip(lin, 0.0, 1.0), "srgb")


def parse_points(text: str, n: int) -> NDArray[np.float64]:
    pts = np.array(
        [[float(v) for v in p.split(",")] for p in text.replace(";", " ").split()], dtype=np.float64
    )
    if pts.shape != (n, 2):
        raise ValueError(f"servono {n} punti x,y")
    return pts


def rectify_hexagon(
    src: Path,
    dst: Path,
    vertices: str | None = None,
    radius_px: int = 512,
    flatten: bool = False,
    inset: float = 0.015,
    gains: tuple[float, float, float] | None = None,
) -> None:
    """Esagono della foto -> texture frontale (2R x √3·R px) con gli angoli riempiti.

    `vertices`: 6 punti "x,y" (punta destra, poi in senso orario); None = rilevamento automatico.
    """
    img = ingest.load_image(src).pixels[..., :3]
    found = parse_points(vertices, 6) if vertices else detect_hexagon(img)
    w, h = 2 * radius_px, round(SQRT3 * radius_px)
    # i vertici trovati vanno su un esagono ideale appena più grande: smusso e sfondo
    # restano fuori dalla texture
    ideal = hex_vertices(radius_px * (1 + inset), w / 2, h / 2)
    m, _ = cv2.findHomography(found.astype(np.float32), ideal.astype(np.float32))
    tex = cv2.warpPerspective(
        np.ascontiguousarray(img),
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inside = np.zeros((h, w), np.uint8)
    poly = np.round(hex_vertices(radius_px - 2, w / 2, h / 2)).astype(np.int32)
    cv2.fillConvexPoly(inside, poly, 1)
    tex = _fill_outside(tex, inside.astype(bool))
    if flatten:
        tex = flatten_lighting(tex)
    if gains:
        tex = apply_gains(tex, gains)
    Image.fromarray(np.ascontiguousarray(tex)).save(dst, format="PNG")
