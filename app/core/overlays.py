"""Immagini di lavoro per gli agenti (mai il risultato finale): funzioni pure che
restituiscono array; la scrittura su disco è compito di `storage`."""

from __future__ import annotations

from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from app.core.models import PlaneGeometry

Arr: TypeAlias = NDArray[np.uint8]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_to_width(img: Arr, width: int) -> tuple[Arr, float]:
    """Ridimensiona a `width` px di larghezza; ritorna (immagine, scala preview/originale)."""
    h, w = img.shape[:2]
    data = img if img.ndim == 2 else img[..., :3]  # maschere 2D: nessun taglio di canali
    scale = min(1.0, width / w)
    if scale == 1.0:
        return data.copy(), 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    out = Image.fromarray(np.ascontiguousarray(data)).resize(size, Image.Resampling.LANCZOS)
    return np.asarray(out, dtype=np.uint8), scale


def coordinate_grid(img: Arr, step: int = 100) -> Arr:
    """Righello di coordinate (pixel della preview) per aiutare gli agenti a indicare punti."""
    pil = Image.fromarray(np.ascontiguousarray(img[..., :3])).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = pil.size
    font = _font(14)
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(255, 255, 0, 70), width=1)
        d.text(
            (x + 3, 3),
            str(x),
            fill=(255, 255, 0, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(255, 255, 0, 70), width=1)
        d.text(
            (3, y + 3),
            str(y),
            fill=(255, 255, 0, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )
    return np.asarray(Image.alpha_composite(pil, layer).convert("RGB"), dtype=np.uint8)


def mask_overlay(img: Arr, mask: NDArray[np.uint8]) -> Arr:
    """Pavimento selezionato in azzurro, contorno in rosso."""
    base = img[..., :3].astype(np.float32)
    a = (mask.astype(np.float32) / 255.0)[..., None] * 0.45
    tint = np.array([0, 200, 255], np.float32)
    out = base * (1 - a) + tint * a
    m = mask >= 128
    edge = m ^ _erode(m)
    out[edge] = [255, 40, 40]
    return np.clip(out, 0, 255).astype(np.uint8)


def _erode(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    e = m.copy()
    e[1:] &= m[:-1]
    e[:-1] &= m[1:]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    e2 = e.copy()
    e2[1:] &= e[:-1]
    e2[:-1] &= e[1:]
    e2[:, 1:] &= e[:, :-1]
    e2[:, :-1] &= e[:, 1:]
    return e2


def plane_grid_overlay(
    img: Arr, geom: PlaneGeometry, mask: NDArray[np.uint8] | None, step_mm: float = 500.0
) -> Arr:
    """Quadrati di `step_mm` sul pavimento, proiettati con l'omografia: se la geometria è
    giusta sembrano quadrati veri posati a terra, allineati ai muri."""
    h_mat = np.asarray(geom.H)
    h, w = img.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    L, W = geom.length_mm, geom.width_mm

    def proj(pts: NDArray[np.float64]) -> list[tuple[float, float]]:
        p = np.c_[pts, np.ones(len(pts))] @ h_mat.T
        ok = p[:, 2] > 1e-9 * np.abs(p[:, 2]).max()
        q = p[ok, :2] / p[ok, 2:3]
        return [(float(x), float(y)) for x, y in q]

    n = 200
    for u in np.arange(0, L + 1e-6, step_mm):
        line = np.stack([np.full(n, u), np.linspace(0, W, n)], 1)
        d.line(proj(line), fill=(255, 230, 0, 230), width=2)
    for v in np.arange(0, W + 1e-6, step_mm):
        line = np.stack([np.linspace(0, L, n), np.full(n, v)], 1)
        d.line(proj(line), fill=(255, 230, 0, 230), width=2)
    rect = np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float)
    pts = []
    for a, b in zip(rect[:-1], rect[1:], strict=True):
        pts += proj(np.linspace(a, b, n))
    d.line(pts, fill=(255, 60, 60, 255), width=4)
    arr = np.asarray(layer, dtype=np.float32)
    if mask is not None:
        arr[..., 3] *= np.where(mask > 0, 1.0, 0.35)
    alpha = arr[..., 3:4] / 255.0
    out = img[..., :3].astype(np.float32) * (1 - alpha) + arr[..., :3] * alpha
    pil = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    dd = ImageDraw.Draw(pil)
    dd.text(
        (10, 10),
        f"griglia {step_mm / 10:.0f} cm - stanza {L / 1000:.2f} x {W / 1000:.2f} m",
        fill=(255, 255, 255),
        font=_font(22),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(pil, dtype=np.uint8)


def side_by_side(before: Arr, after: Arr, labels: tuple[str, str] = ("PRIMA", "DOPO")) -> Arr:
    a = Image.fromarray(np.ascontiguousarray(before[..., :3]))
    b = Image.fromarray(np.ascontiguousarray(after[..., :3]))
    gap = 12
    out = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height)), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + gap, 0))
    d = ImageDraw.Draw(out)
    f = _font(max(18, a.width // 30))
    for x, text in ((12, labels[0]), (a.width + gap + 12, labels[1])):
        d.text((x, 10), text, fill=(255, 255, 255), font=f, stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(out, dtype=np.uint8)


def contact_sheet(items: list[tuple[str, Arr]], thumb: int = 220, cols: int = 3) -> Arr:
    """Miniature etichettate (es. catalogo), per il confronto visivo."""
    rows = (len(items) + cols - 1) // cols
    label_h = 46
    sheet = Image.new(
        "RGB", (cols * (thumb + 10) + 10, rows * (thumb + label_h + 10) + 10), "white"
    )
    d = ImageDraw.Draw(sheet)
    font = _font(15)
    for k, (label, img) in enumerate(items):
        r, c = divmod(k, cols)
        t = Image.fromarray(np.ascontiguousarray(img[..., :3]))
        t.thumbnail((thumb, thumb))
        x = 10 + c * (thumb + 10)
        y = 10 + r * (thumb + label_h + 10)
        sheet.paste(t, (x, y))
        for i, line in enumerate(_wrap(label, 26)[:2]):
            d.text((x, y + thumb + 4 + i * 18), line, fill=(0, 0, 0), font=font)
    return np.asarray(sheet, dtype=np.uint8)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines


def tile_patch(
    texture: Arr, joint_rgb: tuple[int, int, int] | None, reps: int = 2, hexagon: bool = False
) -> Arr:
    """Piccolo campione di posa (2x2 piastrelle con fuga) per mostrare un materiale.
    Per gli esagoni mostra la singola piastrella su fondo bianco."""
    if hexagon:
        h, w = texture.shape[:2]
        mask = Image.new("L", (w, h), 0)
        r = w / 2
        pts = [
            (w / 2 + r * np.cos(np.deg2rad(a)), h / 2 + r * np.sin(np.deg2rad(a)))
            for a in range(0, 360, 60)
        ]
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        m = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]
        return (texture[..., :3] * m + 255 * (1 - m)).astype(np.uint8)
    h, w = texture.shape[:2]
    j = max(2, w // 60) if joint_rgb else 0
    out = np.full(
        (reps * h + (reps + 1) * j, reps * w + (reps + 1) * j, 3), joint_rgb or (0, 0, 0), np.uint8
    )
    for r_ in range(reps):
        for c in range(reps):
            y, x = j + r_ * (h + j), j + c * (w + j)
            out[y : y + h, x : x + w] = texture[..., :3]
    return out


def boundary_details(
    img: Arr, mask: NDArray[np.uint8], n: int = 4, crop: tuple[int, int] = (300, 220), zoom: int = 2
) -> Arr:
    """Tavola con gli ingrandimenti delle zone dove il bordo della maschera è più complicato
    (gambe di sedie e tavoli, piedi dei mobili, angoli): lì si vedono gli errori di contorno."""
    h, w = mask.shape
    cw, ch = min(crop[0], w), min(crop[1], h)
    m = mask >= 128
    edge = (m ^ _erode(m)).astype(np.float32)
    edge[:3], edge[-3:], edge[:, :3], edge[:, -3:] = 0, 0, 0, 0  # il bordo della foto non conta
    density = cv2.boxFilter(edge, -1, (cw, ch), normalize=False, borderType=cv2.BORDER_CONSTANT)
    over = mask_overlay(img, mask)
    tiles: list[Arr] = []
    for _ in range(n):
        y, x = np.unravel_index(int(np.argmax(density)), density.shape)
        if density[y, x] <= 0:
            break
        x0 = int(np.clip(x - cw // 2, 0, w - cw))
        y0 = int(np.clip(y - ch // 2, 0, h - ch))
        tile = Image.fromarray(np.ascontiguousarray(over[y0 : y0 + ch, x0 : x0 + cw]))
        tile = tile.resize((cw * zoom, ch * zoom), Image.Resampling.LANCZOS)
        d = ImageDraw.Draw(tile)
        d.text(
            (8, 6),
            f"dettaglio {len(tiles) + 1}  (x {x0}-{x0 + cw}, y {y0}-{y0 + ch})",
            fill=(255, 255, 255),
            font=_font(18),
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )
        tiles.append(np.asarray(tile, dtype=np.uint8))
        density[max(0, y - ch) : y + ch, max(0, x - cw) : x + cw] = 0  # niente sovrapposizioni
    if not tiles:
        return over
    cols = 2 if len(tiles) > 1 else 1
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sheet = np.full((rows * th + (rows - 1) * 8, cols * tw + (cols - 1) * 8, 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * (th + 8) : r * (th + 8) + th, c * (tw + 8) : c * (tw + 8) + tw] = t
    return sheet
