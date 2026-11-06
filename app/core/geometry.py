"""Geometria della stanza: omografia piano (mm) -> immagine (px) e controlli.

Convenzione: la stanza è il rettangolo [0, L] x [0, W] mm sul piano del pavimento.
L'utente indica gli angoli in ordine: P1 -> P2 è la lunghezza L, P2 -> P3 la larghezza W.
Gli angoli possono stare fuori dall'inquadratura (coordinate negative o oltre i bordi).
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from app.core.errors import GeometryError
from app.core.models import FocalCheck, PlaneGeometry

Mat3: TypeAlias = NDArray[np.float64]

_DIAG_35MM = math.hypot(36.0, 24.0)
MIN_CORNER_SINE = 0.05  # ~3°: angoli del quadrilatero più stretti di così = punti quasi allineati
MAX_CONDITION = 1e7
FOCAL_ASPECT_TOL = 0.15  # scarto massimo sul rapporto lunghezza/larghezza implicito
FOCAL_ANGLE_TOL_DEG = 10.0  # scarto massimo dall'angolo retto tra i muri


def _normalizer(pts: NDArray[np.float64]) -> Mat3:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = math.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def homography(src: NDArray[np.float64], dst: NDArray[np.float64]) -> Mat3:
    """DLT normalizzata (Hartley): H tale che dst ~ H @ src. Con 4 punti è esatta."""
    ts, td = _normalizer(src), _normalizer(dst)
    s = (ts @ np.c_[src, np.ones(len(src))].T).T
    d = (td @ np.c_[dst, np.ones(len(dst))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    return np.asarray(h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h / np.linalg.norm(h))


def project(h: Mat3, pts: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return np.asarray(p[:, :2] / p[:, 2:3])


def room_rect(length_mm: float, width_mm: float) -> NDArray[np.float64]:
    return np.array([[0.0, 0.0], [length_mm, 0.0], [length_mm, width_mm], [0.0, width_mm]])


def _check_quad(corners: NDArray[np.float64]) -> None:
    """Il rettangolo visto in prospettiva (davanti alla camera) è un quadrilatero convesso."""
    signs, sines = [], []
    for k in range(4):
        a, b, c = corners[k - 1], corners[k], corners[(k + 1) % 4]
        e1, e2 = b - a, c - b
        cross = float(e1[0] * e2[1] - e1[1] * e2[0])
        norm = float(np.linalg.norm(e1) * np.linalg.norm(e2))
        if norm == 0:
            raise GeometryError("due angoli coincidono")
        signs.append(math.copysign(1.0, cross))
        sines.append(abs(cross) / norm)
    if min(sines) < MIN_CORNER_SINE:
        raise GeometryError("angoli quasi allineati: il quadrilatero è degenere")
    if len(set(signs)) != 1:
        raise GeometryError(
            "il quadrilatero degli angoli non è convesso o si intreccia: "
            "controlla l'ordine dei punti (P1->P2 lunghezza, P2->P3 larghezza)"
        )


def focal_px_from_35mm(focal_35mm: float, width: int, height: int) -> float:
    """Focale in pixel dalla focale equivalente 35 mm (definita sulla diagonale)."""
    return focal_35mm * math.hypot(width, height) / _DIAG_35MM


def focal_check(h: Mat3, width: int, height: int, focal_35mm: float | None) -> FocalCheck:
    """Coerenza tra angoli, misure e focale EXIF.

    Con K = diag(f, f, 1) e punto principale al centro, le colonne h1, h2 di K^-1 H sono
    le direzioni dei muri scalate per la stessa costante: devono essere ortogonali e di
    uguale norma. Con la focale EXIF misuriamo quanto non lo sono:
    - angolo tra i muri implicito (atteso 90°);
    - rapporto delle scale implicite (atteso 1: altrimenti le misure non tornano).
    """
    t = np.array([[1.0, 0, -width / 2], [0, 1.0, -height / 2], [0, 0, 1.0]])
    hc = t @ h
    a1 = hc[0, 0] * hc[0, 1] + hc[1, 0] * hc[1, 1]
    b1 = hc[2, 0] * hc[2, 1]
    a2 = hc[0, 0] ** 2 + hc[1, 0] ** 2 - hc[0, 1] ** 2 - hc[1, 1] ** 2
    b2 = hc[2, 0] ** 2 - hc[2, 1] ** 2
    estimates = [-a / b for a, b in ((a1, b1), (a2, b2)) if abs(b) > 1e-18 and -a / b > 0]
    f_est = math.sqrt(float(np.median(estimates))) if estimates else None

    if focal_35mm is None:
        return FocalCheck(
            focal_35mm=None,
            f_px_exif=None,
            f_px_estimated=f_est,
            message="EXIF senza focale: controllo di coerenza saltato",
        )
    f = focal_px_from_35mm(focal_35mm, width, height)
    kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2 = kinv @ hc[:, 0], kinv @ hc[:, 1]
    cos = float(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    aspect_err = float(np.linalg.norm(r1) / np.linalg.norm(r2)) - 1.0
    ok = abs(aspect_err) <= FOCAL_ASPECT_TOL and abs(angle - 90) <= FOCAL_ANGLE_TOL_DEG
    msg = (
        "misure e angoli coerenti con la foto"
        if ok
        else "misure o angoli poco coerenti con la foto: ricontrolla lunghezza, larghezza "
        "e posizione degli angoli"
    )
    return FocalCheck(
        focal_35mm=focal_35mm,
        f_px_exif=f,
        f_px_estimated=f_est,
        axes_angle_deg=angle,
        aspect_error=aspect_err,
        consistent=ok,
        message=msg,
    )


def room_geometry(
    corners_px: list[tuple[float, float]],
    length_mm: float,
    width_mm: float,
    image_size: tuple[int, int],
    focal_35mm: float | None = None,
) -> PlaneGeometry:
    if len(corners_px) != 4:
        raise GeometryError("servono esattamente 4 angoli")
    if not (length_mm > 0 and width_mm > 0):
        raise GeometryError("lunghezza e larghezza devono essere positive")
    corners = np.asarray(corners_px, dtype=np.float64)
    if not np.all(np.isfinite(corners)):
        raise GeometryError("coordinate degli angoli non valide")
    _check_quad(corners)
    rect = room_rect(length_mm, width_mm)
    h = homography(rect, corners)
    cond = float(np.linalg.cond(_normalizer(corners) @ h @ np.linalg.inv(_normalizer(rect))))
    if not math.isfinite(cond) or cond > MAX_CONDITION:
        raise GeometryError(f"omografia mal condizionata (cond={cond:.3g})")
    reproj = float(np.abs(project(h, rect) - corners).max())
    w, hgt = image_size
    return PlaneGeometry(
        corners_px=[(float(x), float(y)) for x, y in corners],
        length_mm=length_mm,
        width_mm=width_mm,
        image_size=(w, hgt),
        H=h.tolist(),
        H_inv=np.linalg.inv(h).tolist(),
        condition_number=cond,
        reprojection_error_px=reproj,
        focal=focal_check(h, w, hgt, focal_35mm),
    )


def plane_coords(
    h_inv: Mat3, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pixel -> piano (mm). Ritorna (u, v, w) con w = denominatore proiettivo."""
    w = h_inv[2, 0] * xs + h_inv[2, 1] * ys + h_inv[2, 2]
    u = (h_inv[0, 0] * xs + h_inv[0, 1] * ys + h_inv[0, 2]) / w
    v = (h_inv[1, 0] * xs + h_inv[1, 1] * ys + h_inv[1, 2]) / w
    return u, v, w


def front_sign(geom: PlaneGeometry) -> float:
    """Segno del denominatore proiettivo per i punti del piano davanti alla camera."""
    hi = np.asarray(geom.H_inv)
    c = np.asarray(geom.corners_px)
    _, _, w = plane_coords(hi, c[:, 0], c[:, 1])
    return float(np.sign(w.mean()))


def check_mask_in_front(geom: PlaneGeometry, mask: NDArray[np.uint8]) -> None:
    """Ogni pixel della maschera deve vedere il pavimento, non il cielo oltre l'orizzonte."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return
    hi = np.asarray(geom.H_inv)
    w = hi[2, 0] * (xs + 0.5) + hi[2, 1] * (ys + 0.5) + hi[2, 2]
    s = front_sign(geom)
    # margine: troppo vicino all'orizzonte una piastrella diventa infinita
    w_ref = float(np.abs(w).max())
    bad = int(np.count_nonzero(w * s <= 1e-6 * w_ref))
    if bad:
        raise GeometryError(
            f"{bad} pixel della maschera sono oltre l'orizzonte del pavimento: "
            "controlla gli angoli o la maschera"
        )


def px_per_mm_range(
    geom: PlaneGeometry, mask: NDArray[np.uint8], stride: int = 8
) -> tuple[float, float]:
    """Densità locale (px per mm, media geometrica dei due assi) sulla maschera."""
    ys, xs = np.nonzero(mask[::stride, ::stride])
    if ys.size == 0:
        return (0.0, 0.0)
    x = xs.astype(np.float64) * stride + 0.5
    y = ys.astype(np.float64) * stride + 0.5
    det = np.abs(jacobian_det_image_to_plane(np.asarray(geom.H_inv), x, y))
    ppm = 1.0 / np.sqrt(det)
    return (float(ppm.min()), float(ppm.max()))


def jacobian_image_to_plane(
    h_inv: Mat3, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Derivate (du/dx, du/dy, dv/dx, dv/dy) in mm per pixel."""
    u, v, w = plane_coords(h_inv, xs, ys)
    dudx = (h_inv[0, 0] - u * h_inv[2, 0]) / w
    dudy = (h_inv[0, 1] - u * h_inv[2, 1]) / w
    dvdx = (h_inv[1, 0] - v * h_inv[2, 0]) / w
    dvdy = (h_inv[1, 1] - v * h_inv[2, 1]) / w
    return dudx, dudy, dvdx, dvdy


def jacobian_det_image_to_plane(
    h_inv: Mat3, xs: NDArray[np.float64], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    a, b, c, d = jacobian_image_to_plane(h_inv, xs, ys)
    return np.asarray(a * d - b * c)
