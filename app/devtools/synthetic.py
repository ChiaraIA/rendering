"""Generatori di dati di prova: foto, maschere, EXIF, ICC, catalogo.

Tutto è deterministico (seed espliciti) e scritto via codice: nei test non ci sono binari.
"""

from __future__ import annotations

import io
import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageDraw, ImageFilter

from app.core.ingest import sha256_file

MaskKind = Literal["empty", "full", "pixel", "polygon", "soft", "holes"]


def make_photo(w: int, h: int, channels: int = 3, seed: int = 0) -> NDArray[np.uint8]:
    """Rumore + gradienti + bordi netti: qualunque pixel alterato si nota."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            (xx * 255 // max(w - 1, 1)),
            (yy * 255 // max(h - 1, 1)),
            ((xx + yy) * 127 // max(w + h - 2, 1)),
        ],
        axis=-1,
    ).astype(np.int16)
    noise = rng.integers(-40, 41, size=(h, w, 3), dtype=np.int16)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    img[h // 3 : h // 3 + max(1, h // 20), :, :] = (250, 20, 20)  # bordo netto
    if channels == 4:
        a = rng.integers(0, 256, size=(h, w, 1), dtype=np.uint8)
        a[: h // 4] = 255
        a[h // 4 : h // 4 + 2] = 0  # completamente trasparente
        img = np.concatenate([img, a], axis=-1)
    return img


def make_mask(h: int, w: int, kind: MaskKind = "polygon", seed: int = 0) -> NDArray[np.uint8]:
    m = np.zeros((h, w), dtype=np.uint8)
    if kind == "empty":
        return m
    if kind == "full":
        m[:] = 255
        return m
    if kind == "pixel":
        rng = np.random.default_rng(seed)
        m[rng.integers(0, h), rng.integers(0, w)] = 255
        return m
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    floor = [(0, h - 1), (w * 0.2, h * 0.55), (w * 0.8, h * 0.55), (w - 1, h - 1)]
    draw.polygon(floor, fill=255)
    m = np.asarray(img, dtype=np.uint8).copy()
    if kind == "holes":
        m[int(h * 0.75) : int(h * 0.85), int(w * 0.4) : int(w * 0.6)] = 0  # "mobile" sul pavimento
    if kind == "soft":
        # bordo morbido solo verso l'interno: erosione progressiva
        inner = m > 0
        for v in (64, 128, 192):
            eroded = inner.copy()
            eroded[1:, :] &= inner[:-1, :]
            eroded[:-1, :] &= inner[1:, :]
            eroded[:, 1:] &= inner[:, :-1]
            eroded[:, :-1] &= inner[:, 1:]
            m[inner & ~eroded] = np.where(m[inner & ~eroded] == 255, v, m[inner & ~eroded])
            inner = eroded
    return m


@cache
def srgb_icc() -> bytes:
    """Calcolato una volta: LittleCMS scrive data e ora di creazione nell'header ICC."""
    return bytes(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


def make_exif(
    orientation: int | None = None,
    focal_35mm: int | None = None,
    make: str | None = None,
    model: str | None = None,
) -> Image.Exif:
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    if focal_35mm is not None:
        exif.get_ifd(0x8769)[0xA405] = focal_35mm
    return exif


def save_image(
    arr: NDArray[np.uint8],
    path: Path,
    fmt: str = "PNG",
    icc: bytes | None = None,
    exif_orientation: int | None = None,
    quality: int = 95,
    focal_35mm: int | None = None,
    camera: tuple[str, str] | None = None,
) -> Path:
    img = Image.fromarray(np.ascontiguousarray(arr))
    kwargs: dict[str, object] = {}
    if icc is not None:
        kwargs["icc_profile"] = icc
    if exif_orientation is not None or focal_35mm is not None or camera is not None:
        make, model = camera or (None, None)
        kwargs["exif"] = make_exif(exif_orientation, focal_35mm, make, model)
    if fmt == "JPEG":
        kwargs["quality"] = quality
    img.save(path, format=fmt, **kwargs)
    return path


def save_heic(
    arr: NDArray[np.uint8],
    path: Path,
    exif_orientation: int | None = None,
    primaries: int = 1,
    transfer: int = 13,
    icc: bytes | None = None,
    focal_35mm: int | None = None,
    camera: tuple[str, str] | None = None,
) -> Path:
    """HEIC come da iPhone. Lossless (RGB 4:4:4, matrice identità) per confronti esatti."""
    img = Image.fromarray(np.ascontiguousarray(arr))
    if icc is None:
        img.info["nclx_profile"] = {
            "color_primaries": primaries,
            "transfer_characteristics": transfer,
            "matrix_coefficients": 0,
            "full_range_flag": 1,
        }
    make, model = camera or (None, None)
    exif = make_exif(exif_orientation, focal_35mm, make, model)
    kwargs: dict[str, object] = {"quality": -1, "chroma": 444, "exif": exif.tobytes()}
    if icc is not None:
        kwargs["icc_profile"] = icc
    img.save(path, format="HEIF", **kwargs)
    return path


def save_mpo(frames: list[NDArray[np.uint8]], path: Path, focal_35mm: int | None = None) -> Path:
    """JPEG multi-immagine (come Ultra HDR / Samsung): Pillow lo apre come MPO.

    Pillow perde il segmento MPF se gli si passa l'EXIF in un salvataggio MPO: scriviamo
    l'MPO senza EXIF e inseriamo l'APP1 Exif subito dopo l'APP0. Gli offset MPF sono
    relativi al proprio header, quindi restano validi.
    """
    imgs = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    buf = io.BytesIO()
    imgs[0].save(buf, format="MPO", save_all=True, append_images=imgs[1:], quality=95)
    data = buf.getvalue()
    if focal_35mm is not None:
        payload = make_exif(focal_35mm=focal_35mm).tobytes()
        if not payload.startswith(b"Exif\x00\x00"):
            payload = b"Exif\x00\x00" + payload
        app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        app0_end = 4 + struct.unpack(">H", data[4:6])[0]
        assert data[2:4] == b"\xff\xe0"
        data = data[:app0_end] + app1 + data[app0_end:]
    path.write_bytes(data)
    return path


# Mappatura EXIF -> come appare l'immagine memorizzata (dalla specifica EXIF, non da Pillow).
DISPLAY_FROM_STORED: dict[int, Callable[[NDArray[np.uint8]], NDArray[np.uint8]]] = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1, :],
    5: lambda a: a.transpose(1, 0, 2),
    6: lambda a: np.rot90(a, -1),
    7: lambda a: np.rot90(a, 2).transpose(1, 0, 2),
    8: lambda a: np.rot90(a, 1),
}


def _clouds(h: int, w: int, rng: np.random.Generator, cells: int = 6) -> NDArray[np.float64]:
    """Rumore a bassa frequenza in [-1, 1] (griglia casuale ingrandita bicubica)."""
    small = rng.uniform(0, 255, size=(cells, max(2, cells * w // max(h, 1)))).astype(np.uint8)
    big = Image.fromarray(small).resize((w, h), Image.Resampling.BICUBIC)
    return np.asarray(big, dtype=np.float64) / 127.5 - 1.0


TextureKind = Literal["speckle", "wood", "marble"]


def make_tile_texture(
    size: int | tuple[int, int] = 128,
    base: tuple[int, int, int] = (128, 124, 118),
    seed: int = 0,
    kind: TextureKind = "speckle",
    marker: bool = False,
) -> NDArray[np.uint8]:
    """Texture procedurale di UNA piastrella (w, h in pixel). `marker` = quadratino rosso
    nell'angolo in alto a sinistra, utile per verificare rotazioni e orientamento."""
    w, h = (size, size) if isinstance(size, int) else size
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    tone = rng.uniform(-10, 10)
    if kind == "wood":
        # venature lunghe nel verso della doga (asse x): rumore stirato lungo x
        def streaks(cols: int, amp: float) -> NDArray[np.float64]:
            small = rng.uniform(0, 255, size=(h, max(2, cols))).astype(np.uint8)
            big = Image.fromarray(small).resize((w, h), Image.Resampling.BICUBIC)
            return (np.asarray(big, dtype=np.float64) / 127.5 - 1.0) * amp

        lum = tone + streaks(w // 90, 10) + streaks(w // 25, 6) + 2 * _clouds(h, w, rng, cells=3)
        # fiammatura: archi chiusi lungo la doga
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(-0.5, 1.5) * h
        arc = np.sin(np.hypot((xx - cx) / (w * 0.9), (yy - cy) / (h * 0.25)) * 40)
        lum += 5 * arc * np.exp(-(((xx - cx) / (w * 0.35)) ** 2))
    elif kind == "marble":
        turb = _clouds(h, w, rng, cells=5) * 3.0 + _clouds(h, w, rng, cells=9) * 1.2
        veins = np.abs(np.sin((xx + yy) / w * np.pi * 2.0 + turb))
        lum = tone - 38 * np.power(1 - veins, 8) + 4 * _clouds(h, w, rng, cells=4)
    else:
        lum = tone + 9 * _clouds(h, w, rng, cells=5)
    tex = np.asarray(base, dtype=np.float64) + lum[..., None]
    tex += rng.normal(0, 4 if kind == "wood" else 7, size=(h, w, 1))
    if kind == "wood":  # smusso dei bordi: le giunzioni delle doghe a incastro si leggono
        edge = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy])
        tex *= (1 - 0.35 * np.exp(-edge / 1.2))[..., None]
    out = np.clip(np.rint(tex), 0, 255).astype(np.uint8)
    if marker:
        m = max(2, min(w, h) // 10)
        out[:m, :m] = (220, 30, 30)
    return out


@dataclass(frozen=True)
class SampleMaterial:
    id: str
    name: str
    base: tuple[int, int, int]
    kind: TextureKind
    tile_mm: tuple[float, float]
    joint_mm: float
    joint_color: str | None
    pattern: dict[str, object]
    n_variants: int
    variant_rotations: tuple[int, ...]


SAMPLE_MATERIALS = (
    SampleMaterial(
        "gres-grigio-30x30",
        "Gres grigio 30x30",
        (128, 124, 118),
        "speckle",
        (300, 300),
        3,
        "#b8b4ac",
        {"type": "grid"},
        4,
        (0, 90, 180, 270),
    ),
    SampleMaterial(
        "laminato-rovere-120x20",
        "Laminato rovere a incastro 120x20",
        (164, 120, 78),
        "wood",
        (1200, 200),
        0,
        None,
        {"type": "running_bond", "bond_offset": 0.33},
        3,
        (0, 180),
    ),
    SampleMaterial(
        "marmo-bianco-60x60",
        "Marmo bianco 60x60 in diagonale",
        (226, 224, 219),
        "marble",
        (600, 600),
        2,
        "#d9d6cf",
        {"type": "grid", "rotation_deg": 45},
        3,
        (0, 90, 180, 270),
    ),
)

TEXTURE_PX_PER_MM = 0.8
TEXTURE_MAX_SIDE = 768


def texture_size_px(tile_mm: tuple[float, float]) -> tuple[int, int]:
    scale = min(TEXTURE_PX_PER_MM, TEXTURE_MAX_SIDE / max(tile_mm))
    return (max(8, round(tile_mm[0] * scale)), max(8, round(tile_mm[1] * scale)))


def write_catalog(
    root: Path, materials: tuple[SampleMaterial, ...] = SAMPLE_MATERIALS, marker: bool = False
) -> Path:
    """Scrive catalog.json (schema 2.0) + textures/ con hash calcolati."""
    root = Path(root)
    (root / "textures").mkdir(parents=True, exist_ok=True)
    out: list[dict[str, object]] = []
    for i, m in enumerate(materials):
        variants = []
        for v in range(m.n_variants):
            tex_name = f"{m.id}-v{v + 1}.png"
            tex_path = root / "textures" / tex_name
            tex = make_tile_texture(
                texture_size_px(m.tile_mm), m.base, seed=100 * i + v, kind=m.kind, marker=marker
            )
            save_image(tex, tex_path)
            variants.append({"texture_file": tex_name, "texture_sha256": sha256_file(tex_path)})
        out.append(
            {
                "id": m.id,
                "name": m.name,
                "version": "1",
                "tile_w_mm": m.tile_mm[0],
                "tile_h_mm": m.tile_mm[1],
                "joint_mm": m.joint_mm,
                "joint_color": m.joint_color,
                "pattern": m.pattern,
                "variants": variants,
                "variant_rotations": list(m.variant_rotations),
                "texture_color_space": "srgb",
            }
        )
    path = root / "catalog.json"
    path.write_text(json.dumps({"schema_version": "2.0", "materials": out}, indent=2) + "\n")
    return path


if __name__ == "__main__":  # rigenera il catalogo di esempio distribuito col pacchetto
    print(write_catalog(Path(__file__).resolve().parents[1] / "catalog" / "data"))


# ------------------------------------------------------------------ stanza sintetica


@dataclass(frozen=True, eq=False)
class RoomScene:
    """Foto sintetica di una stanza con verità geometrica nota."""

    photo: NDArray[np.uint8]
    floor_mask: NDArray[np.uint8]  # pavimento visibile (escluso il divano)
    corners_px: NDArray[np.float64]  # angoli della stanza proiettati (possono uscire dall'immagine)
    H: NDArray[np.float64]  # piano pavimento (mm) -> immagine (px)
    K: NDArray[np.float64]
    room_mm: tuple[float, float]
    focal_35mm: float


def _look(pitch_deg: float, yaw_deg: float) -> NDArray[np.float64]:
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    fwd = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), -np.sin(p)])
    right = np.array([np.cos(y), -np.sin(y), 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])  # righe: assi camera in coordinate mondo


def make_room_scene(
    width: int = 1600,
    height: int = 1200,
    room_mm: tuple[float, float] = (4200.0, 3600.0),
    focal_35mm: float = 26.0,
    cam: tuple[float, float, float] = (2300.0, 250.0, 1550.0),
    pitch_deg: float = 27.0,
    yaw_deg: float = -4.0,
    seed: int = 0,
) -> RoomScene:
    """Stanza L x W (mm) vista da vicino al muro d'ingresso: gli angoli vicini sono fuori campo.

    Mondo: pavimento z=0, stanza [0,L]x[0,W], soffitto a 2700 mm. Divano appoggiato al
    muro di fondo, finestra sul muro sinistro, vecchio pavimento a quadri 400 mm con fughe.
    """
    L, W = room_mm
    ceil_h = 2700.0
    f = focal_35mm * np.hypot(width, height) / np.hypot(36, 24)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    R = _look(pitch_deg, yaw_deg)
    C = np.asarray(cam)
    H = K @ np.stack([R[:, 0], R[:, 1], -R @ C], axis=1)
    H = H / H[2, 2]

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    rays_cam = np.stack([(xs + 0.5 - K[0, 2]) / f, (ys + 0.5 - K[1, 2]) / f, np.ones_like(xs)], -1)
    d = rays_cam @ R  # direzione nel mondo (R ortonormale)
    t_best = np.full(xs.shape, np.inf)
    kind = np.zeros(xs.shape, np.int8)  # 1 pav, 2 muro sx, 3 muro dx, 4 fondo, 5 soffitto, 6 divano
    hit = np.zeros(xs.shape + (3,))

    def plane(axis: int, value: float, k: int) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (value - C[axis]) / d[..., axis]
        p = C + t[..., None] * d
        ok = (t > 1e-6) & (t < t_best)
        ok &= (p[..., 0] >= -1) & (p[..., 0] <= L + 1) & (p[..., 1] >= -1) & (p[..., 1] <= W + 1)
        ok &= (p[..., 2] >= -1) & (p[..., 2] <= ceil_h + 1)
        t_best[ok] = t[ok]
        kind[ok] = k
        hit[ok] = p[ok]

    plane(2, 0.0, 1)
    plane(0, 0.0, 2)
    plane(0, L, 3)
    plane(1, W, 4)
    plane(2, ceil_h, 5)

    # divano: scatola [u0,u1] x [v0,v1] x [0,h]
    box_lo = np.array([1350.0, W - 900.0, 0.0])
    box_hi = np.array([3050.0, W - 60.0, 760.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box_lo - C) / d
        t2 = (box_hi - C) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    sofa = (tmax >= tmin) & (tmin > 0) & (tmin < t_best)
    t_best[sofa] = tmin[sofa]
    kind[sofa] = 6
    hit[sofa] = C + tmin[sofa][..., None] * d[sofa]

    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3))
    # luce: finestra sul muro sinistro + decadimento con la distanza
    win = np.array([0.0, W * 0.55, 1500.0])
    dist = np.linalg.norm(hit - win, axis=-1)
    light = 0.55 + 0.75 * np.exp(-((dist / 2600.0) ** 2))

    fl = kind == 1
    u, v = hit[..., 0], hit[..., 1]
    cell = 400.0
    cu, cv_ = np.floor(u / cell), np.floor(v / cell)
    checker = (cu + cv_) % 2 == 0
    grout = (np.mod(u, cell) < 4) | (np.mod(v, cell) < 4)
    floor_col = np.where(checker[..., None], [196, 170, 132], [150, 112, 80]).astype(np.float64)
    floor_col[grout] = [228, 222, 210]
    # ombra di contatto del divano sul pavimento
    near_sofa = np.clip(
        1
        - np.hypot(
            np.maximum(box_lo[0] - u, 0) + np.maximum(u - box_hi[0], 0),
            np.maximum(box_lo[1] - v, 0) + np.maximum(v - box_hi[1], 0),
        )
        / 450.0,
        0,
        1,
    )
    shade = light * (1 - 0.45 * near_sofa)
    img[fl] = floor_col[fl] * shade[fl, None] / 1.1
    wall = (kind >= 2) & (kind <= 4)
    img[wall] = (
        np.array([236, 230, 220])
        * (0.75 + 0.25 * light[wall, None])
        * np.where(kind[wall] == 4, 0.92, 1.0)[:, None]
    )
    win_px = (kind == 2) & (np.abs(v - win[1]) < 600) & (np.abs(hit[..., 2] - 1500) < 650)
    img[win_px] = [250, 252, 255]
    img[kind == 5] = [246, 244, 240]
    sf = kind == 6
    top = sf & (hit[..., 2] > box_hi[2] - 2)
    img[sf] = np.array([70, 92, 120]) * light[sf, None]
    img[top] = np.array([92, 116, 146]) * light[top, None]

    img += rng.normal(0, 3.0, img.shape)
    photo = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    photo = np.asarray(Image.fromarray(photo).filter(ImageFilter.GaussianBlur(0.6)), dtype=np.uint8)
    in_room = (u >= 0) & (u <= L) & (v >= 0) & (v <= W)
    mask = (fl & in_room).astype(np.uint8) * 255
    corners = np.array([[0, 0], [L, 0], [L, W], [0, W]], dtype=np.float64)
    pc = np.c_[corners, np.ones(4)] @ H.T
    return RoomScene(photo, mask, pc[:, :2] / pc[:, 2:3], H, K, (L, W), focal_35mm)
