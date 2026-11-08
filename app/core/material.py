"""Materiale: dal catalogo a un layer in luce lineare nello spazio colore della foto.

- "tiled": piastrelle posate in prospettiva sulla stanza (fughe, schema, varianti dal
  catalogo), supersampling stratificato + mip-map in luce lineare;
- "flat" (debug): colore medio in luce lineare di tutte le varianti.

Coordinate di posa: origine al centro della stanza, assi ruotati di `pattern.rotation_deg`
rispetto ai muri; la piastrella (0, 0) è centrata nella stanza (tagli simmetrici).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core import geometry
from app.core.colorspace import SRGB_LIKE, convert_srgb_to_profile, to_linear, trc_for
from app.core.errors import CatalogError
from app.core.models import (
    SQRT3,
    TILE_NONE,
    BBox,
    MaterialLayer,
    MaterialSpec,
    PlaneGeometry,
    SourceImage,
)

SAMPLES_PER_CHUNK = 2_000_000


def texture_in_photo_space(
    texture_srgb: NDArray[np.uint8], source: SourceImage
) -> tuple[NDArray[np.uint8], list[str]]:
    """Porta la texture (sRGB) nello spazio colore della foto, se diverso."""
    cs = source.meta.color_space
    if cs in SRGB_LIKE or source.icc is None:
        return texture_srgb, []
    try:
        return convert_srgb_to_profile(texture_srgb, source.icc), []
    except Exception as exc:  # profilo non utilizzabile da LittleCMS
        return texture_srgb, [f"conversione ICC della texture fallita ({exc}): usata sRGB"]


def variants_in_photo_space(
    variants_srgb: list[NDArray[np.uint8]], source: SourceImage
) -> tuple[list[NDArray[np.uint8]], list[str]]:
    out, notes = [], []
    for tex in variants_srgb:
        conv, n = texture_in_photo_space(tex, source)
        out.append(conv)
        notes.extend(n)
    return out, sorted(set(notes))


def flat_color_linear(
    variants: list[NDArray[np.uint8]], source: SourceImage
) -> NDArray[np.float64]:
    """Colore medio fisicamente corretto: media in luce lineare, non sui valori codificati.

    Ogni variante pesa allo stesso modo, indipendentemente dalla risoluzione della foto.
    """
    trc = trc_for(source.meta.color_space)
    means = [to_linear(t.reshape(-1, 3), trc).mean(axis=0) for t in variants]
    return np.asarray(np.mean(means, axis=0), dtype=np.float64)


def flat_layer(
    spec: MaterialSpec,
    variants_srgb: list[NDArray[np.uint8]],
    source: SourceImage,
    bbox: BBox,
) -> MaterialLayer:
    variants, notes = variants_in_photo_space(variants_srgb, source)
    color = flat_color_linear(variants, source).astype(np.float32)
    x0, y0, x1, y1 = bbox
    rgb = np.broadcast_to(color, (y1 - y0, x1 - x0, 3))
    return MaterialLayer(
        rgb_linear=rgb,
        bbox=bbox,
        material_id=spec.id,
        render_mode="flat",
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------- posa


@dataclass(frozen=True)
class Laying:
    """Regole di posa del materiale applicate a una stanza L x W.

    Rettangoli: griglia (eventualmente sfalsata) con passo piastrella + fuga.
    Esagoni (lati piatti in alto/basso): reticolo con raggio `r_lat` = raggio della
    piastrella + fuga/√3, così tra due lati affacciati restano esattamente `joint` mm.
    """

    tile_w: float
    tile_h: float
    pitch_w: float
    pitch_h: float
    shift: float  # sfalsamento per riga (running bond), in mm
    cos: float
    sin: float
    cx: float
    cy: float
    hex: bool = False
    r_tile: float = 0.0  # esagoni: raggio (centro-punta) della piastrella
    r_lat: float = 0.0  # esagoni: raggio del reticolo (piastrella + mezza fuga per lato)

    @classmethod
    def of(cls, spec: MaterialSpec, length_mm: float, width_mm: float) -> Laying:
        pw, ph = spec.pitch_mm
        shift = spec.pattern.bond_offset * pw if spec.pattern.type == "running_bond" else 0.0
        th = math.radians(spec.pattern.rotation_deg)
        r_tile = spec.tile_w_mm / 2
        return cls(
            spec.tile_w_mm,
            spec.tile_h_mm,
            pw,
            ph,
            shift,
            math.cos(th),
            math.sin(th),
            length_mm / 2,
            width_mm / 2,
            hex=spec.is_hex,
            r_tile=r_tile,
            r_lat=r_tile + spec.joint_mm / SQRT3,
        )

    @property
    def tile_area(self) -> float:
        if self.hex:
            return 1.5 * SQRT3 * self.r_tile**2
        return self.tile_w * self.tile_h

    def local(
        self, u: NDArray[np.float64], v: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        du, dv = u - self.cx, v - self.cy
        return du * self.cos + dv * self.sin, -du * self.sin + dv * self.cos

    def to_plane(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.cx + a * self.cos - b * self.sin, self.cy + a * self.sin + b * self.cos

    def cell(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """Rettangoli: (i, j, s, t) = indici della piastrella e posizione nella cella in mm."""
        qb = b + self.tile_h / 2
        j = np.floor(qb / self.pitch_h)
        t = qb - j * self.pitch_h
        qa = a + self.tile_w / 2 - j * self.shift
        i = np.floor(qa / self.pitch_w)
        s = qa - i * self.pitch_w
        return i.astype(np.int64), j.astype(np.int64), s, t

    # ------------------------------------------------------------------ esagoni
    def hex_center(self, q: NDArray[np.float64] | float, r: NDArray[np.float64] | float):  # type: ignore[no-untyped-def]
        """Centro (a, b) dell'esagono di coordinate assiali (q, r)."""
        return 1.5 * self.r_lat * q, SQRT3 * self.r_lat * (r + np.asarray(q) / 2)

    def hex_cell(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """(q, r, dx, dy): esagono più vicino (arrotondamento cubico) e posizione dal centro."""
        qf = (2.0 / 3.0) * a / self.r_lat
        rf = (-a / 3.0 + SQRT3 / 3.0 * b) / self.r_lat
        sf = -qf - rf
        q, r, s = np.rint(qf), np.rint(rf), np.rint(sf)
        dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
        fix_q = (dq > dr) & (dq > ds)
        fix_r = ~fix_q & (dr > ds)
        q = np.where(fix_q, -r - s, q)
        r = np.where(fix_r, -q - s, r)
        ca, cb = self.hex_center(q, r)
        return q.astype(np.int64), r.astype(np.int64), a - ca, b - cb

    # ------------------------------------------------------------ interfaccia unica
    def locate(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.int64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
    ]:
        """(i, j, s, t, dentro_piastrella) per qualunque schema di posa."""
        if self.hex:
            q, r, dx, dy = self.hex_cell(a, b)
            ady, adx = np.abs(dy), np.abs(dx)
            inside = (ady <= SQRT3 / 2 * self.r_tile) & (SQRT3 * adx + ady <= SQRT3 * self.r_tile)
            return q, r, dx, dy, inside
        i, j, s, t = self.cell(a, b)
        return i, j, s, t, (s < self.tile_w) & (t < self.tile_h)

    def joint_fraction(
        self,
        s: NDArray[np.float64],
        t: NDArray[np.float64],
        fw_a: NDArray[np.float64],
        fw_b: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Rettangoli: frazione dell'impronta del campione occupata dalla fuga (filtro a box).

        Una fuga di 2-3 mm lontano dalla camera è più stretta di un pixel: il test
        "dentro/fuori" per campione la fa apparire tratteggiata. Qui si integra la fuga
        sull'impronta reale del campione (in mm, lungo i due assi della posa), così una
        fuga sottile diventa una linea continua più tenue, come in una foto vera.
        """
        ja = _box_overlap(s, fw_a, self.tile_w, self.pitch_w)
        jb = _box_overlap(t, fw_b, self.tile_h, self.pitch_h)
        return 1.0 - (1.0 - ja) * (1.0 - jb)

    def bevel(
        self,
        dx: NDArray[np.float64],
        dy: NDArray[np.float64],
        width_mm: float = 2.0,
        depth: float = 0.22,
    ) -> NDArray[np.float32]:
        """Esagoni: leggero scurimento lungo i lati (smusso delle cementine), in mm reali."""
        apothem = SQRT3 / 2 * self.r_tile
        proj = np.maximum.reduce(
            [np.abs(dy), np.abs(SQRT3 / 2 * dx + dy / 2), np.abs(SQRT3 / 2 * dx - dy / 2)]
        )
        dist = np.maximum(apothem - proj, 0.0)
        return (1.0 - depth * np.exp(-dist / width_mm)).astype(np.float32)

    def texcoords(
        self, s: NDArray[np.float64], t: NDArray[np.float64], rot_deg: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Coordinate normalizzate nella texture, con la rotazione della piastrella."""
        if self.hex:
            th = np.deg2rad(rot_deg.astype(np.float64))
            c, sn_ = np.cos(th), np.sin(th)
            dx = s * c + t * sn_
            dy = -s * sn_ + t * c
            return (dx + self.r_tile) / (2 * self.r_tile), (dy + SQRT3 / 2 * self.r_tile) / (
                SQRT3 * self.r_tile
            )
        return _rotate_uv(s / self.tile_w, t / self.tile_h, rot_deg)

    def tile_polygon(self, i: int, j: int) -> NDArray[np.float64]:
        """Contorno della piastrella (i, j) sul piano della stanza, in mm."""
        if self.hex:
            ca, cb = self.hex_center(float(i), float(j))
            ang = np.deg2rad(np.arange(6) * 60.0)
            a = ca + self.r_tile * np.cos(ang)
            b = cb + self.r_tile * np.sin(ang)
        else:
            a0 = i * self.pitch_w - self.tile_w / 2 + j * self.shift
            b0 = j * self.pitch_h - self.tile_h / 2
            a = np.array([a0, a0 + self.tile_w, a0 + self.tile_w, a0])
            b = np.array([b0, b0, b0 + self.tile_h, b0 + self.tile_h])
        u, v = self.to_plane(a, b)
        return np.stack([u, v], axis=1)

    def index_range(
        self, length_mm: float, width_mm: float, margin: int = 2
    ) -> tuple[int, int, int, int]:
        """Indici (i0, i1, j0, j1) inclusivi che coprono la stanza, più un margine."""
        rect = geometry.room_rect(length_mm, width_mm)
        a, b = self.local(rect[:, 0], rect[:, 1])
        if self.hex:
            qf = (2.0 / 3.0) * a / self.r_lat
            rf = (-a / 3.0 + SQRT3 / 3.0 * b) / self.r_lat
            q0, q1 = math.floor(qf.min()) - margin, math.ceil(qf.max()) + margin
            return q0, q1, math.floor(rf.min()) - margin, math.ceil(rf.max()) + margin
        j0 = math.floor((b.min() + self.tile_h / 2) / self.pitch_h) - margin
        j1 = math.floor((b.max() + self.tile_h / 2) / self.pitch_h) + margin
        shifts = (j0 * self.shift, j1 * self.shift)
        i0 = math.floor((a.min() + self.tile_w / 2 - max(shifts)) / self.pitch_w) - margin
        i1 = math.floor((a.max() + self.tile_w / 2 - min(shifts)) / self.pitch_w) + margin
        return i0, i1, j0, j1


def _box_overlap(
    s: NDArray[np.float64], fw: NDArray[np.float64], tile: float, pitch: float
) -> NDArray[np.float64]:
    """Frazione di [s - fw/2, s + fw/2] coperta dalle fughe [tile + k*pitch, (k+1)*pitch]."""
    joint = pitch - tile
    if joint <= 0:
        return np.zeros_like(s)
    fw = np.maximum(fw, 1e-6)
    lo, hi = s - fw / 2, s + fw / 2
    total = np.zeros_like(s)
    k0 = np.floor(lo / pitch) - 1
    for dk in range(int(np.max(np.ceil(fw / pitch), initial=0)) + 3):
        a = tile + (k0 + dk) * pitch
        total += np.clip(np.minimum(hi, a + joint) - np.maximum(lo, a), 0.0, None)
    return np.clip(total / fw, 0.0, 1.0)


def _splitmix(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return np.asarray(x ^ (x >> np.uint64(31)))


@dataclass(frozen=True)
class VariantMap:
    """Variante e rotazione (in gradi) di ogni piastrella. Ancorata alla stanza, non alla maschera.

    Dentro la stanza: estrazione pesata riga per riga con PCG64(seed); con >= 3 varianti una
    piastrella evita la variante di quella "a sinistra" e di quella "sopra" (negli indici della
    posa): divieto se i pesi sono uguali, probabilità ridotta se diversi.
    Fuori (maschera più grande della stanza): hash deterministico.
    """

    i0: int
    j0: int
    variant: NDArray[np.int64]  # (nj, ni)
    rotation: NDArray[np.int64]  # (nj, ni) gradi
    weights: NDArray[np.float64]
    rotations: tuple[tuple[int, ...], ...]  # rotazioni ammesse per variante
    seed: int

    @property
    def n_variants(self) -> int:
        return len(self.weights)

    @classmethod
    def build(
        cls, spec: MaterialSpec, laying: Laying, length_mm: float, width_mm: float, seed: int
    ) -> VariantMap:
        i0, i1, j0, j1 = laying.index_range(length_mm, width_mm)
        ni, nj = i1 - i0 + 1, j1 - j0 + 1
        nv = len(spec.variants)
        weights = np.array([v.weight for v in spec.variants], dtype=np.float64)
        rots = tuple(tuple(spec.rotations_of(k)) for k in range(nv))
        rng = np.random.Generator(np.random.PCG64(seed))
        var = np.zeros((nj, ni), dtype=np.int64)
        rot = np.zeros((nj, ni), dtype=np.int64)
        # pesi uguali: mai due varianti uguali affiancate; pesi diversi (collezioni miste):
        # solo scoraggiate, altrimenti le varianti frequenti non raggiungerebbero la loro quota
        repeat = 0.0 if np.allclose(weights, weights[0]) else 0.5
        for jj in range(nj):
            for ii in range(ni):
                w = weights.copy()
                if nv >= 3:
                    if ii > 0:
                        w[var[jj, ii - 1]] *= repeat
                    if jj > 0:
                        w[var[jj - 1, ii]] *= repeat
                k = int(rng.choice(nv, p=w / w.sum()))
                var[jj, ii] = k
                rot[jj, ii] = rots[k][int(rng.integers(0, len(rots[k])))]
        return cls(i0, j0, var, rot, weights, rots, seed)

    def lookup(
        self, i: NDArray[np.int64], j: NDArray[np.int64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        nj, ni = self.variant.shape
        ii, jj = i - self.i0, j - self.j0
        inside = (ii >= 0) & (ii < ni) & (jj >= 0) & (jj < nj)
        var = np.empty(i.shape, dtype=np.int64)
        rot = np.empty(i.shape, dtype=np.int64)
        var[inside] = self.variant[jj[inside], ii[inside]]
        rot[inside] = self.rotation[jj[inside], ii[inside]]
        out = ~inside
        if np.any(out):
            key = (
                (i[out].astype(np.uint64) << np.uint64(32))
                ^ (j[out].astype(np.uint64) & np.uint64(0xFFFFFFFF))
                ^ np.uint64(self.seed & 0xFFFFFFFFFFFFFFFF)
            )
            h = _splitmix(key)
            u = (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)
            cum = np.cumsum(self.weights) / self.weights.sum()
            k = np.minimum(np.searchsorted(cum, u, side="right"), self.n_variants - 1)
            var[out] = k
            pick = (h >> np.uint64(3)) & np.uint64(0xFF)
            rot[out] = [
                self.rotations[kk][int(pp) % len(self.rotations[kk])]
                for kk, pp in zip(k, pick, strict=True)
            ]
        return var, rot


def _clip_area(poly: NDArray[np.float64], length: float, width: float) -> float:
    """Area del poligono ritagliato sul rettangolo [0, L] x [0, W] (Sutherland-Hodgman)."""
    pts = [tuple(p) for p in poly]
    for axis, bound, keep_le in (
        (0, 0.0, False),
        (0, length, True),
        (1, 0.0, False),
        (1, width, True),
    ):
        if not pts:
            return 0.0
        out: list[tuple[float, float]] = []
        for k in range(len(pts)):
            cur, prev = pts[k], pts[k - 1]
            cin = cur[axis] <= bound if keep_le else cur[axis] >= bound
            pin = prev[axis] <= bound if keep_le else prev[axis] >= bound
            if cin != pin:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cin:
                out.append(cur)
        pts = out
    if len(pts) < 3:
        return 0.0
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def count_tiles(spec: MaterialSpec, length_mm: float, width_mm: float) -> tuple[int, int]:
    """(piastrelle intere, piastrelle tagliate) necessarie per la stanza."""
    laying = Laying.of(spec, length_mm, width_mm)
    i0, i1, j0, j1 = laying.index_range(length_mm, width_mm)
    full = laying.tile_area
    whole = cut = 0
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            area = _clip_area(laying.tile_polygon(i, j), length_mm, width_mm)
            if area >= full * (1 - 1e-9):
                whole += 1
            elif area > full * 1e-6:
                cut += 1
    return whole, cut


def derive_seed(image_sha256: str, material_id: str) -> int:
    import hashlib

    return int(hashlib.sha256(f"{image_sha256}|{material_id}".encode()).hexdigest()[:15], 16)


# ------------------------------------------------------------------------ texture


@dataclass(frozen=True)
class Atlas:
    """Piramidi mip (luce lineare) di tutte le varianti in un unico array piatto."""

    data: NDArray[np.float32]  # (N, 3)
    offsets: NDArray[np.int64]  # (n_variants, n_levels)
    widths: NDArray[np.int64]  # (n_levels,)
    heights: NDArray[np.int64]
    texels_per_mm: float

    @property
    def n_levels(self) -> int:
        return int(self.widths.shape[0])


def _downsample(img: NDArray[np.float32]) -> NDArray[np.float32]:
    h, w = img.shape[:2]
    if h % 2:
        img = np.concatenate([img, img[-1:]], axis=0)
    if w % 2:
        img = np.concatenate([img, img[:, -1:]], axis=1)
    return np.asarray(
        (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2]) * 0.25,
        dtype=np.float32,
    )


def build_atlas(textures_linear: list[NDArray[np.float32]], spec: MaterialSpec) -> Atlas:
    shapes = {t.shape for t in textures_linear}
    if len(shapes) != 1:
        raise CatalogError(f"le varianti di {spec.id} devono avere la stessa risoluzione")
    chains = []
    for tex in textures_linear:
        chain = [tex]
        while min(chain[-1].shape[:2]) > 1:
            chain.append(_downsample(chain[-1]))
        chains.append(chain)
    n_levels = len(chains[0])
    offsets = np.zeros((len(chains), n_levels), dtype=np.int64)
    parts, pos = [], 0
    for v, chain in enumerate(chains):
        for lvl, img in enumerate(chain):
            offsets[v, lvl] = pos
            parts.append(img.reshape(-1, 3))
            pos += img.shape[0] * img.shape[1]
    h0, w0 = textures_linear[0].shape[:2]
    return Atlas(
        data=np.concatenate(parts).astype(np.float32),
        offsets=offsets,
        widths=np.array([c.shape[1] for c in chains[0]], dtype=np.int64),
        heights=np.array([c.shape[0] for c in chains[0]], dtype=np.int64),
        texels_per_mm=max(w0 / spec.tile_w_mm, h0 / spec.tile_h_mm),
    )


def _bilinear(
    atlas: Atlas,
    var: NDArray[np.int64],
    lvl: NDArray[np.int64],
    sn: NDArray[np.float64],
    tn: NDArray[np.float64],
) -> NDArray[np.float32]:
    w = atlas.widths[lvl]
    h = atlas.heights[lvl]
    x = sn * w - 0.5
    y = tn * h - 0.5
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0).astype(np.float32)[:, None]
    fy = (y - y0).astype(np.float32)[:, None]
    xi0 = np.clip(x0.astype(np.int64), 0, w - 1)
    xi1 = np.clip(x0.astype(np.int64) + 1, 0, w - 1)
    yi0 = np.clip(y0.astype(np.int64), 0, h - 1)
    yi1 = np.clip(y0.astype(np.int64) + 1, 0, h - 1)
    base = atlas.offsets[var, lvl]
    d = atlas.data
    c00 = d[base + yi0 * w + xi0]
    c01 = d[base + yi0 * w + xi1]
    c10 = d[base + yi1 * w + xi0]
    c11 = d[base + yi1 * w + xi1]
    top = c00 + (c01 - c00) * fx
    bot = c10 + (c11 - c10) * fx
    return np.asarray(top + (bot - top) * fy, dtype=np.float32)


def sample_texture(
    atlas: Atlas,
    var: NDArray[np.int64],
    sn: NDArray[np.float64],
    tn: NDArray[np.float64],
    lod: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Campionamento trilineare (bilineare su due livelli mip)."""
    top = atlas.n_levels - 1
    lod = np.clip(lod, 0.0, float(top))
    l0 = np.floor(lod).astype(np.int64)
    l1 = np.minimum(l0 + 1, top)
    fr = (lod - l0).astype(np.float32)[:, None]
    c0 = _bilinear(atlas, var, l0, sn, tn)
    if not np.any(fr > 0):
        return c0
    c1 = _bilinear(atlas, var, l1, sn, tn)
    return np.asarray(c0 + (c1 - c0) * fr, dtype=np.float32)


def _rotate_uv(
    sn: NDArray[np.float64], tn: NDArray[np.float64], rot_deg: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s2, t2 = sn.copy(), tn.copy()
    m = rot_deg == 90
    s2[m], t2[m] = tn[m], 1 - sn[m]
    m = rot_deg == 180
    s2[m], t2[m] = 1 - sn[m], 1 - tn[m]
    m = rot_deg == 270
    s2[m], t2[m] = 1 - tn[m], sn[m]
    return s2, t2


def _hex_to_rgb(color: str) -> NDArray[np.uint8]:
    c = color.lstrip("#")
    return np.array([[[int(c[k : k + 2], 16) for k in (0, 2, 4)]]], dtype=np.uint8)


def _stratified_offsets(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    g = (np.arange(n) + 0.5) / n
    ox, oy = np.meshgrid(g, g)
    return ox.ravel(), oy.ravel()


def tiled_layer(
    spec: MaterialSpec,
    variants_srgb: list[NDArray[np.uint8]],
    source: SourceImage,
    geom: PlaneGeometry,
    mask: NDArray[np.uint8],
    seed: int,
    supersampling: int = 4,
) -> MaterialLayer:
    """Rende il materiale posato in prospettiva, solo sui pixel con mask > 0."""
    bbox = _bbox(mask)
    trc = trc_for(source.meta.color_space)
    variants, notes = variants_in_photo_space(variants_srgb, source)
    atlas = build_atlas([to_linear(v, trc).astype(np.float32) for v in variants], spec)
    joint = np.zeros(3, dtype=np.float32)
    if spec.joint_mm > 0 and spec.joint_color is not None:
        jc, jn = texture_in_photo_space(_hex_to_rgb(spec.joint_color), source)
        joint = to_linear(jc, trc)[0, 0].astype(np.float32)
        notes = sorted(set(notes) | set(jn))

    laying = Laying.of(spec, geom.length_mm, geom.width_mm)
    vmap = VariantMap.build(spec, laying, geom.length_mm, geom.width_mm, seed)
    hi = np.asarray(geom.H_inv, dtype=np.float64)

    x0, y0, x1, y1 = bbox
    out = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    tile_ij = np.full((y1 - y0, x1 - x0, 2), TILE_NONE, dtype=np.int32)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    n = supersampling
    ox, oy = _stratified_offsets(n)
    per_chunk = max(1, SAMPLES_PER_CHUNK // (n * n))

    def shade(
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        lod: NDArray[np.float64],
        fw_a: NDArray[np.float64],
        fw_b: NDArray[np.float64],
    ) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        a, b = laying.local(u, v)
        i, j, s, t, in_tile = laying.locate(a, b)
        var, rot = vmap.lookup(i, j)
        if not laying.hex:
            # fuga filtrata sull'impronta del campione: niente fughe tratteggiate in lontananza
            sc = np.clip(s, 0.0, np.nextafter(laying.tile_w, 0.0))
            tc = np.clip(t, 0.0, np.nextafter(laying.tile_h, 0.0))
            sn, tn = laying.texcoords(sc, tc, rot)
            tex = sample_texture(atlas, var, sn, tn, lod)
            jf = laying.joint_fraction(s, t, fw_a, fw_b).astype(np.float32)[:, None]
            return tex * (1.0 - jf) + joint[None, :] * jf, i, j, in_tile
        sn, tn = laying.texcoords(s, t, rot)
        col = np.broadcast_to(joint, (u.size, 3)).copy()
        if np.any(in_tile):
            col[in_tile] = sample_texture(
                atlas, var[in_tile], sn[in_tile], tn[in_tile], lod[in_tile]
            )
            if laying.hex:  # smusso del bordo: ogni esagono resta leggibile anche tra due chiari
                col[in_tile] *= laying.bevel(s[in_tile], t[in_tile])[:, None]
        return col, i, j, in_tile

    for start in range(0, ys.size, per_chunk):
        py = ys[start : start + per_chunk]
        px = xs[start : start + per_chunk]
        gx = px.astype(np.float64) + x0
        gy = py.astype(np.float64) + y0
        # impronta del pixel sul piano -> livello mip per sotto-campione
        dudx, dudy, dvdx, dvdy = geometry.jacobian_image_to_plane(hi, gx + 0.5, gy + 0.5)
        mm_per_px = np.maximum(np.hypot(dudx, dvdx), np.hypot(dudy, dvdy))
        footprint = mm_per_px / n * atlas.texels_per_mm
        lod = np.log2(np.maximum(footprint, 1.0))
        # impronta del sotto-campione lungo i due assi della posa (mm)
        dadx = dudx * laying.cos + dvdx * laying.sin
        dady = dudy * laying.cos + dvdy * laying.sin
        dbdx = -dudx * laying.sin + dvdx * laying.cos
        dbdy = -dudy * laying.sin + dvdy * laying.cos
        fw_a = np.repeat((np.abs(dadx) + np.abs(dady)) / n, n * n)
        fw_b = np.repeat((np.abs(dbdx) + np.abs(dbdy)) / n, n * n)

        sx = (gx[:, None] + ox[None, :]).ravel()
        sy = (gy[:, None] + oy[None, :]).ravel()
        u, v, _ = geometry.plane_coords(hi, sx, sy)
        col, _, _, _ = shade(u, v, np.repeat(lod, n * n), fw_a, fw_b)
        out[py, px] = col.reshape(-1, n * n, 3).mean(axis=1)

        uc, vc, _ = geometry.plane_coords(hi, gx + 0.5, gy + 0.5)
        _, ic, jc, inc = shade_ids(laying, uc, vc)
        tile_ij[py[inc], px[inc], 0] = ic[inc]
        tile_ij[py[inc], px[inc], 1] = jc[inc]

    return MaterialLayer(
        rgb_linear=out,
        bbox=bbox,
        material_id=spec.id,
        render_mode="tiled",
        notes=tuple(notes),
        tile_ij=tile_ij,
    )


def shade_ids(
    laying: Laying, u: NDArray[np.float64], v: NDArray[np.float64]
) -> tuple[None, NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    a, b = laying.local(u, v)
    i, j, _, _, inside = laying.locate(a, b)
    return None, i, j, inside


def _bbox(mask: NDArray[np.uint8]) -> BBox:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("maschera vuota")
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
