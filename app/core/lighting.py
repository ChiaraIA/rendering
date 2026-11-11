"""Luci e ombre della nuova superficie (pavimento o parete), ricavate dalla foto senza il
disegno di quella vecchia.

Tre livelli, tutti misurati sul vecchio pavimento dentro la maschera:
1. luce d'ambiente a bassa frequenza (zone più illuminate e più buie della stanza);
2. ombre e riflessi a media frequenza: l'ombra sotto il tavolo, le sedie, la credenza, i
   riflessi di tende e lampade su un pavimento lucido. Prima si cancella il disegno del
   vecchio pavimento (fughe, puntinature) con una mediana grande quanto una fuga, poi si
   misura quanto ogni zona è più scura (ombra) o più chiara (riflesso) della luce d'ambiente;
3. ombra di contatto metrica vicino a ciò che poggia a terra e ai muri.
Più il colore della luce della stanza. L'agente "luci" regola quanto pesa ciascun livello
(`LightParams`); il codice fa i conti.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.colorspace import to_linear, trc_for
from app.core.models import BBox, Finish, LightMap, LightParams, PlaneKind, SourceImage

GAIN_MIN, GAIN_MAX = 0.35, 1.6
WORK_SIDE = 256  # lato di lavoro: la luce è a bassa frequenza, non serve di più


def light_map(
    source: SourceImage, mask: NDArray[np.uint8], bbox: BBox, strength: float = 1.0
) -> LightMap:
    x0, y0, x1, y1 = bbox
    rgb = source.pixels[y0:y1, x0:x1, :3]
    lin = to_linear(rgb, trc_for(source.meta.color_space))
    lum = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]).astype(np.float32)
    m = (mask[y0:y1, x0:x1] >= 128).astype(np.float32)
    h, w = lum.shape
    if m.sum() < 16:
        return LightMap(gain=np.ones((h, w), np.float32), bbox=bbox)

    scale = min(1.0, WORK_SIDE / max(h, w))
    sw, sh = max(4, round(w * scale)), max(4, round(h * scale))
    small_l = cv2.resize(lum * m, (sw, sh), interpolation=cv2.INTER_AREA)
    small_m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    filled = np.where(small_m > 0.05, small_l / np.maximum(small_m, 1e-6), 0).astype(np.float32)
    # mediana: elimina fughe, venature e piccoli oggetti
    med = np.where(small_m > 0.05, filled, np.median(filled[small_m > 0.05])).astype(np.float32)
    for _ in range(max(1, min(sw, sh) // 40)):  # mediane ripetute = mediana larga
        med = cv2.medianBlur(med, 5)
    sigma = max(sw, sh) / 10
    num = cv2.GaussianBlur(med * small_m, (0, 0), sigma)
    den = cv2.GaussianBlur(small_m, (0, 0), sigma)
    smooth = np.where(den > 1e-3, num / np.maximum(den, 1e-6), 0)
    ref = float(np.median(smooth[small_m > 0.5])) if np.any(small_m > 0.5) else 1.0
    gain_small = np.clip(smooth / max(ref, 1e-6), GAIN_MIN, GAIN_MAX).astype(np.float32)
    gain = cv2.resize(gain_small, (w, h), interpolation=cv2.INTER_CUBIC)
    gain = np.clip(1.0 + (gain - 1.0) * strength, GAIN_MIN, GAIN_MAX).astype(np.float32)
    return LightMap(gain=gain, bbox=bbox)


def contact_shadow(
    mask: NDArray[np.uint8],
    bbox: BBox,
    px_per_mm: NDArray[np.float32] | float,
    radius_mm: float = 260.0,
    depth: float = 0.42,
) -> NDArray[np.float32]:
    """Occlusione ambientale vicino a ciò che sta sul pavimento o lo delimita (mobili, muri).

    La distanza dal bordo della maschera viene misurata in millimetri reali: l'ombra ha la
    stessa larghezza fisica vicino e lontano dalla camera. I bordi dell'immagine non contano.
    """
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    # fuori dall'immagine = pavimento (il bordo della foto non è un oggetto)
    padded = np.pad((mask >= 128).astype(np.uint8), 1, constant_values=1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1][y0:y1, x0:x1]
    dist_mm = dist / np.maximum(px_per_mm, 1e-6)
    ao = 1.0 - depth * np.exp(-((dist_mm / radius_mm) ** 1.3))
    return ao.astype(np.float32)


# un po' meno della dominante misurata: il colore del catalogo resta riconoscibile
TINT_STRENGTH = 0.85
TINT_LIMIT = (0.6, 1.4)  # oltre, la stima è inaffidabile (foto con filtri, luci colorate)


def scene_tint(
    source: SourceImage, exclude: NDArray[np.uint8] | None = None
) -> tuple[float, float, float]:
    """Colore della luce della stanza, stimato dalle superfici bianche della foto.

    Tra i pixel più luminosi (non saturati, fuori dal pavimento da sostituire) si prendono
    quelli meno colorati: tende, soffitto, elettrodomestici, prese, cornici bianche. Il loro
    colore medio è il colore della luce. Ritorna (R, G, B) lineari con luminanza 1; (1, 1, 1)
    se la foto non offre abbastanza superfici chiare.
    """
    rgb = source.pixels[..., :3]
    step = max(1, round(max(rgb.shape[:2]) / 800))  # basta una griglia rada
    sub = rgb[::step, ::step]
    ok = np.ones(sub.shape[:2], bool) if exclude is None else exclude[::step, ::step] == 0
    ok &= sub.max(axis=-1) < 250  # niente zone bruciate (lampadine, riflessi)
    lin = to_linear(sub, trc_for(source.meta.color_space)).reshape(-1, 3)
    ok = ok.reshape(-1)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    if np.count_nonzero(ok) < 200:
        return (1.0, 1.0, 1.0)
    bright = ok & (lum >= np.quantile(lum[ok], 0.85))
    chroma = (lin.max(axis=1) - lin.min(axis=1)) / np.maximum(lin.max(axis=1), 1e-6)
    if np.count_nonzero(bright) < 50:
        return (1.0, 1.0, 1.0)
    white = bright & (chroma <= np.quantile(chroma[bright], 0.2))
    c = lin[white].mean(axis=0)
    c = c / max(float(c @ np.array([0.2126, 0.7152, 0.0722])), 1e-6)
    t = 1.0 + (c - 1.0) * TINT_STRENGTH
    if not np.all((t > TINT_LIMIT[0]) & (t < TINT_LIMIT[1])):
        return (1.0, 1.0, 1.0)
    t = t / float(t @ np.array([0.2126, 0.7152, 0.0722]))
    return (float(t[0]), float(t[1]), float(t[2]))


# riflessi di partenza per finitura (frazione dei riflessi del vecchio pavimento)
REFLECTIONS_BY_FINISH: dict[str, float] = {"opaca": 0.1, "satinata": 0.35, "lucida": 0.7}
DETAIL_WORK_SIDE = 1024  # lato di lavoro per ombre e riflessi a media frequenza
PATTERN_MM = 14.0  # la mediana cancella dettagli più stretti di così (fughe, puntinature)
SHADOW_MM = 18.0  # sfocatura delle ombre: morbide come nella foto, senza grana
HIGHLIGHT_START = 1.1  # sopra il 110% della luce d'ambiente è riflesso, non luce


CONTACT_BY_KIND: dict[str, float] = {"pavimento": 0.6, "parete": 0.35}


def default_params(finish: Finish = "opaca", kind: PlaneKind = "pavimento") -> LightParams:
    """Valori di partenza: i riflessi dipendono dalla finitura; l'ombra di contatto è più
    leggera sulle pareti (un muro non "regge" gli oggetti come un pavimento)."""
    return LightParams(
        riflessi=REFLECTIONS_BY_FINISH.get(finish, 0.1), contatto=CONTACT_BY_KIND.get(kind, 0.6)
    )


def _norm_blur(
    x: NDArray[np.float32], w: NDArray[np.float32], sigma: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Media gaussiana pesata (convoluzione normalizzata). Ritorna (media, peso)."""
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    if x.ndim == 3:
        num = cv2.GaussianBlur(x * w[..., None], (0, 0), sigma)
        return (num / np.maximum(den, 1e-6)[..., None]).astype(np.float32), den
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    return (num / np.maximum(den, 1e-6)).astype(np.float32), den


def _odd(n: float, lo: int, hi: int) -> int:
    k = int(np.clip(round(n), lo, hi))
    return k if k % 2 else k + 1


def photo_shading(
    source: SourceImage,
    mask: NDArray[np.uint8],
    bbox: BBox,
    px_per_mm: NDArray[np.float32] | float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Ombre e riflessi del vecchio pavimento, senza il suo disegno.

    Ritorna (rapporto, eccesso) alla risoluzione di `bbox`:
    - rapporto: luminanza locale / luce d'ambiente (< 1 in ombra, > 1 sui riflessi);
    - eccesso: luce riflessa in più (lineare, a colori), da sommare ai materiali lucidi.
    """
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    floor = mask[y0:y1, x0:x1] >= 250
    if np.count_nonzero(floor) < 64:
        return np.ones((h, w), np.float32), np.zeros((h, w, 3), np.float32)
    s = min(1.0, DETAIL_WORK_SIDE / max(h, w))
    sw, sh = max(8, round(w * s)), max(8, round(h * s))
    enc = cv2.resize(
        np.ascontiguousarray(source.pixels[y0:y1, x0:x1, :3]),
        (sw, sh),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    m = cv2.resize(floor.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.99
    ppm = np.asarray(px_per_mm, dtype=np.float32)
    if ppm.ndim == 2:
        ppm_floor = ppm[floor]
        near, typical = float(np.quantile(ppm_floor, 0.75)), float(np.median(ppm_floor))
    else:
        near = typical = float(ppm)
    # 1. fuori dal pavimento (gambe, mobili) si mette il pavimento vicino: niente "macchie" a
    #    forma di gamba, l'ombra vera resta quella che si vede sul pavimento
    known = m.copy()
    wk = m.astype(np.float32)
    for sigma in (2.0, 6.0, 18.0, 54.0, 160.0):
        fill, den = _norm_blur(enc, wk, sigma)
        new = ~known & (den > 0.02)
        enc[new] = fill[new]
        known |= new
        wk = known.astype(np.float32)
        if known.all():
            break
    if not known.all():
        enc[~known] = np.median(enc[m], axis=0)
    # 2. via il disegno del vecchio pavimento (fughe e venature sottili, chiare o scure)
    k = _odd(PATTERN_MM * near * s, 3, 31)
    med = cv2.medianBlur(np.clip(np.rint(enc), 0, 255).astype(np.uint8), k)
    lin = to_linear(med, trc_for(source.meta.color_space)).astype(np.float32)
    # 3. ombre a media frequenza e luce d'ambiente (stessa scala di `light_map`)
    wf = m.astype(np.float32)
    detail, _ = _norm_blur(lin, wf, max(1.0, SHADOW_MM * typical * s))
    ambient, _ = _norm_blur(lin, wf, max(sw, sh) / 10)
    lum_w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    ratio = (detail @ lum_w) / np.maximum(ambient @ lum_w, 1e-5)
    excess = detail * np.clip(1.0 - HIGHLIGHT_START / np.maximum(ratio, 1e-5), 0.0, None)[..., None]
    ratio_full = cv2.resize(ratio.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    excess_full = cv2.resize(excess.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return ratio_full, np.maximum(excess_full, 0.0)


GRAIN_MAX = 0.08  # oltre, non è rumore ma disegno del vecchio pavimento


def photo_grain(source: SourceImage, mask: NDArray[np.uint8]) -> float:
    """Rumore relativo della fotocamera, misurato sul vecchio pavimento (deviazione robusta)."""
    floor = mask >= 250
    if np.count_nonzero(floor) < 256:
        return 0.0
    g = cv2.cvtColor(np.ascontiguousarray(source.pixels[..., :3]), cv2.COLOR_RGB2GRAY)
    g = g.astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    inner = cv2.erode(floor.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    vals = hp[inner] if np.count_nonzero(inner) >= 256 else hp[floor]
    sigma = 1.4826 * float(np.median(np.abs(vals - np.median(vals))))
    level = float(np.median(g[floor]))
    return float(np.clip(sigma / max(level, 1.0), 0.0, GRAIN_MAX))


def _grain_field(shape: tuple[int, int], bbox: BBox, key: str) -> NDArray[np.float32]:
    """Rumore gaussiano deterministico (stessa foto, stesso rumore)."""
    seed = int.from_bytes(f"{key}:{bbox}".encode()[:32].ljust(32, b"0"), "little") % 2**63
    n = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    # il rumore di una foto da telefono non è a pixel singolo: leggera correlazione
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    return n / max(float(n.std()), 1e-6)


def scene_light(
    source: SourceImage,
    mask: NDArray[np.uint8],
    bbox: BBox,
    px_per_mm: NDArray[np.float32] | float,
    params: LightParams | None = None,
) -> LightMap:
    """Luce completa del nuovo pavimento con le regolazioni dell'agente "luci"."""
    p = params or LightParams()
    base = light_map(source, mask, bbox)
    gain = base.gain
    add = None
    if p.ombre > 0 or p.riflessi > 0:
        ratio, excess = photo_shading(source, mask, bbox, px_per_mm)
        if p.ombre > 0:
            gain = gain * np.clip(ratio, 0.2, 1.0) ** p.ombre
        if p.riflessi > 0:
            add = (excess * p.riflessi).astype(np.float32)
    if p.contatto > 0:
        gain = gain * contact_shadow(mask, bbox, px_per_mm, depth=min(0.8, 0.42 * p.contatto))
    if p.grana > 0:
        rel = photo_grain(source, mask)
        if rel > 0:
            key = source.meta.normalized_sha256
            gain = gain * (1.0 + p.grana * rel * _grain_field(gain.shape, bbox, key))
    t = 1.0 + (np.asarray(scene_tint(source, mask)) - 1.0) * p.luce_calda
    t = t / float(t @ np.array([0.2126, 0.7152, 0.0722]))
    return LightMap(
        gain=np.clip(gain, GAIN_MIN, GAIN_MAX).astype(np.float32),
        bbox=bbox,
        tint=(float(t[0]), float(t[1]), float(t[2])),
        add=add,
    )


def combined_light(
    source: SourceImage,
    mask: NDArray[np.uint8],
    bbox: BBox,
    px_per_mm: NDArray[np.float32] | float,
) -> LightMap:
    """Luce con le regolazioni predefinite (senza agente)."""
    return scene_light(source, mask, bbox, px_per_mm, LightParams())
