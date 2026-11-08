"""Segmentazione semantica delle superfici (opzionale): UperNet-ConvNeXt su ADE20K, licenza MIT.

Riconosce, tra le 150 classi di ADE20K, quelle che servono a "posa": pavimento oggi,
parete, piano di lavoro e mobili della cucina domani. Il modello lavora a bassa
risoluzione (lato lungo 640 px); i bordi vengono poi rifiniti sulla foto vera con
GrabCut, usando la probabilità del modello come guida: così le gambe di sedie e tavoli
e le fessure tra un oggetto e l'altro seguono i contorni reali.

PyTorch e transformers sono importati solo quando serve: senza di loro il resto del
programma funziona come prima (maschera disegnata dagli agenti).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = os.environ.get("POSA_SEG_MODEL", "openmmlab/upernet-convnext-tiny")
WORK_SIDE = 896  # lato lungo per il modello (multiplo di 32): più dettaglio su gambe e piedi
REFINE_SIDE = 1600  # lato lungo per la rifinitura dei bordi

# classi ADE20K (nomi del config del modello, senza spazi finali)
SURFACES = {
    "pavimento": ("floor",),
    "parete": ("wall",),
    "piano_di_lavoro": ("countertop", "counter"),
}


# oggetti che stanno SOPRA una superficie: dove il modello ha un dubbio fondato che ci sia
# uno di questi, il pixel non è superficie (gambe sottili, piedi di mobili e sanitari)
OBJECTS = (
    "chair",
    "seat",
    "sofa",
    "armchair",
    "table",
    "desk",
    "bench",
    "stool",
    "coffee table",
    "swivel chair",
    "person",
    "plant",
    "pot",
    "box",
    "bag",
    "cabinet",
    "chest of drawers",
    "bed",
    "toilet",
    "bathtub",
    "sink",
    "kitchen island",
    "ottoman",
    "cushion",
    "case",
    "basket",
    "vase",
    "radiator",
    "fan",
    "pole",
    "column",
    "rug",
    "television receiver",
    "lamp",
    "shelf",
    "wardrobe",
    "refrigerator",
    "stove",
    "oven",
    "washer",
    "dishwasher",
)
OBJECT_THRESHOLD = 0.25


def available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class Segmentation:
    probs: NDArray[np.float32]  # (C, h, w) probabilità alla risoluzione di lavoro
    labels: dict[str, int]  # nome classe -> indice

    def prob(self, names: tuple[str, ...], size: tuple[int, int]) -> NDArray[np.float32]:
        """Probabilità (somma delle classi) riportata a `size` = (larghezza, altezza)."""
        idx = [self.labels[n] for n in names if n in self.labels]
        p = self.probs[idx].sum(axis=0)
        return np.asarray(cv2.resize(p, size, interpolation=cv2.INTER_LINEAR), dtype=np.float32)


@lru_cache(maxsize=1)
def _load(model_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = UperNetForSemanticSegmentation.from_pretrained(model_name).eval()
    return torch, proc, model


def segment(image_rgb: NDArray[np.uint8], model_name: str = DEFAULT_MODEL) -> Segmentation:
    torch, proc, model = _load(model_name)
    h, w = image_rgb.shape[:2]
    scale = WORK_SIDE / max(h, w)
    nw, nh = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
    small = cv2.resize(
        np.ascontiguousarray(image_rgb[..., :3]), (nw, nh), interpolation=cv2.INTER_AREA
    )
    mean = np.asarray(proc.image_mean, dtype=np.float32)
    std = np.asarray(proc.image_std, dtype=np.float32)
    x = (small.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(x.transpose(2, 0, 1)[None].copy())
    with torch.no_grad():
        logits = model(pixel_values=tensor).logits[0]
    probs = torch.softmax(logits, dim=0).numpy().astype(np.float32)
    labels = {str(v).strip(): int(k) for k, v in model.config.id2label.items()}
    return Segmentation(probs=probs, labels=labels)


def surface_mask(
    image_rgb: NDArray[np.uint8],
    surface: str = "pavimento",
    seg: Segmentation | None = None,
    refine: bool = True,
) -> NDArray[np.uint8]:
    """Maschera uint8 (0..255) della superficie, alla risoluzione della foto."""
    h, w = image_rgb.shape[:2]
    seg = seg or segment(image_rgb)
    s = min(1.0, REFINE_SIDE / max(h, w))
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    work = cv2.resize(
        np.ascontiguousarray(image_rgb[..., :3]), (rw, rh), interpolation=cv2.INTER_AREA
    )
    p = seg.prob(SURFACES[surface], (rw, rh))
    p_obj = seg.prob(OBJECTS, (rw, rh))
    # gli oggetti "vincono": la probabilità di superficie si annulla dove c'è un oggetto
    p = np.where(p_obj >= OBJECT_THRESHOLD, np.minimum(p, 0.3), p).astype(np.float32)
    hard = p >= 0.5
    if refine and hard.any() and not hard.all():
        hard = _grabcut(work, p, near_objects=p_obj >= 0.1)
    hard = _clean(hard)
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 0.7)
    full = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.clip(np.rint(full * 255), 0, 255).astype(np.uint8)
    out[full < 0.02] = 0
    return out


def _grabcut(
    rgb: NDArray[np.uint8], p: NDArray[np.float32], near_objects: NDArray[np.bool_] | None = None
) -> NDArray[np.bool_]:
    """GrabCut guidato dalla probabilità: il modello decide dove, la foto decide il bordo.
    Vicino agli oggetti niente è "sicuro": lì GrabCut può ritagliare gambe e fessure."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sure_fg = cv2.erode((p > 0.9).astype(np.uint8), k).astype(bool)
    if near_objects is not None and near_objects.any():
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        sure_fg &= ~cv2.dilate(near_objects.astype(np.uint8), k2).astype(bool)
    sure_bg = cv2.erode((p < 0.1).astype(np.uint8), k).astype(bool)
    gc = np.where(p >= 0.5, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc[sure_fg] = cv2.GC_FGD
    gc[sure_bg] = cv2.GC_BGD
    if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_BGD).any():
        return p >= 0.5
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(rgb[..., ::-1]), gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    return np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))


def _clean(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Toglie isole minuscole e chiude buchi minuscoli (rumore), lasciando le fessure vere."""
    m = mask.astype(np.uint8)
    area = m.size
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= area * 0.002
    m = keep[lab].astype(np.uint8)
    inv = 1 - m
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    small_holes = np.zeros(n, bool)
    small_holes[1:] = stats[1:, cv2.CC_STAT_AREA] < area * 0.00005
    m[small_holes[lab]] = 1
    return m.astype(bool)
