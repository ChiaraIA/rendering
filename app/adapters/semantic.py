"""Segmentazione semantica delle superfici (opzionale): UperNet-ConvNeXt su ADE20K, licenza MIT.

Riconosce, tra le 150 classi di ADE20K, quelle che servono a "posa": pavimento, parete
(per i rivestimenti) e piano di lavoro. Il modello lavora a risoluzione ridotta (lato
lungo 896 px); i bordi vengono poi portati sui contorni reali della foto con un filtro
guidato e ricalcolati alla risoluzione piena: così le gambe di sedie e tavoli e le fessure
tra un oggetto e l'altro seguono i contorni veri.

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

DEFAULT_MODEL = os.environ.get("POSA_SEG_MODEL", "openmmlab/upernet-convnext-base")
# versione esatta dei pesi (commit di Hugging Face): un repository alterato non cambia il
# modello in uso. Si aggiorna di proposito, dopo averlo verificato.
MODEL_REVISION = os.environ.get("POSA_SEG_REVISIONE", "13a9c410ff779f25236a8d24802d2742e4694e9c")
WORK_SIDE = 896  # lato lungo per il modello (multiplo di 32): più dettaglio su gambe e piedi
REFINE_SIDE = 1600  # lato lungo per la rifinitura dei bordi

# classi ADE20K (nomi del config del modello, senza spazi finali)
SURFACES = {
    "pavimento": ("floor",),
    "parete": ("wall",),
    "piano_di_lavoro": ("countertop", "counter"),
}


# Classi che NON sono oggetti appoggiati o appesi a una superficie: tutto il resto (mobili,
# sedie, vasi, elettrodomestici, quadri... 130 classi) "vince" sulla superficie quando il
# modello ha un dubbio fondato. Misurato: con soglia 0,3 il rivestimento finito sopra gli
# oggetti cala di un quinto, al prezzo di poca superficie in più da far aggiungere agli agenti.
STRUCTURE = (
    "wall",
    "floor",
    "ceiling",
    "windowpane",
    "door",
    "rug",
    "stairs",
    "stairway",
    "step",
    "column",
    "countertop",
    "counter",
    "road",
    "sidewalk",
    "path",
    "earth",
    "grass",
    "field",
    "sand",
    "land",
    "dirt track",
)
OBJECT_THRESHOLD = 0.3
# raggio del filtro che porta il bordo del modello sui contorni veri: ~19 px su 1600, cioè
# recupera errori del modello fino a ~8 px (il modello lavora a 896 px)
SNAP_FRACTION = 0.012


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

    def object_prob(self, size: tuple[int, int]) -> NDArray[np.float32]:
        """Probabilità che il pixel sia un oggetto (qualunque classe non strutturale)."""
        structure = {self.labels[n] for n in STRUCTURE if n in self.labels}
        idx = [i for i in range(self.probs.shape[0]) if i not in structure]
        p = self.probs[idx].sum(axis=0)
        return np.asarray(cv2.resize(p, size, interpolation=cv2.INTER_LINEAR), dtype=np.float32)


@lru_cache(maxsize=1)
def _load(model_name: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    proc = AutoImageProcessor.from_pretrained(model_name, revision=MODEL_REVISION)
    # solo safetensors: i pesi in formato pickle (.bin) possono eseguire codice al caricamento
    model = UperNetForSemanticSegmentation.from_pretrained(
        model_name, revision=MODEL_REVISION, use_safetensors=True
    ).eval()
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
    """Maschera uint8 (0..255) della superficie, alla risoluzione della foto.

    1. il modello dice DOVE è la superficie (probabilità oltre 0,5), e dove ha un dubbio
       fondato che ci sia un oggetto (sedia, vaso, mobile...) l'oggetto vince;
    2. il bordo viene portato sui contorni reali della foto (filtro guidato, solo in una
       fascia stretta attorno al bordo del modello);
    3. la maschera finale è calcolata alla risoluzione piena, con bordi di un pixel.
    Misurato su 109 stanze vere con molti oggetti (ADE20K validation): rispetto alla
    rifinitura precedente (GrabCut) meno rivestimento sopra gli oggetti e meno superficie
    vecchia lasciata scoperta.
    """
    from app.core import masking

    h, w = image_rgb.shape[:2]
    seg = seg or segment(image_rgb)
    s = min(1.0, REFINE_SIDE / max(h, w))
    rw, rh = max(1, round(w * s)), max(1, round(h * s))
    work = cv2.resize(
        np.ascontiguousarray(image_rgb[..., :3]), (rw, rh), interpolation=cv2.INTER_AREA
    )
    p = seg.prob(SURFACES[surface], (rw, rh))
    p = np.where(seg.object_prob((rw, rh)) >= OBJECT_THRESHOLD, np.minimum(p, 0.3), p)
    p = p.astype(np.float32)
    hard = p >= 0.5
    if refine and hard.any() and not hard.all():
        hard = masking.snap_to_edges(work, p, radius=max(3, round(SNAP_FRACTION * max(rw, rh))))
    hard = _clean(hard)
    if not refine:
        full = cv2.resize(hard.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.asarray(full, dtype=np.uint8)
    return masking.edge_alpha(image_rgb, hard)


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
