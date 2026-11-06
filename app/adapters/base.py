"""Contratti degli adapter.

Regola 5: un adapter non restituisce MAI un'immagine finale. Può solo proporre
una maschera (che va confermata) o un suggerimento di luce (che passa sempre dal
filtro passa-basso e dal clamp di `core/lighting.py`). Implementazioni in M4 e M6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from app.core.models import SourceImage


@dataclass(frozen=True, eq=False)
class MaskProposal:
    mask: NDArray[np.uint8]  # sempre stato "proposed": serve la conferma dell'utente
    adapter: str


@dataclass(frozen=True, eq=False)
class LightHint:
    luminance: NDArray[np.float32]  # solo luminanza, mai RGB
    adapter: str


class SegmentationAdapter(Protocol):
    name: str

    def propose(
        self, image: SourceImage, positive: list[tuple[int, int]], negative: list[tuple[int, int]]
    ) -> MaskProposal: ...


class LightHintAdapter(Protocol):
    name: str

    def light_hint(self, image: SourceImage, mask: NDArray[np.uint8]) -> LightHint: ...
