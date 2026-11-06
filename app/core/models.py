"""Contratti dati condivisi.

- Pydantic per i metadati serializzabili (finiscono in job.json e report.json).
- dataclass frozen per i contenitori di array NumPy.

`CompositeResult` NON è qui: vive in `compositor.py`, l'unico modulo che può crearlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

BBox = tuple[int, int, int, int]
"""(x0, y0, x1, y1), estremi x1/y1 esclusi."""


class ColorSpace(StrEnum):
    SRGB = "srgb"
    DISPLAY_P3 = "display_p3"
    ADOBE_RGB = "adobe_rgb"
    ASSUMED_SRGB = "assumed_srgb"  # nessun profilo nel file
    UNKNOWN = "unknown"  # profilo presente ma non riconosciuto: TRC sRGB + warning


class ImageMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_sha256: str
    normalized_sha256: str
    width: int
    height: int
    channels: Literal[3, 4]
    bit_depth: Literal[8, 16]
    source_format: str
    source_mode: str
    exif_orientation: int = Field(ge=1, le=8)
    icc_sha256: str | None
    icc_description: str | None
    icc_dropped: bool = False
    color_space: ColorSpace


@dataclass(frozen=True, eq=False)
class SourceImage:
    """Originale normalizzato. `pixels` è in sola lettura."""

    pixels: NDArray[np.uint8]
    icc: bytes | None
    meta: ImageMeta

    def __post_init__(self) -> None:
        if self.pixels.flags.writeable:
            raise ValueError("SourceImage.pixels deve essere in sola lettura")

    @property
    def shape_hw(self) -> tuple[int, int]:
        return (int(self.pixels.shape[0]), int(self.pixels.shape[1]))


class MaskOrigin(StrEnum):
    IMPORTED = "imported"
    POLYGON = "polygon"
    GRABCUT = "grabcut"
    SAM = "sam"


class MaskStatus(StrEnum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"


class MaskRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    mask_sha256: str
    image_sha256: str
    origin: MaskOrigin
    status: MaskStatus
    width: int
    height: int
    area_px: int
    soft_edge_px: int
    bbox: BBox | None
    confirmed_at: str | None = None


class MaterialSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{1,63}$")
    name: str
    version: str
    texture_file: str
    texture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    texture_color_space: Literal["srgb"] = "srgb"
    tile_w_mm: float = Field(gt=0)
    tile_h_mm: float = Field(gt=0)
    joint_default_mm: float = Field(ge=0)
    joint_range_mm: tuple[float, float]
    joint_color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    allowed_patterns: list[Literal["grid", "running_bond_50"]] = ["grid"]

    @model_validator(mode="after")
    def _joint_in_range(self) -> MaterialSpec:
        lo, hi = self.joint_range_mm
        if not (0 <= lo <= self.joint_default_mm <= hi):
            raise ValueError("joint_default_mm fuori da joint_range_mm")
        if "/" in self.texture_file or "\\" in self.texture_file or ".." in self.texture_file:
            raise ValueError("texture_file deve essere un nome di file nella cartella textures/")
        return self


class LayoutParams(BaseModel):
    """Parametri di posa scelti dall'utente. Misure e texture arrivano solo dal catalogo."""

    model_config = ConfigDict(frozen=True)

    material_id: str
    joint_mm: float | None = None
    rotation_deg: float = 0.0
    offset_mm: tuple[float, float] = (0.0, 0.0)
    pattern: Literal["grid", "running_bond_50"] = "grid"
    seed: int = 0


class LightingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["none"] = "none"  # M3: "lowpass_rectified", M6: "generative_hint"


class RenderRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    layout: LayoutParams
    lighting: LightingParams = LightingParams()
    render_mode: Literal["flat"] = "flat"  # M2: "tiled"
    output_format: Literal["png", "tiff"] = "png"


@dataclass(frozen=True, eq=False)
class MaterialLayer:
    """Materiale proiettato, in luce lineare, limitato a `bbox`. Shape (h, w, 3)."""

    rgb_linear: NDArray[np.float32]
    bbox: BBox
    material_id: str
    render_mode: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class LightMap:
    """Guadagno di luce a bassa frequenza (M3). Shape (h, w), limitato a `bbox`."""

    gain: NDArray[np.float32]
    bbox: BBox


class CheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    blocking: bool
    passed: bool
    value: float | int | str | bool | None = None
    threshold: float | int | str | None = None
    message: str = ""


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    final_sha256: str
    checks: list[CheckResult]

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    def with_checks(self, extra: list[CheckResult]) -> ValidationReport:
        return ValidationReport(final_sha256=self.final_sha256, checks=[*self.checks, *extra])


class OutputFile(BaseModel):
    kind: Literal["master", "preview", "mask", "quarantine"]
    path: str
    sha256: str


class MaterialInfo(BaseModel):
    id: str
    name: str
    version: str
    texture_sha256: str
    catalog_sha256: str
    flat_color_linear: tuple[float, float, float] | None = None


class RenderReport(BaseModel):
    schema_version: str
    render_id: str
    job_id: str
    created_at: str
    valid: bool
    input: ImageMeta
    mask: MaskRecord
    request: RenderRequest
    material: MaterialInfo
    checks: list[CheckResult]
    warnings: list[str]
    outputs: list[OutputFile]
    final_sha256: str
    determinism_key: str
    versions: dict[str, str]
    timings_ms: dict[str, float]
