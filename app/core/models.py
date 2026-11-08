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
    source_bit_depth: int = 8
    source_format: str
    source_mode: str
    exif_orientation: int = Field(ge=1, le=8)
    icc_sha256: str | None
    icc_description: str | None
    icc_dropped: bool = False
    color_space: ColorSpace
    color_origin: Literal["icc", "nclx", "none"] = "none"
    focal_35mm: float | None = None
    camera: str | None = None
    extra_images_ignored: bool = False


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
    POLYGON = "polygon"  # poligono proposto da un agente
    REFINED = "refined"  # poligono rifinito sui bordi reali (GrabCut)
    SEMANTIC = "semantic"  # modello di segmentazione + rifinitura (+ correzioni dell'agente)


class MaskStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"  # approvata da un controllore indipendente


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
    approved_at: str | None = None
    approved_by: str | None = None


Rotation = Literal[0, 60, 90, 120, 180, 240, 270, 300]
SQRT3 = 3.0**0.5


class TextureVariant(BaseModel):
    """Foto di una singola piastrella, frontale e ritagliata, senza fuga."""

    model_config = ConfigDict(frozen=True)

    texture_file: str
    texture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weight: float = Field(1.0, gt=0)  # frequenza relativa nella posa (collezioni miste)
    rotations: list[Rotation] | None = None  # None = quelle del materiale

    @model_validator(mode="after")
    def _plain_name(self) -> TextureVariant:
        f = self.texture_file
        if "/" in f or "\\" in f or ".." in f or not f:
            raise ValueError("texture_file deve essere un nome di file nella cartella textures/")
        return self


class LayingPattern(BaseModel):
    """Schema di posa: fissato dal materiale, mai scelto dal cliente.

    `hex`: esagoni con lati piatti in alto e in basso; tile_w_mm = da punta a punta,
    tile_h_mm = da lato a lato (= tile_w_mm * √3/2).
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["grid", "running_bond", "hex"] = "grid"
    bond_offset: float = Field(0.5, gt=0, lt=1)  # frazione di piastrella, solo running_bond
    rotation_deg: Literal[0, 45, 90] = 0  # rispetto ai muri (lati del rettangolo stanza)


class MaterialSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{1,63}$")
    name: str
    version: str
    tile_w_mm: float = Field(gt=0)
    tile_h_mm: float = Field(gt=0)
    joint_mm: float = Field(ge=0, le=30)  # 0 = posa a incastro / senza fuga
    joint_color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    pattern: LayingPattern = LayingPattern()
    variants: list[TextureVariant] = Field(min_length=1)
    variant_rotations: list[Rotation] = [0]
    texture_color_space: Literal["srgb"] = "srgb"

    @property
    def is_hex(self) -> bool:
        return self.pattern.type == "hex"

    def rotations_of(self, variant: int) -> list[int]:
        own = self.variants[variant].rotations
        return list(own if own else self.variant_rotations)

    @model_validator(mode="after")
    def _consistent(self) -> MaterialSpec:
        if self.joint_mm > 0 and self.joint_color is None:
            raise ValueError("joint_color obbligatorio se joint_mm > 0")
        if self.is_hex and abs(self.tile_h_mm - self.tile_w_mm * SQRT3 / 2) > 0.02 * self.tile_h_mm:
            raise ValueError("esagono: tile_h_mm (lato-lato) = tile_w_mm (punta-punta) × √3/2")
        sets = [self.variant_rotations] + [v.rotations for v in self.variants if v.rotations]
        for rots in sets:
            if not rots or len(set(rots)) != len(rots):
                raise ValueError("le rotazioni devono essere valori distinti")
            if self.is_hex and any(r % 60 for r in rots):
                raise ValueError("esagoni: rotazioni solo a multipli di 60°")
            if not self.is_hex and any(r % 90 for r in rots):
                raise ValueError("rotazioni solo a multipli di 90°")
            if (
                not self.is_hex
                and self.tile_w_mm != self.tile_h_mm
                and any(r in (90, 270) for r in rots)
            ):
                raise ValueError("rotazioni 90/270 ammesse solo per piastrelle quadrate")
        files = [v.texture_file for v in self.variants]
        if len(set(files)) != len(files):
            raise ValueError("varianti duplicate")
        return self

    @property
    def pitch_mm(self) -> tuple[float, float]:
        """Passo: piastrella + fuga. Per gli esagoni: distanza tra centri (colonne, righe)."""
        if self.is_hex:
            r_lat = self.tile_w_mm / 2 + self.joint_mm / SQRT3
            return (1.5 * r_lat, SQRT3 * r_lat)
        return (self.tile_w_mm + self.joint_mm, self.tile_h_mm + self.joint_mm)


class FocalCheck(BaseModel):
    """Coerenza tra angoli cliccati, misure della stanza e focale EXIF."""

    model_config = ConfigDict(frozen=True)

    focal_35mm: float | None
    f_px_exif: float | None
    f_px_estimated: float | None
    axes_angle_deg: float | None = None  # angolo tra i muri implicito (atteso 90)
    aspect_error: float | None = None  # errore relativo sul rapporto L/W implicito
    consistent: bool | None = None  # None = non verificabile (manca l'EXIF)
    message: str = ""


class PlaneGeometry(BaseModel):
    """Stanza rettangolare L x W mm; H mappa il piano (mm) nell'immagine (px)."""

    model_config = ConfigDict(frozen=True)

    corners_px: list[tuple[float, float]] = Field(min_length=4, max_length=4)
    length_mm: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    image_size: tuple[int, int]
    H: list[list[float]]
    H_inv: list[list[float]]
    condition_number: float
    reprojection_error_px: float
    focal: FocalCheck


TILE_NONE = np.iinfo(np.int32).min


@dataclass(frozen=True, eq=False)
class MaterialLayer:
    """Materiale proiettato, in luce lineare, limitato a `bbox`. Shape (h, w, 3)."""

    rgb_linear: NDArray[np.float32]
    bbox: BBox
    material_id: str
    render_mode: str
    notes: tuple[str, ...] = ()
    tile_ij: NDArray[np.int32] | None = None  # (h, w, 2) indici piastrella; TILE_NONE = fuga


@dataclass(frozen=True, eq=False)
class LightMap:
    """Guadagno di luce a bassa frequenza (M3). Shape (h, w), limitato a `bbox`.

    `tint` è il colore della luce della stanza (R, G, B lineari, luminanza 1): con la luce
    calda di una lampada un pavimento bianco nella foto vera appare caldo, non bianco puro.
    """

    gain: NDArray[np.float32]
    bbox: BBox
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


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


class LayoutStats(BaseModel):
    """Numeri della posa, utili al cliente e al posatore."""

    seed: int
    tiles_whole: int
    tiles_cut: int
    px_per_mm_min: float
    px_per_mm_max: float
