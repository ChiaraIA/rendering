"""Stato della pratica e cancelli.

I cancelli sono funzioni pure sullo stato: decidono se si può renderizzare e se si può
consegnare. Nessun agente può scavalcarli: i tool li interrogano prima di agire, e le
approvazioni possono essere scritte solo dal codice dopo il verdetto di un controllore.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_ATTEMPTS
from app.core.models import (
    CheckResult,
    ImageMeta,
    LayoutStats,
    LightParams,
    MaskRecord,
    PlaneGeometry,
    Surface,
)

Step = Literal["maschera", "geometria", "materiale", "luce", "posa"]
STEPS: tuple[Step, ...] = ("maschera", "geometria", "materiale", "luce", "posa")


class Verdict(BaseModel):
    """Verdetto di un controllore indipendente."""

    checker: str
    approved: bool
    reason: str
    issues: list[str] = []
    cost_usd: float | None = None


class Attempt(BaseModel):
    n: int
    artifact_sha256: str
    overlay: str | None = None
    verdict: Verdict


Zone = Literal["automatica", "paraschizzi", "parete"]


class ClientRequest(BaseModel):
    material_id: str | None = None  # codice a catalogo scelto dal cliente
    sample_file: str | None = None  # foto di un campione (alternativa al codice)
    # misure date dal cliente: pavimento = lunghezza x larghezza della stanza;
    # rivestimento = larghezza x altezza della zona di parete da rivestire
    room_mm: tuple[float, float] | None = None
    surface: Surface = "pavimento"  # cosa rivestire: il pavimento o le pareti
    # solo rivestimento: fascia tra piano di lavoro e pensili/finestre, o parete intera
    zone: Zone = "automatica"


class MaterialChoice(BaseModel):
    material_id: str
    how: Literal["codice_cliente", "abbinamento_campione"]
    motivation: str


class RenderOutcome(BaseModel):
    final_sha256: str
    mask_sha256: str
    geometry_sha256: str
    material_id: str
    light_sha256: str = ""
    checks: list[CheckResult]
    integrity_ok: bool
    layout: LayoutStats
    before_after: str
    result_preview: str


class JobState(BaseModel):
    job_id: str
    created_at: str
    image: ImageMeta
    request: ClientRequest
    preview_scale: float
    preview_size: tuple[int, int]

    mask_attempts: list[Attempt] = []
    mask: MaskRecord | None = None  # solo se approvata

    geometry_attempts: list[Attempt] = []
    geometry: PlaneGeometry | None = None  # solo se approvata (primo piano)
    more_planes: list[PlaneGeometry] = []  # rivestimento: le altre pareti (cucina ad angolo)
    geometry_sha256: str | None = None  # di tutti i piani insieme
    room_estimated: bool = False

    material_attempts: list[Attempt] = []
    material: MaterialChoice | None = None  # solo se approvato

    light_attempts: list[Attempt] = []
    light: LightParams | None = None  # solo se approvata dal controllo-luce
    light_sha256: str | None = None
    light_material_id: str | None = None  # la luce vale per quel materiale (finitura)

    render_attempts: list[Attempt] = []
    render: RenderOutcome | None = None  # solo se integrità ok E revisore approva

    delivered: list[str] = []
    escalation: str | None = None
    events: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def planes(self) -> list[PlaneGeometry]:
        return [self.geometry, *self.more_planes] if self.geometry else []

    @property
    def is_wall(self) -> bool:
        return self.request.surface == "rivestimento"

    def attempts(self, step: Step) -> list[Attempt]:
        return {
            "maschera": self.mask_attempts,
            "geometria": self.geometry_attempts,
            "materiale": self.material_attempts,
            "luce": self.light_attempts,
            "posa": self.render_attempts,
        }[step]


def can_attempt(state: JobState, step: Step) -> str | None:
    """None se si può fare un altro tentativo, altrimenti il motivo del blocco."""
    if state.escalation:
        return "pratica passata all'operatore"
    if len(state.attempts(step)) >= MAX_ATTEMPTS:
        return (
            f"raggiunto il limite di {MAX_ATTEMPTS} tentativi per '{step}': "
            "chiama richiedi_operatore"
        )
    return None


def missing_for_render(state: JobState) -> list[str]:
    missing = []
    if state.mask is None:
        missing.append("maschera della superficie approvata dal controllore")
    if state.geometry is None:
        missing.append("geometria approvata dal controllore")
    if state.material is None:
        missing.append("materiale confermato")
    if not light_ok(state):
        missing.append("luci e ombre approvate dal controllo-luce per il materiale confermato")
    return missing


def light_ok(state: JobState) -> bool:
    return (
        state.light is not None
        and state.material is not None
        and state.light_material_id == state.material.material_id
    )


def missing_for_delivery(state: JobState) -> list[str]:
    missing = missing_for_render(state)
    if state.render is None:
        missing.append("posa eseguita, integrità verificata e approvata dal revisore finale")
    else:
        if not state.render.integrity_ok:
            missing.append("controllo di integrità superato")
        if state.mask and state.render.mask_sha256 != state.mask.mask_sha256:
            missing.append("posa rifatta con la maschera approvata corrente")
        if state.render.geometry_sha256 != state.geometry_sha256:
            missing.append("posa rifatta con la geometria approvata corrente")
        if state.material and state.render.material_id != state.material.material_id:
            missing.append("posa rifatta con il materiale confermato")
        if state.render.light_sha256 != (state.light_sha256 or ""):
            missing.append("posa rifatta con le luci approvate correnti")
    if state.escalation:
        missing.append("pratica non in escalation")
    return missing


def next_step(state: JobState) -> str:
    if state.escalation:
        return "fermo: pratica passata all'operatore"
    if state.delivered:
        return "completato: risultato consegnato"
    if state.mask is None:
        return f"maschera: delega all'agente '{'parete' if state.is_wall else 'pavimento'}'"
    if state.geometry is None:
        return "geometria: delega all'agente 'geometra'"
    if state.material is None:
        return "materiale: delega all'agente 'materiali'"
    if not light_ok(state):
        return "luce: delega all'agente 'luci'"
    if state.render is None:
        return "posa: chiama esegui_posa"
    return "consegna: chiama consegna"
