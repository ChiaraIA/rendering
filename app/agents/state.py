"""Stato della pratica e cancelli.

I cancelli sono funzioni pure sullo stato: decidono se si può renderizzare e se si può
consegnare. Nessun agente può scavalcarli: i tool li interrogano prima di agire, e le
approvazioni possono essere scritte solo dal codice dopo il verdetto di un controllore.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.config import MAX_ATTEMPTS
from app.core.models import CheckResult, ImageMeta, LayoutStats, MaskRecord, PlaneGeometry

Step = Literal["maschera", "geometria", "materiale", "posa"]


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


class ClientRequest(BaseModel):
    material_id: str | None = None  # codice a catalogo scelto dal cliente
    sample_file: str | None = None  # foto di un campione (alternativa al codice)
    room_mm: tuple[float, float] | None = None  # misure date dal cliente (lunghezza, larghezza)


class MaterialChoice(BaseModel):
    material_id: str
    how: Literal["codice_cliente", "abbinamento_campione"]
    motivation: str


class RenderOutcome(BaseModel):
    final_sha256: str
    mask_sha256: str
    geometry_sha256: str
    material_id: str
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
    geometry: PlaneGeometry | None = None  # solo se approvata
    geometry_sha256: str | None = None
    room_estimated: bool = False

    material_attempts: list[Attempt] = []
    material: MaterialChoice | None = None  # solo se approvato

    render_attempts: list[Attempt] = []
    render: RenderOutcome | None = None  # solo se integrità ok E revisore approva

    delivered: list[str] = []
    escalation: str | None = None
    events: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0

    def attempts(self, step: Step) -> list[Attempt]:
        return {
            "maschera": self.mask_attempts,
            "geometria": self.geometry_attempts,
            "materiale": self.material_attempts,
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
        missing.append("maschera del pavimento approvata dal controllore")
    if state.geometry is None:
        missing.append("geometria approvata dal controllore")
    if state.material is None:
        missing.append("materiale confermato")
    return missing


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
    if state.escalation:
        missing.append("pratica non in escalation")
    return missing


def next_step(state: JobState) -> str:
    if state.escalation:
        return "fermo: pratica passata all'operatore"
    if state.delivered:
        return "completato: risultato consegnato"
    if state.mask is None:
        return "maschera: delega all'agente 'pavimento'"
    if state.geometry is None:
        return "geometria: delega all'agente 'geometra'"
    if state.material is None:
        return "materiale: delega all'agente 'materiali'"
    if state.render is None:
        return "posa: chiama esegui_posa"
    return "consegna: chiama consegna"
