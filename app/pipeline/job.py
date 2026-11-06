"""Documento di job (job.json) e macchina a stati.

Stati persistiti:
    INGESTED -> MASK_PROPOSED -> MASK_CONFIRMED -> EXPORTED | REJECTED
RENDERED e VALIDATED sono fasi transitorie dentro una singola chiamata di render.
M2 aggiungerà GEOMETRY_SET tra MASK_CONFIRMED e il render.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.core.errors import JobStateError
from app.core.models import ImageMeta, MaskRecord


class JobState(StrEnum):
    INGESTED = "ingested"
    MASK_PROPOSED = "mask_proposed"
    MASK_CONFIRMED = "mask_confirmed"
    EXPORTED = "exported"
    REJECTED = "rejected"


_RENDERABLE = {JobState.MASK_CONFIRMED, JobState.EXPORTED, JobState.REJECTED}

_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.INGESTED: {JobState.MASK_PROPOSED},
    JobState.MASK_PROPOSED: {JobState.MASK_PROPOSED, JobState.MASK_CONFIRMED},
    JobState.MASK_CONFIRMED: {JobState.MASK_PROPOSED} | _RENDERABLE,
    JobState.EXPORTED: {JobState.MASK_PROPOSED} | _RENDERABLE,
    JobState.REJECTED: {JobState.MASK_PROPOSED} | _RENDERABLE,
}


class RenderSummary(BaseModel):
    render_id: str
    valid: bool
    final_sha256: str
    report: str


class JobDocument(BaseModel):
    schema_version: str
    job_id: str
    created_at: str
    state: JobState
    source_file: str
    original_path: str
    image: ImageMeta
    active_mask: str | None = None
    masks: dict[str, MaskRecord] = {}
    renders: list[RenderSummary] = []

    def advance(self, new_state: JobState) -> JobDocument:
        if new_state not in _TRANSITIONS[self.state]:
            raise JobStateError(f"transizione non ammessa: {self.state} -> {new_state}")
        return self.model_copy(update={"state": new_state})

    @property
    def renderable(self) -> bool:
        return self.state in _RENDERABLE
