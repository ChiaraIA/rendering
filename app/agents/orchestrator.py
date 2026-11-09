"""Orchestratore multiagente su Claude Agent SDK.

Struttura:
- orchestratore (questa sessione): pianifica, delega, gestisce i retry e l'escalation;
- agenti esecutori (sotto-agenti): `pavimento`, `geometra`, `materiali`;
- controllori indipendenti: avviati dal codice dentro gli strumenti (vedi checkers.py);
- cancelli: nel codice (state.py), nessun agente può scavalcarli.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app import config
from app.agents import prompts
from app.agents.checkers import CheckerRunner, ClaudeChecker
from app.agents.security import INJECTION_GUARD, locked_options, redact
from app.agents.state import ClientRequest
from app.agents.tools import Toolbox
from app.catalog.loader import Catalog
from app.storage.workspace import Workspace

SUBAGENT_LABELS = {
    "pavimento": "🟦 pavimento",
    "geometra": "📐 geometra",
    "materiali": "🧱 materiali",
}


@dataclass
class JobOutcome:
    workspace: Workspace
    delivered: bool
    summary: str
    cost_usd: float


def _subagents(model: str) -> dict[str, Any]:
    from claude_agent_sdk import AgentDefinition

    def t(*names: str) -> list[str]:
        return [f"mcp__posa__{n}" for n in ("guarda", *names)]

    return {
        "pavimento": AgentDefinition(
            description="Delimita il pavimento visibile (maschera), escludendo mobili e muri.",
            prompt=prompts.MASK_AGENT + INJECTION_GUARD,
            tools=t("proponi_maschera", "stato_pratica"),
            model=model,
        ),
        "geometra": AgentDefinition(
            description="Trova la prospettiva del pavimento: 4 angoli della stanza e misure.",
            prompt=prompts.GEOMETRY_AGENT + INJECTION_GUARD,
            tools=t("proponi_geometria", "stato_pratica"),
            model=model,
        ),
        "materiali": AgentDefinition(
            description="Sceglie l'articolo: codice del cliente o abbinamento al campione.",
            prompt=prompts.MATERIAL_AGENT + INJECTION_GUARD,
            tools=t("catalogo", "scegli_materiale", "stato_pratica"),
            model=model,
        ),
        "luci": AgentDefinition(
            description="Regola luci, ombre e riflessi del nuovo pavimento perché sia naturale.",
            prompt=prompts.LIGHT_AGENT + INJECTION_GUARD,
            tools=t("proponi_luce", "stato_pratica"),
            model=model,
        ),
    }


def _job_prompt(ws: Workspace) -> str:
    req = ws.state.request
    lines = [
        f"Nuova pratica {ws.state.job_id}.",
        f"Foto del cliente: {ws.state.image.width}x{ws.state.image.height} px "
        f"(per gli agenti: preview.jpg e preview_griglia.jpg, {ws.state.preview_size[0]}x"
        f"{ws.state.preview_size[1]} px).",
    ]
    if req.material_id:
        lines.append(f"Il cliente ha scelto l'articolo a catalogo: {req.material_id}.")
    if req.sample_file:
        lines.append("Il cliente ha mandato la foto di un campione: campione.jpg.")
    if req.room_mm:
        lines.append(
            f"Misure della stanza date dal cliente: {req.room_mm[0] / 1000:.2f} x "
            f"{req.room_mm[1] / 1000:.2f} m."
        )
    else:
        lines.append("Il cliente non ha dato le misure della stanza: andranno stimate.")
    lines.append("Inizia chiamando stato_pratica, poi segui la procedura.")
    return "\n".join(lines)


async def run_job(
    ws: Workspace,
    catalog: Catalog,
    checker: CheckerRunner | None = None,
    model: str | None = None,
) -> JobOutcome:
    from claude_agent_sdk import AssistantMessage, ResultMessage, query

    model = model or config.AGENT_MODEL
    toolbox = Toolbox(ws, catalog, checker or ClaudeChecker(model))
    tools = [
        "guarda",
        "stato_pratica",
        "proponi_maschera",
        "proponi_geometria",
        "catalogo",
        "scegli_materiale",
        "proponi_luce",
        "esegui_posa",
        "consegna",
        "richiedi_operatore",
    ]
    options = locked_options(
        system_prompt=prompts.ORCHESTRATOR + INJECTION_GUARD,
        model=model,
        cwd=str(ws.root),
        mcp_servers={"posa": toolbox.mcp_server()},
        agents=_subagents(model),
        tools=["Task", "Agent"],  # solo deleghe ai sotto-agenti: niente file, shell o rete
        allowed_tools=["Task", "Agent", *[f"mcp__posa__{n}" for n in tools]],
        max_turns=config.ORCHESTRATOR_MAX_TURNS,
        max_budget_usd=config.JOB_BUDGET_USD,
    )
    ws.log(f"🧭 orchestratore: pratica avviata (modello {model})")
    names: dict[str, str] = {}  # tool_use_id della delega -> nome del sotto-agente
    seen: set[str] = set()
    summary, cost = "", 0.0
    async for msg in query(prompt=_job_prompt(ws), options=options):
        if isinstance(msg, AssistantMessage):
            _narrate(ws, msg, names, seen)
        elif isinstance(msg, ResultMessage):
            summary = redact(msg.result or "")
            cost = msg.total_cost_usd or 0.0
    ws.state.cost_usd += cost
    ws.persist()
    delivered = bool(ws.state.delivered)
    ws.log(
        ("🏁 completata" if delivered else "⏸️ non consegnata")
        + f" — costo stimato {ws.state.cost_usd:.2f} USD"
    )
    return JobOutcome(ws, delivered, summary, ws.state.cost_usd)


def _narrate(ws: Workspace, msg: Any, names: dict[str, str], seen: set[str]) -> None:
    """Traduce i messaggi degli agenti in righe di log leggibili."""
    parent = getattr(msg, "parent_tool_use_id", None)
    who = SUBAGENT_LABELS.get(names.get(parent or "", ""), "🧭 orchestratore")
    for block in msg.content:
        kind = type(block).__name__
        bid = getattr(block, "id", None)
        if bid:  # l'SDK può riemettere lo stesso blocco: lo raccontiamo una volta sola
            if bid in seen:
                continue
            seen.add(bid)
        if kind == "TextBlock" and block.text.strip():
            ws.log(f"{who}: {redact(' '.join(block.text.split()), 400)}")
        elif kind == "ToolUseBlock":
            inp = getattr(block, "input", {}) or {}
            if block.name in ("Task", "Agent"):
                sub = str(inp.get("subagent_type", "?"))
                names[block.id] = sub
                what = redact(str(inp.get("description", "")), 200)
                ws.log(f"🧭 orchestratore → delega a «{sub}»: {what}")
            elif block.name == "mcp__posa__guarda":
                ws.log(f"{who} guarda {Path(str(inp.get('file', ''))).name}")
            elif block.name.startswith("mcp__posa__"):
                short = block.name.removeprefix("mcp__posa__")
                if short not in ("stato_pratica",):
                    ws.log(f"{who} → {short}")


def start(
    photo: Path,
    out_dir: Path,
    material_id: str | None = None,
    sample: Path | None = None,
    room_m: tuple[float, float] | None = None,
    listener: Callable[[str], None] | None = None,
) -> Workspace:
    """Crea la pratica (ingest in sola lettura dell'originale)."""
    if not material_id and not sample:
        raise ValueError("serve un codice a catalogo o la foto di un campione")
    req = ClientRequest(
        material_id=material_id,
        sample_file=str(sample) if sample else None,
        room_mm=(room_m[0] * 1000, room_m[1] * 1000) if room_m else None,
    )
    ws = Workspace.create(photo, out_dir, req)
    if listener:
        ws.listeners.append(listener)
    return ws


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
