"""Orchestratore multiagente su Claude Agent SDK.

Struttura:
- orchestratore (questa sessione): pianifica, delega, gestisce i retry e l'escalation;
- agenti esecutori (sotto-agenti): `pavimento` oppure `parete` (maschera), `geometra`,
  `materiali`, `luci`; prompt e strumenti dipendono dalla superficie della pratica;
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
from app.agents.state import ClientRequest, Zone
from app.agents.tools import Toolbox
from app.catalog.loader import Catalog
from app.core.models import Surface
from app.storage.workspace import Workspace

SUBAGENT_LABELS = {
    "pavimento": "🟦 pavimento",
    "parete": "🧱 parete",
    "geometra": "📐 geometra",
    "materiali": "🎨 materiali",
    "luci": "💡 luci",
}


@dataclass
class JobOutcome:
    workspace: Workspace
    delivered: bool
    summary: str
    cost_usd: float


def geometry_tool(surface: str) -> str:
    return "proponi_geometria_parete" if surface == "rivestimento" else "proponi_geometria"


def _subagents(model: str, surface: str = "pavimento") -> dict[str, Any]:
    from claude_agent_sdk import AgentDefinition

    pr = prompts.for_surface(surface)
    wall = surface == "rivestimento"

    def t(*names: str) -> list[str]:
        return [f"mcp__posa__{n}" for n in ("guarda", *names)]

    return {
        pr.mask_agent_name: AgentDefinition(
            description=(
                "Delimita la zona di parete da rivestire (maschera), escludendo piano di "
                "lavoro, pensili, finestre, prese e oggetti."
                if wall
                else "Delimita il pavimento visibile (maschera), escludendo mobili, oggetti e muri."
            ),
            prompt=pr.mask_agent + INJECTION_GUARD,
            tools=t("proponi_maschera", "stato_pratica"),
            model=model,
        ),
        "geometra": AgentDefinition(
            description=(
                "Trova la prospettiva delle pareti da rivestire: 4 angoli e misure per parete."
                if wall
                else "Trova la prospettiva del pavimento: 4 angoli della stanza e misure."
            ),
            prompt=pr.geometry_agent + INJECTION_GUARD,
            tools=t(geometry_tool(surface), "stato_pratica"),
            model=model,
        ),
        "materiali": AgentDefinition(
            description="Sceglie l'articolo: codice del cliente o abbinamento al campione.",
            prompt=prompts.MATERIAL_AGENT + INJECTION_GUARD,
            tools=t("catalogo", "scegli_materiale", "stato_pratica"),
            model=model,
        ),
        "luci": AgentDefinition(
            description="Regola luci, ombre e riflessi della nuova superficie perché sia naturale.",
            prompt=pr.light_agent + INJECTION_GUARD,
            tools=t("proponi_luce", "stato_pratica"),
            model=model,
        ),
    }


def _job_prompt(ws: Workspace) -> str:
    req = ws.state.request
    wall = req.surface == "rivestimento"
    lines = [
        f"Nuova pratica {ws.state.job_id}: "
        + (
            f"rivestimento a PARETE (zona richiesta: {req.zone})."
            if wall
            else "posa del PAVIMENTO."
        ),
        f"Foto del cliente: {ws.state.image.width}x{ws.state.image.height} px "
        f"(per gli agenti: preview.jpg e preview_griglia.jpg, {ws.state.preview_size[0]}x"
        f"{ws.state.preview_size[1]} px).",
    ]
    if req.material_id:
        lines.append(f"Il cliente ha scelto l'articolo a catalogo: {req.material_id}.")
    if req.sample_file:
        lines.append("Il cliente ha mandato la foto di un campione: campione.jpg.")
    if req.room_mm:
        what = "della zona di parete (larghezza x altezza)" if wall else "della stanza"
        lines.append(
            f"Misure {what} date dal cliente: {req.room_mm[0] / 1000:.2f} x "
            f"{req.room_mm[1] / 1000:.2f} m."
        )
    else:
        lines.append("Il cliente non ha dato le misure: andranno stimate.")
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
    surface = ws.state.request.surface
    tools = [
        "guarda",
        "stato_pratica",
        "proponi_maschera",
        geometry_tool(surface),
        "catalogo",
        "scegli_materiale",
        "proponi_luce",
        "esegui_posa",
        "consegna",
        "richiedi_operatore",
    ]
    options = locked_options(
        system_prompt=prompts.for_surface(surface).orchestrator + INJECTION_GUARD,
        model=model,
        cwd=str(ws.root),
        mcp_servers={"posa": toolbox.mcp_server()},
        agents=_subagents(model, surface),
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
    surface: Surface = "pavimento",
    zone: Zone = "automatica",
) -> Workspace:
    """Crea la pratica (ingest in sola lettura dell'originale).

    `surface`: "pavimento" o "rivestimento" (parete). Per un rivestimento `room_m` sono
    larghezza e altezza della zona di parete, e `zone` dice quale zona rivestire."""
    if not material_id and not sample:
        raise ValueError("serve un codice a catalogo o la foto di un campione")
    req = ClientRequest(
        material_id=material_id,
        sample_file=str(sample) if sample else None,
        room_mm=(room_m[0] * 1000, room_m[1] * 1000) if room_m else None,
        surface=surface,
        zone=zone,
    )
    ws = Workspace.create(photo, out_dir, req)
    if listener:
        ws.listeners.append(listener)
    return ws


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
