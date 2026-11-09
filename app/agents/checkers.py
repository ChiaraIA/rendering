"""Controllori indipendenti.

Ogni controllore è una sessione Claude separata, avviata dal CODICE (non dall'orchestratore):
- vede solo le immagini che gli passiamo (strumento `guarda`, elenco chiuso);
- non conosce la conversazione dell'agente che ha prodotto il lavoro;
- può rispondere solo chiamando il tool `verdetto`; se non lo fa, il verdetto è negativo.
Così nessun agente esecutore può approvarsi da solo o saltare un controllo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app import config
from app.agents import prompts
from app.agents.security import INJECTION_GUARD, locked_options, redact, view_image
from app.agents.state import Verdict


@dataclass(frozen=True)
class CheckRequest:
    checker: str  # nome del controllore (es. "controllo-maschera")
    instructions: str  # cosa verificare, con i criteri
    images: list[str]  # path relativi alla cartella della pratica
    context: str = ""  # dati numerici utili (misure, focale, statistiche)


class CheckerRunner(Protocol):
    async def __call__(self, req: CheckRequest, cwd: Path) -> Verdict: ...


class ClaudeChecker:
    """Controllore reale su Claude Agent SDK."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.AGENT_MODEL

    async def __call__(self, req: CheckRequest, cwd: Path) -> Verdict:
        from claude_agent_sdk import (
            ResultMessage,
            create_sdk_mcp_server,
            query,
            tool,
        )

        box: dict[str, Any] = {}

        @tool(
            "verdetto",
            "Registra il tuo verdetto finale. Chiamalo una sola volta.",
            {
                "type": "object",
                "properties": {
                    "approvato": {"type": "boolean"},
                    "motivo": {"type": "string"},
                    "problemi": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["approvato", "motivo"],
            },
        )
        async def verdetto(args: dict[str, Any]) -> dict[str, Any]:
            if "approvato" not in box:
                box.update(args)
            return {"content": [{"type": "text", "text": "Verdetto registrato. Hai finito."}]}

        allowed_images = set(req.images)

        @tool(
            "guarda",
            "Mostra una delle immagini da esaminare.",
            {"type": "object", "properties": {"file": {"type": "string"}}, "required": ["file"]},
        )
        async def guarda(args: dict[str, Any]) -> dict[str, Any]:
            return view_image(cwd, str(args.get("file", "")), allowed_images)

        server = create_sdk_mcp_server(name="verdetto", version="1.0.0", tools=[verdetto, guarda])
        images = "\n".join(f"- {p}" for p in req.images)
        prompt = (
            f"{req.instructions}\n\nImmagini da esaminare (aprile con `guarda`):\n{images}\n\n"
            f"{req.context}\n\nQuando hai deciso, chiama il tool `verdetto`."
        )
        options = locked_options(
            system_prompt=prompts.CHECKER_SYSTEM.format(nome=req.checker) + INJECTION_GUARD,
            model=self.model,
            cwd=str(cwd),
            mcp_servers={"verdetto": server},
            allowed_tools=["mcp__verdetto__verdetto", "mcp__verdetto__guarda"],
            max_turns=8,
            max_budget_usd=config.CHECK_BUDGET_USD,
        )
        cost = None
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, ResultMessage):
                cost = msg.total_cost_usd
        if "approvato" not in box:
            return Verdict(
                checker=req.checker,
                approved=False,
                reason="il controllore non ha emesso un verdetto",
                cost_usd=cost,
            )
        return Verdict(
            checker=req.checker,
            approved=bool(box["approvato"]),
            reason=redact(str(box.get("motivo", "")), 600),
            issues=[redact(str(p), 300) for p in (box.get("problemi", []) or [])[:12]],
            cost_usd=cost,
        )
