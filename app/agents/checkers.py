"""Controllori indipendenti.

Ogni controllore è una sessione Claude separata, avviata dal CODICE (non dall'orchestratore):
- vede solo le immagini che gli passiamo, con lo strumento Read;
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
            ClaudeAgentOptions,
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

        server = create_sdk_mcp_server(name="verdetto", version="1.0.0", tools=[verdetto])
        images = "\n".join(f"- {p}" for p in req.images)
        prompt = (
            f"{req.instructions}\n\nImmagini da esaminare (aprile con Read):\n{images}\n\n"
            f"{req.context}\n\nQuando hai deciso, chiama il tool `verdetto`."
        )
        options = ClaudeAgentOptions(
            system_prompt=prompts.CHECKER_SYSTEM.format(nome=req.checker),
            model=self.model,
            cwd=str(cwd),
            mcp_servers={"verdetto": server},
            allowed_tools=["Read", "mcp__verdetto__verdetto"],
            tools=["Read"],
            permission_mode="dontAsk",
            max_turns=8,
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
            reason=str(box.get("motivo", "")),
            issues=[str(p) for p in box.get("problemi", []) or []],
            cost_usd=cost,
        )
