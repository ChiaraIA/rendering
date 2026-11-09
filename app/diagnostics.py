"""Diagnosi dell'installazione (`posa verifica`): cosa c'è, cosa manca, come rimediare."""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Level = Literal["ok", "avviso", "errore"]

CREDENTIAL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    detail: str
    fix: str = ""


def credentials_problem() -> str | None:
    """None se gli agenti hanno credenziali, altrimenti cosa fare."""
    if any(os.environ.get(v) for v in CREDENTIAL_VARS) or os.environ.get("ANTHROPIC_BASE_URL"):
        return None
    return (
        "manca ANTHROPIC_API_KEY. Su Codespaces: Settings di GitHub → Codespaces → Secrets, "
        "aggiungi ANTHROPIC_API_KEY per questa repo e ricrea il Codespace; "
        "oppure nel terminale: export ANTHROPIC_API_KEY=sk-ant-..."
    )


def _python() -> Check:
    v = sys.version_info
    ok = v >= (3, 11)
    return Check(
        "Python",
        "ok" if ok else "errore",
        f"{v.major}.{v.minor}.{v.micro}",
        "" if ok else "serve Python 3.11 o più recente",
    )


def _module(label: str, module: str) -> Check:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # ImportError o librerie di sistema mancanti
        return Check(label, "errore", f"non importabile: {exc}", "esegui: pip install -e .")
    return Check(label, "ok", str(getattr(mod, "__version__", "presente")))


def _heif() -> Check:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception as exc:
        return Check("Foto HEIC (iPhone)", "errore", str(exc), "esegui: pip install -e .")
    return Check("Foto HEIC (iPhone)", "ok", f"pillow-heif {pillow_heif.__version__}")


def _agent_cli() -> Check:
    try:
        import claude_agent_sdk
    except Exception as exc:
        return Check("Agenti (Claude Agent SDK)", "errore", str(exc), "esegui: pip install -e .")
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    if bundled.is_file():
        return Check(
            "Agenti (Claude Agent SDK)",
            "ok",
            f"{getattr(claude_agent_sdk, '__version__', '')} con eseguibile incluso",
        )
    return Check(
        "Agenti (Claude Agent SDK)",
        "avviso",
        "eseguibile incluso non trovato: verrà usato `claude` se installato",
        "reinstalla con: pip install --force-reinstall claude-agent-sdk",
    )


def _segmentation() -> Check:
    from app.adapters import semantic

    name = "Segmentazione automatica"
    if os.environ.get("POSA_SEGMENTAZIONE", "1") == "0":
        return Check(name, "avviso", "disattivata (POSA_SEGMENTAZIONE=0)", "")
    if not semantic.available():
        return Check(
            name,
            "avviso",
            "non installata: la maschera la disegnano gli agenti",
            "per i bordi precisi (gambe di sedie e tavoli): "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
            'pip install -e ".[segmentazione]"',
        )
    return Check(name, "ok", f"disponibile ({semantic.DEFAULT_MODEL})")


def model_check() -> Check:
    """Carica (e se serve scarica) il modello di segmentazione e lo prova su un'immagine."""
    import numpy as np

    from app.adapters import semantic

    name = "Modello di segmentazione"
    if not semantic.available():
        return Check(name, "avviso", "non installato", "")
    try:
        img = np.full((240, 320, 3), 180, np.uint8)
        semantic.segment(img)
    except Exception as exc:
        return Check(
            name,
            "errore",
            str(exc)[:300],
            "controlla la connessione a huggingface.co (primo download ~230 MB)",
        )
    return Check(name, "ok", "scaricato e funzionante")


def _catalog() -> Check:
    try:
        from app.catalog.loader import DEFAULT_CATALOG, Catalog

        cat = Catalog.load(DEFAULT_CATALOG)  # il catalogo del negozio si conta in _shop
        for spec in cat.materials:
            cat.load_variants(spec)
    except Exception as exc:
        return Check(
            "Catalogo",
            "errore",
            str(exc),
            "controlla app/catalog/data (texture mancanti o modificate)",
        )
    return Check("Catalogo", "ok", f"{len(cat.materials)} articoli, texture verificate")


def _shop() -> Check:
    from app.catalog.loader import shop_catalog_dir
    from app.storage.shopify import configured_store

    store, _ = configured_store(shop_catalog_dir())
    if not store:
        return Check(
            "Negozio Shopify",
            "ok",
            "non collegato (facoltativo)",
            "per usare le foto del negozio: posa shopify --negozio mio-negozio.myshopify.com",
        )
    path = shop_catalog_dir() / "catalog.json"
    n = len(json.loads(path.read_text())["materials"]) if path.is_file() else 0
    return Check("Negozio Shopify", "ok", f"{store}: {n} articoli importati")


def _credentials() -> Check:
    problem = credentials_problem()
    if problem is None:
        source = next((v for v in CREDENTIAL_VARS if os.environ.get(v)), "proxy/gateway")
        return Check("Credenziali API", "ok", f"trovate ({source})")
    return Check("Credenziali API", "avviso", "nessuna chiave trovata", problem)


def offline_checks() -> list[Check]:
    return [
        _python(),
        _module("NumPy", "numpy"),
        _module("OpenCV", "cv2"),
        _module("Pillow", "PIL"),
        _heif(),
        _module("Interfaccia web (Gradio)", "gradio"),
        _agent_cli(),
        _catalog(),
        _shop(),
        _segmentation(),
        _credentials(),
    ]


async def online_check(model: str) -> Check:
    """Chiamata minima all'API attraverso l'Agent SDK (stesso percorso degli agenti)."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    try:
        result = None
        async for msg in query(
            prompt="Rispondi solo con la parola: pronto",
            options=ClaudeAgentOptions(model=model, max_turns=1, allowed_tools=[], tools=[]),
        ):
            if isinstance(msg, ResultMessage):
                result = msg
    except Exception as exc:
        return Check(
            "Chiamata di prova agli agenti",
            "errore",
            str(exc)[:300],
            "controlla la chiave API e la connessione",
        )
    if result is None or result.is_error:
        detail = (result.result or result.subtype) if result else "nessuna risposta"
        return Check(
            "Chiamata di prova agli agenti",
            "errore",
            str(detail)[:300],
            "controlla la chiave API (valida, con credito) e la connessione",
        )
    cost = f", costo {result.total_cost_usd:.3f} USD" if result.total_cost_usd else ""
    return Check("Chiamata di prova agli agenti", "ok", f"risposta ricevuta da {model}{cost}")
