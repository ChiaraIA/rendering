"""Soglie e default in un unico punto."""

from __future__ import annotations

import os

# Ingest
MAX_PIXELS = 80_000_000
ALLOWED_INPUT_FORMATS = frozenset({"JPEG", "MPO", "PNG", "TIFF", "WEBP", "HEIF"})


# Export
PNG_COMPRESS_LEVEL = 6
PREVIEW_MAX_SIDE = 1600
PREVIEW_JPEG_QUALITY = 90


# Agenti
AGENT_MODEL = os.environ.get("POSA_MODEL", "claude-opus-5-5")
# tetto di spesa per pratica e per controllo (difesa da foto/abusi che fanno girare gli agenti)
JOB_BUDGET_USD = float(os.environ.get("POSA_BUDGET_PRATICA_USD", "4.0"))
CHECK_BUDGET_USD = float(os.environ.get("POSA_BUDGET_CONTROLLO_USD", "0.6"))
MAX_ATTEMPTS = 3  # tentativi per passo prima dell'escalation a un operatore
ORCHESTRATOR_MAX_TURNS = 60

# Render
SUPERSAMPLING = int(os.environ.get("POSA_SUPERSAMPLING", "3"))  # 3x3 campioni per pixel
