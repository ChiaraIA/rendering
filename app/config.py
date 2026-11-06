"""Soglie e default in un unico punto."""

from __future__ import annotations

# Ingest
MAX_PIXELS = 80_000_000
ALLOWED_INPUT_FORMATS = frozenset({"JPEG", "PNG", "TIFF", "WEBP"})

# Maschera
CONFIRM_PREFIX_MIN_LEN = 8

# Export
PNG_COMPRESS_LEVEL = 6
PREVIEW_MAX_SIDE = 1600
PREVIEW_JPEG_QUALITY = 90

# Report
REPORT_SCHEMA_VERSION = "1.0"
JOB_SCHEMA_VERSION = "1.0"
