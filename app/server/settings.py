"""Configurazione del servizio, solo da variabili d'ambiente (nessun segreto nel codice)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("POSA_DATI", "dati")))
    # segreto dell'app Shopify (Partner Dashboard): firma App Proxy e webhook
    shopify_secret: str = field(default_factory=lambda: os.environ.get("SHOPIFY_API_SECRET", ""))
    # token per le API di amministrazione (registrazione negozi)
    admin_token: str = field(default_factory=lambda: os.environ.get("POSA_ADMIN_TOKEN", ""))
    # chiave per firmare i permessi di caricamento (generata se assente: vale fino al riavvio)
    upload_key: str = field(
        default_factory=lambda: os.environ.get("POSA_CHIAVE_CARICAMENTI") or secrets.token_hex(32)
    )
    # dietro un reverse proxy fidato (es. Caddy/Nginx) che imposta X-Forwarded-For
    trust_proxy: bool = field(default_factory=lambda: os.environ.get("POSA_PROXY_FIDATO") == "1")
    public_url: str = field(default_factory=lambda: os.environ.get("POSA_URL_PUBBLICO", ""))
    max_upload_mb: int = field(default_factory=lambda: _int("POSA_MAX_FOTO_MB", 15))
    max_pixels: int = field(default_factory=lambda: _int("POSA_MAX_PIXEL", 50_000_000))
    jobs_per_shop_day: int = field(
        default_factory=lambda: _int("POSA_PRATICHE_NEGOZIO_GIORNO", 100)
    )
    jobs_per_ip_hour: int = field(default_factory=lambda: _int("POSA_PRATICHE_IP_ORA", 6))
    max_concurrent: int = field(default_factory=lambda: _int("POSA_PRATICHE_IN_PARALLELO", 2))
    max_queue: int = field(default_factory=lambda: _int("POSA_CODA_MASSIMA", 50))
    job_timeout_s: int = field(default_factory=lambda: _int("POSA_TIMEOUT_PRATICA_S", 900))
    retention_days: int = field(default_factory=lambda: _int("POSA_CONSERVAZIONE_GIORNI", 7))
    token_ttl_s: int = field(default_factory=lambda: _int("POSA_DURATA_PERMESSO_S", 900))
    proxy_max_age_s: int = field(default_factory=lambda: _int("POSA_ETA_FIRMA_PROXY_S", 300))

    def scrub_environment(self) -> None:
        """Toglie i segreti del servizio dall'ambiente: i processi figli (le sessioni degli
        agenti) non li ereditano, quindi non sono raggiungibili nemmeno in teoria."""
        for name in ("SHOPIFY_API_SECRET", "POSA_ADMIN_TOKEN", "POSA_CHIAVE_CARICAMENTI"):
            os.environ.pop(name, None)

    def check(self) -> list[str]:
        """Problemi di configurazione che impediscono di andare in produzione."""
        problems = []
        if len(self.shopify_secret) < 16:
            problems.append("SHOPIFY_API_SECRET mancante: senza, nessuna richiesta è accettata")
        if len(self.admin_token) < 24:
            problems.append("POSA_ADMIN_TOKEN mancante o troppo corto (minimo 24 caratteri)")
        if not os.environ.get("POSA_CHIAVE_CARICAMENTI"):
            problems.append("POSA_CHIAVE_CARICAMENTI assente: i permessi scadono a ogni riavvio")
        return problems
