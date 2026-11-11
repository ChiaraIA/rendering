"""Configurazione del servizio, solo da variabili d'ambiente (nessun segreto nel codice)."""
# i limiti per cliente (quota mensile, superfici) stanno nei piani: vedi store.py

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
    # segreto dei webhook di Stripe (whsec_...): pagamenti -> attiva/sospende i clienti
    stripe_secret: str = field(default_factory=lambda: os.environ.get("STRIPE_WEBHOOK_SECRET", ""))
    # token per le API di amministrazione (clienti, piani, chiavi)
    admin_token: str = field(default_factory=lambda: os.environ.get("POSA_ADMIN_TOKEN", ""))
    # chiave per firmare i permessi di caricamento (generata se assente: vale fino al riavvio)
    upload_key: str = field(
        default_factory=lambda: os.environ.get("POSA_CHIAVE_CARICAMENTI") or secrets.token_hex(32)
    )
    # la chiave arriva dalla configurazione (e non è stata generata a questo avvio)
    upload_key_configured: bool = field(
        default_factory=lambda: bool(os.environ.get("POSA_CHIAVE_CARICAMENTI"))
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
    # dopo la consegna restano solo i risultati: foto originale e immagini di lavoro vengono
    # cancellate subito. Con POSA_TIENI_LAVORAZIONE=1 restano fino alla scadenza (debug).
    keep_work_files: bool = field(
        default_factory=lambda: os.environ.get("POSA_TIENI_LAVORAZIONE") == "1"
    )

    def scrub_environment(self) -> None:
        """Toglie i segreti del servizio dall'ambiente del processo.

        I processi figli (le sessioni degli agenti) non li ereditano. In più il valore viene
        azzerato nella memoria d'origine: altrimenti resterebbe leggibile in
        /proc/<pid>/environ da qualunque processo dello stesso utente, anche dopo la
        rimozione. Resta solo ANTHROPIC_API_KEY, che agli agenti serve per funzionare."""
        for name in (
            "SHOPIFY_API_SECRET",
            "POSA_ADMIN_TOKEN",
            "POSA_CHIAVE_CARICAMENTI",
            "STRIPE_WEBHOOK_SECRET",
        ):
            _wipe_env(name)

    def check(self) -> list[str]:
        """Problemi di configurazione che impediscono di andare in produzione."""
        problems = []
        if len(self.shopify_secret) < 16:
            problems.append("SHOPIFY_API_SECRET mancante: i negozi Shopify non possono collegarsi")
        if not self.public_url.startswith("https://"):
            problems.append("POSA_URL_PUBBLICO mancante o non https: il widget non può caricare")
        if len(self.stripe_secret) < 16:
            problems.append(
                "STRIPE_WEBHOOK_SECRET assente: i pagamenti non attivano/sospendono i clienti "
                "(si può fare a mano da /admin/clienti/<id>/stato)"
            )
        if len(self.admin_token) < 24:
            problems.append("POSA_ADMIN_TOKEN mancante o troppo corto (minimo 24 caratteri)")
        if len(self.upload_key) < 32:
            problems.append(
                "POSA_CHIAVE_CARICAMENTI troppo corta (minimo 32 caratteri casuali): i "
                "permessi di caricamento si potrebbero falsificare"
            )
        elif not self.upload_key_configured:
            problems.append("POSA_CHIAVE_CARICAMENTI assente: i permessi scadono a ogni riavvio")
        return problems


def _wipe_env(name: str) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.getenv.restype = ctypes.c_void_p
        libc.getenv.argtypes = [ctypes.c_char_p]
        where = libc.getenv(name.encode())
        if where:  # sovrascrive il valore dov'è, poi toglie la variabile
            ctypes.memset(where, ord("x"), len(value.encode()))
    except Exception:  # nosec B110 - sistemi senza libc compatibile: resta la rimozione
        pass
    os.environ.pop(name, None)
