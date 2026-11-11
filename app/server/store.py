"""Archivio del servizio (SQLite): clienti, chiavi API, piani, uso mensile, pratiche.

Un "cliente" è chi paga il canone: un sito che integra posa (canale `sito`, con chiavi API)
oppure un negozio Shopify (canale `shopify`, identificato dal dominio .myshopify.com).

Solo query parametrizzate. Nessun dato personale del cliente finale: niente nomi, email o
indirizzi; l'IP è conservato solo come hash con sale, per i limiti anti-abuso. Le chiavi
API sono conservate solo come hash SHA-256: la chiave in chiaro si vede una volta sola.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    channel TEXT NOT NULL,
    site TEXT NOT NULL,
    origins TEXT NOT NULL DEFAULT '[]',
    plan TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'attivo',
    daily_quota INTEGER NOT NULL,
    monthly_quota INTEGER,
    billing_ref TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    billing_token TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS api_keys (
    hash TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    kind TEXT NOT NULL,
    prefix TEXT NOT NULL,
    created_at REAL NOT NULL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS api_keys_tenant ON api_keys(tenant);
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    monthly_jobs INTEGER NOT NULL,
    surfaces TEXT NOT NULL,
    overage INTEGER NOT NULL DEFAULT 0,
    monthly_fee_eur REAL,
    extra_job_eur REAL
);
CREATE TABLE IF NOT EXISTS usage (
    tenant TEXT NOT NULL,
    month TEXT NOT NULL,
    created INTEGER NOT NULL DEFAULT 0,
    delivered INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (tenant, month)
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    shop TEXT NOT NULL,
    article TEXT NOT NULL,
    status TEXT NOT NULL,
    step TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    ip_hash TEXT NOT NULL,
    room TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    surface TEXT NOT NULL DEFAULT 'pavimento',
    zone TEXT NOT NULL DEFAULT 'automatica'
);
CREATE INDEX IF NOT EXISTS jobs_shop_time ON jobs(shop, created_at);
CREATE INDEX IF NOT EXISTS jobs_ip_time ON jobs(ip_hash, created_at);
CREATE TABLE IF NOT EXISTS nonces (nonce TEXT PRIMARY KEY, used_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

STATUSES = ("in_coda", "in_lavorazione", "pronto", "non_riuscito")
TENANT_STATUSES = ("attivo", "sospeso", "disdetto")
CHANNELS = ("sito", "shopify")
SURFACES = ("pavimento", "rivestimento")
KEY_KINDS = {"pubblica": "pk_live_", "segreta": "sk_live_"}

# Piani di partenza (modificabili da /admin/piani). I prezzi NON sono decisi qui: il canone
# si imposta dall'amministrazione; il servizio applica solo quote e superfici.
DEFAULT_PLANS = (
    ("prova", "Prova", 30, ("pavimento", "rivestimento"), False),
    ("base", "Base", 150, ("pavimento",), False),
    ("professionale", "Professionale", 600, ("pavimento", "rivestimento"), True),
    ("azienda", "Azienda", 3000, ("pavimento", "rivestimento"), True),
)
DEFAULT_PLAN = "prova"
# le pratiche fallite costano comunque (agenti): oltre questo multiplo della quota mensile
# il cliente viene fermato anche se le consegnate sono poche (foto inadatte, abusi)
ATTEMPTS_FACTOR = 2


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    monthly_jobs: int
    surfaces: tuple[str, ...]
    overage: bool  # oltre la quota: True = si continua (a consumo), False = ci si ferma
    monthly_fee_eur: float | None = None
    extra_job_eur: float | None = None


@dataclass(frozen=True)
class Tenant:
    id: str
    name: str
    channel: str
    site: str
    origins: tuple[str, ...]
    plan: str
    status: str
    daily_quota: int
    monthly_quota: int | None  # None = quella del piano
    billing_ref: str
    # codice di attivazione da mettere nel link di pagamento (client_reference_id): è
    # segreto e casuale, perché quel campo lo può scrivere chi paga
    billing_token: str = ""

    @property
    def active(self) -> bool:
        return self.status == "attivo"


@dataclass(frozen=True)
class Usage:
    month: str
    created: int
    delivered: int
    failed: int
    cost_usd: float


@dataclass(frozen=True)
class Job:
    id: str
    shop: str  # id del cliente (per Shopify: il dominio del negozio)
    article: str
    status: str
    step: str
    created_at: float
    updated_at: float
    cost_usd: float
    surface: str = "pavimento"
    zone: str = "automatica"


def month_of(ts: float | None = None) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts if ts is not None else time.time()))


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


_TENANT_COLS = (
    "id, name, channel, site, origins, plan, status, daily_quota, monthly_quota, billing_ref, "
    "billing_token"
)
_JOB_COLS = "id, shop, article, status, step, created_at, updated_at, cost_usd, surface, zone"


def _tenant(r: tuple[Any, ...]) -> Tenant:
    return Tenant(
        r[0], r[1], r[2], r[3], tuple(json.loads(r[4])), r[5], r[6], r[7], r[8], r[9], r[10]
    )


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._migrate()
            row = self._db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO meta(key, value) VALUES('salt', ?)", (secrets.token_hex(16),)
                )
                row = self._db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            self._salt = row[0]
            for pid, name, jobs, surfaces, overage in DEFAULT_PLANS:
                self._db.execute(
                    "INSERT OR IGNORE INTO plans(id, name, monthly_jobs, surfaces, overage) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (pid, name, jobs, json.dumps(surfaces), int(overage)),
                )

    def _migrate(self) -> None:
        """Archivi creati prima dei clienti/piani: colonne nuove e negozi -> clienti."""
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(jobs)")}
        if "surface" not in cols:
            self._db.execute(
                "ALTER TABLE jobs ADD COLUMN surface TEXT NOT NULL DEFAULT 'pavimento'"
            )
        if "zone" not in cols:
            self._db.execute("ALTER TABLE jobs ADD COLUMN zone TEXT NOT NULL DEFAULT 'automatica'")
        tcols = {r[1] for r in self._db.execute("PRAGMA table_info(tenants)")}
        if "billing_token" not in tcols:
            self._db.execute(
                "ALTER TABLE tenants ADD COLUMN billing_token TEXT NOT NULL DEFAULT ''"
            )
        old = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shops'"
        ).fetchone()
        if old:
            for domain, url, active, quota, created in self._db.execute(
                "SELECT domain, store_url, active, daily_quota, created_at FROM shops"
            ).fetchall():
                self._db.execute(
                    "INSERT OR IGNORE INTO tenants(id, name, channel, site, plan, status, "
                    "daily_quota, created_at) VALUES(?, ?, 'shopify', ?, ?, ?, ?, ?)",
                    (
                        domain,
                        domain,
                        url,
                        "professionale",
                        "attivo" if active else "sospeso",
                        quota,
                        created,
                    ),
                )
            self._db.execute("DROP TABLE shops")
        for (tid,) in self._db.execute("SELECT id FROM tenants WHERE billing_token=''").fetchall():
            self._db.execute(
                "UPDATE tenants SET billing_token=? WHERE id=?", (secrets.token_urlsafe(32), tid)
            )

    # ------------------------------------------------------------------- piani
    def plans(self) -> list[Plan]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, monthly_jobs, surfaces, overage, monthly_fee_eur, extra_job_eur "
                "FROM plans ORDER BY monthly_jobs"
            ).fetchall()
        return [
            Plan(r[0], r[1], r[2], tuple(json.loads(r[3])), bool(r[4]), r[5], r[6]) for r in rows
        ]

    def plan(self, plan_id: str) -> Plan | None:
        return next((p for p in self.plans() if p.id == plan_id), None)

    def upsert_plan(self, plan: Plan) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO plans(id, name, monthly_jobs, surfaces, overage, monthly_fee_eur, "
                "extra_job_eur) VALUES(?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "name=excluded.name, monthly_jobs=excluded.monthly_jobs, "
                "surfaces=excluded.surfaces, overage=excluded.overage, "
                "monthly_fee_eur=excluded.monthly_fee_eur, extra_job_eur=excluded.extra_job_eur",
                (
                    plan.id,
                    plan.name,
                    plan.monthly_jobs,
                    json.dumps(plan.surfaces),
                    int(plan.overage),
                    plan.monthly_fee_eur,
                    plan.extra_job_eur,
                ),
            )

    # ----------------------------------------------------------------- clienti
    def upsert_tenant(
        self,
        tenant_id: str,
        *,
        name: str,
        channel: str,
        site: str,
        daily_quota: int,
        origins: tuple[str, ...] = (),
        plan: str | None = None,
        monthly_quota: int | None = None,
    ) -> None:
        """Crea il cliente o ne aggiorna i dati. Un cliente esistente resta nel suo stato
        (un cliente sospeso per mancato pagamento non torna attivo ri-registrandolo)."""
        if channel not in CHANNELS:
            raise ValueError(f"canale sconosciuto: {channel}")
        with self._lock:
            exists = self._db.execute(
                "SELECT plan FROM tenants WHERE id=?", (tenant_id,)
            ).fetchone()
            if exists is None:
                self._db.execute(
                    "INSERT INTO tenants(id, name, channel, site, origins, plan, status, "
                    "daily_quota, monthly_quota, created_at, billing_token) "
                    "VALUES(?, ?, ?, ?, ?, ?, 'attivo', ?, ?, ?, ?)",
                    (
                        tenant_id,
                        name,
                        channel,
                        site,
                        json.dumps(origins),
                        plan or DEFAULT_PLAN,
                        daily_quota,
                        monthly_quota,
                        time.time(),
                        secrets.token_urlsafe(32),
                    ),
                )
            else:
                self._db.execute(
                    "UPDATE tenants SET name=?, site=?, origins=?, plan=?, daily_quota=?, "
                    "monthly_quota=? WHERE id=?",
                    (
                        name,
                        site,
                        json.dumps(origins),
                        plan or exists[0],
                        daily_quota,
                        monthly_quota,
                        tenant_id,
                    ),
                )

    def tenant(self, tenant_id: str) -> Tenant | None:
        with self._lock:
            r = self._db.execute(
                f"SELECT {_TENANT_COLS} FROM tenants WHERE id=?",  # nosec B608
                (tenant_id,),
            ).fetchone()
        return _tenant(r) if r else None

    def tenants(self) -> list[Tenant]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_TENANT_COLS} FROM tenants ORDER BY id"  # nosec B608
            ).fetchall()
        return [_tenant(r) for r in rows]

    def tenant_by_billing_ref(self, ref: str) -> Tenant | None:
        if not ref:
            return None
        with self._lock:
            r = self._db.execute(
                f"SELECT {_TENANT_COLS} FROM tenants WHERE billing_ref=?",  # nosec B608
                (ref,),
            ).fetchone()
        return _tenant(r) if r else None

    def tenant_by_billing_token(self, token: str) -> Tenant | None:
        """Cliente a cui appartiene un codice di attivazione (confronto esatto, indicizzato
        dal database: il codice è casuale e lungo, non si indovina)."""
        if len(token) < 32 or len(token) > 128:
            return None
        with self._lock:
            r = self._db.execute(
                f"SELECT {_TENANT_COLS} FROM tenants WHERE billing_token=?",  # nosec B608
                (token,),
            ).fetchone()
        return _tenant(r) if r else None

    def link_billing(self, tenant_id: str, ref: str) -> bool:
        """Collega il cliente Stripe SOLO se non ce n'è già uno: un secondo pagamento non
        può sostituire quello vero. Per cambiarlo serve l'amministratore (`set_billing`)."""
        with self._lock:
            taken = self._db.execute(
                "SELECT 1 FROM tenants WHERE billing_ref=? AND id<>?", (ref, tenant_id)
            ).fetchone()
            if taken:
                return False
            cur = self._db.execute(
                "UPDATE tenants SET billing_ref=? WHERE id=? AND billing_ref=''", (ref, tenant_id)
            )
        return cur.rowcount == 1

    def set_status(self, tenant_id: str, status: str) -> None:
        if status not in TENANT_STATUSES:
            raise ValueError(f"stato sconosciuto: {status}")
        with self._lock:
            self._db.execute("UPDATE tenants SET status=? WHERE id=?", (status, tenant_id))

    def set_billing(
        self, tenant_id: str, *, ref: str | None = None, plan: str | None = None
    ) -> None:
        with self._lock:
            if ref is not None:
                self._db.execute("UPDATE tenants SET billing_ref=? WHERE id=?", (ref, tenant_id))
            if plan is not None:
                self._db.execute("UPDATE tenants SET plan=? WHERE id=?", (plan, tenant_id))

    def delete_tenant(self, tenant_id: str) -> list[str]:
        """Cancella il cliente e ritorna gli id delle sue pratiche (per cancellarne i file)."""
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM jobs WHERE shop=?", (tenant_id,))]
            self._db.execute("DELETE FROM jobs WHERE shop=?", (tenant_id,))
            self._db.execute("DELETE FROM api_keys WHERE tenant=?", (tenant_id,))
            self._db.execute("DELETE FROM usage WHERE tenant=?", (tenant_id,))
            self._db.execute("DELETE FROM tenants WHERE id=?", (tenant_id,))
        return ids

    # ------------------------------------------------------------- chiavi API
    def create_key(self, tenant_id: str, kind: str) -> str:
        """Nuova chiave del cliente: ritorna la chiave in chiaro, che non è conservata."""
        key = KEY_KINDS[kind] + secrets.token_urlsafe(32)
        with self._lock:
            self._db.execute(
                "INSERT INTO api_keys(hash, tenant, kind, prefix, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (hash_key(key), tenant_id, kind, key[:14], time.time()),
            )
        return key

    def key_owner(self, key: str, kind: str) -> Tenant | None:
        """Cliente a cui appartiene una chiave valida e non revocata del tipo richiesto."""
        if not key.startswith(KEY_KINDS[kind]) or len(key) > 128:
            return None
        with self._lock:
            r = self._db.execute(
                "SELECT tenant FROM api_keys WHERE hash=? AND kind=? AND revoked_at IS NULL",
                (hash_key(key), kind),
            ).fetchone()
        return self.tenant(r[0]) if r else None

    def keys(self, tenant_id: str) -> list[dict[str, object]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT kind, prefix, created_at, revoked_at FROM api_keys WHERE tenant=? "
                "ORDER BY created_at",
                (tenant_id,),
            ).fetchall()
        return [
            {"tipo": r[0], "inizio": r[1], "creata": r[2], "revocata": r[3] is not None}
            for r in rows
        ]

    def revoke_key(self, tenant_id: str, prefix: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "UPDATE api_keys SET revoked_at=? WHERE tenant=? AND prefix=? "
                "AND revoked_at IS NULL",
                (time.time(), tenant_id, prefix),
            )
        return cur.rowcount

    # --------------------------------------------------------------------- uso
    def usage(self, tenant_id: str, month: str | None = None) -> Usage:
        m = month or month_of()
        with self._lock:
            r = self._db.execute(
                "SELECT created, delivered, failed, cost_usd FROM usage WHERE tenant=? AND month=?",
                (tenant_id, m),
            ).fetchone()
        return Usage(m, *r) if r else Usage(m, 0, 0, 0, 0.0)

    def usage_history(self, tenant_id: str, limit: int = 12) -> list[Usage]:
        with self._lock:
            rows = self._db.execute(
                "SELECT month, created, delivered, failed, cost_usd FROM usage WHERE tenant=? "
                "ORDER BY month DESC LIMIT ?",
                (tenant_id, limit),
            ).fetchall()
        return [Usage(*r) for r in rows]

    def _bump(self, tenant_id: str, month: str, column: str, cost: float = 0.0) -> None:
        # `column` arriva solo da costanti di questo modulo
        self._db.execute(
            "INSERT INTO usage(tenant, month) VALUES(?, ?) ON CONFLICT(tenant, month) DO NOTHING",
            (tenant_id, month),
        )
        sql = (
            f"UPDATE usage SET {column}={column}+1, cost_usd=cost_usd+? WHERE tenant=? AND month=?"  # nosec B608
        )
        self._db.execute(sql, (cost, tenant_id, month))

    def monthly_allowance(self, tenant: Tenant) -> tuple[int, bool]:
        """(prove comprese nel mese, si può andare oltre a consumo)."""
        plan = self.plan(tenant.plan)
        base = tenant.monthly_quota if tenant.monthly_quota is not None else None
        if plan is None:
            return (base or 0, False)
        return (base if base is not None else plan.monthly_jobs, plan.overage)

    def quota_problem(self, tenant: Tenant) -> str | None:
        """None se il cliente può avviare un'altra prova questo mese, altrimenti il motivo."""
        with self._lock:
            return self._quota_problem(tenant)

    def _quota_problem(self, tenant: Tenant) -> str | None:
        """Come `quota_problem`, da chiamare con il lucchetto già preso."""
        plan = self._db.execute(
            "SELECT monthly_jobs, overage FROM plans WHERE id=?", (tenant.plan,)
        ).fetchone()
        overage = bool(plan[1]) if plan else False
        allowed = tenant.monthly_quota
        if allowed is None:
            allowed = int(plan[0]) if plan else 0
        r = self._db.execute(
            "SELECT created, delivered, failed FROM usage WHERE tenant=? AND month=?",
            (tenant.id, month_of()),
        ).fetchone()
        created, delivered, failed = r if r else (0, 0, 0)
        if overage:
            return None
        if created >= ATTEMPTS_FACTOR * max(allowed, 1):
            return "troppe prove non riuscite questo mese: contatta l'assistenza"
        if delivered + (created - delivered - failed) >= allowed:
            return "prove del mese esaurite per questo piano"
        return None

    # ---------------------------------------------------------------- pratiche
    def ip_hash(self, ip: str) -> str:
        return hashlib.sha256(f"{self._salt}:{ip}".encode()).hexdigest()[:32]

    def nonce_used(self, nonce: str) -> bool:
        with self._lock:
            r = self._db.execute("SELECT 1 FROM nonces WHERE nonce=?", (nonce,)).fetchone()
        return r is not None

    def use_nonce(self, nonce: str) -> bool:
        """Un permesso di caricamento vale una volta sola."""
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO nonces(nonce, used_at) VALUES(?, ?)", (nonce, time.time())
                )
            except sqlite3.IntegrityError:
                return False
        return True

    def count_jobs(
        self, *, shop: str | None = None, ip_hash: str | None = None, since: float
    ) -> int:
        with self._lock:
            if shop is not None:
                q = "SELECT COUNT(*) FROM jobs WHERE shop=? AND created_at>=?"
                return int(self._db.execute(q, (shop, since)).fetchone()[0])
            q = "SELECT COUNT(*) FROM jobs WHERE ip_hash=? AND created_at>=?"
            return int(self._db.execute(q, (ip_hash, since)).fetchone()[0])

    def pending_jobs(self) -> int:
        with self._lock:
            q = "SELECT COUNT(*) FROM jobs WHERE status IN ('in_coda', 'in_lavorazione')"
            return int(self._db.execute(q).fetchone()[0])

    def create_job(
        self,
        shop: str,
        article: str,
        ip_hash: str,
        room: str | None,
        surface: str = "pavimento",
        zone: str = "automatica",
    ) -> str:
        with self._lock:
            return self._insert_job(shop, article, ip_hash, room, surface, zone)

    def _insert_job(
        self, shop: str, article: str, ip_hash: str, room: str | None, surface: str, zone: str
    ) -> str:
        job_id = secrets.token_urlsafe(24)  # 192 bit: non si indovina
        now = time.time()
        self._db.execute(
            "INSERT INTO jobs(id, shop, article, status, created_at, updated_at, ip_hash, "
            "room, surface, zone) VALUES(?, ?, ?, 'in_coda', ?, ?, ?, ?, ?, ?)",
            (job_id, shop, article, now, now, ip_hash, room, surface, zone),
        )
        self._bump(shop, month_of(now), "created")
        return job_id

    def admit_job(
        self,
        tenant: Tenant,
        article: str,
        ip_hash: str,
        room: str | None,
        surface: str,
        zone: str,
        *,
        ip_limit: int | None,
        max_queue: int,
        nonce: str | None,
    ) -> tuple[str | None, int, str]:
        """Tutti i limiti e la creazione della pratica in un'unica operazione.

        Controllare e poi creare in due passi lasciava una finestra: molte richieste in
        parallelo passavano tutte il controllo e superavano quota mensile, limite
        giornaliero e limite per indirizzo. Qui nessun'altra richiesta può infilarsi in
        mezzo. Ritorna (id pratica, 201, "") oppure (None, codice HTTP, motivo).
        """
        now = time.time()
        with self._lock:
            row = self._db.execute("SELECT status FROM tenants WHERE id=?", (tenant.id,)).fetchone()
            if row is None or row[0] != "attivo":
                return None, 403, "abbonamento non attivo"
            if reason := self._quota_problem(tenant):
                return None, 402, reason
            q = "SELECT COUNT(*) FROM jobs WHERE shop=? AND created_at>=?"
            if self._db.execute(q, (tenant.id, now - 86400)).fetchone()[0] >= tenant.daily_quota:
                return None, 429, "limite giornaliero di prove raggiunto"
            if ip_limit is not None:
                q = "SELECT COUNT(*) FROM jobs WHERE ip_hash=? AND created_at>=?"
                if self._db.execute(q, (ip_hash, now - 3600)).fetchone()[0] >= ip_limit:
                    return None, 429, "troppe prove: riprova più tardi"
            q = "SELECT COUNT(*) FROM jobs WHERE status IN ('in_coda', 'in_lavorazione')"
            if self._db.execute(q).fetchone()[0] >= max_queue:
                return None, 503, "servizio molto richiesto: riprova tra qualche minuto"
            if nonce is not None:
                try:
                    self._db.execute(
                        "INSERT INTO nonces(nonce, used_at) VALUES(?, ?)", (nonce, now)
                    )
                except sqlite3.IntegrityError:
                    return None, 401, "permesso già usato"
            return self._insert_job(tenant.id, article, ip_hash, room, surface, zone), 201, ""

    def delete_job(self, job_id: str) -> None:
        """Toglie la pratica dall'archivio (l'uso del mese resta: serve a fatturare)."""
        with self._lock:
            self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def job(self, job_id: str) -> Job | None:
        with self._lock:
            r = self._db.execute(
                f"SELECT {_JOB_COLS} FROM jobs WHERE id=?",  # nosec B608
                (job_id,),
            ).fetchone()
        return Job(*r) if r else None

    def job_room(self, job_id: str) -> str | None:
        with self._lock:
            r = self._db.execute("SELECT room FROM jobs WHERE id=?", (job_id,)).fetchone()
        return r[0] if r else None

    def update_job(
        self, job_id: str, *, status: str | None = None, step: str | None = None,
        cost_usd: float | None = None,
    ) -> None:  # fmt: skip
        sets: list[str] = ["updated_at=?"]
        args: list[float | str] = [time.time()]
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"stato sconosciuto: {status}")
            sets.append("status=?")
            args.append(status)
        if step is not None:
            sets.append("step=?")
            args.append(step)
        if cost_usd is not None:
            sets.append("cost_usd=?")
            args.append(cost_usd)
        with self._lock:
            before = self._db.execute(
                "SELECT shop, status, created_at FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            # solo nomi di colonna fissi nel testo SQL: i valori passano come parametri
            sql = f"UPDATE jobs SET {', '.join(sets)} WHERE id=?"  # nosec B608
            self._db.execute(sql, (*args, job_id))
            # l'uso del mese si chiude una volta sola, quando la pratica finisce
            done = {"pronto": "delivered", "non_riuscito": "failed"}
            if before and status in done and before[1] not in done:
                self._bump(before[0], month_of(before[2]), done[status], cost_usd or 0.0)

    def queued_job_ids(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM jobs WHERE status='in_coda' ORDER BY created_at"
            ).fetchall()
        return [r[0] for r in rows]

    def interrupted_job_ids(self) -> list[str]:
        """Pratiche rimaste "in lavorazione" da un riavvio del servizio."""
        with self._lock:
            rows = self._db.execute("SELECT id FROM jobs WHERE status='in_lavorazione'").fetchall()
        return [r[0] for r in rows]

    def expired_jobs(self, before: float) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT id FROM jobs WHERE created_at<?", (before,)).fetchall()
            self._db.execute("DELETE FROM jobs WHERE created_at<?", (before,))
            self._db.execute("DELETE FROM nonces WHERE used_at<?", (before,))
        return [r[0] for r in rows]
