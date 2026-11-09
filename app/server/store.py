"""Archivio del servizio (SQLite): negozi, pratiche, limiti di frequenza.

Solo query parametrizzate. Nessun dato personale del cliente finale: niente nomi, email o
indirizzi; l'IP è conservato solo come hash con sale, per i limiti anti-abuso.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS shops (
    domain TEXT PRIMARY KEY,
    store_url TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    daily_quota INTEGER NOT NULL,
    created_at REAL NOT NULL
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
    cost_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_shop_time ON jobs(shop, created_at);
CREATE INDEX IF NOT EXISTS jobs_ip_time ON jobs(ip_hash, created_at);
CREATE TABLE IF NOT EXISTS nonces (nonce TEXT PRIMARY KEY, used_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

STATUSES = ("in_coda", "in_lavorazione", "pronto", "non_riuscito")


@dataclass(frozen=True)
class Shop:
    domain: str
    store_url: str
    active: bool
    daily_quota: int


@dataclass(frozen=True)
class Job:
    id: str
    shop: str
    article: str
    status: str
    step: str
    created_at: float
    updated_at: float
    cost_usd: float


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            row = self._db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO meta(key, value) VALUES('salt', ?)", (secrets.token_hex(16),)
                )
                row = self._db.execute("SELECT value FROM meta WHERE key='salt'").fetchone()
            self._salt = row[0]

    # ------------------------------------------------------------------ negozi
    def upsert_shop(self, domain: str, store_url: str, daily_quota: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO shops(domain, store_url, active, daily_quota, created_at) "
                "VALUES(?, ?, 1, ?, ?) ON CONFLICT(domain) DO UPDATE SET "
                "store_url=excluded.store_url, daily_quota=excluded.daily_quota, active=1",
                (domain, store_url, daily_quota, time.time()),
            )

    def shop(self, domain: str) -> Shop | None:
        with self._lock:
            r = self._db.execute(
                "SELECT domain, store_url, active, daily_quota FROM shops WHERE domain=?",
                (domain,),
            ).fetchone()
        return Shop(r[0], r[1], bool(r[2]), r[3]) if r else None

    def shops(self) -> list[Shop]:
        with self._lock:
            rows = self._db.execute(
                "SELECT domain, store_url, active, daily_quota FROM shops ORDER BY domain"
            ).fetchall()
        return [Shop(r[0], r[1], bool(r[2]), r[3]) for r in rows]

    def set_active(self, domain: str, active: bool) -> None:
        with self._lock:
            self._db.execute("UPDATE shops SET active=? WHERE domain=?", (int(active), domain))

    def delete_shop(self, domain: str) -> list[str]:
        """Cancella il negozio e ritorna gli id delle sue pratiche (per cancellarne i file)."""
        with self._lock:
            ids = [r[0] for r in self._db.execute("SELECT id FROM jobs WHERE shop=?", (domain,))]
            self._db.execute("DELETE FROM jobs WHERE shop=?", (domain,))
            self._db.execute("DELETE FROM shops WHERE domain=?", (domain,))
        return ids

    # ---------------------------------------------------------------- pratiche
    def ip_hash(self, ip: str) -> str:
        return hashlib.sha256(f"{self._salt}:{ip}".encode()).hexdigest()[:32]

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

    def create_job(self, shop: str, article: str, ip_hash: str, room: str | None) -> str:
        job_id = secrets.token_urlsafe(24)  # 192 bit: non si indovina
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs(id, shop, article, status, created_at, updated_at, ip_hash, room)"
                " VALUES(?, ?, ?, 'in_coda', ?, ?, ?, ?)",
                (job_id, shop, article, now, now, ip_hash, room),
            )
        return job_id

    def job(self, job_id: str) -> Job | None:
        with self._lock:
            r = self._db.execute(
                "SELECT id, shop, article, status, step, created_at, updated_at, cost_usd "
                "FROM jobs WHERE id=?",
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
            # solo nomi di colonna fissi nel testo SQL: i valori passano come parametri
            sql = f"UPDATE jobs SET {', '.join(sets)} WHERE id=?"  # nosec B608
            self._db.execute(sql, (*args, job_id))

    def queued_job_ids(self) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id FROM jobs WHERE status='in_coda' ORDER BY created_at"
            ).fetchall()
        return [r[0] for r in rows]

    def expired_jobs(self, before: float) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT id FROM jobs WHERE created_at<?", (before,)).fetchall()
            self._db.execute("DELETE FROM jobs WHERE created_at<?", (before,))
            self._db.execute("DELETE FROM nonces WHERE used_at<?", (before,))
        return [r[0] for r in rows]
