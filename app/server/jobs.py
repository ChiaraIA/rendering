"""Esecuzione delle pratiche: coda, worker in parallelo, risultati, pulizia.

Il cliente vede solo uno stato e un passo da un elenco fisso: mai testo prodotto dagli
agenti (che potrebbe essere stato manipolato da una foto malevola).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.agents.security import redact
from app.agents.state import next_step
from app.catalog.loader import Catalog
from app.server.settings import Settings
from app.server.store import Store
from app.storage import exporter
from app.storage.images import jpeg_bytes

log = logging.getLogger("posa.server")

# passo della procedura -> messaggio per il cliente (elenco chiuso)
STEPS = {
    "maschera": "Riconosco il pavimento nella foto",
    "geometria": "Calcolo la prospettiva della stanza",
    "materiale": "Preparo il rivestimento scelto",
    "luce": "Regolo luci e ombre",
    "posa": "Poso il pavimento",
    "consegna": "Ultimi controlli di qualità",
    "completato": "Pronto",
    "fermo": "Serve un controllo manuale",
}
OUTPUT_SIDE = 2400

Runner = Callable[
    [Path, Path, str, tuple[float, float] | None, Catalog, Callable[[str], None], float],
    tuple[bool, float],
]


def shop_dir(settings: Settings, domain: str) -> Path:
    return settings.data_dir / "negozi" / domain


def job_dir(settings: Settings, job_id: str) -> Path:
    return settings.data_dir / "pratiche" / job_id


def agent_runner(
    photo: Path,
    work: Path,
    article: str,
    room_m: tuple[float, float] | None,
    catalog: Catalog,
    on_step: Callable[[str], None],
    timeout_s: float,
) -> tuple[bool, float]:
    """Esecuzione reale: sistema multiagente, interrotto se supera il tempo massimo."""
    from app.agents.orchestrator import run_job, start

    ws = start(photo, work, article, None, room_m)
    ws.listeners.append(lambda _line: on_step(next_step(ws.state).split(":")[0]))
    outcome = asyncio.run(asyncio.wait_for(run_job(ws, catalog), timeout_s))
    return outcome.delivered, outcome.cost_usd


class JobQueue:
    def __init__(self, settings: Settings, store: Store, runner: Runner | None = None) -> None:
        self.settings = settings
        self.store = store
        self.runner = runner or agent_runner
        self._q: queue.Queue[str] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        for job_id in self.store.queued_job_ids():  # riprende quelle rimaste in coda
            self._q.put(job_id)
        for k in range(self.settings.max_concurrent):
            t = threading.Thread(target=self._work, name=f"posa-worker-{k}", daemon=True)
            t.start()
            self._threads.append(t)
        cleaner = threading.Thread(target=self._clean_loop, name="posa-pulizia", daemon=True)
        cleaner.start()
        self._threads.append(cleaner)

    def stop(self) -> None:
        self._stop.set()

    def submit(self, job_id: str) -> None:
        self._q.put(job_id)

    def run_now(self, job_id: str) -> None:
        """Esegue subito una pratica (usato dai test)."""
        self._run(job_id)

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(job_id)
            finally:
                self._q.task_done()

    def _run(self, job_id: str) -> None:
        job = self.store.job(job_id)
        if job is None or job.status != "in_coda":
            return
        d = job_dir(self.settings, job_id)
        photo = next(iter(sorted((d / "input").glob("foto.*"))), None)
        catalog_file = shop_dir(self.settings, job.shop) / "catalogo" / "catalog.json"
        if photo is None or not catalog_file.is_file():
            self.store.update_job(job_id, status="non_riuscito", step="fermo")
            return
        self.store.update_job(job_id, status="in_lavorazione", step="maschera")
        room = self.store.job_room(job_id)
        room_m = (float(room.split("x")[0]), float(room.split("x")[1])) if room else None

        def on_step(step: str) -> None:
            if step in STEPS:
                self.store.update_job(job_id, step=step)

        try:
            catalog = Catalog.load(catalog_file)
            delivered, cost = self.runner(
                photo,
                d / "pratica",
                job.article,
                room_m,
                catalog,
                on_step,
                float(self.settings.job_timeout_s),
            )
        except Exception as exc:  # la pratica fallisce, il servizio no
            log.warning("pratica %s non riuscita: %s", job_id, redact(str(exc), 300))
            self.store.update_job(job_id, status="non_riuscito", step="fermo")
            return
        if delivered and self._publish(d):
            self.store.update_job(job_id, status="pronto", step="completato", cost_usd=cost)
        else:
            self.store.update_job(job_id, status="non_riuscito", step="fermo", cost_usd=cost)

    def _publish(self, d: Path) -> bool:
        """Risultati per il cliente: JPEG ricodificati, senza metadati della foto originale."""
        final = exporter.master_path(d / "pratica" / "consegna")
        before_after = d / "pratica" / "consegna" / "prima_dopo.jpg"
        if not final.is_file():
            return False
        out = d / "pubblico"
        out.mkdir(exist_ok=True)
        for src, name in ((final, "risultato.jpg"), (before_after, "prima_dopo.jpg")):
            if src.is_file():
                with Image.open(src) as im:
                    im.load()
                    im.thumbnail((OUTPUT_SIDE, OUTPUT_SIDE))
                    (out / name).write_bytes(jpeg_bytes(im, quality=90))
        return True

    def _clean_loop(self) -> None:
        while not self._stop.wait(3600):
            self.clean()

    def clean(self) -> int:
        """Cancella pratiche e foto più vecchie del periodo di conservazione."""
        before = time.time() - self.settings.retention_days * 86400
        ids = self.store.expired_jobs(before)
        for job_id in ids:
            shutil.rmtree(job_dir(self.settings, job_id), ignore_errors=True)
        return len(ids)
