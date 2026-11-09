"""API del servizio (FastAPI).

Percorsi:
- /proxy/...   chiamati dal negozio tramite App Proxy di Shopify (firmati da Shopify);
- /api/...     chiamati dal browser del cliente (permesso di caricamento a tempo, CORS
               limitato ai negozi registrati);
- /webhooks/shopify   notifiche di Shopify (firma HMAC);
- /admin/...   amministrazione (token segreto).
"""

from __future__ import annotations

import hmac
import io
import json
import logging
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from app.server import shopify_auth as auth
from app.server.jobs import STEPS, JobQueue, Runner, job_dir, shop_dir
from app.server.limits import BodyLimitMiddleware
from app.server.settings import Settings
from app.server.store import Store

log = logging.getLogger("posa.server")

ALLOWED_FORMATS = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "HEIF": "heic"}
ROOM = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)x(\d{1,2}(?:\.\d{1,2})?)$")
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{32}$")
PUBLIC_FILES = {"risultato.jpg", "prima_dopo.jpg"}
HANDLE = re.compile(r"^[a-z0-9][a-z0-9-]{0,254}$")
VARIANT = re.compile(r"^\d{1,20}$")
MAX_WEBHOOK_BYTES = 1_000_000

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; img-src 'self'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Cache-Control": "no-store",
}


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"errore": msg}, status_code=status)


def create_app(
    settings: Settings | None = None, runner: Runner | None = None, start_workers: bool = True
) -> FastAPI:
    st = settings or Settings()
    for problem in st.check():
        log.warning("configurazione: %s", problem)
    store = Store(st.data_dir / "servizio.db")
    jobs = JobQueue(st, store, runner)
    Image.MAX_IMAGE_PIXELS = st.max_pixels  # oltre: "bomba di decompressione", rifiutata
    sync_timers: dict[str, threading.Timer] = {}
    sync_lock = threading.Lock()

    app = FastAPI(title="posa", docs_url=None, redoc_url=None, openapi_url=None)
    # limite sul corpo applicato mentre arriva (anche senza Content-Length)
    app.add_middleware(BodyLimitMiddleware, max_bytes=st.max_upload_mb * 1024 * 1024 + 64 * 1024)
    app.state.store, app.state.jobs, app.state.settings = store, jobs, st
    if start_workers:
        jobs.start()

    # ----------------------------------------------------------------- middleware
    def allowed_origin(origin: str) -> bool:
        if not origin:
            return False
        host = (urlparse(origin).hostname or "").lower()
        if urlparse(origin).scheme != "https":
            return False
        for shop in store.shops():
            if not shop.active:
                continue
            if host == shop.domain or host == (urlparse(shop.store_url).hostname or ""):
                return True
        return False

    @app.middleware("http")
    async def harden(request: Request, call_next: Any) -> Response:
        origin = request.headers.get("origin", "")
        is_api = request.url.path.startswith("/api/")
        if request.method == "OPTIONS" and is_api:
            resp: Response = Response(status_code=204)
        else:
            length = request.headers.get("content-length")
            limit = st.max_upload_mb * 1024 * 1024 + 64 * 1024
            if length and (not length.isdigit() or int(length) > limit):
                resp = _err(413, "richiesta troppo grande")
            else:
                resp = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        if is_api and allowed_origin(origin):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Posa-Permesso"
            resp.headers["Access-Control-Max-Age"] = "600"
        return resp

    def client_ip(request: Request) -> str:
        if st.trust_proxy:  # solo dietro un proxy fidato che riscrive l'intestazione
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return fwd.split(",")[-1].strip()
        return request.client.host if request.client else "?"

    def proxy_shop(request: Request) -> str | None:
        params = parse_qs(request.url.query, keep_blank_values=True)
        shop = auth.verify_proxy(params, st.shopify_secret, st.proxy_max_age_s)
        if shop is None:
            return None
        s = store.shop(shop)
        return shop if s and s.active else None

    def shop_index(domain: str) -> dict[str, Any]:
        path = shop_dir(st, domain) / "catalogo" / "index.json"
        if not path.is_file():
            return {"handles": {}, "variants": {}}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    # ------------------------------------------------------------------- pubblico
    @app.get("/salute")
    def health() -> dict[str, str]:
        return {"stato": "ok"}

    # --------------------------------------------------------------- App Proxy
    @app.get("/proxy/sessione")
    def session(request: Request, prodotto: str = "", variante: str = "") -> Response:
        """Permesso di caricamento per un articolo, richiesto dalla pagina prodotto."""
        shop = proxy_shop(request)
        if shop is None:
            return _err(401, "richiesta non firmata da Shopify")
        if not HANDLE.match(prodotto) or (variante and not VARIANT.match(variante)):
            return _err(400, "articolo non valido")
        index = shop_index(shop)
        article = index["variants"].get(variante) if variante else None
        if article is None:
            options = index["handles"].get(prodotto) or []
            article = options[0] if options else None
        if article is None:
            return _err(404, "articolo non disponibile per la prova")
        token = auth.make_upload_token(
            st.upload_key, shop, article, st.token_ttl_s, secrets.token_hex(12)
        )
        base = st.public_url.rstrip("/")
        return JSONResponse(
            {"permesso": token, "carica": f"{base}/api/pratiche", "scade_in_s": st.token_ttl_s}
        )

    # ---------------------------------------------------------- API del browser
    @app.post("/api/pratiche")
    async def create_job(request: Request) -> Response:
        # il permesso si verifica PRIMA di leggere il corpo: senza, il server non legge nulla
        token = request.headers.get("x-posa-permesso", "")
        data = auth.read_upload_token(st.upload_key, token)
        if data is None:
            return _err(401, "permesso non valido o scaduto")
        shop = store.shop(data["shop"])
        if shop is None or not shop.active:
            return _err(403, "negozio non attivo")
        form = await request.form(max_files=1, max_fields=3)
        room = str(form.get("stanza", "") or "").strip().lower().replace(",", ".")
        if room:
            m = ROOM.match(room)
            if not m or not all(1.0 <= float(v) <= 50.0 for v in m.groups()):
                return _err(400, "misure della stanza non valide (es. 4.2x3.6)")
        upload = form.get("foto")
        if upload is None or isinstance(upload, str):
            return _err(400, "manca la foto")
        raw = await upload.read(st.max_upload_mb * 1024 * 1024 + 1)
        if len(raw) > st.max_upload_mb * 1024 * 1024:
            return _err(413, f"foto troppo grande (massimo {st.max_upload_mb} MB)")
        ext = _image_ext(raw)
        if ext is None:
            return _err(415, "il file non è una foto supportata (JPEG, HEIC, PNG, WebP)")
        # limiti anti-abuso: per negozio al giorno, per IP all'ora, coda globale
        now = time.time()
        if store.count_jobs(shop=shop.domain, since=now - 86400) >= shop.daily_quota:
            return _err(429, "limite giornaliero di prove raggiunto per questo negozio")
        iph = store.ip_hash(client_ip(request))
        if store.count_jobs(ip_hash=iph, since=now - 3600) >= st.jobs_per_ip_hour:
            return _err(429, "troppe prove: riprova più tardi")
        if store.pending_jobs() >= st.max_queue:
            return _err(503, "servizio molto richiesto: riprova tra qualche minuto")
        if not store.use_nonce(data["nonce"]):
            return _err(401, "permesso già usato")
        job_id = store.create_job(shop.domain, data["article"], iph, room or None)
        d = job_dir(st, job_id) / "input"
        d.mkdir(parents=True)
        (d / f"foto.{ext}").write_bytes(raw)
        jobs.submit(job_id)
        return JSONResponse({"pratica": job_id, "stato": "in_coda"}, status_code=201)

    @app.get("/api/pratiche/{job_id}")
    def job_status(job_id: str) -> Response:
        if not JOB_ID.match(job_id):
            return _err(404, "pratica inesistente")
        job = store.job(job_id)
        if job is None:
            return _err(404, "pratica inesistente")
        body: dict[str, Any] = {
            "stato": job.status,
            "messaggio": STEPS.get(job.step, "In attesa"),
        }
        if job.status == "pronto":
            body["risultato"] = f"/api/pratiche/{job_id}/risultato.jpg"
            body["prima_dopo"] = f"/api/pratiche/{job_id}/prima_dopo.jpg"
        return JSONResponse(body)

    @app.get("/api/pratiche/{job_id}/{name}")
    def job_file(job_id: str, name: str) -> Response:
        if not JOB_ID.match(job_id) or name not in PUBLIC_FILES:
            return _err(404, "file inesistente")
        job = store.job(job_id)
        path = job_dir(st, job_id) / "pubblico" / name
        if job is None or job.status != "pronto" or not path.is_file():
            return _err(404, "file inesistente")
        return FileResponse(path, media_type="image/jpeg")

    # ----------------------------------------------------------------- webhook
    @app.post("/webhooks/shopify")
    async def webhook(request: Request) -> Response:
        body = await request.body()
        if len(body) > MAX_WEBHOOK_BYTES:
            return _err(413, "troppo grande")
        if not auth.verify_webhook(
            body, request.headers.get("x-shopify-hmac-sha256", ""), st.shopify_secret
        ):
            return _err(401, "firma non valida")
        hook_id = request.headers.get("x-shopify-webhook-id", "")
        if hook_id and not store.use_nonce(f"webhook:{hook_id[:100]}"):
            return Response(status_code=200)  # già ricevuto: un replay non ha effetto
        topic = request.headers.get("x-shopify-topic", "")
        domain = request.headers.get("x-shopify-shop-domain", "").lower()
        if not auth.SHOP_DOMAIN.match(domain):
            return _err(400, "negozio non valido")
        if topic.startswith("products/"):
            schedule_sync(domain)
        elif topic == "app/uninstalled":
            store.set_active(domain, False)
        elif topic == "shop/redact":  # GDPR: cancellazione di tutti i dati del negozio
            for job_id in store.delete_shop(domain):
                _rmtree(job_dir(st, job_id))
            _rmtree(shop_dir(st, domain))
        # customers/data_request e customers/redact: il servizio non conserva dati dei
        # clienti finali (nessun nome, email o ordine): niente da restituire o cancellare
        return Response(status_code=200)

    def schedule_sync(domain: str, delay_s: float = 60.0) -> None:
        """Le modifiche ai prodotti arrivano a raffica: una sola sincronizzazione."""
        shop = store.shop(domain)
        if shop is None or not shop.active:
            return
        with sync_lock:
            old = sync_timers.pop(domain, None)
            if old:
                old.cancel()
            t = threading.Timer(delay_s, run_sync, args=(domain,))
            t.daemon = True
            sync_timers[domain] = t
            t.start()

    def run_sync(domain: str) -> dict[str, Any]:
        from app.storage import shopify

        shop = store.shop(domain)
        if shop is None:
            return {"errore": "negozio sconosciuto"}
        out = shop_dir(st, domain) / "catalogo"
        fetch = shopify.safe_fetcher(shop.store_url, domain)
        report = shopify.sync(shop.store_url, out, fetch=fetch)
        return {"riepilogo": report.summary()}

    # ----------------------------------------------------------- amministrazione
    def is_admin(request: Request) -> bool:
        if len(st.admin_token) < 24:
            return False
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return hmac.compare_digest(given.encode(), st.admin_token.encode())

    @app.get("/admin/negozi")
    def admin_shops(request: Request) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        return JSONResponse([s.__dict__ for s in store.shops()])

    @app.post("/admin/negozi")
    async def admin_add_shop(request: Request) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        data = await request.json()
        domain = str(data.get("dominio", "")).lower()
        site = str(data.get("sito", "")) or f"https://{domain}"
        quota = int(data.get("prove_al_giorno", st.jobs_per_shop_day))
        if not auth.SHOP_DOMAIN.match(domain) or urlparse(site).scheme != "https":
            return _err(400, "dominio .myshopify.com e sito https obbligatori")
        store.upsert_shop(domain, site.rstrip("/"), max(1, min(quota, 10_000)))
        return JSONResponse({"ok": True})

    @app.post("/admin/negozi/{domain}/sincronizza")
    def admin_sync(request: Request, domain: str) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        if not auth.SHOP_DOMAIN.match(domain) or store.shop(domain) is None:
            return _err(404, "negozio sconosciuto")
        try:
            return JSONResponse(run_sync(domain))
        except Exception as exc:
            log.warning("sincronizzazione %s non riuscita: %s", domain, exc)
            return _err(502, "sincronizzazione non riuscita")

    app.state.run_sync = run_sync
    return app


def _image_ext(raw: bytes) -> str | None:
    """Formato reale del file (dai byte, mai dal nome), dopo una decodifica completa."""
    from app.core import ingest  # registra HEIC

    _ = ingest
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = im.format or ""
            if im.width * im.height > (Image.MAX_IMAGE_PIXELS or 0):
                return None  # Pillow da solo blocca solo oltre il doppio del limite
            im.load()
    except Exception:  # file rotto, formato ignoto, bomba di decompressione
        return None
    return ALLOWED_FORMATS.get(fmt)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
