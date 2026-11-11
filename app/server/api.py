"""API del servizio (FastAPI).

Percorsi:
- /widget/posa.js   pulsante "Prova nella tua stanza" da incollare in qualunque sito;
- /api/...     chiamati dal browser del cliente finale (chiave pubblica + origine del sito,
               poi permesso di caricamento a tempo; CORS limitato ai siti registrati);
- /v1/...      chiamati dal server del sito cliente (chiave segreta): catalogo, pratiche, uso;
- /proxy/...   chiamati dai negozi Shopify tramite App Proxy (firmati da Shopify);
- /webhooks/shopify, /webhooks/stripe   notifiche firmate (catalogo, GDPR, pagamenti);
- /admin/...   amministrazione del servizio: clienti, piani, chiavi, uso (token segreto).
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
from app.server.jobs import JobQueue, Runner, job_dir, step_message, tenant_dir
from app.server.limits import BodyLimitMiddleware
from app.server.settings import Settings
from app.server.store import (
    CHANNELS,
    KEY_KINDS,
    SURFACES,
    TENANT_STATUSES,
    Plan,
    Store,
    Tenant,
)

log = logging.getLogger("posa.server")

ALLOWED_FORMATS = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp", "HEIF": "heic"}
ROOM = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)x(\d{1,2}(?:\.\d{1,2})?)$")
JOB_ID = re.compile(r"^[A-Za-z0-9_-]{32}$")
PUBLIC_FILES = {"risultato.jpg", "prima_dopo.jpg"}
HANDLE = re.compile(r"^[a-z0-9][a-z0-9-]{0,254}$")
VARIANT = re.compile(r"^\d{1,20}$")
ARTICLE = re.compile(r"^[a-z0-9][a-z0-9\-]{1,63}$")
TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,39}$")
PLAN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,29}$")
KEY_PREFIX = re.compile(r"^[ps]k_live_[A-Za-z0-9_-]{6}$")
ZONES = ("automatica", "paraschizzi", "parete")
MAX_WEBHOOK_BYTES = 1_000_000
MAX_CATALOG_PHOTOS = 8
STATIC = Path(__file__).parent / "static"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; img-src 'self'; frame-ancestors 'none'",
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Cache-Control": "no-store",
}

# stato dell'abbonamento Stripe -> stato del cliente
STRIPE_STATUS = {
    "active": "attivo",
    "trialing": "attivo",
    "past_due": "sospeso",
    "unpaid": "sospeso",
    "paused": "sospeso",
    "incomplete": "sospeso",
    "incomplete_expired": "disdetto",
    "canceled": "disdetto",
}


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"errore": msg}, status_code=status)


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


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
    max_bytes = st.max_upload_mb * 1024 * 1024

    def body_limit(path: str) -> int:
        """Una foto per richiesta; solo il catalogo ne porta di più (fino a 8)."""
        many = path.startswith("/v1/catalogo/")
        return (MAX_CATALOG_PHOTOS if many else 1) * max_bytes + 64 * 1024

    app = FastAPI(title="posa", docs_url=None, redoc_url=None, openapi_url=None)
    # limite sul corpo applicato mentre arriva (anche senza Content-Length)
    app.add_middleware(BodyLimitMiddleware, max_bytes=body_limit(""), limit_for=body_limit)
    app.state.store, app.state.jobs, app.state.settings = store, jobs, st
    if start_workers:
        jobs.start()

    # ----------------------------------------------------------------- middleware
    def tenant_hosts(t: Tenant) -> set[str]:
        hosts = {_host(t.site)} | {_host(o) for o in t.origins}
        if t.channel == "shopify":
            hosts.add(t.id)
        return hosts - {""}

    def origin_allowed(origin: str, tenant: Tenant | None = None) -> bool:
        """L'origine (https) è un sito registrato: di `tenant`, o di un cliente attivo."""
        if not origin or urlparse(origin).scheme != "https":
            return False
        host = _host(origin)
        candidates = [tenant] if tenant else [t for t in store.tenants() if t.active]
        return any(host in tenant_hosts(t) for t in candidates if t is not None)

    @app.middleware("http")
    async def harden(request: Request, call_next: Any) -> Response:
        origin = request.headers.get("origin", "")
        path = request.url.path
        is_api = path.startswith("/api/")
        if request.method == "OPTIONS" and is_api:
            resp: Response = Response(status_code=204)
        else:
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > body_limit(path)):
                resp = _err(413, "richiesta troppo grande")
            else:
                resp = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            resp.headers.setdefault(k, v)
        if is_api and origin_allowed(origin):
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST"
            resp.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, X-Posa-Permesso, X-Posa-Chiave"
            )
            resp.headers["Access-Control-Max-Age"] = "600"
        return resp

    def client_ip(request: Request) -> str:
        if st.trust_proxy:  # solo dietro un proxy fidato che riscrive l'intestazione
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return fwd.split(",")[-1].strip()
        return request.client.host if request.client else "?"

    def proxy_tenant(request: Request) -> Tenant | None:
        params = parse_qs(request.url.query, keep_blank_values=True)
        shop = auth.verify_proxy(params, st.shopify_secret, st.proxy_max_age_s)
        if shop is None:
            return None
        t = store.tenant(shop)
        return t if t and t.active and t.channel == "shopify" else None

    def secret_tenant(request: Request) -> Tenant | None:
        """Cliente autenticato con la chiave segreta (da server a server)."""
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return store.key_owner(given, "segreta")

    def catalog_dir(tenant_id: str) -> Path:
        return tenant_dir(st, tenant_id) / "catalogo"

    def shop_index(tenant_id: str) -> dict[str, Any]:
        path = catalog_dir(tenant_id) / "index.json"
        if not path.is_file():
            return {"handles": {}, "variants": {}}
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def surface_problem(tenant: Tenant, article: str, surface: str, zone: str) -> Response | None:
        """Controlli comuni prima di dare un permesso o accettare una foto."""
        from app.storage import catalog_store

        if surface not in SURFACES or zone not in ZONES:
            return _err(400, "superficie non valida (pavimento o rivestimento)")
        plan = store.plan(tenant.plan)
        if plan is None or surface not in plan.surfaces:
            return _err(403, f"il piano non comprende la prova su: {surface}")
        uses = catalog_store.article_uses(catalog_dir(tenant.id), article)
        if uses is None:
            return _err(404, "articolo non disponibile per la prova")
        if surface not in uses:
            return _err(409, f"questo articolo non si posa come {surface}")
        return None

    def issue_token(tenant: Tenant, article: str, surface: str, zone: str) -> Response:
        if problem := surface_problem(tenant, article, surface, zone):
            return problem
        if reason := store.quota_problem(tenant):
            return _err(402, reason)
        token = auth.make_upload_token(
            st.upload_key, tenant.id, article, st.token_ttl_s, secrets.token_hex(12), surface, zone
        )
        base = st.public_url.rstrip("/")
        return JSONResponse(
            {"permesso": token, "carica": f"{base}/api/pratiche", "scade_in_s": st.token_ttl_s}
        )

    async def accept_photo(
        request: Request,
        tenant: Tenant,
        article: str,
        surface: str,
        zone: str,
        per_ip: bool,
        nonce: str | None = None,
    ) -> Response:
        """Riceve la foto e mette in coda la pratica, dopo tutti i limiti. `nonce`: permesso
        monouso da consumare (solo se la foto è accettata: un file sbagliato non lo brucia)."""
        if not tenant.active:
            return _err(403, "abbonamento non attivo")
        if problem := surface_problem(tenant, article, surface, zone):
            return problem
        form = await request.form(max_files=1, max_fields=6)
        room = str(form.get("stanza", "") or "").strip().lower().replace(",", ".")
        if room:
            m = ROOM.match(room)
            low = 0.2 if surface == "rivestimento" else 1.0
            if not m or not all(low <= float(v) <= 50.0 for v in m.groups()):
                return _err(400, "misure non valide (es. 4.2x3.6)")
        upload = form.get("foto")
        if upload is None or isinstance(upload, str):
            return _err(400, "manca la foto")
        raw = await upload.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return _err(413, f"foto troppo grande (massimo {st.max_upload_mb} MB)")
        ext = _image_ext(raw)
        if ext is None:
            return _err(415, "il file non è una foto supportata (JPEG, HEIC, PNG, WebP)")
        # limiti: canone (quota del mese), anti-abuso (giorno, IP), capacità (coda)
        now = time.time()
        if reason := store.quota_problem(tenant):
            return _err(402, reason)
        if store.count_jobs(shop=tenant.id, since=now - 86400) >= tenant.daily_quota:
            return _err(429, "limite giornaliero di prove raggiunto")
        iph = store.ip_hash(client_ip(request))
        if per_ip and store.count_jobs(ip_hash=iph, since=now - 3600) >= st.jobs_per_ip_hour:
            return _err(429, "troppe prove: riprova più tardi")
        if store.pending_jobs() >= st.max_queue:
            return _err(503, "servizio molto richiesto: riprova tra qualche minuto")
        if nonce is not None and not store.use_nonce(nonce):
            return _err(401, "permesso già usato")
        job_id = store.create_job(tenant.id, article, iph, room or None, surface, zone)
        d = job_dir(st, job_id) / "input"
        d.mkdir(parents=True)
        (d / f"foto.{ext}").write_bytes(raw)
        jobs.submit(job_id)
        return JSONResponse({"pratica": job_id, "stato": "in_coda"}, status_code=201)

    def job_body(job_id: str, prefix: str) -> dict[str, Any] | None:
        job = store.job(job_id) if JOB_ID.match(job_id) else None
        if job is None:
            return None
        body: dict[str, Any] = {
            "stato": job.status,
            "messaggio": step_message(job.step, job.surface),
            "superficie": job.surface,
        }
        if job.status == "pronto":
            body["risultato"] = f"{prefix}/{job_id}/risultato.jpg"
            body["prima_dopo"] = f"{prefix}/{job_id}/prima_dopo.jpg"
        return body

    def job_public_file(job_id: str, name: str) -> Response:
        if not JOB_ID.match(job_id) or name not in PUBLIC_FILES:
            return _err(404, "file inesistente")
        job = store.job(job_id)
        path = job_dir(st, job_id) / "pubblico" / name
        if job is None or job.status != "pronto" or not path.is_file():
            return _err(404, "file inesistente")
        return FileResponse(path, media_type="image/jpeg")

    # ------------------------------------------------------------------- pubblico
    @app.get("/salute")
    def health() -> dict[str, str]:
        return {"stato": "ok"}

    @app.get("/widget/posa.js")
    def widget() -> Response:
        return FileResponse(
            STATIC / "posa.js",
            media_type="text/javascript; charset=utf-8",
            headers={"Cache-Control": "public, max-age=300"},
        )

    # --------------------------------------------------------------- App Proxy
    @app.get("/proxy/sessione")
    def session(
        request: Request, prodotto: str = "", variante: str = "", superficie: str = "pavimento"
    ) -> Response:
        """Permesso di caricamento per un articolo, richiesto dalla pagina prodotto."""
        tenant = proxy_tenant(request)
        if tenant is None:
            return _err(401, "richiesta non firmata da Shopify")
        if not HANDLE.match(prodotto) or (variante and not VARIANT.match(variante)):
            return _err(400, "articolo non valido")
        index = shop_index(tenant.id)
        article = index["variants"].get(variante) if variante else None
        if article is None:
            options = index["handles"].get(prodotto) or []
            article = options[0] if options else None
        if article is None:
            return _err(404, "articolo non disponibile per la prova")
        return issue_token(tenant, str(article), superficie, "automatica")

    # ---------------------------------------------------------- API del browser
    @app.post("/api/sessioni")
    async def browser_session(request: Request) -> Response:
        """Permesso di caricamento chiesto dal widget con la chiave pubblica del sito.
        La chiave pubblica vale solo dalle pagine dei siti registrati per quel cliente."""
        tenant = store.key_owner(request.headers.get("x-posa-chiave", ""), "pubblica")
        if tenant is None or not tenant.active:
            return _err(401, "chiave non valida o abbonamento non attivo")
        if not origin_allowed(request.headers.get("origin", ""), tenant):
            return _err(403, "sito non autorizzato per questa chiave")
        data = await _json(request)
        article = str(data.get("articolo", ""))
        if not ARTICLE.match(article):
            return _err(400, "articolo non valido")
        return issue_token(
            tenant,
            article,
            str(data.get("superficie", "pavimento")),
            str(data.get("zona", "automatica")),
        )

    @app.post("/api/pratiche")
    async def create_job(request: Request) -> Response:
        # il permesso si verifica PRIMA di leggere il corpo: senza, il server non legge nulla
        token = request.headers.get("x-posa-permesso", "")
        data = auth.read_upload_token(st.upload_key, token)
        if data is None:
            return _err(401, "permesso non valido o scaduto")
        tenant = store.tenant(data["shop"])
        if tenant is None or not tenant.active:
            return _err(403, "abbonamento non attivo")
        return await accept_photo(
            request,
            tenant,
            data["article"],
            data["surface"],
            data["zone"],
            per_ip=True,
            nonce=data["nonce"],
        )

    @app.get("/api/pratiche/{job_id}")
    def job_status(job_id: str) -> Response:
        body = job_body(job_id, "/api/pratiche")
        return JSONResponse(body) if body else _err(404, "pratica inesistente")

    @app.get("/api/pratiche/{job_id}/{name}")
    def job_file(job_id: str, name: str) -> Response:
        return job_public_file(job_id, name)

    # ------------------------------------------- API del cliente (chiave segreta)
    @app.get("/v1/account")
    def v1_account(request: Request) -> Response:
        tenant = secret_tenant(request)
        if tenant is None:
            return _err(401, "chiave segreta non valida")
        return JSONResponse(_account(store, tenant))

    @app.get("/v1/catalogo")
    def v1_catalog(request: Request) -> Response:
        from app.storage import catalog_store

        tenant = secret_tenant(request)
        if tenant is None:
            return _err(401, "chiave segreta non valida")
        return JSONResponse({"articoli": catalog_store.list_articles(catalog_dir(tenant.id))})

    @app.put("/v1/catalogo/{article}")
    async def v1_put_article(request: Request, article: str) -> Response:
        """Aggiunge o sostituisce un articolo: foto della piastrella + regole di posa."""
        from app.storage import catalog_store

        tenant = secret_tenant(request)
        if tenant is None or not tenant.active:
            return _err(401, "chiave segreta non valida o abbonamento non attivo")
        if tenant.channel != "sito":
            return _err(409, "il catalogo di questo cliente è sincronizzato dal negozio Shopify")
        if not ARTICLE.match(article):
            return _err(400, "codice articolo non valido")
        form = await request.form(max_files=MAX_CATALOG_PHOTOS, max_fields=16)
        photos: list[bytes] = []
        for upload in form.getlist("foto"):
            if isinstance(upload, str):
                continue
            raw = await upload.read(max_bytes + 1)
            if len(raw) > max_bytes or _image_ext(raw) is None:
                return _err(415, "ogni foto deve essere JPEG, PNG, WebP o HEIC entro il limite")
            photos.append(raw)
        fields = {k: str(v) for k, v in form.items() if isinstance(v, str)}
        try:
            rules = catalog_store.parse_rules(fields)
            entry, notes = catalog_store.put_article(
                catalog_dir(tenant.id), article, fields.get("nome", ""), rules, photos
            )
        except catalog_store.ArticleError as exc:
            return _err(422, str(exc))
        return JSONResponse({"articolo": entry, "note": notes})

    @app.delete("/v1/catalogo/{article}")
    def v1_delete_article(request: Request, article: str) -> Response:
        from app.storage import catalog_store

        tenant = secret_tenant(request)
        if tenant is None:
            return _err(401, "chiave segreta non valida")
        if tenant.channel != "sito" or not ARTICLE.match(article):
            return _err(409, "articolo non modificabile")
        if not catalog_store.delete_article(catalog_dir(tenant.id), article):
            return _err(404, "articolo inesistente")
        return JSONResponse({"ok": True})

    @app.post("/v1/pratiche")
    async def v1_create_job(request: Request) -> Response:
        """Prova avviata dal server del sito cliente (foto già ricevuta dal sito)."""
        tenant = secret_tenant(request)
        if tenant is None:
            return _err(401, "chiave segreta non valida")
        q = request.query_params
        article = q.get("articolo", "")
        if not ARTICLE.match(article):
            return _err(400, "articolo non valido")
        return await accept_photo(
            request,
            tenant,
            article,
            q.get("superficie", "pavimento"),
            q.get("zona", "automatica"),
            per_ip=False,
        )

    @app.get("/v1/pratiche/{job_id}")
    def v1_job(request: Request, job_id: str) -> Response:
        tenant = secret_tenant(request)
        if tenant is None:
            return _err(401, "chiave segreta non valida")
        job = store.job(job_id) if JOB_ID.match(job_id) else None
        if job is None or job.shop != tenant.id:  # mai le pratiche di un altro cliente
            return _err(404, "pratica inesistente")
        return JSONResponse(job_body(job_id, "/api/pratiche"))

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
            store.set_status(domain, "sospeso")
        elif topic == "shop/redact":  # GDPR: cancellazione di tutti i dati del negozio
            wipe_tenant(domain)
        # customers/data_request e customers/redact: il servizio non conserva dati dei
        # clienti finali (nessun nome, email o ordine): niente da restituire o cancellare
        return Response(status_code=200)

    @app.post("/webhooks/stripe")
    async def stripe_webhook(request: Request) -> Response:
        """Pagamenti: l'abbonamento Stripe decide se il cliente è attivo, sospeso o disdetto."""
        body = await request.body()
        if len(body) > MAX_WEBHOOK_BYTES:
            return _err(413, "troppo grande")
        if not auth.verify_stripe(
            body, request.headers.get("stripe-signature", ""), st.stripe_secret
        ):
            return _err(401, "firma non valida")
        try:
            event = json.loads(body)
            kind = str(event.get("type", ""))
            obj = event["data"]["object"]
            event_id = str(event.get("id", ""))[:100]
        except (ValueError, KeyError, TypeError):
            return _err(400, "evento non valido")
        if not isinstance(obj, dict):
            return _err(400, "evento non valido")
        if event_id and not store.use_nonce(f"stripe:{event_id}"):
            return Response(status_code=200)
        raw_meta = obj.get("metadata")
        meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        customer = str(obj.get("customer") or "")
        if kind == "checkout.session.completed":
            # il link di pagamento porta l'id del cliente in client_reference_id
            tenant = store.tenant(str(obj.get("client_reference_id") or ""))
            if tenant and customer:
                store.set_billing(tenant.id, ref=customer)
                store.set_status(tenant.id, "attivo")
        elif kind.startswith("customer.subscription."):
            tenant = store.tenant(str(meta.get("cliente", ""))) or store.tenant_by_billing_ref(
                customer
            )
            if tenant:
                if customer and not tenant.billing_ref:
                    store.set_billing(tenant.id, ref=customer)
                plan = str(meta.get("piano", ""))
                if plan and store.plan(plan):
                    store.set_billing(tenant.id, plan=plan)
                status = (
                    "disdetto"
                    if kind.endswith(".deleted")
                    else STRIPE_STATUS.get(str(obj.get("status", "")))
                )
                if status:
                    store.set_status(tenant.id, status)
                    log.info("cliente %s: abbonamento -> %s", tenant.id, status)
        return Response(status_code=200)

    def wipe_tenant(tenant_id: str) -> None:
        for job_id in store.delete_tenant(tenant_id):
            _rmtree(job_dir(st, job_id))
        _rmtree(tenant_dir(st, tenant_id))

    def schedule_sync(domain: str, delay_s: float = 60.0) -> None:
        """Le modifiche ai prodotti arrivano a raffica: una sola sincronizzazione."""
        shop = store.tenant(domain)
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

        shop = store.tenant(domain)
        if shop is None or shop.channel != "shopify":
            return {"errore": "negozio sconosciuto"}
        fetch = shopify.safe_fetcher(shop.site, domain)
        report = shopify.sync(shop.site, catalog_dir(domain), fetch=fetch)
        return {"riepilogo": report.summary()}

    # ----------------------------------------------------------- amministrazione
    def is_admin(request: Request) -> bool:
        if len(st.admin_token) < 24:
            return False
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return hmac.compare_digest(given.encode(), st.admin_token.encode())

    def tenant_view(t: Tenant) -> dict[str, Any]:
        return {
            "id": t.id,
            "nome": t.name,
            "canale": t.channel,
            "sito": t.site,
            "origini": list(t.origins),
            "stato": t.status,
            "prove_al_giorno": t.daily_quota,
            "riferimento_pagamento": t.billing_ref,
            **_account(store, t),
        }

    @app.get("/admin/clienti")
    def admin_tenants(request: Request) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        return JSONResponse([tenant_view(t) for t in store.tenants()])

    @app.post("/admin/clienti")
    async def admin_add_tenant(request: Request) -> Response:
        """Crea un cliente (sito con chiavi API) o ne aggiorna dati, piano e quote."""
        if not is_admin(request):
            return _err(401, "non autorizzato")
        data = await _json(request)
        tid = str(data.get("id", "")).lower()
        channel = str(data.get("canale", "sito"))
        site = str(data.get("sito", "")).rstrip("/")
        origins = data.get("origini", [])
        plan = str(data.get("piano", "")) or None
        if channel not in CHANNELS:
            return _err(400, "canale: sito o shopify")
        valid_id = auth.SHOP_DOMAIN.match(tid) if channel == "shopify" else TENANT_ID.match(tid)
        if not valid_id:
            return _err(400, "id non valido (minuscole, cifre e trattini; 3-40 caratteri)")
        if channel == "shopify" and not site:
            site = f"https://{tid}"
        if urlparse(site).scheme != "https" or not _host(site):
            return _err(400, "sito https obbligatorio")
        if not isinstance(origins, list) or len(origins) > 20:
            return _err(400, "origini: elenco di al massimo 20 indirizzi https")
        clean = []
        for o in origins:
            u = urlparse(str(o))
            if u.scheme != "https" or not u.hostname:
                return _err(400, "origini: solo indirizzi https")
            clean.append(f"https://{u.netloc.lower()}")
        if plan and store.plan(plan) is None:
            return _err(400, "piano sconosciuto")
        try:
            daily = int(data.get("prove_al_giorno", st.jobs_per_shop_day))
            monthly = data.get("prove_al_mese")
            monthly = None if monthly is None else max(0, min(int(monthly), 1_000_000))
        except (TypeError, ValueError):
            return _err(400, "quote non valide")
        store.upsert_tenant(
            tid,
            name=" ".join(str(data.get("nome", "") or tid).split())[:120],
            channel=channel,
            site=site,
            daily_quota=max(1, min(daily, 10_000)),
            origins=tuple(clean),
            plan=plan,
            monthly_quota=monthly,
        )
        return JSONResponse({"ok": True, "cliente": tid})

    def admin_tenant(request: Request, tid: str) -> Tenant | Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        t = store.tenant(tid.lower())
        return t if t else _err(404, "cliente sconosciuto")

    @app.post("/admin/clienti/{tid}/stato")
    async def admin_status(request: Request, tid: str) -> Response:
        t = admin_tenant(request, tid)
        if isinstance(t, Response):
            return t
        status = str((await _json(request)).get("stato", ""))
        if status not in TENANT_STATUSES:
            return _err(400, "stato: attivo, sospeso o disdetto")
        store.set_status(t.id, status)
        return JSONResponse({"ok": True})

    @app.post("/admin/clienti/{tid}/fatturazione")
    async def admin_billing(request: Request, tid: str) -> Response:
        """Collega il cliente al suo cliente Stripe (cus_...) e/o cambia piano."""
        t = admin_tenant(request, tid)
        if isinstance(t, Response):
            return t
        data = await _json(request)
        ref = str(data.get("riferimento", ""))[:100] or None
        plan = str(data.get("piano", "")) or None
        if plan and store.plan(plan) is None:
            return _err(400, "piano sconosciuto")
        store.set_billing(t.id, ref=ref, plan=plan)
        return JSONResponse({"ok": True})

    @app.get("/admin/clienti/{tid}/chiavi")
    def admin_keys(request: Request, tid: str) -> Response:
        t = admin_tenant(request, tid)
        return t if isinstance(t, Response) else JSONResponse(store.keys(t.id))

    @app.post("/admin/clienti/{tid}/chiavi")
    async def admin_new_key(request: Request, tid: str) -> Response:
        """Crea una chiave: `pubblica` (widget nel browser) o `segreta` (server del sito).
        La chiave si vede solo in questa risposta."""
        t = admin_tenant(request, tid)
        if isinstance(t, Response):
            return t
        kind = str((await _json(request)).get("tipo", ""))
        if kind not in KEY_KINDS:
            return _err(400, "tipo: pubblica o segreta")
        return JSONResponse({"tipo": kind, "chiave": store.create_key(t.id, kind)})

    @app.delete("/admin/clienti/{tid}/chiavi/{prefix}")
    def admin_revoke_key(request: Request, tid: str, prefix: str) -> Response:
        t = admin_tenant(request, tid)
        if isinstance(t, Response):
            return t
        if not KEY_PREFIX.match(prefix) or not store.revoke_key(t.id, prefix):
            return _err(404, "chiave sconosciuta")
        return JSONResponse({"ok": True})

    @app.get("/admin/clienti/{tid}/uso")
    def admin_usage(request: Request, tid: str) -> Response:
        t = admin_tenant(request, tid)
        if isinstance(t, Response):
            return t
        return JSONResponse(
            {
                **_account(store, t),
                "mesi": [
                    {
                        "mese": u.month,
                        "avviate": u.created,
                        "consegnate": u.delivered,
                        "non_riuscite": u.failed,
                        "costo_agenti_usd": round(u.cost_usd, 2),
                    }
                    for u in store.usage_history(t.id)
                ],
            }
        )

    @app.delete("/admin/clienti/{tid}")
    def admin_delete_tenant(request: Request, tid: str) -> Response:
        """Fine del rapporto: cancella cliente, chiavi, catalogo, foto e risultati."""
        t = admin_tenant(request, tid)
        if isinstance(t, Response):
            return t
        wipe_tenant(t.id)
        return JSONResponse({"ok": True})

    @app.get("/admin/piani")
    def admin_plans(request: Request) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        return JSONResponse([_plan_view(p) for p in store.plans()])

    @app.put("/admin/piani/{plan_id}")
    async def admin_put_plan(request: Request, plan_id: str) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        if not PLAN_ID.match(plan_id):
            return _err(400, "id del piano non valido")
        data = await _json(request)
        surfaces = data.get("superfici", list(SURFACES))
        try:
            jobs_month = int(data["prove_al_mese"])
            fee = data.get("canone_eur")
            extra = data.get("prova_extra_eur")
            plan = Plan(
                id=plan_id,
                name=" ".join(str(data.get("nome", plan_id)).split())[:60],
                monthly_jobs=jobs_month,
                surfaces=tuple(surfaces),
                overage=bool(data.get("oltre_quota_a_consumo", False)),
                monthly_fee_eur=None if fee is None else float(fee),
                extra_job_eur=None if extra is None else float(extra),
            )
        except (KeyError, TypeError, ValueError):
            return _err(400, "servono prove_al_mese (intero) e valori numerici validi")
        if (
            not 0 < plan.monthly_jobs <= 1_000_000
            or not plan.surfaces
            or any(s not in SURFACES for s in plan.surfaces)
        ):
            return _err(400, "prove_al_mese o superfici non validi")
        store.upsert_plan(plan)
        return JSONResponse({"ok": True})

    # negozi Shopify: stesse operazioni di prima, ora sono clienti del canale "shopify"
    @app.get("/admin/negozi")
    def admin_shops(request: Request) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        return JSONResponse([tenant_view(t) for t in store.tenants() if t.channel == "shopify"])

    @app.post("/admin/negozi")
    async def admin_add_shop(request: Request) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        data = await _json(request)
        domain = str(data.get("dominio", "")).lower()
        site = str(data.get("sito", "")) or f"https://{domain}"
        if not auth.SHOP_DOMAIN.match(domain) or urlparse(site).scheme != "https":
            return _err(400, "dominio .myshopify.com e sito https obbligatori")
        plan = str(data.get("piano", "")) or None
        if plan and store.plan(plan) is None:
            return _err(400, "piano sconosciuto")
        try:
            quota = int(data.get("prove_al_giorno", st.jobs_per_shop_day))
        except (TypeError, ValueError):
            return _err(400, "quote non valide")
        store.upsert_tenant(
            domain,
            name=str(data.get("nome", "") or domain)[:120],
            channel="shopify",
            site=site.rstrip("/"),
            daily_quota=max(1, min(quota, 10_000)),
            plan=plan,
        )
        return JSONResponse({"ok": True})

    @app.post("/admin/negozi/{domain}/sincronizza")
    def admin_sync(request: Request, domain: str) -> Response:
        if not is_admin(request):
            return _err(401, "non autorizzato")
        if not auth.SHOP_DOMAIN.match(domain) or store.tenant(domain) is None:
            return _err(404, "negozio sconosciuto")
        try:
            return JSONResponse(run_sync(domain))
        except Exception as exc:
            log.warning("sincronizzazione %s non riuscita: %s", domain, exc)
            return _err(502, "sincronizzazione non riuscita")

    app.state.run_sync = run_sync
    return app


async def _json(request: Request) -> dict[str, Any]:
    """Corpo JSON come dizionario; qualunque altra cosa diventa un dizionario vuoto."""
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _plan_view(p: Plan) -> dict[str, Any]:
    return {
        "id": p.id,
        "nome": p.name,
        "prove_al_mese": p.monthly_jobs,
        "superfici": list(p.surfaces),
        "oltre_quota_a_consumo": p.overage,
        "canone_eur": p.monthly_fee_eur,
        "prova_extra_eur": p.extra_job_eur,
    }


def _account(store: Store, t: Tenant) -> dict[str, Any]:
    """Piano e uso del mese in corso: quello che serve per fatturare e per avvisare."""
    allowed, overage = store.monthly_allowance(t)
    used = store.usage(t.id)
    plan = store.plan(t.plan)
    return {
        "cliente": t.id,
        "stato": t.status,
        "piano": t.plan,
        "superfici": list(plan.surfaces) if plan else [],
        "mese": used.month,
        "prove_comprese": allowed,
        "prove_consegnate": used.delivered,
        "prove_non_riuscite": used.failed,
        "prove_in_corso": used.created - used.delivered - used.failed,
        "prove_extra": max(0, used.delivered - allowed) if overage else 0,
        "oltre_quota_a_consumo": overage,
    }


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
