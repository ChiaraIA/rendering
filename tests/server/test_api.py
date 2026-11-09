"""Servizio multi-negozio: flusso completo, firme, isolamento, limiti, webhook."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.devtools import synthetic
from app.server import shopify_auth as auth
from app.server.api import create_app
from app.server.jobs import shop_dir
from app.server.settings import Settings
from app.storage import exporter

SECRET = "segreto-app-shopify-di-prova-123"
ADMIN = "token-amministratore-molto-lungo-0123456789"
SHOP = "ceramiche-rossi.myshopify.com"
OTHER = "concorrente.myshopify.com"


def jpeg(w: int = 320, h: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(synthetic.make_photo(w, h)).save(buf, format="JPEG")
    return buf.getvalue()


def signed(
    path: str, shop: str = SHOP, secret: str = SECRET, ts: int | None = None, **q: str
) -> str:
    params = {"shop": shop, "path_prefix": "/apps/posa", "timestamp": str(ts or int(time.time()))}
    params.update(q)
    params["signature"] = auth.proxy_signature({k: [v] for k, v in params.items()}, secret)
    return f"{path}?{urlencode(params)}"


class FakeRunner:
    """Al posto degli agenti: produce un risultato finto e registra cosa ha ricevuto."""

    def __init__(self, deliver: bool = True) -> None:
        self.deliver = deliver
        self.calls: list[dict[str, Any]] = []

    def __call__(self, photo, work, article, room_m, catalog, on_step, timeout_s):  # type: ignore[no-untyped-def]
        self.calls.append({"photo": photo, "article": article, "room": room_m})
        on_step("posa")
        if self.deliver:
            out = work / "consegna"
            out.mkdir(parents=True)
            Image.new("RGB", (64, 48), (120, 80, 60)).save(exporter.master_path(out))
            Image.new("RGB", (128, 48), (120, 80, 60)).save(out / "prima_dopo.jpg")
        return self.deliver, 0.42


def _catalog(root: Path, domain: str) -> None:
    """Catalogo del negozio con un articolo e l'indice prodotto -> articolo."""
    out = shop_dir_for(root, domain) / "catalogo"
    path = synthetic.write_catalog(out)
    data = json.loads(path.read_text())
    article = data["materials"][0]["id"]
    (out / "index.json").write_text(
        json.dumps({"handles": {"gres-grigio": [article]}, "variants": {"111": article}})
    )


def shop_dir_for(root: Path, domain: str) -> Path:
    return root / "negozi" / domain


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("POSA_CHIAVE_CARICAMENTI", "k" * 64)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)  # ripristinato dopo
    st = Settings(
        data_dir=tmp_path,
        shopify_secret=SECRET,
        admin_token=ADMIN,
        upload_key="k" * 64,
        public_url="https://posa.example.com",
        max_upload_mb=2,
        jobs_per_ip_hour=3,
        max_pixels=4_000_000,
    )
    runner = FakeRunner()
    app = create_app(st, runner, start_workers=False)
    client = TestClient(app)
    h = {"Authorization": f"Bearer {ADMIN}"}
    for d in (SHOP, OTHER):
        assert client.post("/admin/negozi", json={"dominio": d}, headers=h).status_code == 200
        _catalog(tmp_path, d)
    return client, app, runner, st


def _upload(client: TestClient, token: str, data: bytes | None = None, **form: str):  # type: ignore[no-untyped-def]
    files = {"foto": ("stanza.jpg", data if data is not None else jpeg(), "image/jpeg")}
    return client.post("/api/pratiche", data=form, files=files, headers={"X-Posa-Permesso": token})


def _token(client: TestClient, shop: str = SHOP) -> str:
    r = client.get(signed("/proxy/sessione", shop=shop, prodotto="gres-grigio"))
    assert r.status_code == 200, r.text
    return str(r.json()["permesso"])


def test_full_flow_from_shop_button_to_result(env) -> None:  # type: ignore[no-untyped-def]
    client, app, runner, st = env
    token = _token(client)
    r = _upload(client, token, stanza="4,2x3.6")
    assert r.status_code == 201, r.text
    job = r.json()["pratica"]
    assert client.get(f"/api/pratiche/{job}").json()["stato"] == "in_coda"
    app.state.jobs.run_now(job)
    body = client.get(f"/api/pratiche/{job}").json()
    assert body["stato"] == "pronto" and body["messaggio"] == "Pronto"
    img = client.get(body["risultato"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert runner.calls[0]["room"] == (4.2, 3.6)
    assert runner.calls[0]["photo"].suffix == ".jpg"  # dal contenuto, non dal nome


def test_requests_without_valid_shopify_signature_are_refused(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    good = signed("/proxy/sessione", prodotto="gres-grigio")
    assert client.get(good).status_code == 200
    forged = signed("/proxy/sessione", secret="segreto-sbagliato", prodotto="gres-grigio")
    assert client.get(forged).status_code == 401
    tampered = good.replace("gres-grigio", "gres-grigio-x")  # parametro alterato
    assert client.get(tampered).status_code == 401
    old = signed("/proxy/sessione", ts=int(time.time()) - 3600, prodotto="gres-grigio")
    assert client.get(old).status_code == 401  # richiesta riusata
    unknown = signed("/proxy/sessione", shop="sconosciuto.myshopify.com", prodotto="gres-grigio")
    assert client.get(unknown).status_code == 401
    assert client.get("/proxy/sessione?prodotto=gres-grigio").status_code == 401


def test_upload_tokens_are_single_use_and_bound(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    token = _token(client)
    assert _upload(client, token).status_code == 201
    assert _upload(client, token).status_code == 401  # già usato
    raw, sig = token.split(".")
    assert _upload(client, raw + "." + "0" * 64).status_code == 401
    expired = auth.make_upload_token("k" * 64, SHOP, "x", -10, "n1")
    assert _upload(client, expired).status_code == 401
    other_key = auth.make_upload_token("z" * 64, SHOP, "x", 600, "n2")
    assert _upload(client, other_key).status_code == 401


def test_malicious_files_are_rejected(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    html = b"<html><script>alert(1)</script></html>"
    assert _upload(client, _token(client), html).status_code == 415
    polyglot = jpeg()[:200]  # JPEG troncato
    assert _upload(client, _token(client), polyglot).status_code == 415
    # bomba di decompressione: PNG minuscolo che si espande a 6000x6000 pixel
    buf = io.BytesIO()
    Image.new("L", (6000, 6000)).save(buf, format="PNG", optimize=True)
    assert len(buf.getvalue()) < 200_000
    assert _upload(client, _token(client), buf.getvalue()).status_code == 415
    too_big = b"\xff" * (3 * 1024 * 1024)
    assert _upload(client, _token(client), too_big).status_code == 413
    assert _upload(client, _token(client), stanza="999x1").status_code == 400
    assert _upload(client, _token(client), stanza="4x3; rm -rf /").status_code == 400


def test_shops_are_isolated_and_job_ids_unguessable(env) -> None:  # type: ignore[no-untyped-def]
    client, app, *_ = env
    job = _upload(client, _token(client)).json()["pratica"]
    assert len(job) == 32
    app.state.jobs.run_now(job)
    for bad in ("../../servizio.db", "%2e%2e%2fservizio.db", job[:-1] + "A", "x" * 32):
        assert client.get(f"/api/pratiche/{bad}").status_code == 404
    master = exporter.master_path(Path(".")).name
    assert client.get(f"/api/pratiche/{job}/{master}").status_code == 404  # solo JPEG pubblici
    assert client.get(f"/api/pratiche/{job}/..%2f..%2fservizio.db").status_code == 404


def test_rate_limits_and_daily_quota(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    codes = [_upload(client, _token(client)).status_code for _ in range(4)]
    assert codes == [201, 201, 201, 429]  # 3 per IP all'ora


def test_cors_only_for_registered_shops(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    ok = client.get("/salute", headers={"Origin": f"https://{SHOP}"})
    assert ok.headers.get("x-content-type-options") == "nosniff"
    r = client.options("/api/pratiche", headers={"Origin": f"https://{SHOP}"})
    assert r.headers.get("access-control-allow-origin") == f"https://{SHOP}"
    evil = client.options("/api/pratiche", headers={"Origin": "https://attaccante.example"})
    assert "access-control-allow-origin" not in evil.headers
    plain = client.options("/api/pratiche", headers={"Origin": f"http://{SHOP}"})
    assert "access-control-allow-origin" not in plain.headers


def test_admin_and_docs_are_closed(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    assert client.get("/admin/negozi").status_code == 401
    assert client.get("/admin/negozi", headers={"Authorization": "Bearer x"}).status_code == 401
    assert client.post("/admin/negozi", json={"dominio": "x.myshopify.com"}).status_code == 401
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def _hook(client: TestClient, topic: str, shop: str, body: dict[str, Any], secret: str = SECRET):  # type: ignore[no-untyped-def]
    raw = json.dumps(body).encode()
    sig = base64.b64encode(hmac.new(secret.encode(), raw, hashlib.sha256).digest()).decode()
    return client.post(
        "/webhooks/shopify",
        content=raw,
        headers={
            "X-Shopify-Hmac-Sha256": sig,
            "X-Shopify-Topic": topic,
            "X-Shopify-Shop-Domain": shop,
            "Content-Type": "application/json",
        },
    )


def test_webhooks_signed_and_gdpr_redact(env) -> None:  # type: ignore[no-untyped-def]
    client, app, _runner, st = env
    job = _upload(client, _token(client, OTHER)).json()["pratica"]
    assert _hook(client, "shop/redact", OTHER, {}, secret="falso").status_code == 401
    assert _hook(client, "app/uninstalled", SHOP, {}).status_code == 200
    assert app.state.store.shop(SHOP).active is False
    assert client.get(signed("/proxy/sessione", prodotto="gres-grigio")).status_code == 401
    assert _hook(client, "shop/redact", OTHER, {}).status_code == 200
    assert app.state.store.shop(OTHER) is None
    assert not shop_dir(st, OTHER).exists()
    assert client.get(f"/api/pratiche/{job}").status_code == 404


def test_customer_sees_only_fixed_messages(env) -> None:  # type: ignore[no-untyped-def]
    client, app, runner, _st = env
    runner.deliver = False
    job = _upload(client, _token(client)).json()["pratica"]
    app.state.jobs.run_now(job)
    body = client.get(f"/api/pratiche/{job}").json()
    assert body == {"stato": "non_riuscito", "messaggio": "Serve un controllo manuale"}


def test_retention_deletes_old_photos(env) -> None:  # type: ignore[no-untyped-def]
    client, app, _runner, st = env
    job = _upload(client, _token(client)).json()["pratica"]
    app.state.store._db.execute("UPDATE jobs SET created_at=0 WHERE id=?", (job,))
    assert app.state.jobs.clean() == 1
    assert not (st.data_dir / "pratiche" / job).exists()


def test_pixel_limit_is_enforced_even_below_pillow_double_threshold(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    buf = io.BytesIO()
    Image.fromarray(np.zeros((2100, 2100), np.uint8)).save(buf, format="PNG", optimize=True)
    assert _upload(client, _token(client), buf.getvalue()).status_code == 415  # 4.4 MP > 4 MP


def test_webhook_replay_is_ignored_and_secrets_scrubbed(env, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    client, app, _runner, st = env
    raw = b"{}"
    sig = base64.b64encode(hmac.new(SECRET.encode(), raw, hashlib.sha256).digest()).decode()
    h = {
        "X-Shopify-Hmac-Sha256": sig,
        "X-Shopify-Topic": "app/uninstalled",
        "X-Shopify-Shop-Domain": SHOP,
        "X-Shopify-Webhook-Id": "evento-1",
    }
    assert client.post("/webhooks/shopify", content=raw, headers=h).status_code == 200
    assert app.state.store.shop(SHOP).active is False
    app.state.store.set_active(SHOP, True)  # il negozio reinstalla
    assert client.post("/webhooks/shopify", content=raw, headers=h).status_code == 200
    assert app.state.store.shop(SHOP).active is True  # il replay non lo disattiva di nuovo
    import os

    monkeypatch.setenv("SHOPIFY_API_SECRET", "x" * 32)
    monkeypatch.setenv("POSA_ADMIN_TOKEN", "y" * 32)
    st.scrub_environment()
    assert "SHOPIFY_API_SECRET" not in os.environ and "POSA_ADMIN_TOKEN" not in os.environ
