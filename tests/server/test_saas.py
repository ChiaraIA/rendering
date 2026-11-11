"""Servizio in abbonamento: clienti con chiavi API, piani e quote mensili, widget per
qualunque sito, catalogo via API, pagamenti (Stripe) e rivestimenti a parete."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.devtools import synthetic
from app.server.api import create_app
from app.server.settings import Settings
from app.server.store import Store
from tests.server.test_api import ADMIN, SECRET, FakeRunner, jpeg

STRIPE = "whsec_segreto_di_prova_0123456789"
SITE = "https://www.ceramiche-bianchi.it"
H = {"Authorization": f"Bearer {ADMIN}"}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("POSA_CHIAVE_CARICAMENTI", "k" * 64)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", Image.MAX_IMAGE_PIXELS)
    st = Settings(
        data_dir=tmp_path,
        shopify_secret=SECRET,
        admin_token=ADMIN,
        upload_key="k" * 64,
        stripe_secret=STRIPE,
        public_url="https://posa.example.com",
        max_upload_mb=2,
        jobs_per_ip_hour=50,
        max_pixels=4_000_000,
    )
    runner = FakeRunner()
    app = create_app(st, runner, start_workers=False)
    return TestClient(app), app, runner, st


def _tenant(client: TestClient, tid: str = "bianchi", **extra: Any) -> tuple[str, str]:
    """Crea un cliente e ritorna (chiave pubblica, chiave segreta)."""
    body = {"id": tid, "nome": "Ceramiche Bianchi", "sito": SITE, "piano": "professionale"}
    assert client.post("/admin/clienti", json=body | extra, headers=H).status_code == 200
    keys = []
    for kind in ("pubblica", "segreta"):
        r = client.post(f"/admin/clienti/{tid}/chiavi", json={"tipo": kind}, headers=H)
        assert r.status_code == 200
        keys.append(r.json()["chiave"])
    return keys[0], keys[1]


def _tile(color: tuple[int, int, int] = (40, 90, 170)) -> bytes:
    buf = io.BytesIO()
    tex = synthetic.make_tile_texture((200, 200), color, kind="speckle")
    Image.fromarray(tex).save(buf, format="PNG")
    return buf.getvalue()


def _article(client: TestClient, sk: str, code: str = "blu-20x20", **fields: str) -> Any:
    data = {"nome": "Ceramica blu 20x20", "formato": "20x20", "fuga": "2"} | fields
    return client.put(
        f"/v1/catalogo/{code}",
        data=data,
        files=[("foto", ("piastrella.png", _tile(), "image/png"))],
        headers={"Authorization": f"Bearer {sk}"},
    )


def _session(client: TestClient, pk: str, origin: str = SITE, **body: str):  # type: ignore[no-untyped-def]
    return client.post(
        "/api/sessioni",
        json={"articolo": "blu-20x20"} | body,
        headers={"X-Posa-Chiave": pk, "Origin": origin},
    )


def _upload(client: TestClient, token: str):  # type: ignore[no-untyped-def]
    files = {"foto": ("stanza.jpg", jpeg(), "image/jpeg")}
    return client.post("/api/pratiche", files=files, headers={"X-Posa-Permesso": token})


def test_any_site_integrates_with_widget_key_and_catalog_api(env) -> None:  # type: ignore[no-untyped-def]
    client, app, runner, _st = env
    pk, sk = _tenant(client)
    assert pk.startswith("pk_live_") and sk.startswith("sk_live_")
    r = _article(client, sk)
    assert r.status_code == 200, r.text
    assert r.json()["articolo"]["formato_cm"] == [20.0, 20.0]
    listed = client.get("/v1/catalogo", headers={"Authorization": f"Bearer {sk}"}).json()
    assert [a["codice"] for a in listed["articoli"]] == ["blu-20x20"]

    # il widget del sito chiede il permesso con la chiave pubblica, dalla pagina del sito
    r = _session(client, pk, superficie="rivestimento", zona="paraschizzi")
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == SITE
    job = _upload(client, r.json()["permesso"]).json()["pratica"]
    app.state.jobs.run_now(job)
    body = client.get(f"/api/pratiche/{job}").json()
    assert body["stato"] == "pronto" and body["superficie"] == "rivestimento"
    assert runner.calls[0]["surface"] == "rivestimento" and runner.calls[0]["zone"] == "paraschizzi"
    assert client.get(body["risultato"]).headers["content-type"] == "image/jpeg"

    widget = client.get("/widget/posa.js")
    assert widget.status_code == 200 and "javascript" in widget.headers["content-type"]
    assert "data-posa" in widget.text and "innerHTML" not in widget.text.split("*/", 1)[1]


def test_publishable_key_only_works_from_registered_sites(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    pk, sk = _tenant(client)
    _article(client, sk)
    assert _session(client, pk, origin="https://sito-copione.example").status_code == 403
    assert _session(client, pk, origin="http://www.ceramiche-bianchi.it").status_code == 403
    assert _session(client, "pk_live_" + "x" * 43).status_code == 401
    assert _session(client, sk).status_code == 401  # la chiave segreta non va nel browser
    # la chiave pubblica non apre le API del server
    assert client.get("/v1/account", headers={"Authorization": f"Bearer {pk}"}).status_code == 401
    # origini aggiuntive (es. sito di prova) si registrano dall'amministrazione
    _tenant(client, origini=["https://staging.ceramiche-bianchi.it"])
    assert _session(client, pk, origin="https://staging.ceramiche-bianchi.it").status_code == 200
    # chiave revocata
    prefix = pk[:14]
    assert client.delete(f"/admin/clienti/bianchi/chiavi/{prefix}", headers=H).status_code == 200
    assert _session(client, pk).status_code == 401
    listed = client.get("/admin/clienti/bianchi/chiavi", headers=H).json()
    assert all("chiave" not in k for k in listed)  # mai le chiavi in chiaro


def test_tenants_cannot_see_each_other(env) -> None:  # type: ignore[no-untyped-def]
    client, app, *_ = env
    pk, sk = _tenant(client)
    _pk2, sk2 = _tenant(client, "verdi", sito="https://www.verdi-pavimenti.it")
    _article(client, sk)
    job = _upload(client, _session(client, pk).json()["permesso"]).json()["pratica"]
    mine = client.get(f"/v1/pratiche/{job}", headers={"Authorization": f"Bearer {sk}"})
    theirs = client.get(f"/v1/pratiche/{job}", headers={"Authorization": f"Bearer {sk2}"})
    assert mine.status_code == 200 and theirs.status_code == 404
    other = client.get("/v1/catalogo", headers={"Authorization": f"Bearer {sk2}"}).json()
    assert other["articoli"] == []  # il catalogo di un cliente non si vede dall'altro
    # il sito di un cliente non può usare la chiave di un altro
    assert _session(client, pk, origin="https://www.verdi-pavimenti.it").status_code == 403


def test_plan_decides_surfaces_and_article_use(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    pk, sk = _tenant(client, piano="base")  # il piano base comprende solo i pavimenti
    _article(client, sk)
    assert _session(client, pk, superficie="pavimento").status_code == 200
    r = _session(client, pk, superficie="rivestimento")
    assert r.status_code == 403 and "piano" in r.json()["errore"]
    _tenant(client, piano="professionale")
    assert _session(client, pk, superficie="rivestimento").status_code == 200
    # un articolo solo da pavimento non si prova a parete
    assert _article(client, sk, "gres-terra", uso="pavimento").status_code == 200
    r = _session(client, pk, articolo="gres-terra", superficie="rivestimento")
    assert r.status_code == 409
    assert _session(client, pk, articolo="inesistente").status_code == 404
    assert _session(client, pk, superficie="soffitto").status_code == 400


def test_monthly_quota_counts_delivered_jobs_only(env) -> None:  # type: ignore[no-untyped-def]
    client, app, runner, _st = env
    plan = {"nome": "Mini", "prove_al_mese": 2, "superfici": ["pavimento"], "canone_eur": 49}
    assert client.put("/admin/piani/mini", json=plan, headers=H).status_code == 200
    pk, sk = _tenant(client, piano="mini")
    _article(client, sk)

    def run(deliver: bool) -> int:
        runner.deliver = deliver
        r = _session(client, pk)
        if r.status_code != 200:
            return r.status_code
        up = _upload(client, r.json()["permesso"])
        if up.status_code == 201:
            app.state.jobs.run_now(up.json()["pratica"])
        return up.status_code

    assert run(True) == 201
    assert run(False) == 201  # non riuscita: non consuma la quota
    assert run(True) == 201
    assert run(True) == 402  # 2 consegnate: quota del mese finita
    account = client.get("/v1/account", headers={"Authorization": f"Bearer {sk}"}).json()
    assert account["prove_comprese"] == 2 and account["prove_consegnate"] == 2
    assert account["prove_non_riuscite"] == 1 and account["piano"] == "mini"
    usage = client.get("/admin/clienti/bianchi/uso", headers=H).json()
    assert usage["mesi"][0]["consegnate"] == 2 and usage["mesi"][0]["costo_agenti_usd"] > 0
    # una quota su misura per il cliente vale più di quella del piano
    _tenant(client, piano="mini", prove_al_mese=5)
    assert run(True) == 201
    # piano a consumo: oltre la quota si continua e le prove extra sono contate
    over = {"prove_al_mese": 1, "oltre_quota_a_consumo": True, "prova_extra_eur": 1.5}
    client.put("/admin/piani/consumo", json=over, headers=H)
    _tenant(client, piano="consumo", prove_al_mese=None)
    assert run(True) == 201
    account = client.get("/v1/account", headers={"Authorization": f"Bearer {sk}"}).json()
    assert account["prove_extra"] == 3 and account["oltre_quota_a_consumo"] is True


def test_too_many_failed_jobs_stop_a_tenant(env) -> None:  # type: ignore[no-untyped-def]
    """Le prove fallite costano comunque: oltre il doppio della quota ci si ferma."""
    client, app, runner, _st = env
    client.put("/admin/piani/mini", json={"prove_al_mese": 2}, headers=H)
    pk, sk = _tenant(client, piano="mini")
    _article(client, sk)
    runner.deliver = False
    codes = []
    for _ in range(5):
        r = _session(client, pk)
        if r.status_code != 200:
            codes.append(r.status_code)
            continue
        up = _upload(client, r.json()["permesso"])
        app.state.jobs.run_now(up.json()["pratica"])
        codes.append(up.status_code)
    assert codes == [201, 201, 201, 201, 402]


def _stripe(client: TestClient, event: dict[str, Any], secret: str = STRIPE, ts: int | None = None):  # type: ignore[no-untyped-def]
    raw = json.dumps(event).encode()
    t = str(ts or int(time.time()))
    sig = hmac.new(secret.encode(), t.encode() + b"." + raw, hashlib.sha256).hexdigest()
    return client.post(
        "/webhooks/stripe", content=raw, headers={"Stripe-Signature": f"t={t},v1={sig}"}
    )


def test_payments_activate_and_suspend_tenants(env) -> None:  # type: ignore[no-untyped-def]
    client, app, *_ = env
    pk, sk = _tenant(client)
    _article(client, sk)
    store: Store = app.state.store
    code = store.tenant("bianchi").billing_token  # type: ignore[union-attr]
    session = {
        "client_reference_id": code,  # il codice di attivazione segreto, non l'id
        "customer": "cus_123",
        "mode": "subscription",
        "payment_status": "paid",
    }
    paid = {"id": "evt_1", "type": "checkout.session.completed", "data": {"object": session}}
    assert _stripe(client, paid, secret="whsec_falso_falso_falso").status_code == 401
    assert _stripe(client, paid, ts=int(time.time()) - 3600).status_code == 401  # evento vecchio
    assert _stripe(client, paid).status_code == 200
    assert store.tenant("bianchi").billing_ref == "cus_123"  # type: ignore[union-attr]

    def sub(event_id: str, status: str, kind: str = "customer.subscription.updated", **meta: str):  # type: ignore[no-untyped-def]
        obj = {"customer": "cus_123", "status": status, "metadata": meta}
        return _stripe(client, {"id": event_id, "type": kind, "data": {"object": obj}})

    assert sub("evt_2", "past_due").status_code == 200
    assert store.tenant("bianchi").status == "sospeso"  # type: ignore[union-attr]
    assert _session(client, pk).status_code == 401  # niente prove finché non paga
    assert client.get("/v1/account", headers={"Authorization": f"Bearer {sk}"}).status_code == 200
    assert sub("evt_3", "active", piano="azienda").status_code == 200
    t = store.tenant("bianchi")
    assert t is not None and t.status == "attivo" and t.plan == "azienda"
    assert _session(client, pk).status_code == 200
    assert sub("evt_2", "past_due").status_code == 200  # replay dello stesso evento: ignorato
    assert store.tenant("bianchi").status == "attivo"  # type: ignore[union-attr]
    assert sub("evt_4", "canceled", kind="customer.subscription.deleted").status_code == 200
    assert store.tenant("bianchi").status == "disdetto"  # type: ignore[union-attr]
    # ri-registrare il cliente non lo riattiva: serve il pagamento o l'amministratore
    _tenant(client)
    assert store.tenant("bianchi").status == "disdetto"  # type: ignore[union-attr]
    r = client.post("/admin/clienti/bianchi/stato", json={"stato": "attivo"}, headers=H)
    assert r.status_code == 200 and store.tenant("bianchi").active  # type: ignore[union-attr]
    assert client.post("/webhooks/stripe", content=b"{}").status_code == 401


def test_catalog_api_validates_articles(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    _pk, sk = _tenant(client)
    auth = {"Authorization": f"Bearer {sk}"}
    assert _article(client, sk, formato="").status_code == 422
    assert _article(client, sk, posa="spina").status_code == 422
    assert _article(client, sk, "Codice Non Valido").status_code == 400
    bad = client.put(
        "/v1/catalogo/x-1",
        data={"nome": "x", "formato": "20x20"},
        files=[("foto", ("a.png", b"<svg onload=alert(1)>", "image/png"))],
        headers=auth,
    )
    assert bad.status_code == 415
    # una foto d'ambiente (non una piastrella) viene scartata con una spiegazione
    room = io.BytesIO()
    Image.fromarray(synthetic.make_room_scene(320, 240).photo).save(room, format="JPEG")
    r = client.put(
        "/v1/catalogo/x-2",
        data={"nome": "x", "formato": "20x20"},
        files=[("foto", ("a.jpg", room.getvalue(), "image/jpeg"))],
        headers=auth,
    )
    assert r.status_code == 422 and "foto" in r.json()["errore"]
    assert client.put("/v1/catalogo/x-3", data={"nome": "x"}).status_code == 401
    assert _article(client, sk).status_code == 200
    assert client.delete("/v1/catalogo/blu-20x20", headers=auth).status_code == 200
    assert client.delete("/v1/catalogo/blu-20x20", headers=auth).status_code == 404


def test_server_to_server_job_and_admin_overview(env) -> None:  # type: ignore[no-untyped-def]
    client, app, runner, _st = env
    _pk, sk = _tenant(client)
    _article(client, sk)
    auth = {"Authorization": f"Bearer {sk}"}
    r = client.post(
        "/v1/pratiche?articolo=blu-20x20&superficie=pavimento",
        data={"stanza": "4x3"},
        files={"foto": ("s.jpg", jpeg(), "image/jpeg")},
        headers=auth,
    )
    assert r.status_code == 201, r.text
    job = r.json()["pratica"]
    app.state.jobs.run_now(job)
    assert client.get(f"/v1/pratiche/{job}", headers=auth).json()["stato"] == "pronto"
    assert runner.calls[0]["room"] == (4.0, 3.0)
    bad = client.post("/v1/pratiche?articolo=blu-20x20", files={"foto": ("s.jpg", b"x", "x")})
    assert bad.status_code == 401
    overview = client.get("/admin/clienti", headers=H).json()
    assert overview[0]["id"] == "bianchi" and overview[0]["prove_consegnate"] == 1
    assert client.get("/admin/clienti").status_code == 401
    plans = {p["id"]: p for p in client.get("/admin/piani", headers=H).json()}
    assert {"prova", "base", "professionale", "azienda"} <= set(plans)
    # fine del rapporto: tutti i dati del cliente spariscono
    assert client.delete("/admin/clienti/bianchi", headers=H).status_code == 200
    assert client.get("/v1/account", headers=auth).status_code == 401
    assert client.get(f"/api/pratiche/{job}").status_code == 404


def test_admin_input_is_validated(env) -> None:  # type: ignore[no-untyped-def]
    client, *_ = env
    bad = [
        {"id": "A B", "sito": SITE},
        {"id": "ok-id", "sito": "http://non-sicuro.it"},
        {"id": "ok-id", "sito": SITE, "piano": "inesistente"},
        {"id": "ok-id", "sito": SITE, "origini": ["javascript:alert(1)"]},
        {"id": "ok-id", "sito": SITE, "canale": "altro"},
    ]
    for body in bad:
        assert client.post("/admin/clienti", json=body, headers=H).status_code == 400, body
    assert client.post("/admin/clienti", content=b"non json", headers=H).status_code == 400
    assert client.put("/admin/piani/x1", json={"prove_al_mese": -5}, headers=H).status_code == 400
    assert client.put("/admin/piani/x1", json={}, headers=H).status_code == 400
    r = client.post("/admin/clienti/sconosciuto/chiavi", json={"tipo": "segreta"}, headers=H)
    assert r.status_code == 404


def test_old_shop_database_is_migrated_and_interrupted_jobs_closed(tmp_path: Path) -> None:
    """Un archivio creato per i soli negozi Shopify continua a funzionare."""
    db = sqlite3.connect(tmp_path / "servizio.db")
    db.executescript(
        """
        CREATE TABLE shops (domain TEXT PRIMARY KEY, store_url TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1, daily_quota INTEGER NOT NULL,
            created_at REAL NOT NULL);
        CREATE TABLE jobs (id TEXT PRIMARY KEY, shop TEXT NOT NULL, article TEXT NOT NULL,
            status TEXT NOT NULL, step TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
            updated_at REAL NOT NULL, ip_hash TEXT NOT NULL, room TEXT,
            cost_usd REAL NOT NULL DEFAULT 0);
        INSERT INTO shops VALUES('rossi.myshopify.com', 'https://www.rossi.it', 1, 80, 1.0);
        INSERT INTO shops VALUES('spento.myshopify.com', 'https://www.spento.it', 0, 80, 1.0);
        """
    )
    now = time.time()
    db.execute(
        "INSERT INTO jobs VALUES('j1', 'rossi.myshopify.com', 'a', 'in_lavorazione', 'posa', "
        "?, ?, 'h', NULL, 0)",
        (now, now),
    )
    db.commit()
    db.close()
    st = Settings(data_dir=tmp_path, shopify_secret=SECRET, admin_token=ADMIN, upload_key="k" * 64)
    app = create_app(st, FakeRunner(), start_workers=False)
    store: Store = app.state.store
    t = store.tenant("rossi.myshopify.com")
    assert t is not None and t.channel == "shopify" and t.active and t.daily_quota == 80
    assert store.tenant("spento.myshopify.com").status == "sospeso"  # type: ignore[union-attr]
    job = store.job("j1")
    assert job is not None and job.surface == "pavimento"
    app.state.jobs.start()  # al riavvio la pratica interrotta viene chiusa
    app.state.jobs.stop()
    assert store.job("j1").status == "non_riuscito"  # type: ignore[union-attr]
    assert np.isclose(store.usage("rossi.myshopify.com").failed, 1)
