"""Verifica delle firme di Shopify (App Proxy e webhook) e dei permessi di caricamento.

Tutti i confronti sono a tempo costante (`hmac.compare_digest`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from collections.abc import Mapping, Sequence

SHOP_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}\.myshopify\.com$")


def proxy_signature(params: Mapping[str, Sequence[str]], secret: str) -> str:
    """Firma dell'App Proxy: parametri ordinati 'chiave=valore' concatenati, valori multipli
    uniti da virgola, HMAC-SHA256 esadecimale (documentazione Shopify App Proxy)."""
    items = sorted((k, ",".join(v)) for k, v in params.items() if k != "signature")
    message = "".join(f"{k}={v}" for k, v in items)
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def verify_proxy(
    params: Mapping[str, Sequence[str]], secret: str, max_age_s: int, now: float | None = None
) -> str | None:
    """Dominio del negozio se la richiesta arriva davvero da Shopify, altrimenti None."""
    if not secret:
        return None
    sig = (params.get("signature") or [""])[0]
    if not re.fullmatch(r"[0-9a-f]{64}", sig):
        return None
    if not hmac.compare_digest(proxy_signature(params, secret), sig):
        return None
    try:
        ts = int((params.get("timestamp") or ["0"])[0])
    except ValueError:
        return None
    if abs((now or time.time()) - ts) > max_age_s:  # niente riuso di richieste vecchie
        return None
    shop = (params.get("shop") or [""])[0].lower()
    return shop if SHOP_DOMAIN.match(shop) else None


def verify_webhook(body: bytes, header_hmac: str, secret: str) -> bool:
    if not secret or not header_hmac:
        return False
    digest = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    return hmac.compare_digest(digest, header_hmac)


# ------------------------------------------------------------ permessi di caricamento
def make_upload_token(key: str, shop: str, article: str, ttl_s: int, nonce: str) -> str:
    payload = {"s": shop, "a": article, "e": int(time.time()) + ttl_s, "n": nonce}
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    sig = hmac.new(key.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def read_upload_token(key: str, token: str, now: float | None = None) -> dict[str, str] | None:
    if not token or len(token) > 1024 or token.count(".") != 1:
        return None
    raw, sig = token.split(".")
    expected = hmac.new(key.encode(), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(raw.encode()))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("e", 0)) < (now or time.time()):
        return None
    return {"shop": str(payload["s"]), "article": str(payload["a"]), "nonce": str(payload["n"])}
