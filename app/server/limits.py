"""Limiti sul corpo delle richieste, applicati mentre i byte arrivano.

Il controllo su Content-Length non basta: con l'invio "a pezzi" (chunked) la dimensione
non è dichiarata e il parser dei moduli scriverebbe tutto su disco prima di qualunque
verifica. Qui la richiesta viene interrotta appena supera il limite, o se il corpo arriva
troppo lentamente (connessioni tenute aperte apposta, "slowloris").
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

Scope = dict[str, Any]
Message = dict[str, Any]


class _Abort(Exception):
    def __init__(self, status: int, msg: str) -> None:
        self.status, self.msg = status, msg


class BodyLimitMiddleware:
    def __init__(
        self,
        app: Any,
        max_bytes: int,
        chunk_timeout_s: float = 30.0,
        total_timeout_s: float = 300.0,
        min_rate_bps: float = 4096.0,
        grace_s: float = 10.0,
        limit_for: Callable[[str], int] | None = None,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.limit_for = limit_for  # limite per percorso (es. più foto nel catalogo)
        self.chunk_timeout_s = chunk_timeout_s
        self.total_timeout_s = total_timeout_s
        self.min_rate_bps = min_rate_bps  # sotto questa velocità media il corpo è un attacco
        self.grace_s = grace_s

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0
        max_bytes = self.limit_for(scope.get("path", "")) if self.limit_for else self.max_bytes
        started = time.monotonic()
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            remaining = self.total_timeout_s - (time.monotonic() - started)
            try:
                msg: Message = await asyncio.wait_for(
                    receive(), timeout=max(0.1, min(self.chunk_timeout_s, remaining))
                )
            except TimeoutError:
                raise _Abort(408, "invio troppo lento") from None
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > max_bytes:
                    raise _Abort(413, "richiesta troppo grande")
                elapsed = time.monotonic() - started
                if (
                    msg.get("more_body")
                    and elapsed > self.grace_s
                    and received / elapsed < self.min_rate_bps
                ):
                    raise _Abort(408, "invio troppo lento")
            return msg

        async def tracking_send(msg: Message) -> None:
            nonlocal response_started
            if msg["type"] == "http.response.start":
                response_started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _Abort as abort:
            if response_started:
                return
            body = json.dumps({"errore": abort.msg}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": abort.status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"connection", b"close"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
