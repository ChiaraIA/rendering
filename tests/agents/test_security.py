"""Pentest dei confini degli agenti: cosa può fare un agente manipolato da una foto malevola."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from PIL import Image

from app.agents.security import FORBIDDEN_BUILTINS, locked_options, redact, view_image
from app.agents.tools import Toolbox
from tests.agents.test_toolbox import ScriptedChecker, _ws, photo  # noqa: F401, F811


def _job(tmp_path: Path) -> Path:
    root = tmp_path / "pratica"
    (root / "lavoro").mkdir(parents=True)
    Image.new("RGB", (40, 30), (100, 100, 100)).save(root / "preview.jpg")
    Image.new("RGB", (40, 30), (100, 100, 100)).save(root / "lavoro" / "maschera_1.jpg")
    secret = tmp_path / "segreti.txt"
    secret.write_text("sk-ant-api03-SEGRETISSIMO")
    other = tmp_path / "altra_pratica"
    other.mkdir()
    Image.new("RGB", (40, 30), (1, 2, 3)).save(other / "foto_cliente.jpg")
    return root


@pytest.mark.parametrize(
    "name",
    [
        "../segreti.txt",
        "../altra_pratica/foto_cliente.jpg",
        "/etc/passwd",
        "lavoro/../../segreti.txt",
        "..\\altra_pratica\\foto_cliente.jpg",
        "%2e%2e/segreti.txt",
        "lavoro/sub/../../../segreti.txt",
        "originale/foto.jpg",
        "stato.json",
        "lavoro/tracce.jsonl",
        "",
        "a" * 5000 + ".jpg",
    ],
)
def test_guarda_cannot_leave_the_job(tmp_path: Path, name: str) -> None:
    root = _job(tmp_path)
    out = view_image(root, name)
    assert out.get("isError"), name
    assert "SEGRETISSIMO" not in str(out)


def test_guarda_blocks_symlinks_and_non_images(tmp_path: Path) -> None:
    root = _job(tmp_path)
    os.symlink(tmp_path / "altra_pratica" / "foto_cliente.jpg", root / "lavoro" / "link.jpg")
    assert view_image(root, "lavoro/link.jpg").get("isError")
    (root / "lavoro" / "finto.jpg").write_text("sk-ant-api03-SEGRETISSIMO")
    out = view_image(root, "lavoro/finto.jpg")
    assert out.get("isError") and "SEGRETISSIMO" not in str(out)
    ok = view_image(root, "lavoro/maschera_1.jpg")
    assert not ok.get("isError") and ok["content"][1]["type"] == "image"


def test_checker_sees_only_its_images(tmp_path: Path) -> None:
    root = _job(tmp_path)
    allowed = {"lavoro/maschera_1.jpg"}
    assert not view_image(root, "lavoro/maschera_1.jpg", allowed).get("isError")
    assert view_image(root, "preview.jpg", allowed).get("isError")


def test_locked_options_forbid_everything_dangerous() -> None:
    opts = locked_options(system_prompt="x", tools=["Read", "Bash"], allowed_tools=["Bash"])
    assert set(FORBIDDEN_BUILTINS) <= set(opts.disallowed_tools)
    assert {"Bash", "Read", "Write", "WebFetch"} <= set(opts.disallowed_tools)
    assert opts.setting_sources == [] and opts.strict_mcp_config
    assert opts.permission_mode == "dontAsk"


def test_redact_removes_secrets_and_paths() -> None:
    text = (
        "fatto. chiave sk-ant-api03-abcdefghijklmnop token shpat_0123456789abcdef "
        "password=hunter2 file /home/user/dati/servizio.db"
    )
    out = redact(text)
    for leak in ("sk-ant", "shpat_", "hunter2", "/home/user"):
        assert leak not in out
    assert redact("parola " * 1000, 100).endswith("…")


def test_tool_arguments_are_bounded(tmp_path: Path, photo: Path, catalog) -> None:  # type: ignore[no-untyped-def]  # noqa: F811
    tb = Toolbox(_ws(tmp_path, photo), catalog, ScriptedChecker())
    huge = [[[float(i % 640), float(i % 480)] for i in range(200_000)]]
    r = asyncio.run(tb.proponi_maschera(huge, None, False, "poligoni"))
    assert "errore" in r or r.get("esito") == "BOCCIATO"
    many = [[[0, 0], [10, 0], [10, 10]]] * 500
    r = asyncio.run(tb.proponi_maschera(many, None, False, "poligoni"))
    assert "errore" in r
    nan = asyncio.run(tb.proponi_geometria([[float("nan"), 0], [1, 0], [1, 1], [0, 1]]))
    assert "errore" in nan or nan.get("esito") == "BOCCIATO"
    big = asyncio.run(tb.catalogo("x" * 100_000))
    assert big["mostrati"] == 0
    r = asyncio.run(tb.richiedi_operatore("A" * 1_000_000))
    assert len((tb.ws.root / "ESCALATION.txt").read_text()) < 5000
