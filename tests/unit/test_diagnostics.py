"""`posa verifica`: diagnosi dell'installazione e avviso sulle credenziali."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from app import diagnostics
from app.cli import app

runner = CliRunner()


def _no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (*diagnostics.CREDENTIAL_VARS, "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_offline_checks_pass_in_a_working_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    checks = {c.name: c for c in diagnostics.offline_checks()}
    assert all(c.level == "ok" for c in checks.values()), checks
    assert "sk-ant-test" not in " ".join(c.detail for c in checks.values())  # mai stampata


def test_missing_key_is_a_warning_with_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_credentials(monkeypatch)
    assert "ANTHROPIC_API_KEY" in (diagnostics.credentials_problem() or "")
    r = runner.invoke(app, ["verifica"])
    assert r.exit_code == 0  # l'installazione è a posto: non deve far fallire il Codespace
    assert "Credenziali API" in r.output and "Secrets" in r.output
    assert "risolvi gli avvisi" in r.output


def test_run_stops_early_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_credentials(monkeypatch)
    r = runner.invoke(app, ["run", "foto.jpg", "-m", "gres-grigio-30x30"])
    assert r.exit_code == 2
    assert "ANTHROPIC_API_KEY" in r.output


def test_broken_module_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    check = diagnostics._module("Modulo finto", "modulo_che_non_esiste_123")
    assert check.level == "errore" and "pip install" in check.fix


def test_first_free_port_skips_busy_ports() -> None:
    import socket

    from app.cli import first_free_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("0.0.0.0", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        free = first_free_port(taken)
        assert free is not None and free != taken and taken < free < taken + 20
