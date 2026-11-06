"""Confini architetturali verificati staticamente.

- solo il compositor crea l'immagine finale;
- solo storage/devtools scrivono immagini, e il master solo l'exporter;
- solo gli strumenti (dopo il verdetto di un controllore) possono approvare una maschera;
- il core non dipende da agenti, web o AI.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"


def _modules() -> list[tuple[Path, ast.Module]]:
    return [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in sorted(APP.rglob("*.py"))]


def _rel(p: Path) -> str:
    return p.relative_to(APP).as_posix()


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _calls(name: str, allowed: tuple[str, ...]) -> list[str]:
    return [
        f"{_rel(p)}:{n.lineno}"
        for p, tree in _modules()
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _call_name(n) == name and not _rel(p).startswith(allowed)
    ]


def test_only_compositor_builds_composite_result() -> None:
    assert _calls("CompositeResult", ("core/compositor.py",)) == []


def test_seal_is_private_to_compositor() -> None:
    offenders = [
        _rel(p)
        for p, tree in _modules()
        for n in ast.walk(tree)
        if (isinstance(n, ast.Name) and n.id == "_SEAL")
        or (isinstance(n, ast.Attribute) and n.attr == "_SEAL")
        if _rel(p) != "core/compositor.py"
    ]
    assert offenders == []


def test_image_writes_only_in_storage_and_devtools() -> None:
    allowed = ("storage/", "devtools/")
    assert _calls("save", allowed) == []
    assert _calls("imwrite", allowed) == []


def test_master_written_only_by_exporter() -> None:
    assert _calls("export_master", ("storage/exporter.py", "agents/tools.py")) == []
    offenders = [
        f"{_rel(p)}:{n.lineno}"
        for p, tree in _modules()
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith("final.")
        and _rel(p) != "storage/exporter.py"
    ]
    assert offenders == []


def test_only_tools_approve_masks() -> None:
    """Le approvazioni nascono solo dentro gli strumenti, dopo un verdetto del controllore."""
    assert _calls("approve", ("core/masking.py", "agents/tools.py")) == []


def _imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out.update(f"{n.module}.{a.name}" for a in n.names)
    return out


def test_core_does_not_depend_on_outer_layers() -> None:
    forbidden = (
        "app.agents",
        "app.storage",
        "app.cli",
        "app.web",
        "app.catalog",
        "claude_agent_sdk",
        "gradio",
    )
    for p, tree in _modules():
        if _rel(p).startswith("core/"):
            bad = [i for i in _imports(tree) if i.startswith(forbidden)]
            assert bad == [], f"{_rel(p)} importa {bad}"


def test_core_import_pulls_no_ai_or_web_deps() -> None:
    code = (
        "import sys, app.core.compositor, app.core.validator, app.core.material, app.core.lighting;"
        "heavy=('torch','claude_agent_sdk','gradio','fastapi');"
        "bad=[m for m in heavy if m in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
