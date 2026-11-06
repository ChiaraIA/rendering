"""Confini architetturali verificati staticamente (regole 4, 5 e core senza AI)."""

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


def test_only_compositor_builds_composite_result() -> None:
    offenders = [
        _rel(p)
        for p, tree in _modules()
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and _call_name(n) == "CompositeResult"
        and _rel(p) != "core/compositor.py"
    ]
    assert offenders == []


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
    offenders = [
        f"{_rel(p)}:{n.lineno}"
        for p, tree in _modules()
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and _call_name(n) in {"save", "imwrite"}
        and not _rel(p).startswith(allowed)
    ]
    assert offenders == []


def test_final_master_written_only_by_exporter() -> None:
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


def _imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out.update(f"{n.module}.{a.name}" for a in n.names)
    return out


def test_adapters_cannot_reach_compositor_or_exporter() -> None:
    forbidden = ("app.core.compositor", "app.storage")
    for p, tree in _modules():
        if _rel(p).startswith("adapters/"):
            bad = [i for i in _imports(tree) if i.startswith(forbidden)]
            assert bad == [], f"{_rel(p)} importa {bad}"


def test_core_does_not_depend_on_outer_layers() -> None:
    forbidden = ("app.adapters", "app.api", "app.pipeline", "app.storage", "app.cli")
    for p, tree in _modules():
        if _rel(p).startswith("core/"):
            bad = [i for i in _imports(tree) if i.startswith(forbidden)]
            assert bad == [], f"{_rel(p)} importa {bad}"


def test_core_import_pulls_no_ai_or_web_deps() -> None:
    code = (
        "import sys, app.core.compositor, app.core.validator, app.pipeline.render, app.cli;"
        "heavy=('torch','google.genai','fastapi','segment_anything');"
        "bad=[m for m in heavy if m in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
