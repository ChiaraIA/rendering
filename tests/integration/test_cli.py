from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from app.catalog.loader import Catalog
from app.cli import app
from app.core import compositor
from app.devtools import synthetic
from tests.conftest import MATERIAL, forge

runner = CliRunner()


@pytest.fixture
def files(tmp_path: Path, catalog: Catalog) -> dict[str, Path]:
    return {
        "photo": synthetic.save_image(
            synthetic.make_photo(80, 60), tmp_path / "foto.jpg", fmt="JPEG"
        ),
        "mask": synthetic.save_image(synthetic.make_mask(60, 80, "holes"), tmp_path / "m.png"),
        "badmask": synthetic.save_image(synthetic.make_mask(10, 10), tmp_path / "bad.png"),
        "catalog": catalog.path,
        "job": tmp_path / "job",
    }


def _sha_from(output: str) -> str:
    line = next(ln for ln in output.splitlines() if "sha256" in ln)
    return line.split()[-1]


def test_cli_flow_and_exit_codes(files: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    job, cat = str(files["job"]), str(files["catalog"])
    render_cmd = ["render", job, "--material", MATERIAL, "--catalog", cat]

    r = runner.invoke(app, ["ingest", str(files["photo"]), "--job", job])
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["ingest", str(files["photo"]), "--job", job]).exit_code == 2

    assert runner.invoke(app, render_cmd).exit_code == 3  # nessuna maschera
    assert runner.invoke(app, ["mask", "import", job, str(files["badmask"])]).exit_code == 2

    r = runner.invoke(app, ["mask", "import", job, str(files["mask"])])
    assert r.exit_code == 0, r.output
    sha = _sha_from(r.output)
    assert runner.invoke(app, render_cmd).exit_code == 3  # non confermata
    assert runner.invoke(app, ["mask", "confirm", job, "--sha", "deadbeefdeadbeef"]).exit_code == 3

    assert runner.invoke(app, ["mask", "confirm", job, "--sha", sha[:12]]).exit_code == 0
    r = runner.invoke(app, render_cmd)
    assert r.exit_code == 0, r.output
    assert "VALIDO" in r.output
    assert (files["job"] / "renders" / "render-001" / "final.png").is_file()

    assert (
        runner.invoke(app, ["render", job, "--material", "nope", "--catalog", cat]).exit_code == 2
    )
    assert runner.invoke(app, [*render_cmd, "--format", "bmp"]).exit_code == 2

    real = compositor.composite

    def broken(*a, **k):  # type: ignore[no-untyped-def]
        res = real(*a, **k)
        bad = res.pixels.copy()
        bad[0, 0, 0] ^= 1
        return forge(bad, res.alpha)

    monkeypatch.setattr(compositor, "composite", broken)
    r = runner.invoke(app, render_cmd)
    assert r.exit_code == 5
    assert "NON VALIDO" in r.output
    assert not list((files["job"] / "renders" / "render-002").glob("final.*"))

    r = runner.invoke(app, ["status", job])
    assert r.exit_code == 0 and "render-002 NON valido" in r.output
    assert runner.invoke(app, ["status", str(files["job"].parent / "nojob")]).exit_code == 2


def test_cli_catalog_list(files: dict[str, Path]) -> None:
    r = runner.invoke(app, ["catalog", "list", "--catalog", str(files["catalog"])])
    assert r.exit_code == 0 and MATERIAL in r.output
    assert runner.invoke(app, ["catalog", "list", "--catalog", "/nope.json"]).exit_code == 2


def test_demo_runs_fast(tmp_path: Path) -> None:
    t0 = time.perf_counter()
    r = runner.invoke(app, ["demo", "--out", str(tmp_path / "demo")])
    elapsed = time.perf_counter() - t0
    assert r.exit_code == 0, r.output
    render_dir = tmp_path / "demo" / "job" / "renders" / "render-001"
    assert {p.name for p in render_dir.iterdir()} == {
        "final.png",
        "mask.png",
        "preview.jpg",
        "report.json",
    }
    assert elapsed < 10, f"demo 2000x1500 troppo lenta: {elapsed:.1f}s"
    from PIL import Image

    assert np.asarray(Image.open(render_dir / "final.png")).shape == (1500, 2000, 3)
