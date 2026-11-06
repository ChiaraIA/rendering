"""Strumenti e cancelli del sistema multiagente, con controllori finti (niente modelli)."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.agents import state as st
from app.agents.checkers import CheckRequest
from app.agents.state import ClientRequest, Verdict
from app.agents.tools import Toolbox
from app.catalog.loader import Catalog
from app.core import compositor, icc, ingest
from app.devtools import synthetic
from app.storage.workspace import Workspace
from tests.conftest import forge

SCENE = synthetic.make_room_scene(width=640, height=480)
# poligono del pavimento e del divano in coordinate della preview (640 px = scala 1)
FLOOR = [[0, 215], [50, 170], [585, 185], [640, 205], [640, 480], [0, 480]]


class ScriptedChecker:
    """Controllore finto: approva o boccia secondo un copione, e registra cosa ha visto."""

    def __init__(self, approve: bool | dict[str, list[bool]] = True) -> None:
        self.approve = approve
        self.seen: list[CheckRequest] = []

    async def __call__(self, req: CheckRequest, cwd: Path) -> Verdict:
        self.seen.append(req)
        for img in req.images:
            assert (cwd / img).is_file(), f"il controllore deve poter aprire {img}"
        ok = self.approve
        if isinstance(ok, dict):
            ok = ok[req.checker].pop(0) if ok.get(req.checker) else True
        return Verdict(
            checker=req.checker,
            approved=bool(ok),
            reason="copione",
            issues=[] if ok else ["problema di prova"],
            cost_usd=0.01,
        )


def _floor_mask_polys() -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    ys, xs = np.nonzero(SCENE.floor_mask)
    sofa = [[x, y] for x, y in _sofa_hull()]
    return [FLOOR], [sofa]


def _sofa_hull() -> list[tuple[float, float]]:
    import cv2

    floor_region = SCENE.floor_mask == 0
    below = np.zeros_like(floor_region)
    below[150:, :] = True
    sofa = (floor_region & below).astype(np.uint8)
    sofa[:, :40] = 0
    sofa[:, 600:] = 0
    cnts, _ = cv2.findContours(sofa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(big)[:, 0, :]
    return [(float(x), float(y)) for x, y in hull]


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    return synthetic.save_image(
        SCENE.photo, tmp_path / "stanza.jpg", fmt="JPEG", quality=97, focal_35mm=26
    )


def _ws(tmp_path: Path, photo: Path, **req: object) -> Workspace:
    request = ClientRequest(**({"material_id": "gres-grigio-30x30", "room_mm": (4200, 3600)} | req))
    return Workspace.create(photo, tmp_path / "pratica", request)


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


async def _happy_path(tb: Toolbox) -> dict:
    floor, excl = _floor_mask_polys()
    assert (await tb.proponi_maschera(floor, excl, True))["esito"] == "APPROVATO"
    corners = SCENE.corners_px.tolist()  # preview = foto (640 px)
    assert (await tb.proponi_geometria(corners))["esito"] == "APPROVATO"
    assert (await tb.scegli_materiale("gres-grigio-30x30"))["esito"] == "APPROVATO"
    assert (await tb.esegui_posa())["esito"] == "APPROVATO"
    return await tb.consegna()


def test_happy_path_delivers_and_keeps_original(
    tmp_path: Path, photo: Path, catalog: Catalog
) -> None:
    before = ingest.sha256_file(photo)
    ws = _ws(tmp_path, photo)
    checker = ScriptedChecker()
    tb = Toolbox(ws, catalog, checker)
    out = _run(_happy_path(tb))
    assert "consegnato" in out
    files = {Path(f).name for f in out["consegnato"]}
    assert {"final.png", "mask.png", "preview.jpg", "prima_dopo.jpg", "report.json"} <= files

    # integrità sul file consegnato: fuori maschera identico all'originale normalizzato
    final = np.asarray(Image.open(ws.root / "consegna" / "final.png"))
    alpha = np.asarray(Image.open(ws.root / "consegna" / "mask.png"))
    original = ws.source.pixels
    assert final.shape == original.shape
    assert np.array_equal(final[alpha == 0], original[alpha == 0])
    assert not np.array_equal(final[alpha == 255], original[alpha == 255])

    # originale del cliente intatto, copia in sola lettura
    assert ingest.sha256_file(photo) == before
    copy = ws.root / "originale" / photo.name
    assert stat.S_IMODE(copy.stat().st_mode) == 0o444
    assert ws.verify_original()

    # ogni controllore ha visto il lavoro, nell'ordine giusto
    assert [r.checker for r in checker.seen] == [
        "controllo-maschera",
        "controllo-geometria",
        "revisore-finale",
    ]
    report = json.loads((ws.root / "consegna" / "report.json").read_text())
    assert report["sha256_risultato"] == ws.state.render.final_sha256  # type: ignore[union-attr]
    assert report["posa"]["tiles_whole"] > 0


def test_gates_block_render_and_delivery(tmp_path: Path, photo: Path, catalog: Catalog) -> None:
    tb = Toolbox(_ws(tmp_path, photo), catalog, ScriptedChecker())
    assert "cancello chiuso" in _run(tb.esegui_posa())["errore"]
    assert "cancello chiuso" in _run(tb.consegna())["errore"]
    assert "maschera" in _run(tb.proponi_geometria([[0, 0]] * 4))["errore"]


def test_rejection_is_not_approval_and_attempts_are_limited(
    tmp_path: Path, photo: Path, catalog: Catalog
) -> None:
    ws = _ws(tmp_path, photo)
    tb = Toolbox(ws, catalog, ScriptedChecker(approve=False))
    floor, excl = _floor_mask_polys()
    for _ in range(3):
        r = _run(tb.proponi_maschera(floor, excl, False))
        assert r["esito"] == "BOCCIATO" and r["problemi"] == ["problema di prova"]
    assert ws.state.mask is None
    assert "limite" in _run(tb.proponi_maschera(floor, excl, False))["errore"]
    assert _run(tb.richiedi_operatore("prova"))["ok"]
    assert (ws.root / "ESCALATION.txt").is_file()
    assert "pratica non in escalation" in st.missing_for_delivery(ws.state)


def test_retry_after_rejection_then_approval(tmp_path: Path, photo: Path, catalog: Catalog) -> None:
    ws = _ws(tmp_path, photo)
    tb = Toolbox(ws, catalog, ScriptedChecker({"controllo-maschera": [False, True]}))
    floor, excl = _floor_mask_polys()
    assert _run(tb.proponi_maschera(floor, excl))["esito"] == "BOCCIATO"
    assert _run(tb.proponi_maschera(floor, excl))["esito"] == "APPROVATO"
    assert ws.state.mask is not None and ws.state.mask.approved_by == "controllo-maschera"
    assert len(ws.state.mask_attempts) == 2


def test_integrity_failure_blocks_delivery(
    tmp_path: Path, photo: Path, catalog: Catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = compositor.composite

    def broken(*a, **k):  # type: ignore[no-untyped-def]
        res = real(*a, **k)
        bad = res.pixels.copy()
        ys, xs = np.nonzero(res.alpha == 0)
        bad[ys[0], xs[0], 0] ^= 1  # 1 bit fuori dal pavimento
        return forge(bad, res.alpha)

    monkeypatch.setattr(compositor, "composite", broken)
    ws = _ws(tmp_path, photo)
    checker = ScriptedChecker()
    tb = Toolbox(ws, catalog, checker)
    floor, excl = _floor_mask_polys()
    _run(tb.proponi_maschera(floor, excl))
    _run(tb.proponi_geometria(SCENE.corners_px.tolist()))
    _run(tb.scegli_materiale("gres-grigio-30x30"))
    r = _run(tb.esegui_posa())
    assert r["esito"] == "BOCCIATO" and r["controllore"] == "controllo-integrità"
    assert "revisore-finale" not in [c.checker for c in checker.seen]  # nemmeno mostrato
    assert "cancello chiuso" in _run(tb.consegna())["errore"]
    assert not any((ws.root / "consegna").iterdir())


def test_client_code_is_binding(tmp_path: Path, photo: Path, catalog: Catalog) -> None:
    tb = Toolbox(_ws(tmp_path, photo), catalog, ScriptedChecker())
    assert "usa quello" in _run(tb.scegli_materiale("marmo-bianco-60x60"))["errore"]
    assert "non presente" in _run(tb.scegli_materiale("legno-inventato"))["errore"]


def test_sample_matching_goes_through_checker(
    tmp_path: Path, photo: Path, catalog: Catalog
) -> None:
    sample = synthetic.save_image(
        synthetic.make_tile_texture((240, 40), (164, 120, 78), kind="wood"),
        tmp_path / "campione.png",
    )
    ws = _ws(tmp_path, photo, material_id=None, sample_file=str(sample))
    checker = ScriptedChecker()
    tb = Toolbox(ws, catalog, checker)
    r = _run(tb.scegli_materiale("laminato-rovere-120x20", "venatura rovere"))
    assert r["esito"] == "APPROVATO" and checker.seen[-1].checker == "controllo-materiale"
    assert "campione.jpg" in checker.seen[-1].images
    assert ws.state.material.how == "abbinamento_campione"  # type: ignore[union-attr]


def test_geometry_inconsistent_with_exif_focal_is_rejected(
    tmp_path: Path, photo: Path, catalog: Catalog
) -> None:
    ws = _ws(tmp_path, photo, room_mm=(4200, 2500))  # larghezza sbagliata di ~30%
    tb = Toolbox(ws, catalog, ScriptedChecker())
    floor, excl = _floor_mask_polys()
    _run(tb.proponi_maschera(floor, excl))
    r = _run(tb.proponi_geometria(SCENE.corners_px.tolist()))
    assert r["esito"] == "BOCCIATO" and ws.state.geometry is None
    assert "focale" in r["motivo"]


def test_degenerate_geometry_is_rejected_by_code(
    tmp_path: Path, photo: Path, catalog: Catalog
) -> None:
    ws = _ws(tmp_path, photo)
    tb = Toolbox(ws, catalog, ScriptedChecker())
    floor, excl = _floor_mask_polys()
    _run(tb.proponi_maschera(floor, excl))
    r = _run(tb.proponi_geometria([[0, 0], [100, 0], [200, 0], [300, 1]]))
    assert r["esito"] == "BOCCIATO" and r["controllore"] == "controllo-automatico"


def test_tool_traces_and_p3_photo_keeps_icc(tmp_path: Path, catalog: Catalog) -> None:
    p3 = synthetic.save_heic(SCENE.photo, tmp_path / "IMG.HEIC", primaries=12, focal_35mm=26)
    ws = Workspace.create(
        p3, tmp_path / "p3", ClientRequest(material_id="gres-grigio-30x30", room_mm=(4200, 3600))
    )
    assert ws.source.icc == icc.display_p3()
    tb = Toolbox(ws, catalog, ScriptedChecker())
    _run(_happy_path(tb))
    final = Image.open(ws.root / "consegna" / "final.png")
    assert final.info.get("icc_profile") == icc.display_p3()


def test_next_step_follows_the_procedure(tmp_path: Path, photo: Path, catalog: Catalog) -> None:
    ws = _ws(tmp_path, photo)
    assert st.next_step(ws.state).startswith("maschera")
    tb = Toolbox(ws, catalog, ScriptedChecker())
    _run(_happy_path(tb))
    assert st.next_step(ws.state).startswith("completato")
    status = _run(tb.stato_pratica())
    assert status["consegnato"] and status["mancano_per_consegna"] == []
