"""Rivestimenti a parete: maschera della zona, più pareti, posa dal basso, cancelli."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.agents import prompts
from app.agents import state as st
from app.agents.state import ClientRequest
from app.agents.tools import Toolbox
from app.catalog.loader import Catalog
from app.core import geometry, material
from app.devtools import synthetic
from app.storage.workspace import Workspace
from tests.agents.test_toolbox import ScriptedChecker

K = synthetic.make_kitchen_scene()
MATERIAL = "gres-grigio-30x30"


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    return synthetic.save_image(
        K.photo, tmp_path / "cucina.jpg", fmt="JPEG", quality=97, focal_35mm=26
    )


def _ws(tmp_path: Path, photo: Path, **req: object) -> Workspace:
    request = ClientRequest(
        **({"material_id": MATERIAL, "surface": "rivestimento", "zone": "paraschizzi"} | req)
    )
    return Workspace.create(photo, tmp_path / "pratica", request)


def _wall_segmenter(img: np.ndarray) -> np.ndarray:
    """Come il modello vero: riconosce TUTTO il muro (anche sopra i pensili), e non sa
    che cosa sia una presa (resta dentro)."""
    m = K.wall_mask.copy()
    x0, y0, x1, y1 = K.socket_box
    m[y0:y1, x0:x1] = 255
    m[:60] = 255  # un altro pezzo di muro, lontano dalla zona richiesta
    return m


def _zone() -> list[list[list[float]]]:
    """Poligono abbondante attorno al paraschizzi (preview = foto, scala 1)."""
    h, w = K.wall_mask.shape
    return [[[0, 290], [w, 290], [w, 520], [0, 520]]]


def _walls() -> list[dict[str, object]]:
    return [dict(w, angoli=[list(p) for p in w["angoli"]]) for w in K.walls]  # type: ignore[call-overload]


async def _mask(tb: Toolbox) -> dict:
    return await tb.proponi_maschera(
        base="automatica", area=_zone(), escludi_oggetti=[list(_pad(K.socket_box, 10))]
    )


def _pad(box: tuple[int, int, int, int], m: int) -> tuple[int, int, int, int]:
    return (box[0] - m, box[1] - m, box[2] + m, box[3] + m)


def test_wall_job_tiles_only_the_wall_zone(tmp_path: Path, photo: Path, catalog: Catalog) -> None:
    ws = _ws(tmp_path, photo)
    checker = ScriptedChecker()
    tb = Toolbox(ws, catalog, checker, segmenter=_wall_segmenter)
    assert st.next_step(ws.state).startswith("maschera: delega all'agente 'parete'")
    # senza `area` il modello darebbe tutti i muri della foto: il codice lo rifiuta
    assert "area" in _run(tb.proponi_maschera(base="automatica"))["errore"]
    assert _run(_mask(tb))["esito"] == "APPROVATO"
    assert checker.seen[-1].instructions == prompts.CHECK_MASK_WALL
    assert "paraschizzi" in checker.seen[-1].context
    mask = ws.mask(ws.state.mask.mask_sha256)  # type: ignore[union-attr]
    x0, y0, x1, y1 = K.socket_box
    assert mask[y0 + 3 : y1 - 3, x0 + 3 : x1 - 3].max() == 0  # la presa è fuori
    assert mask[:60].max() == 0  # il muro fuori dalla zona richiesta è fuori
    vx0, vy0, vx1, vy1 = K.vase_box
    assert mask[vy0 + 8 : vy1 - 20, vx0 + 8 : vx1 - 8].max() == 0  # il vaso davanti al muro
    agree = np.mean((mask >= 128) == (K.wall_mask >= 128))
    assert agree > 0.995

    # geometria: lo strumento dei pavimenti non vale per le pareti, e viceversa
    assert "parete" in _run(tb.proponi_geometria([[0, 0]] * 4, 4, 3))["errore"]
    r = _run(tb.proponi_geometria_parete(_walls()))
    assert r["esito"] == "APPROVATO", r
    assert checker.seen[-1].instructions == prompts.CHECK_GEOMETRY_WALL
    assert [p.kind for p in ws.state.planes] == ["parete", "parete"]
    assert ws.state.planes[0].focal.consistent is True  # misure coerenti con la focale

    assert _run(tb.scegli_materiale(MATERIAL))["esito"] == "APPROVATO"
    r = _run(tb.proponi_luce())
    assert r["esito"] == "APPROVATO" and r["regolazioni"]["contatto"] == 0.35
    r = _run(tb.esegui_posa())
    assert r["esito"] == "APPROVATO"
    final_check = checker.seen[-1]
    assert final_check.checker == "revisore-finale" and len(final_check.images) == 3
    assert "bordi" in final_check.images[1]  # ingrandimenti dei bordi a piena risoluzione
    out = _run(tb.consegna())
    assert "consegnato" in out

    final = np.asarray(Image.open(ws.root / "consegna" / "final.png"))
    alpha = np.asarray(Image.open(ws.root / "consegna" / "mask.png"))
    original = ws.source.pixels
    assert np.array_equal(final[alpha == 0], original[alpha == 0])  # fuori: identico
    core = (slice(y0 + 3, y1 - 3), slice(x0 + 3, x1 - 3))
    assert np.array_equal(final[core], original[core])  # la presa è intatta
    inside = alpha == 255
    changed = np.any(final[inside] != original[inside], axis=-1).mean()
    assert changed > 0.95  # il muro della zona è davvero rivestito
    report = json.loads((ws.root / "consegna" / "report.json").read_text())
    assert report["superficie"] == "rivestimento" and len(report["pareti_m"]) == 2
    assert report["stanza_m"] is None and report["posa"]["tiles_whole"] > 0


def test_wall_rows_start_full_at_the_bottom_and_continue_round_the_corner() -> None:
    size = (K.photo.shape[1], K.photo.shape[0])
    planes = [
        geometry.room_geometry(
            [tuple(p) for p in w["angoli"]],  # type: ignore[union-attr]
            float(w["larghezza_m"]) * 1000,  # type: ignore[arg-type]
            float(w["altezza_m"]) * 1000,  # type: ignore[arg-type]
            size,
            26,
            kind="parete",
        )
        for w in K.walls
    ]
    labels = geometry.assign_planes(planes, K.photo.shape[:2])
    back = (K.wall_mask > 0) & (labels == 0)
    left = (K.wall_mask > 0) & (labels == 1)
    assert back.sum() > 5 * left.sum() > 0  # la parete sinistra è in scorcio
    # ogni pixel del paraschizzi cade nel rettangolo reale della sua parete
    for k, sel in ((0, back), (1, left)):
        ys, xs = np.nonzero(sel)
        u, v, _ = geometry.plane_coords(np.asarray(planes[k].H_inv), xs + 0.5, ys + 0.5)
        assert u.min() > -30 and u.max() < planes[k].length_mm + 30
        assert v.min() > -30 and v.max() < planes[k].width_mm + 30
    geometry.check_planes_in_front(planes, labels, K.wall_mask)


def test_wall_laying_anchors_first_row_on_the_bottom_edge(catalog: Catalog) -> None:
    spec = catalog.get(MATERIAL)
    height = 550.0
    lay = material.Laying.of(spec, 4000.0, height, "parete")
    # il bordo inferiore della zona (v = altezza) coincide con il bordo di una piastrella:
    # appena sopra si è dentro la piastrella, appena sotto nella fuga o nella fila dopo
    a, b = lay.local(np.array([2000.0, 2000.0]), np.array([height - 0.5, height + 0.5]))
    i, j, s, t, inside = lay.locate(a, b)
    assert inside[0] and t[0] > spec.tile_h_mm - 1.0
    assert j[1] == j[0] + 1 or not inside[1]
    floor = material.Laying.of(spec, 4000.0, height)
    assert floor.cy == height / 2 and lay.cy != floor.cy  # a terra la posa resta centrata
    whole, cut = material.count_tiles(spec, 4000.0, height, "parete")
    assert whole > 0 and cut > 0  # i tagli stanno in alto


def test_material_must_suit_the_surface(tmp_path: Path, photo: Path) -> None:
    path = synthetic.write_catalog(tmp_path / "cat")
    doc = json.loads(path.read_text())
    doc["materials"][0]["uses"] = ["pavimento"]
    only_floor = doc["materials"][0]["id"]
    path.write_text(json.dumps(doc))
    ws = _ws(tmp_path, photo, material_id=only_floor)
    tb = Toolbox(ws, Catalog.load(path), ScriptedChecker(), segmenter=_wall_segmenter)
    assert "non è adatto" in _run(tb.scegli_materiale(only_floor))["errore"]
    assert ws.state.material is None


def test_redoing_the_mask_after_final_review_forces_a_new_render(
    tmp_path: Path, photo: Path, catalog: Catalog
) -> None:
    """Il revisore boccia per un oggetto coperto: si rifà la maschera e la posa vecchia
    non può più essere consegnata."""
    ws = _ws(tmp_path, photo)
    tb = Toolbox(ws, catalog, ScriptedChecker(), segmenter=_wall_segmenter)
    # prima maschera SENZA escludere la presa: verrebbe coperta
    assert _run(tb.proponi_maschera(base="automatica", area=_zone()))["esito"] == "APPROVATO"
    _run(tb.proponi_geometria_parete(_walls()))
    _run(tb.scegli_materiale(MATERIAL))
    _run(tb.proponi_luce())
    assert _run(tb.esegui_posa())["esito"] == "APPROVATO"
    first = ws.state.render.final_sha256  # type: ignore[union-attr]
    assert _run(_mask(tb))["esito"] == "APPROVATO"  # maschera corretta
    assert ws.state.geometry is not None  # la geometria resta valida
    assert "maschera approvata corrente" in " ".join(st.missing_for_delivery(ws.state))
    assert "cancello chiuso" in _run(tb.consegna())["errore"]
    assert _run(tb.esegui_posa())["esito"] == "APPROVATO"
    assert ws.state.render.final_sha256 != first  # type: ignore[union-attr]
    assert "consegnato" in _run(tb.consegna())
