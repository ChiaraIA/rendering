"""Catalogo dal negozio Shopify, con un negozio finto (niente rete)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from app.catalog.loader import Catalog
from app.devtools import synthetic
from app.storage import shopify as shop
from app.storage.catalog_import import detect_quad, hex_vertices

STORE = "https://piastrelle-prova.myshopify.com"


def _jpeg(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _on_white(
    tex: np.ndarray, canvas: tuple[int, int] = (900, 760), tilt: float = 18
) -> np.ndarray:
    """Piastrella fotografata su sfondo bianco, leggermente in prospettiva."""
    h, w = tex.shape[:2]
    W, H = canvas
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    x0, y0 = (W - w) / 2, (H - h) / 2
    dst = np.float32([[x0 + tilt, y0], [x0 + w - tilt, y0 + 6], [x0 + w, y0 + h], [x0, y0 + h - 4]])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(tex, m, (W, H), borderValue=(250, 250, 248))
    return np.asarray(out, dtype=np.uint8)


def _hex_on_white(tex: np.ndarray, radius: int = 300) -> np.ndarray:
    W, H = 2 * radius + 160, int(np.sqrt(3) * radius) + 160
    canvas = np.full((H, W, 3), 248, np.uint8)
    poly = np.round(hex_vertices(radius, W / 2, H / 2)).astype(np.int32)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(inside, poly, 1)
    big = cv2.resize(tex, (W, H))
    canvas[inside > 0] = big[inside > 0]
    return canvas


TILE_BLUE = synthetic.make_tile_texture(420, (40, 70, 150), seed=1)
TILE_GREEN = synthetic.make_tile_texture(420, (60, 130, 70), seed=2)
TILE_DARK = synthetic.make_tile_texture(480, (55, 55, 58), seed=3)
PLANK = synthetic.make_tile_texture((1200, 200), (164, 120, 78), seed=4, kind="wood")
HEX_A = synthetic.make_tile_texture(400, (190, 120, 90), seed=5)
HEX_B = synthetic.make_tile_texture(400, (220, 205, 180), seed=6)
ROOM = synthetic.make_room_scene(width=900, height=600).photo

IMAGES: dict[str, bytes] = {
    "room.jpg": _jpeg(ROOM),
    "antracite.jpg": _jpeg(_on_white(TILE_DARK)),
    "rovere.jpg": _jpeg(PLANK),
    "hex-a.jpg": _jpeg(_hex_on_white(HEX_A)),
    "hex-b.jpg": _jpeg(_hex_on_white(HEX_B)),
    "blu.jpg": _jpeg(_on_white(TILE_BLUE)),
    "verde.jpg": _jpeg(_on_white(TILE_GREEN)),
}


def _img(i: int, name: str, pos: int, alt: str | None = None) -> dict[str, Any]:
    return {"id": i, "src": f"//cdn.shopify.com/s/files/{name}?v=1", "position": pos, "alt": alt}


def _products() -> list[dict[str, Any]]:
    default = [{"id": 1, "option1": "Default Title", "image_id": None}]
    return [
        {
            "id": 101,
            "title": "Gres Antracite 60x60 lappato",
            "handle": "gres-antracite-60x60",
            "product_type": "Piastrelle",
            "tags": ["fuga:2"],
            "options": [{"name": "Title"}],
            "variants": default,
            "images": [_img(11, "room.jpg", 1), _img(12, "antracite.jpg", 2)],
        },
        {
            "id": 102,
            "title": "Rovere naturale 20x120",
            "handle": "rovere-naturale",
            "product_type": "Parquet",
            "tags": "legno, interni",
            "options": [{"name": "Title"}],
            "variants": default,
            "images": [_img(21, "rovere.jpg", 1), _img(22, "room.jpg", 2)],
        },
        {
            "id": 103,
            "title": "Cementina esagona Mix 25 cm",
            "handle": "cementina-mix",
            "tags": ["misto"],
            "options": [{"name": "Title"}],
            "variants": default,
            "images": [
                _img(31, "room.jpg", 1),
                _img(32, "hex-a.jpg", 2, "texture cotto"),
                _img(33, "hex-b.jpg", 3, "Texture crema"),
            ],
        },
        {
            "id": 104,
            "title": "Ceramica Colori 20x20",
            "handle": "ceramica-colori",
            "options": [{"name": "Colore"}],
            "variants": [
                {"id": 41, "option1": "Blu", "image_id": 401},
                {"id": 42, "option1": "Verde", "image_id": 402},
            ],
            "images": [_img(401, "blu.jpg", 1), _img(402, "verde.jpg", 2)],
        },
        {
            "id": 105,
            "title": "Collante professionale",
            "handle": "collante",
            "options": [{"name": "Title"}],
            "variants": default,
            "images": [_img(51, "room.jpg", 1)],
        },
        {
            "id": 106,
            "title": "Gres Bianco 30x30",
            "handle": "gres-bianco",
            "tags": ["render:no"],
            "options": [{"name": "Title"}],
            "variants": default,
            "images": [_img(61, "blu.jpg", 1)],
        },
    ]


class FakeShop:
    def __init__(self, products: list[dict[str, Any]]) -> None:
        self.products = products
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if "/products.json" in url:
            page = int(url.split("page=")[1])
            return json.dumps({"products": self.products if page == 1 else []}).encode()
        name = url.split("/files/")[1].split("?")[0]
        assert "width=" in url  # le foto si chiedono già ridotte al CDN
        return IMAGES[name]


def test_rules_from_title_tags_and_type() -> None:
    assert shop.parse_size("Gres 60x60 rettificato") == (60.0, 60.0)
    assert shop.parse_size("Doga 20 x 120 cm") == (20.0, 120.0)
    assert shop.parse_size("Lastra 600x1200") == (60.0, 120.0)  # senza unità: mm
    assert shop.parse_size("Piastrella blu") is None
    r = shop.product_rules({"title": "Rovere 20x120", "product_type": "Parquet"}, None)
    assert not isinstance(r, str)
    assert (r.w_cm, r.h_cm, r.pattern, r.joint_mm) == (120.0, 20.0, "running_bond", 0.0)
    r = shop.product_rules({"title": "Marmo 60x60 lucido", "tags": ["fuga:1.5"]}, None)
    assert not isinstance(r, str) and r.finish == "lucida" and r.joint_mm == 1.5
    r = shop.product_rules({"title": "Esagona 25x22"}, None)
    assert not isinstance(r, str) and r.hexagon and r.w_cm == 25.0
    assert "formato" in str(shop.product_rules({"title": "Piastrella blu"}, None))


def test_detect_quad_finds_the_tile_on_a_white_background() -> None:
    quad = detect_quad(_on_white(TILE_BLUE))
    assert quad is not None
    assert quad[0][0] < quad[1][0] and quad[1][1] < quad[2][1]  # alto-sx, alto-dx, basso-dx
    assert detect_quad(ROOM) is None  # un'ambientazione non ha sfondo uniforme


def test_sync_builds_a_catalog_ready_for_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeShop(_products())
    out = tmp_path / "negozio"
    report = shop.sync("piastrelle-prova.myshopify.com", out, fetch=fake)
    status = {p.title: p for p in report.products}

    gres = status["Gres Antracite 60x60 lappato"]
    assert gres.status == "importato" and gres.articles == ["shop-gres-antracite-60x60"]
    assert any("foto 1 ignorata" in n for n in gres.notes)  # l'ambientazione è scartata
    assert any("foto 2: piastrella su sfondo" in n for n in gres.notes)

    assert status["Rovere naturale 20x120"].articles == ["shop-rovere-naturale"]
    assert status["Cementina esagona Mix 25 cm"].articles == ["shop-cementina-mix"]
    assert status["Ceramica Colori 20x20"].articles == [
        "shop-ceramica-colori-blu",
        "shop-ceramica-colori-verde",
    ]
    assert status["Collante professionale"].status == "saltato"
    assert any("non è un rivestimento" in n for n in status["Collante professionale"].notes)
    assert status["Gres Bianco 30x30"].status == "saltato"

    monkeypatch.setenv("POSA_SHOPIFY_DIR", str(out))
    cat = Catalog.load()  # catalogo della repo + quello del negozio
    assert "gres-grigio-30x30" in {m.id for m in cat.materials}
    gres_spec = cat.get("shop-gres-antracite-60x60")
    assert (gres_spec.tile_w_mm, gres_spec.joint_mm, gres_spec.finish) == (600.0, 2.0, "lucida")
    tex = cat.load_variants(gres_spec)[0]
    assert abs(float(tex.mean()) - float(TILE_DARK.mean())) < 25  # è la piastrella, non lo sfondo
    wood = cat.get("shop-rovere-naturale")
    assert wood.pattern.type == "running_bond" and wood.joint_mm == 0 and wood.tile_w_mm == 1200
    hexa = cat.get("shop-cementina-mix")
    assert hexa.is_hex and len(hexa.variants) == 2  # due foto "texture" = misto
    blu, verde = (
        cat.load_variants(cat.get(f"shop-ceramica-colori-{c}"))[0] for c in ("blu", "verde")
    )
    assert blu[..., 2].mean() > blu[..., 1].mean() and verde[..., 1].mean() > verde[..., 2].mean()

    # seconda sincronizzazione: niente cambiato, niente riscaricato
    fake.calls.clear()
    again = shop.sync(STORE, out, fetch=fake)
    assert {p.status for p in again.products if p.articles} == {"invariato"}
    assert {p.title for p in again.products if p.status == "saltato"} == {
        "Collante professionale",
        "Gres Bianco 30x30",
    }
    assert all("/products.json" in c for c in fake.calls)

    # articolo tolto dal negozio -> sparisce dal catalogo, con le sue texture
    fake.products = [p for p in fake.products if p["id"] != 101]
    gone = shop.sync(STORE, out, fetch=fake)
    assert gone.removed == ["shop-gres-antracite-60x60"]
    assert not list((out / "textures").glob("shop-gres-antracite*"))
    assert "shop-gres-antracite-60x60" not in {m.id for m in Catalog.load().materials}
    assert shop.configured_store(out)[0] == STORE  # il negozio resta memorizzato


def test_shop_article_can_be_rendered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from app.agents.state import ClientRequest
    from app.agents.tools import Toolbox
    from app.storage.workspace import Workspace
    from tests.agents.test_toolbox import SCENE, ScriptedChecker, _floor_mask_polys

    out = tmp_path / "negozio"
    shop.sync(STORE, out, fetch=FakeShop(_products()))
    monkeypatch.setenv("POSA_SHOPIFY_DIR", str(out))
    photo = synthetic.save_image(SCENE.photo, tmp_path / "stanza.jpg", fmt="JPEG", focal_35mm=26)
    ws = Workspace.create(
        photo,
        tmp_path / "pratica",
        ClientRequest(material_id="shop-ceramica-colori-blu", room_mm=(4200, 3600)),
    )
    tb = Toolbox(ws, Catalog.load(), ScriptedChecker())
    floor, excl = _floor_mask_polys()

    async def flow() -> dict[str, Any]:
        assert (await tb.proponi_maschera(floor, excl))["esito"] == "APPROVATO"
        assert (await tb.proponi_geometria(SCENE.corners_px.tolist()))["esito"] == "APPROVATO"
        assert (await tb.scegli_materiale("shop-ceramica-colori-blu"))["esito"] == "APPROVATO"
        assert (await tb.proponi_luce())["esito"] == "APPROVATO"
        assert (await tb.esegui_posa())["esito"] == "APPROVATO"
        return await tb.consegna()

    assert "consegnato" in asyncio.run(flow())


def test_store_url_and_errors() -> None:
    assert shop.store_url("www.miosito.it/") == "https://www.miosito.it"
    assert shop.store_url("https://x.myshopify.com/collections/a") == "https://x.myshopify.com"
    with pytest.raises(ValueError):
        shop.store_url("https://")


def test_store_tag_conventions() -> None:
    """Tag nello stile del negozio: Formato_60x120, Superficie_Lucida, Utilizzo_Rivestimento."""
    product = {
        "title": "Calacatta Gold - Gres effetto marmo",
        "product_type": "Gres Porcellanato",
        "tags": [
            "Formato_Grande",
            "Formato_60x120",
            "Superficie_Lucida",
            "Utilizzo_Rivestimento",
            "Collezione_Spina di Pesce",  # tag di collezione: non decide la posa
        ],
    }
    r = shop.product_rules(product, None)
    assert not isinstance(r, str)
    assert (r.w_cm, r.h_cm, r.finish, r.uses) == (120.0, 60.0, "lucida", ["rivestimento"])
    hexa = shop.product_rules(
        {"title": "Terra Giglio - Gres", "tags": ["Formato_22x25", "Formato_Esagonale"]}, None
    )
    assert not isinstance(hexa, str) and hexa.hexagon and hexa.w_cm == 25.0
    assert "spina di pesce" in str(
        shop.product_rules({"title": "Rovere Chevron Spina di Pesce 7.5x45 gres"}, None)
    )
    assert "non è un rivestimento" in str(
        shop.product_rules({"title": "Box doccia 80x190", "product_type": "Box"}, None)
    )
    plans, why = shop.plan_product(
        {"title": "Gres 60x60", "tags": ["Fuori Produzione"], "images": [_img(1, "x", 1)]}, STORE
    )
    assert plans == [] and "Fuori Produzione" in why[0]


def test_joint_lines_tell_compositions_from_single_tiles() -> None:
    from app.storage.catalog_import import has_joint_lines

    tile = synthetic.make_tile_texture(600, (70, 70, 72), seed=8)
    assert not has_joint_lines(tile)
    grid = np.concatenate([np.concatenate([tile[:200, :200]] * 3, 1)] * 3, 0).copy()
    grid[199::200, :] = 150  # fughe chiare sottili, poco contrastate come nelle foto vere
    grid[:, 199::200] = 150
    assert has_joint_lines(grid)
    grooves = tile.copy()
    grooves[::12] = 40  # rigatura fitta (rilievo): non sono fughe
    assert not has_joint_lines(grooves)


def test_composition_photo_is_split_into_figures_and_printed_grids_stay_whole() -> None:
    from app.storage import catalog_import as ci

    a = synthetic.make_tile_texture(200, (180, 90, 70), seed=21)
    b = synthetic.make_tile_texture(200, (70, 110, 160), seed=22)
    # composizione 2x2 di piastrelle 20x20 (due figure) con fughe chiare
    comp = np.concatenate([np.concatenate([a, b], 1), np.concatenate([b, a], 1)], 0).copy()
    comp[198:202, :] = 235
    comp[:, 198:202] = 235
    rules = shop.product_rules({"title": "Cementina Decoro Mix 20x20"}, None)
    assert not isinstance(rules, str) and rules.mix
    figures, how = shop.tile_textures(comp, rules, False)
    assert len(figures) == 2 and "composizione di 4" in how
    # piastrella 50x20 con un disegno a riquadri quadrati stampato: resta intera
    squares = [synthetic.make_tile_texture(100, (120 + 20 * k, 120, 120), seed=k) for k in range(5)]
    printed = np.concatenate(
        [np.concatenate(squares, 1), np.concatenate(squares[::-1], 1)], 0
    ).copy()
    for x in range(100, 500, 100):
        printed[:, x - 1 : x + 1] = 230
    printed[99:101, :] = 230
    r50 = shop.product_rules({"title": "Maiolica Patchwork 20x50"}, None)
    assert not isinstance(r50, str)
    figures, how = shop.tile_textures(printed, r50, False)
    assert len(figures) == 1 and "riquadri" in how
    assert figures[0].shape[1] > 2 * figures[0].shape[0]  # 50x20 intera
    assert len(ci.distinct_tiles([a, a.copy(), b])) == 2  # le copie si tolgono
