"""Interfaccia web per la demo (Gradio).

La foto viene caricata come FILE (non come componente immagine): i byte originali arrivano
intatti alla pratica, che li identifica con SHA-256 e non li modifica mai.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

from app.agents.orchestrator import run_job, start
from app.catalog.loader import Catalog
from app.core import overlays
from app.diagnostics import credentials_problem

WORKDIR = Path("pratiche")
NONE_CHOICE = "— nessuno: uso la foto del campione —"

CSS = """
#log textarea {font-family: ui-monospace, monospace; font-size: 13px}
.posa-title h1 {margin-bottom: 0}
"""


GALLERY_MAX = 24  # miniature mostrate (il catalogo del negozio può avere migliaia di articoli)
CHOICES_MAX = 300


def _catalog_gallery(cat: Catalog, query: str = "") -> list[tuple[Any, str]]:
    items = []
    for spec in cat.search(query)[:GALLERY_MAX]:
        tex = cat.load_variants(spec)[0]
        joint = None
        if spec.joint_mm > 0 and spec.joint_color:
            c = spec.joint_color.lstrip("#")
            joint = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
        caption = f"{spec.name} · {spec.tile_w_mm / 10:g}×{spec.tile_h_mm / 10:g} cm"
        items.append((overlays.tile_patch(tex, joint, hexagon=spec.is_hex), caption))
    return items


def _parse_room(text: str | None) -> tuple[float, float] | None:
    if not text or not text.strip():
        return None
    a, b = text.lower().replace(",", ".").replace(" ", "").split("x")
    return float(a), float(b)


def _example_photo() -> str:
    from app.devtools import synthetic

    path = WORKDIR / "esempio_stanza.jpg"
    if not path.exists():
        WORKDIR.mkdir(parents=True, exist_ok=True)
        scene = synthetic.make_room_scene()
        synthetic.save_image(scene.photo, path, fmt="JPEG", focal_35mm=26, camera=("Demo", "Phone"))
    return str(path)


def _choices(cat: Catalog, query: str = "") -> list[tuple[str, str]]:
    found = cat.search(query)[:CHOICES_MAX]
    return [(f"{m.name} ({m.id})", m.id) for m in found] + [(NONE_CHOICE, "")]


def _sync_shop() -> tuple[str, Catalog]:
    """Riscarica gli articoli dal negozio Shopify e ricarica il catalogo."""
    from app.catalog.loader import shop_catalog_dir
    from app.storage import shopify as shop

    store, coll = shop.configured_store(shop_catalog_dir())
    if not store:
        msg = (
            "⚠️ Nessun negozio configurato: avvia una volta `posa shopify --negozio "
            "mio-negozio.myshopify.com` (oppure imposta POSA_SHOPIFY_NEGOZIO)."
        )
        return msg, Catalog.load()
    try:
        host = shop.store_url(store).split("://", 1)[1]
        report = shop.sync(store, shop_catalog_dir(), coll, fetch=shop.safe_fetcher(store, host))
    except Exception as exc:
        msg = f"❌ Negozio non raggiungibile ({exc}): resta il catalogo precedente."
        return msg, Catalog.load()
    detail = shop.format_report(report).replace("\n", "  \n")
    return f"✅ {detail}", Catalog.load()


def build_app(catalog: Catalog | None = None) -> gr.Blocks:
    holder = {"cat": catalog or Catalog.load()}  # il catalogo cambia con "Aggiorna dal negozio"
    cat = holder["cat"]
    choices = _choices(cat)

    async def run(
        photo: str | None,
        material_id: str,
        sample: str | None,
        room: str,
        surface: str = "Pavimento",
    ):
        if problem := credentials_problem():
            yield f"⚠️ Gli agenti non possono partire: {problem}", "", None, [], None
            return
        if not photo:
            yield "⚠️ Carica una foto della stanza.", "", None, [], None
            return
        if not material_id and not sample:
            yield (
                "⚠️ Scegli un materiale dal catalogo oppure carica la foto di un campione.",
                "",
                None,
                [],
                None,
            )
            return
        try:
            room_m = _parse_room(room)
        except ValueError:
            yield "⚠️ Misure stanza: usa il formato 4.2x3.6 (metri).", "", None, [], None
            return

        queue: asyncio.Queue[str] = asyncio.Queue()
        out_dir = WORKDIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        loop = asyncio.get_running_loop()

        def push(text: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, text)

        sample_path = Path(sample) if sample else None
        wall = str(surface or "").startswith("Parete")
        ws = start(
            Path(photo),
            out_dir,
            material_id or None,
            sample_path,
            room_m,
            listener=push,
            surface="rivestimento" if wall else "pavimento",
        )
        lines: list[str] = []
        task = asyncio.create_task(run_job(ws, holder["cat"]))
        status = "⏳ Gli agenti sono al lavoro…"
        while not task.done() or not queue.empty():
            with contextlib.suppress(TimeoutError):
                lines.append(await asyncio.wait_for(queue.get(), timeout=1.0))
            yield status, "\n".join(lines), None, _work_gallery(ws), None
        try:
            outcome = task.result()
        except Exception as exc:  # errore tecnico: lo mostriamo, niente risultato
            yield f"❌ Errore: {exc}", "\n".join(lines), None, _work_gallery(ws), None
            return

        before = str(ws.root / "preview.jpg")
        if outcome.delivered:
            after = str(ws.root / ws.state.render.result_preview)  # type: ignore[union-attr]
            files = [str(ws.root / f) for f in ws.state.delivered]
            note = " (misure stimate dagli agenti)" if ws.state.room_estimated else ""
            status = (
                f"✅ **Consegnato**{note} — costo agenti ≈ {outcome.cost_usd:.2f} USD\n\n"
                f"{outcome.summary}"
            )
            yield status, "\n".join(lines), (before, after), _work_gallery(ws), files
        else:
            reason = ws.state.escalation or "il flusso non ha superato tutti i controlli"
            status = (
                f"🙋 **Non consegnato: passato a un operatore.**\n\nMotivo: {reason}\n\n"
                f"{outcome.summary}"
            )
            yield status, "\n".join(lines), None, _work_gallery(ws), [str(ws.root / "stato.json")]

    with gr.Blocks(title="posa — rivestimenti nella tua stanza") as demo:
        gr.Markdown(
            "# posa\n**Il pavimento o il rivestimento nuovo, nella foto della tua stanza.** "
            "Un orchestratore coordina agenti specializzati (superficie, prospettiva, materiale, "
            "luci) e ogni passo viene verificato da un controllore indipendente prima di "
            "arrivare a te. La foto originale non viene mai modificata fuori dalla superficie.",
            elem_classes=["posa-title"],
        )
        with gr.Row():
            with gr.Column(scale=4):
                photo = gr.File(
                    label="1 · Foto della stanza (JPEG, HEIC, PNG)",
                    type="filepath",
                    file_types=["image", ".heic", ".heif", ".HEIC"],
                )
                example = gr.Button("Usa una stanza di esempio", size="sm")
                search = gr.Textbox(
                    label=f"Cerca nel catalogo ({len(cat.materials)} articoli)",
                    placeholder="es. grigio cemento 60x60, esagona, rovere…",
                )
                material = gr.Dropdown(
                    choices=choices,
                    value=cat.materials[0].id,
                    label="2 · Rivestimento dal catalogo",
                )
                gallery = gr.Gallery(
                    value=_catalog_gallery(cat),
                    label="Catalogo",
                    columns=5,
                    height=300,
                    object_fit="contain",  # niente ritaglio: gli esagoni restano interi
                    allow_preview=False,
                )
                with gr.Row():
                    refresh = gr.Button("🔄 Aggiorna dal negozio Shopify", size="sm")
                shop_status = gr.Markdown(visible=False)
                sample = gr.File(
                    label="…oppure la foto di un campione visto in negozio",
                    type="filepath",
                    file_types=["image", ".heic", ".heif"],
                )
                surface = gr.Radio(
                    choices=["Pavimento", "Parete (rivestimento, es. paraschizzi della cucina)"],
                    value="Pavimento",
                    label="3 · Dove posarlo",
                )
                room = gr.Textbox(
                    label="4 · Misure in metri (facoltativo): stanza, o larghezza x altezza "
                    "della zona di parete",
                    placeholder="es. 4.2x3.6 — se vuoto le stimano gli agenti",
                )
                go = gr.Button("Posa il rivestimento", variant="primary")
            with gr.Column(scale=6):
                status = gr.Markdown("Carica una foto e premi **Posa il rivestimento**.")
                slider = gr.ImageSlider(label="Prima / dopo", type="filepath")
                log = gr.Textbox(
                    label="Cosa stanno facendo gli agenti",
                    lines=14,
                    max_lines=14,
                    autoscroll=True,
                    elem_id="log",
                )
                work = gr.Gallery(
                    label="Lavoro degli agenti (maschere, griglie, verifiche)",
                    columns=4,
                    height=200,
                    object_fit="contain",
                )
                files = gr.File(
                    label="Scarica (risultato a piena risoluzione, maschera, report)",
                    file_count="multiple",
                )
        example.click(lambda: _example_photo(), outputs=photo).then(lambda: "4.2x3.6", outputs=room)

        async def refresh_shop():
            msg, new_cat = await asyncio.to_thread(_sync_shop)
            holder["cat"] = new_cat
            first = new_cat.materials[0].id if new_cat.materials else ""
            return (
                gr.update(choices=_choices(new_cat), value=first),
                _catalog_gallery(new_cat),
                gr.update(value=msg, visible=True),
            )

        refresh.click(refresh_shop, outputs=[material, gallery, shop_status])

        def do_search(q: str):
            c = holder["cat"]
            found = c.search(q)
            first = found[0].id if found else ""
            return gr.update(choices=_choices(c, q), value=first), _catalog_gallery(c, q)

        search.submit(do_search, inputs=search, outputs=[material, gallery])
        search.change(do_search, inputs=search, outputs=[material, gallery])

        def pick(q: str, evt: gr.SelectData):
            found = holder["cat"].search(q)[:GALLERY_MAX]
            return found[evt.index].id if 0 <= evt.index < len(found) else gr.update()

        gallery.select(pick, inputs=search, outputs=material)
        go.click(
            run,
            inputs=[photo, material, sample, room, surface],
            outputs=[status, log, slider, work, files],
        )
    return demo


def _work_gallery(ws: Any) -> list[tuple[str, str]]:
    labels = {
        "maschera": "maschera",
        "geometria": "prospettiva",
        "prima_dopo": "prima/dopo",
        "materiale": "materiale",
        "luce": "luci e ombre",
    }
    out = []
    for p in sorted((ws.root / "lavoro").glob("*.jpg"), key=lambda q: q.stat().st_mtime):
        stem = p.stem.rsplit("_", 1)[0]
        if stem in labels:
            out.append((str(p), f"{labels[stem]} {p.stem.rsplit('_', 1)[-1]}"))
    return out


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", css=CSS)
