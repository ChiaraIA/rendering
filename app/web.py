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

WORKDIR = Path("pratiche")
NONE_CHOICE = "— nessuno: uso la foto del campione —"

CSS = """
#log textarea {font-family: ui-monospace, monospace; font-size: 13px}
.posa-title h1 {margin-bottom: 0}
"""


def _catalog_gallery(cat: Catalog) -> list[tuple[Any, str]]:
    items = []
    for spec in cat.materials:
        tex = cat.load_variants(spec)[0]
        joint = None
        if spec.joint_mm > 0 and spec.joint_color:
            c = spec.joint_color.lstrip("#")
            joint = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
        caption = f"{spec.name} · {spec.tile_w_mm / 10:g}×{spec.tile_h_mm / 10:g} cm"
        items.append((overlays.tile_patch(tex, joint), caption))
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


def build_app(catalog: Catalog | None = None) -> gr.Blocks:
    cat = catalog or Catalog.load()
    choices = [(f"{m.name} ({m.id})", m.id) for m in cat.materials] + [(NONE_CHOICE, "")]

    async def run(photo: str | None, material_id: str, sample: str | None, room: str):
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
        ws = start(Path(photo), out_dir, material_id or None, sample_path, room_m, listener=push)
        lines: list[str] = []
        task = asyncio.create_task(run_job(ws, cat))
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
            "# posa\n**Il tuo pavimento nuovo, nella foto della tua stanza.** "
            "Un orchestratore coordina agenti specializzati (pavimento, prospettiva, materiale) e "
            "ogni passo viene verificato da un controllore indipendente prima di arrivare a te. "
            "La foto originale non viene mai modificata fuori dal pavimento.",
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
                material = gr.Dropdown(
                    choices=choices,
                    value=cat.materials[0].id,
                    label="2 · Rivestimento dal catalogo",
                )
                gr.Gallery(
                    value=_catalog_gallery(cat),
                    label="Catalogo",
                    columns=3,
                    height=230,
                    allow_preview=False,
                )
                sample = gr.File(
                    label="…oppure la foto di un campione visto in negozio",
                    type="filepath",
                    file_types=["image", ".heic", ".heif"],
                )
                room = gr.Textbox(
                    label="3 · Misure della stanza in metri (facoltativo)",
                    placeholder="es. 4.2x3.6 — se vuoto le stimano gli agenti",
                )
                go = gr.Button("Posa il pavimento", variant="primary")
            with gr.Column(scale=6):
                status = gr.Markdown("Carica una foto e premi **Posa il pavimento**.")
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
                )
                files = gr.File(
                    label="Scarica (risultato a piena risoluzione, maschera, report)",
                    file_count="multiple",
                )
        example.click(lambda: _example_photo(), outputs=photo).then(lambda: "4.2x3.6", outputs=room)
        go.click(
            run, inputs=[photo, material, sample, room], outputs=[status, log, slider, work, files]
        )
    return demo


def _work_gallery(ws: Any) -> list[tuple[str, str]]:
    labels = {
        "maschera": "maschera",
        "geometria": "prospettiva",
        "prima_dopo": "prima/dopo",
        "materiale": "materiale",
    }
    out = []
    for p in sorted((ws.root / "lavoro").glob("*.jpg"), key=lambda q: q.stat().st_mtime):
        stem = p.stem.rsplit("_", 1)[0]
        if stem in labels:
            out.append((str(p), f"{labels[stem]} {p.stem.rsplit('_', 1)[-1]}"))
    return out


if __name__ == "__main__":
    build_app().queue().launch(server_name="0.0.0.0", css=CSS)
