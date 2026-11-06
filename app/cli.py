"""CLI `posa`: avvia il sistema multiagente da terminale, o l'interfaccia web."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from app.catalog.loader import Catalog

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


def _room(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    try:
        a, b = value.lower().replace(",", ".").split("x")
        return float(a), float(b)
    except ValueError:
        raise typer.BadParameter(
            "formato misure: LUNGHEZZAxLARGHEZZA in metri, es. 4.2x3.6"
        ) from None


def _default_out() -> Path:
    return Path("pratiche") / datetime.now().strftime("%Y%m%d-%H%M%S")


@app.command()
def run(
    foto: Annotated[Path, typer.Argument(help="Foto della stanza (JPEG, HEIC, PNG…)")],
    materiale: Annotated[
        str | None, typer.Option("--materiale", "-m", help="Codice a catalogo")
    ] = None,
    campione: Annotated[Path | None, typer.Option("--campione", help="Foto di un campione")] = None,
    stanza: Annotated[
        str | None, typer.Option("--stanza", help="Misure in metri, es. 4.2x3.6")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Cartella della pratica")] = None,
) -> None:
    """Esegue una pratica completa con il sistema multiagente."""
    from app.agents.orchestrator import run_job, start

    out = out or _default_out()
    ws = start(foto, out, materiale, campione, _room(stanza), listener=typer.echo)
    outcome = asyncio.run(run_job(ws, Catalog.load()))
    typer.echo("\n" + (outcome.summary or ""))
    if outcome.delivered:
        typer.secho(f"\nRisultato in {out / 'consegna'}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\nPratica non consegnata: vedi {out / 'stato.json'}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out")] = Path("pratiche/demo"),
    materiale: Annotated[str, typer.Option("--materiale", "-m")] = "laminato-rovere-120x20",
) -> None:
    """Genera una stanza sintetica ed esegue il sistema multiagente su quella."""
    from app.devtools import synthetic

    out.parent.mkdir(parents=True, exist_ok=True)
    scene = synthetic.make_room_scene()
    photo = out.parent / f"{out.name}_stanza.jpg"
    synthetic.save_image(scene.photo, photo, fmt="JPEG", focal_35mm=26, camera=("Demo", "Phone"))
    run(photo, materiale, None, "4.2x3.6", out)


@app.command("catalogo")
def catalogo() -> None:
    """Elenca gli articoli a catalogo."""
    for m in Catalog.load().materials:
        joint = f"fuga {m.joint_mm:g} mm" if m.joint_mm > 0 else "a incastro"
        typer.echo(f"{m.id:26s} {m.tile_w_mm / 10:g}x{m.tile_h_mm / 10:g} cm  {joint:12s} {m.name}")


@app.command("aggiungi-materiale")
def aggiungi_materiale(
    foto: Annotated[
        list[Path],
        typer.Argument(
            help="Foto di UNA piastrella/doga, frontale e ritagliata (una per variante)"
        ),
    ],
    codice: Annotated[
        str, typer.Option("--codice", help="Codice articolo, es. gres-antracite-60x60")
    ],
    nome: Annotated[str, typer.Option("--nome", help="Nome commerciale")],
    formato: Annotated[str, typer.Option("--formato", help="Formato in cm, es. 60x60 o 120x20")],
    fuga: Annotated[float, typer.Option("--fuga", help="Fuga in mm (0 = a incastro)")] = 0.0,
    colore_fuga: Annotated[str | None, typer.Option("--colore-fuga", help="Es. #c8c4bc")] = None,
    posa: Annotated[str, typer.Option("--posa", help="grid | running_bond")] = "grid",
    sfalsamento: Annotated[
        float, typer.Option("--sfalsamento", help="Solo running_bond: 0.5 = metà")
    ] = 0.5,
    rotazione: Annotated[int, typer.Option("--rotazione", help="0 | 45 | 90 rispetto ai muri")] = 0,
    angoli: Annotated[
        list[str] | None,
        typer.Option(
            "--angoli",
            help="Foto in prospettiva: 4 angoli della faccia della piastrella in pixel, "
            "ordine alto-sx alto-dx basso-dx basso-sx, es. --angoli '128,63 718,60 800,640 45,640' "
            "(uno per foto, nello stesso ordine)",
        ),
    ] = None,
    catalogo_path: Annotated[Path | None, typer.Option("--catalogo")] = None,
) -> None:
    """Aggiunge un articolo al catalogo a partire dalle foto delle piastrelle."""
    import json
    import shutil

    from app.catalog.loader import DEFAULT_CATALOG
    from app.core.ingest import sha256_file
    from app.core.models import MaterialSpec

    path = catalogo_path or DEFAULT_CATALOG
    w_cm, h_cm = (float(x) for x in formato.lower().replace(",", ".").split("x"))
    tex_dir = path.parent / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    if angoli and len(angoli) != len(foto):
        raise typer.BadParameter("serve un --angoli per ogni foto")
    variants = []
    for k, f in enumerate(foto, 1):
        if angoli:
            dst = tex_dir / f"{codice}-v{k}.png"
            _rectify(f, angoli[k - 1], w_cm, h_cm, dst)
        else:
            dst = tex_dir / f"{codice}-v{k}{f.suffix.lower()}"
            shutil.copyfile(f, dst)
        variants.append({"texture_file": dst.name, "texture_sha256": sha256_file(dst)})
    square = abs(w_cm - h_cm) < 1e-6
    entry = {
        "id": codice,
        "name": nome,
        "version": "1",
        "tile_w_mm": w_cm * 10,
        "tile_h_mm": h_cm * 10,
        "joint_mm": fuga,
        "joint_color": colore_fuga if fuga > 0 else None,
        "pattern": {"type": posa, "bond_offset": sfalsamento, "rotation_deg": rotazione},
        "variants": variants,
        "variant_rotations": [0, 90, 180, 270] if square else [0, 180],
        "texture_color_space": "srgb",
    }
    MaterialSpec.model_validate(entry)  # errore chiaro prima di toccare il catalogo
    doc = json.loads(path.read_text())
    doc["materials"] = [m for m in doc["materials"] if m["id"] != codice] + [entry]
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    Catalog.load(path).load_variants(Catalog.load(path).get(codice))
    typer.secho(f"Aggiunto {codice} ({len(variants)} varianti) a {path}", fg=typer.colors.GREEN)


def _rectify(src: Path, corners: str, w_cm: float, h_cm: float, dst: Path) -> None:
    """Raddrizza la faccia della piastrella fotografata in prospettiva (vista frontale)."""
    import cv2
    import numpy as np
    from PIL import Image

    from app.core import ingest

    pts = np.array(
        [[float(v) for v in p.split(",")] for p in corners.replace(";", " ").split()],
        dtype=np.float32,
    )
    if pts.shape != (4, 2):
        raise typer.BadParameter("--angoli: servono 4 punti x,y")
    img = ingest.load_image(src).pixels[..., :3]
    scale = min(1024 / w_cm, 1024 / h_cm, 30.0)  # px per cm, lato massimo 1024 px
    out_w, out_h = round(w_cm * scale), round(h_cm * scale)
    target = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, target)
    flat = cv2.warpPerspective(
        np.ascontiguousarray(img),
        m,
        (out_w, out_h),
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_REPLICATE,
    )
    inset = max(1, round(min(out_w, out_h) * 0.01))  # via il filo del bordo/smusso
    flat = flat[inset:-inset, inset:-inset]
    Image.fromarray(np.ascontiguousarray(flat)).resize(
        (out_w, out_h), Image.Resampling.LANCZOS
    ).save(dst, format="PNG")


@app.command()
def web(
    port: Annotated[int, typer.Option(help="Porta")] = 7860,
    share: Annotated[bool, typer.Option(help="Link pubblico temporaneo (Gradio)")] = False,
) -> None:
    """Avvia l'interfaccia web per la demo."""
    from app.web import CSS, build_app

    build_app().queue().launch(server_name="0.0.0.0", server_port=port, share=share, css=CSS)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
