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


MODEL_ALIASES = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5",
}

ModelOpt = Annotated[
    str | None,
    typer.Option(
        "--modello",
        help="Modello degli agenti: opus (predefinito), sonnet (circa metà costo), haiku, "
        "oppure un ID completo. In alternativa la variabile POSA_MODEL",
    ),
]


def _set_model(modello: str | None) -> None:
    from app import config

    if modello:
        config.AGENT_MODEL = MODEL_ALIASES.get(modello.lower(), modello)
    typer.echo(f"Modello degli agenti: {config.AGENT_MODEL}")


def _default_out() -> Path:
    return Path("pratiche") / datetime.now().strftime("%Y%m%d-%H%M%S")


def _require_credentials() -> None:
    from app.diagnostics import credentials_problem

    if problem := credentials_problem():
        typer.secho(f"Impossibile avviare gli agenti: {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)


@app.command()
def verifica(
    online: Annotated[
        bool, typer.Option("--online", help="Fa anche una chiamata di prova agli agenti (~0,1 USD)")
    ] = False,
    modelli: Annotated[
        bool,
        typer.Option("--modelli", help="Scarica e prova il modello di segmentazione (~230 MB)"),
    ] = False,
    modello: ModelOpt = None,
) -> None:
    """Controlla l'installazione e dice come risolvere ciò che manca."""
    from app import config
    from app.diagnostics import model_check, offline_checks, online_check

    _set_model(modello)

    checks = offline_checks()
    if modelli:
        checks.append(model_check())
    if online:
        checks.append(asyncio.run(online_check(config.AGENT_MODEL)))
    icon = {"ok": "✅", "avviso": "⚠️ ", "errore": "❌"}
    for c in checks:
        typer.echo(f"{icon[c.level]} {c.name}: {c.detail}")
        if c.fix:
            typer.echo(f"     → {c.fix}")
    errors = [c for c in checks if c.level == "errore"]
    if errors:
        typer.secho(f"\n{len(errors)} problemi da risolvere.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if any(c.level == "avviso" for c in checks):
        typer.secho(
            "\nInstallazione a posto, ma risolvi gli avvisi qui sopra prima di avviare gli agenti.",
            fg=typer.colors.YELLOW,
        )
        return
    typer.secho("\nTutto pronto: avvia la demo con  posa web", fg=typer.colors.GREEN)


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
    modello: ModelOpt = None,
) -> None:
    """Esegue una pratica completa con il sistema multiagente."""
    from app.agents.orchestrator import run_job, start

    _require_credentials()
    _set_model(modello)
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
    modello: ModelOpt = None,
) -> None:
    """Genera una stanza sintetica ed esegue il sistema multiagente su quella."""
    from app.devtools import synthetic

    out.parent.mkdir(parents=True, exist_ok=True)
    scene = synthetic.make_room_scene()
    photo = out.parent / f"{out.name}_stanza.jpg"
    synthetic.save_image(scene.photo, photo, fmt="JPEG", focal_35mm=26, camera=("Demo", "Phone"))
    run(photo, materiale, None, "4.2x3.6", out, modello)


@app.command("catalogo")
def catalogo() -> None:
    """Elenca gli articoli a catalogo."""
    for m in Catalog.load().materials:
        joint = f"fuga {m.joint_mm:g} mm" if m.joint_mm > 0 else "a incastro"
        typer.echo(
            f"{m.id:26s} {m.tile_w_mm / 10:.4g}x{m.tile_h_mm / 10:.4g} cm  {joint:12s} {m.name}"
        )


@app.command("aggiungi-materiale")
def aggiungi_materiale(
    foto: Annotated[
        list[Path],
        typer.Argument(help="Foto delle piastrelle della collezione (una per variante)"),
    ],
    codice: Annotated[
        str, typer.Option("--codice", help="Codice articolo, es. gres-antracite-60x60")
    ],
    nome: Annotated[str, typer.Option("--nome", help="Nome commerciale")],
    formato: Annotated[
        str,
        typer.Option(
            "--formato",
            help="Formato in cm, es. 60x60 o 120x20; esagoni: larghezza da punta a punta, es. 25",
        ),
    ],
    fuga: Annotated[float, typer.Option("--fuga", help="Fuga in mm (0 = a incastro)")] = 0.0,
    colore_fuga: Annotated[str | None, typer.Option("--colore-fuga", help="Es. #c8c4bc")] = None,
    posa: Annotated[str, typer.Option("--posa", help="grid | running_bond | hex")] = "grid",
    sfalsamento: Annotated[
        float, typer.Option("--sfalsamento", help="Solo running_bond: 0.5 = metà")
    ] = 0.5,
    rotazione: Annotated[int, typer.Option("--rotazione", help="0 | 45 | 90 rispetto ai muri")] = 0,
    angoli: Annotated[
        list[str] | None,
        typer.Option(
            "--angoli",
            help="Foto in prospettiva (rettangoli): 4 angoli della faccia in pixel, ordine "
            "alto-sx alto-dx basso-dx basso-sx, es. '128,63 718,60 800,640 45,640' (uno per foto)",
        ),
    ] = None,
    vertici: Annotated[
        list[str] | None,
        typer.Option(
            "--vertici",
            help="Esagoni: 'auto' oppure 6 vertici x,y (punta destra, poi senso orario), "
            "uno per foto. Senza questa opzione: tutti 'auto'",
        ),
    ] = None,
    pesi: Annotated[
        str | None,
        typer.Option("--pesi", help="Frequenza di ogni variante nella posa, es. '3,2,1,1'"),
    ] = None,
    rotazioni: Annotated[
        list[str] | None,
        typer.Option(
            "--rotazioni",
            help="Rotazioni ammesse per ogni foto, es. '0,180' o '0,60,120,180,240,300'",
        ),
    ] = None,
    correzione: Annotated[
        list[str] | None,
        typer.Option(
            "--correzione",
            help="Esagoni: guadagni R,G,B in luce lineare per ogni foto (es. '1.3,1.2,1.05'), "
            "oppure 'no'. Serve a uniformare foto scattate con luce diversa",
        ),
    ] = None,
    finitura: Annotated[
        str,
        typer.Option(
            "--finitura",
            help="Finitura della superficie: opaca, satinata o lucida (decide i riflessi)",
        ),
    ] = "opaca",
    mantieni_luce: Annotated[
        bool,
        typer.Option(
            "--mantieni-luce",
            help="Con --angoli: non togliere riflessi e luce dello studio fotografico",
        ),
    ] = False,
    catalogo_path: Annotated[Path | None, typer.Option("--catalogo")] = None,
) -> None:
    """Aggiunge un articolo (o una collezione mista) al catalogo dalle foto delle piastrelle."""
    import json
    import shutil

    from app.catalog.loader import DEFAULT_CATALOG
    from app.core.ingest import sha256_file
    from app.core.models import SQRT3, MaterialSpec
    from app.storage.catalog_import import rectify_hexagon, rectify_tile

    def per_photo(values: list[str] | None, name: str) -> list[str] | None:
        if values is not None and len(values) != len(foto):
            raise typer.BadParameter(f"serve un --{name} per ogni foto ({len(foto)})")
        return values

    path = catalogo_path or DEFAULT_CATALOG
    hexagon = posa == "hex"
    sizes = [float(x) for x in formato.lower().replace(",", ".").split("x")]
    w_cm = sizes[0]
    h_cm = sizes[1] if len(sizes) > 1 else (w_cm * SQRT3 / 2 if hexagon else w_cm)
    angoli = per_photo(angoli, "angoli")
    vertici = per_photo(vertici, "vertici")
    rotazioni = per_photo(rotazioni, "rotazioni")
    correzione = per_photo(correzione, "correzione")
    weights = [float(x) for x in pesi.split(",")] if pesi else [1.0] * len(foto)
    if len(weights) != len(foto):
        raise typer.BadParameter(f"--pesi: servono {len(foto)} valori")
    tex_dir = path.parent / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for k, f in enumerate(foto, 1):
        try:
            if hexagon:
                dst = tex_dir / f"{codice}-v{k}.png"
                v = vertici[k - 1] if vertici else "auto"
                g = correzione[k - 1] if correzione else "no"
                gains: tuple[float, float, float] | None = None
                if g != "no":
                    vals = [float(x) for x in g.split(",")]
                    if len(vals) != 3:
                        raise ValueError("--correzione: servono 3 valori R,G,B")
                    gains = (vals[0], vals[1], vals[2])
                rectify_hexagon(f, dst, None if v == "auto" else v, gains=gains)
            elif angoli:
                dst = tex_dir / f"{codice}-v{k}.png"
                rectify_tile(f, angoli[k - 1], w_cm, h_cm, dst, flatten=not mantieni_luce)
            else:
                dst = tex_dir / f"{codice}-v{k}{f.suffix.lower()}"
                shutil.copyfile(f, dst)
        except ValueError as exc:
            raise typer.BadParameter(f"{f.name}: {exc}") from exc
        entry_v: dict[str, object] = {
            "texture_file": dst.name,
            "texture_sha256": sha256_file(dst),
            "weight": weights[k - 1],
        }
        if rotazioni:
            entry_v["rotations"] = [int(r) for r in rotazioni[k - 1].split(",")]
        variants.append(entry_v)
    if hexagon:
        default_rot = [0, 60, 120, 180, 240, 300]
    else:
        default_rot = [0, 90, 180, 270] if abs(w_cm - h_cm) < 1e-6 else [0, 180]
    entry = {
        "id": codice,
        "name": nome,
        "version": "1",
        "tile_w_mm": round(w_cm * 10, 3),
        "tile_h_mm": round(h_cm * 10, 3),
        "joint_mm": fuga,
        "joint_color": colore_fuga if fuga > 0 else None,
        "pattern": {"type": posa, "bond_offset": sfalsamento, "rotation_deg": rotazione},
        "variants": variants,
        "variant_rotations": default_rot,
        "texture_color_space": "srgb",
        "finish": finitura,
    }
    MaterialSpec.model_validate(entry)  # errore chiaro prima di toccare il catalogo
    doc = json.loads(path.read_text())
    doc["materials"] = [m for m in doc["materials"] if m["id"] != codice] + [entry]
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    Catalog.load(path).load_variants(Catalog.load(path).get(codice))
    typer.secho(f"Aggiunto {codice} ({len(variants)} varianti) a {path}", fg=typer.colors.GREEN)


@app.command()
def web(
    port: Annotated[int, typer.Option(help="Porta")] = 7860,
    share: Annotated[bool, typer.Option(help="Link pubblico temporaneo (Gradio)")] = False,
    modello: ModelOpt = None,
) -> None:
    """Avvia l'interfaccia web per la demo."""
    from app.diagnostics import credentials_problem
    from app.web import CSS, build_app

    _set_model(modello)
    if problem := credentials_problem():  # la pagina si apre comunque, ma avvisiamo subito
        typer.secho(f"Attenzione: {problem}", fg=typer.colors.YELLOW, err=True)
    free = first_free_port(port)
    if free is None:
        typer.secho(
            f"Le porte da {port} a {port + PORT_TRIES - 1} sono tutte occupate. "
            'Ferma i "posa web" già accesi con:  pkill -f "posa web"',
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    if free != port:
        typer.secho(
            f'La porta {port} è già in uso: probabilmente "posa web" è già acceso in un altro '
            f"terminale (puoi aprire quella pagina dalla scheda PORTS, porta {port}, "
            'oppure fermarlo con:  pkill -f "posa web").\n'
            f"Avvio questa interfaccia sulla porta {free}.",
            fg=typer.colors.YELLOW,
            err=True,
        )
    build_app().queue().launch(server_name="0.0.0.0", server_port=free, share=share, css=CSS)


PORT_TRIES = 20


def first_free_port(start: int, tries: int = PORT_TRIES) -> int | None:
    """Prima porta libera da `start` in poi (None se sono tutte occupate)."""
    import socket

    for candidate in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", candidate))
            except OSError:
                continue
            return candidate
    return None


def main() -> None:
    app()


if __name__ == "__main__":
    main()
