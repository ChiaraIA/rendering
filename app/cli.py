"""CLI `posa`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Literal

import typer

from app.catalog.loader import Catalog
from app.core.errors import PosaError
from app.core.models import LayoutParams, RenderRequest
from app.pipeline import render as pipeline
from app.storage import jobstore

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
mask_app = typer.Typer(no_args_is_help=True, help="Gestione della maschera.")
catalog_app = typer.Typer(no_args_is_help=True, help="Catalogo materiali.")
app.add_typer(mask_app, name="mask")
app.add_typer(catalog_app, name="catalog")

JobDir = Annotated[Path, typer.Argument(help="Cartella del job")]
CatalogOpt = Annotated[Path | None, typer.Option("--catalog", help="catalog.json alternativo")]


def _fail(exc: PosaError) -> typer.Exit:
    typer.secho(f"errore: {exc}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=exc.exit_code)


@app.command()
def ingest(
    photo: Annotated[Path, typer.Argument(help="Foto da elaborare (non viene modificata)")],
    job: Annotated[Path, typer.Option("--job", help="Cartella del nuovo job")],
) -> None:
    """Valida la foto, applica l'orientamento EXIF, calcola gli hash e crea il job."""
    try:
        j = pipeline.ingest_job(photo, job)
    except PosaError as exc:
        raise _fail(exc) from exc
    m = j.doc.image
    typer.echo(f"job {j.doc.job_id} creato in {job}")
    typer.echo(
        f"  {m.width}x{m.height} {m.source_format} orient={m.exif_orientation} {m.color_space}"
    )
    typer.echo(f"  sha256 sorgente  {m.source_sha256}")


@mask_app.command("import")
def mask_import(
    job: JobDir, mask: Annotated[Path, typer.Argument(help="PNG in scala di grigi")]
) -> None:
    """Importa una maschera (stato: proposta). Va confermata prima del render."""
    try:
        rec = pipeline.import_mask(job, mask)
    except PosaError as exc:
        raise _fail(exc) from exc
    typer.echo(f"maschera proposta: {rec.area_px} px, bordo morbido {rec.soft_edge_px} px")
    typer.echo(f"  sha256 {rec.mask_sha256}")
    typer.echo(f"conferma con:  posa mask confirm {job} --sha {rec.mask_sha256[:12]}")


@mask_app.command("confirm")
def mask_confirm(
    job: JobDir, sha: Annotated[str, typer.Option("--sha", help="Prefisso dello sha (>= 8)")]
) -> None:
    """Conferma la maschera attiva ripetendone lo sha."""
    try:
        rec = pipeline.confirm_mask(job, sha)
    except PosaError as exc:
        raise _fail(exc) from exc
    typer.echo(f"maschera {rec.mask_sha256[:12]} confermata ({rec.confirmed_at})")


@app.command()
def render(
    job: JobDir,
    material: Annotated[str, typer.Option("--material", help="id del materiale a catalogo")],
    catalog: CatalogOpt = None,
    fmt: Annotated[str, typer.Option("--format", help="png | tiff")] = "png",
    keep_invalid: Annotated[
        bool, typer.Option("--keep-invalid", help="Salva i risultati non validi in quarantine/")
    ] = False,
) -> None:
    """Posa il materiale sulla maschera confermata, valida ed esporta."""
    if fmt not in ("png", "tiff"):
        typer.secho("errore: --format deve essere png o tiff", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    output_format: Literal["png", "tiff"] = "png" if fmt == "png" else "tiff"
    try:
        cat = Catalog.load(catalog)
        req = RenderRequest(layout=LayoutParams(material_id=material), output_format=output_format)
        out = pipeline.render(job, req, cat, keep_invalid=keep_invalid)
    except PosaError as exc:
        raise _fail(exc) from exc
    _print_outcome(out)
    if not out.valid:
        raise typer.Exit(code=5)


def _print_outcome(out: pipeline.RenderOutcome) -> None:
    color = typer.colors.GREEN if out.valid else typer.colors.RED
    typer.secho(f"{out.render_id}: {'VALIDO' if out.valid else 'NON VALIDO'}", fg=color)
    for c in out.report.checks:
        mark = "ok " if c.passed else "KO "
        typer.echo(f"  {mark} {c.name}{'' if c.blocking else ' (warning)'}")
    for w in out.report.warnings:
        typer.echo(f"  !  {w}")
    for o in out.report.outputs:
        typer.echo(f"  -> {o.kind:10s} {o.path}")
    typer.echo(f"  final sha256 {out.report.final_sha256}")


@app.command()
def status(job: JobDir) -> None:
    """Mostra lo stato del job."""
    try:
        j = jobstore.load(job)
    except PosaError as exc:
        raise _fail(exc) from exc
    typer.echo(f"job {j.doc.job_id}  stato={j.doc.state}")
    if j.doc.active_mask:
        rec = j.doc.masks[j.doc.active_mask]
        typer.echo(f"  maschera {rec.mask_sha256[:12]} {rec.status}")
    for r in j.doc.renders:
        typer.echo(f"  {r.render_id} {'valido' if r.valid else 'NON valido'} {r.final_sha256[:12]}")


@catalog_app.command("list")
def catalog_list(catalog: CatalogOpt = None) -> None:
    """Elenca i materiali a catalogo."""
    try:
        cat = Catalog.load(catalog)
    except PosaError as exc:
        raise _fail(exc) from exc
    for m in cat.materials:
        typer.echo(
            f"{m.id:24s} {m.tile_w_mm:g}x{m.tile_h_mm:g} mm"
            f"  fuga {m.joint_default_mm:g} mm  {m.name}"
        )


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", help="Cartella di output")] = Path("runs/demo"),
    material: Annotated[str, typer.Option("--material")] = "gres-grigio-30x30",
    width: int = 2000,
    height: int = 1500,
) -> None:
    """Genera una scena sintetica ed esegue l'intera pipeline."""
    from app.devtools import synthetic

    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    photo = synthetic.save_image(
        synthetic.make_photo(width, height, seed=7), out / "foto.png", icc=synthetic.srgb_icc()
    )
    mask = synthetic.save_image(synthetic.make_mask(height, width, "holes"), out / "maschera.png")
    try:
        job = out / "job"
        pipeline.ingest_job(photo, job)
        rec = pipeline.import_mask(job, mask)
        pipeline.confirm_mask(job, rec.mask_sha256[:12])
        result = pipeline.render(
            job, RenderRequest(layout=LayoutParams(material_id=material)), Catalog.load()
        )
    except PosaError as exc:
        raise _fail(exc) from exc
    _print_outcome(result)
    typer.echo(f"  output in {result.render_dir}  ({time.perf_counter() - t0:.2f} s)")
    if not result.valid:
        raise typer.Exit(code=5)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
