"""Strumenti degli agenti.

La logica sta in `Toolbox` (funzioni async normali, testabili senza modelli); `mcp_server()`
le espone all'Agent SDK. Ogni strumento che produce un lavoro lo fa giudicare SUBITO a un
controllore indipendente e registra l'esito: l'approvazione la scrive solo il codice.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np

from app import config
from app.agents import prompts
from app.agents.checkers import CheckerRunner, CheckRequest
from app.agents.state import (
    Attempt,
    JobState,
    MaterialChoice,
    RenderOutcome,
    Step,
    Verdict,
    can_attempt,
    missing_for_delivery,
    missing_for_render,
    next_step,
)
from app.catalog.loader import Catalog
from app.core import compositor, geometry, lighting, masking, material, overlays, validator
from app.core.compositor import CompositeResult
from app.core.errors import CatalogError, GeometryError
from app.core.ingest import sha256_bytes
from app.core.masking import Polygon
from app.core.models import LayoutStats, MaskOrigin, ValidationReport
from app.storage import exporter
from app.storage.workspace import Workspace

Result = dict[str, Any]
MAX_ATTEMPTS = config.MAX_ATTEMPTS


def _polys(raw: Any) -> list[Polygon]:
    out: list[Polygon] = []
    for poly in raw or []:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if len(pts) >= 3:
            out.append(pts)
    return out


class Toolbox:
    def __init__(self, ws: Workspace, catalog: Catalog, checker: CheckerRunner) -> None:
        self.ws = ws
        self.catalog = catalog
        self.checker = checker
        self._result: CompositeResult | None = None
        self._validation: ValidationReport | None = None

    # ------------------------------------------------------------------ supporto
    @property
    def state(self) -> JobState:
        return self.ws.state

    async def _check(self, req: CheckRequest) -> Verdict:
        self.ws.log(f"🔍 {req.checker}: esamina il lavoro…")
        verdict = await self.checker(req, self.ws.root)
        if verdict.cost_usd:
            self.state.cost_usd += verdict.cost_usd
        icon = "✅" if verdict.approved else "❌"
        word = "APPROVATO" if verdict.approved else "BOCCIATO"
        self.ws.log(f"{icon} {req.checker}: {word} — {verdict.reason}")
        return verdict

    def _record(self, step: Step, sha: str, overlay: str | None, verdict: Verdict) -> Attempt:
        att = Attempt(
            n=len(self.state.attempts(step)) + 1,
            artifact_sha256=sha,
            overlay=overlay,
            verdict=verdict,
        )
        self.state.attempts(step).append(att)
        self.ws.persist()
        return att

    def _reply(self, step: Step, att: Attempt, extra: Result | None = None) -> Result:
        left = config.MAX_ATTEMPTS - len(self.state.attempts(step))
        out: Result = {
            "esito": "APPROVATO" if att.verdict.approved else "BOCCIATO",
            "controllore": att.verdict.checker,
            "motivo": att.verdict.reason,
            "problemi": att.verdict.issues,
            "tentativi_rimasti": left,
            "prossimo_passo": next_step(self.state),
        }
        if att.overlay:
            out["overlay"] = att.overlay
        out.update(extra or {})
        return out

    def _code_reject(self, step: Step, reason: str) -> Result:
        att = self._record(
            step, "-", None, Verdict(checker="controllo-automatico", approved=False, reason=reason)
        )
        self.ws.log(f"❌ controllo-automatico ({step}): {reason}")
        return self._reply(step, att)

    # --------------------------------------------------------------------- stato
    async def stato_pratica(self) -> Result:
        s = self.state
        req = s.request
        return {
            "pratica": s.job_id,
            "foto": {
                "preview": "preview.jpg",
                "preview_con_righello": "preview_griglia.jpg",
                "dimensioni_preview_px": list(s.preview_size),
            },
            "richiesta_cliente": {
                "codice_materiale": req.material_id,
                "foto_campione": req.sample_file,
                "misure_stanza_m": [round(x / 1000, 3) for x in req.room_mm]
                if req.room_mm
                else None,
            },
            "maschera_approvata": s.mask is not None,
            "overlay_maschera_approvata": next(
                (a.overlay for a in reversed(s.mask_attempts) if a.verdict.approved), None
            ),
            "geometria_approvata": s.geometry is not None,
            "materiale": s.material.material_id if s.material else None,
            "posa_approvata": s.render is not None,
            "consegnato": bool(s.delivered),
            "tentativi": {
                k: len(s.attempts(k)) for k in ("maschera", "geometria", "materiale", "posa")
            },
            "mancano_per_consegna": missing_for_delivery(s),
            "prossimo_passo": next_step(s),
        }

    # ------------------------------------------------------------------ maschera
    async def proponi_maschera(
        self, pavimento: Any, escludi: Any = None, rifinisci: bool = True
    ) -> Result:
        if block := can_attempt(self.state, "maschera"):
            return {"errore": block}
        include = [self.ws.to_full(p) for p in _polys(pavimento)]
        exclude = [self.ws.to_full(p) for p in _polys(escludi)]
        if not include:
            return {"errore": "serve almeno un poligono 'pavimento' con 3 o più punti"}
        src = self.ws.source
        coarse = masking.rasterize(src.shape_hw, include, exclude)
        mask = await asyncio.to_thread(masking.refine, src.pixels, coarse) if rifinisci else coarse
        area = float(np.count_nonzero(mask >= 128)) / mask.size
        if area < 0.01:
            return self._code_reject("maschera", f"maschera quasi vuota ({area:.1%} della foto)")
        sha = self.ws.keep_mask(mask)
        n = len(self.state.mask_attempts) + 1
        overlay = self.ws.write_image(
            overlays.mask_overlay(self.ws.preview_of(src.pixels), self.ws.preview_of(mask)),
            f"maschera_{n}.jpg",
        )
        self.ws.log(f"🟦 pavimento: maschera #{n} proposta ({area:.0%} della foto)")
        verdict = await self._check(
            CheckRequest(
                "controllo-maschera",
                prompts.CHECK_MASK,
                ["preview.jpg", overlay],
                f"Dati: la maschera copre il {area:.1%} della foto.",
            )
        )
        att = self._record("maschera", sha, overlay, verdict)
        if verdict.approved:
            origin = MaskOrigin.REFINED if rifinisci else MaskOrigin.POLYGON
            rec = masking.make_record(mask, src.meta.normalized_sha256, origin)
            self.state.mask = masking.approve(rec, mask, verdict.checker)
            self.ws.persist()
        return self._reply("maschera", att)

    # ----------------------------------------------------------------- geometria
    async def proponi_geometria(
        self, angoli: Any, lunghezza_m: float | None = None, larghezza_m: float | None = None
    ) -> Result:
        if block := can_attempt(self.state, "geometria"):
            return {"errore": block}
        if self.state.mask is None:
            return {"errore": "prima serve la maschera approvata"}
        try:
            pts = [(float(p[0]), float(p[1])) for p in angoli]
        except (TypeError, ValueError, IndexError):
            return {"errore": "angoli: servono 4 punti [x, y]"}
        if len(pts) != 4:
            return {"errore": "angoli: servono esattamente 4 punti [x, y]"}
        req = self.state.request
        estimated = req.room_mm is None
        if req.room_mm:
            length, width = req.room_mm
        elif lunghezza_m and larghezza_m:
            length, width = float(lunghezza_m) * 1000, float(larghezza_m) * 1000
        else:
            return {"errore": "il cliente non ha dato le misure: stima lunghezza_m e larghezza_m"}
        src = self.ws.source
        try:
            geom = geometry.room_geometry(
                self.ws.to_full(pts),
                length,
                width,
                (src.meta.width, src.meta.height),
                src.meta.focal_35mm,
            )
            geometry.check_mask_in_front(geom, self.ws.mask(self.state.mask.mask_sha256))
        except GeometryError as exc:
            return self._code_reject("geometria", f"geometria non valida: {exc}")
        n = len(self.state.geometry_attempts) + 1
        s = self.state.preview_scale
        h_prev = (np.diag([s, s, 1.0]) @ np.asarray(geom.H)).tolist()
        mask_prev = self.ws.preview_of(self.ws.mask(self.state.mask.mask_sha256))
        overlay = self.ws.write_image(
            overlays.plane_grid_overlay(
                self.ws.preview_of(src.pixels), geom.model_copy(update={"H": h_prev}), mask_prev
            ),
            f"geometria_{n}.jpg",
        )
        f = geom.focal
        focal_txt = f.message
        if f.consistent is not None:
            focal_txt += (
                f" (angolo tra i muri implicito {f.axes_angle_deg:.1f}°, errore sul "
                f"rapporto lunghezza/larghezza {f.aspect_error:+.0%})"
            )
        self.ws.log(
            f"📐 geometra: proposta #{n}, stanza {length / 1000:.2f} x {width / 1000:.2f} m"
            + (" (stimata)" if estimated else "")
        )
        verdict = await self._check(
            CheckRequest(
                "controllo-geometria",
                prompts.CHECK_GEOMETRY,
                ["preview.jpg", overlay],
                f"Dati: stanza {length / 1000:.2f} x {width / 1000:.2f} m "
                f"({'stimata dall agente' if estimated else 'misure del cliente'}). "
                f"Coerenza con la focale EXIF: {focal_txt}.",
            )
        )
        if verdict.approved and f.consistent is False:
            verdict = verdict.model_copy(
                update={
                    "approved": False,
                    "reason": verdict.reason + " — ma il controllo numerico sulla focale è fallito",
                    "issues": [*verdict.issues, focal_txt],
                }
            )
            self.ws.log("❌ controllo-automatico: incoerenza con la focale della fotocamera")
        sha = sha256_bytes(geom.model_dump_json().encode())
        att = self._record("geometria", sha, overlay, verdict)
        if verdict.approved:
            self.state.geometry = geom
            self.state.geometry_sha256 = sha
            self.state.room_estimated = estimated
            self.ws.persist()
        return self._reply("geometria", att, {"focale": focal_txt})

    # ------------------------------------------------------------------ materiale
    async def catalogo(self) -> Result:
        items, out = [], []
        for spec in self.catalog.materials:
            tex = self.catalog.load_variants(spec)[0]
            joint = None
            if spec.joint_mm > 0 and spec.joint_color:
                c = spec.joint_color.lstrip("#")
                joint = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16))
            items.append(
                (f"{spec.id} — {spec.name}", overlays.tile_patch(tex, joint, hexagon=spec.is_hex))
            )
            out.append(
                {
                    "codice": spec.id,
                    "nome": spec.name,
                    "formato_cm": [spec.tile_w_mm / 10, spec.tile_h_mm / 10],
                    "fuga_mm": spec.joint_mm,
                    "posa": spec.pattern.type,
                    "rotazione": spec.pattern.rotation_deg,
                    "varianti": len(spec.variants),
                }
            )
        sheet = self.ws.write_image(overlays.contact_sheet(items), "catalogo.jpg")
        return {"articoli": out, "foglio_miniature": sheet}

    async def scegli_materiale(self, codice: str, motivazione: str = "") -> Result:
        if block := can_attempt(self.state, "materiale"):
            return {"errore": block}
        try:
            spec = self.catalog.get(codice)
        except CatalogError as exc:
            return {"errore": str(exc)}
        req = self.state.request
        if req.material_id:
            if codice != req.material_id:
                return {"errore": f"il cliente ha scelto il codice {req.material_id!r}: usa quello"}
            verdict = Verdict(
                checker="codice-cliente",
                approved=True,
                reason="codice a catalogo indicato dal cliente",
            )
            att = self._record("materiale", spec.id, None, verdict)
            self.state.material = MaterialChoice(
                material_id=spec.id,
                how="codice_cliente",
                motivation=motivazione or "codice del cliente",
            )
            self.ws.log(f"🧱 materiali: {spec.name} (codice del cliente)")
            self.ws.persist()
            return self._reply("materiale", att)
        if not req.sample_file:
            return {"errore": "il cliente non ha indicato né un codice né un campione"}
        tex = self.catalog.load_variants(spec)[0]
        patch = self.ws.write_image(
            overlays.tile_patch(tex, None, hexagon=spec.is_hex), f"materiale_{spec.id}.jpg"
        )
        self.ws.log(f"🧱 materiali: propone {spec.name} — {motivazione}")
        verdict = await self._check(
            CheckRequest(
                "controllo-materiale",
                prompts.CHECK_MATERIAL,
                [req.sample_file, patch],
                f"Articolo scelto: {spec.name} ({spec.id}), {spec.tile_w_mm / 10:g}x"
                f"{spec.tile_h_mm / 10:g} cm. Motivazione dell'agente: {motivazione}",
            )
        )
        att = self._record("materiale", spec.id, patch, verdict)
        if verdict.approved:
            self.state.material = MaterialChoice(
                material_id=spec.id, how="abbinamento_campione", motivation=motivazione
            )
            self.ws.persist()
        return self._reply("materiale", att)

    # ----------------------------------------------------------------------- posa
    def _render_sync(self) -> tuple[CompositeResult, ValidationReport, LayoutStats]:
        s, src = self.state, self.ws.source
        assert s.mask and s.geometry and s.material
        mask = self.ws.mask(s.mask.mask_sha256)
        spec = self.catalog.get(s.material.material_id)
        seed = material.derive_seed(src.meta.normalized_sha256, spec.id)
        layer = material.tiled_layer(
            spec,
            self.catalog.load_variants(spec),
            src,
            s.geometry,
            mask,
            seed,
            supersampling=config.SUPERSAMPLING,
        )
        ppm = geometry.px_per_mm_map(s.geometry, layer.bbox)
        light = lighting.combined_light(src, mask, layer.bbox, ppm)
        result = compositor.composite(src, mask, s.mask, layer, light)
        copy = self.ws.root / "originale" / s.image.source_name
        report = validator.validate(src, result, mask, s.mask, copy)
        lo, hi = geometry.px_per_mm_range(s.geometry, mask)
        whole, cut = material.count_tiles(spec, s.geometry.length_mm, s.geometry.width_mm)
        return (
            result,
            report,
            LayoutStats(
                seed=seed, tiles_whole=whole, tiles_cut=cut, px_per_mm_min=lo, px_per_mm_max=hi
            ),
        )

    async def esegui_posa(self) -> Result:
        if block := can_attempt(self.state, "posa"):
            return {"errore": block}
        if missing := missing_for_render(self.state):
            return {"errore": "cancello chiuso, manca: " + "; ".join(missing)}
        chosen, geom, mask_rec = self.state.material, self.state.geometry, self.state.mask
        assert chosen is not None and geom is not None and mask_rec is not None  # cancello
        spec = self.catalog.get(chosen.material_id)
        self.ws.log(f"🎨 posa: render di {spec.name} con texture del catalogo…")
        result, report, stats = await asyncio.to_thread(self._render_sync)
        integrity = report.valid and self.ws.verify_original()
        n = len(self.state.render_attempts) + 1
        before = self.ws.preview_of(self.ws.source.pixels)
        after = self.ws.preview_of(result.pixels)
        result_prev = self.ws.write_image(after, f"posa_{n}.jpg")
        ba = self.ws.write_image(overlays.side_by_side(before, after), f"prima_dopo_{n}.jpg")
        failed = [c.name for c in report.checks if c.blocking and not c.passed]
        if not integrity:
            att = self._record(
                "posa",
                result.final_sha256,
                ba,
                Verdict(
                    checker="controllo-integrità",
                    approved=False,
                    reason="integrità fallita: " + (", ".join(failed) or "originale modificato"),
                ),
            )
            self.ws.log(f"❌ controllo-integrità: FALLITO ({', '.join(failed)})")
            return self._reply("posa", att)
        self.ws.log(
            "✅ controllo-integrità: pixel fuori dal pavimento identici all'originale, "
            "risoluzione e profilo colore invariati"
        )
        verdict = await self._check(
            CheckRequest(
                "revisore-finale",
                prompts.CHECK_RENDER,
                [ba],
                f"Materiale: {spec.name}, {spec.tile_w_mm / 10:g}x{spec.tile_h_mm / 10:g} cm, fuga "
                f"{spec.joint_mm:g} mm. Stanza {geom.length_mm / 1000:.2f} x "
                f"{geom.width_mm / 1000:.2f} m. Controlli di integrità: tutti superati "
                f"({len(report.checks)}).",
            )
        )
        att = self._record("posa", result.final_sha256, ba, verdict)
        if verdict.approved:
            self._result, self._validation = result, report
            self.state.render = RenderOutcome(
                final_sha256=result.final_sha256,
                mask_sha256=mask_rec.mask_sha256,
                geometry_sha256=self.state.geometry_sha256 or "",
                material_id=spec.id,
                checks=report.checks,
                integrity_ok=True,
                layout=stats,
                before_after=ba,
                result_preview=result_prev,
            )
            self.ws.persist()
        return self._reply("posa", att, {"prima_dopo": ba})

    # ------------------------------------------------------------------- consegna
    async def consegna(self) -> Result:
        if missing := missing_for_delivery(self.state):
            return {"errore": "cancello chiuso, manca: " + "; ".join(missing)}
        result, report, render = self._result, self._validation, self.state.render
        if (
            result is None
            or report is None
            or render is None
            or result.final_sha256 != render.final_sha256
        ):
            return {"errore": "risultato non disponibile in memoria: riesegui esegui_posa"}
        out = self.ws.root / "consegna"
        master, checks = exporter.export_master(result, report, self.ws.source.icc, out, "png")
        if master is None:
            failed = ", ".join(c.name for c in checks if not c.passed)
            self.ws.log(f"❌ consegna bloccata: verifica del file fallita ({failed})")
            return {"errore": f"verifica del file esportato fallita: {failed}"}
        files = [master.path] + [
            o.path for o in exporter.export_extras(result, self.ws.source.icc, out)
        ]
        ba_src = self.ws.root / render.before_after
        (out / "prima_dopo.jpg").write_bytes(ba_src.read_bytes())
        files.append("prima_dopo.jpg")
        self.state.delivered = [f"consegna/{f}" for f in files]
        report_json = self.report()
        (out / "report.json").write_text(json.dumps(report_json, indent=2, ensure_ascii=False))
        self.state.delivered.append("consegna/report.json")
        self.ws.persist()
        self.ws.log("📦 consegna: risultato esportato e verificato")
        return {"consegnato": self.state.delivered}

    async def richiedi_operatore(self, motivo: str) -> Result:
        self.state.escalation = motivo
        self.ws.persist()
        (self.ws.root / "ESCALATION.txt").write_text(motivo + "\n", encoding="utf-8")
        self.ws.log(f"🙋 pratica passata a un operatore: {motivo}")
        return {"ok": True, "nota": "pratica sospesa e segnalata all'operatore"}

    # --------------------------------------------------------------------- report
    def report(self) -> Result:
        s = self.state
        spec = self.catalog.get(s.material.material_id) if s.material else None
        return {
            "pratica": s.job_id,
            "creata": s.created_at,
            "foto": s.image.model_dump(mode="json"),
            "richiesta": s.request.model_dump(mode="json"),
            "materiale": spec.model_dump(mode="json") if spec else None,
            "scelta_materiale": s.material.model_dump(mode="json") if s.material else None,
            "stanza_m": [s.geometry.length_mm / 1000, s.geometry.width_mm / 1000]
            if s.geometry
            else None,
            "misure_stimate": s.room_estimated,
            "posa": s.render.layout.model_dump(mode="json") if s.render else None,
            "controlli_integrita": [c.model_dump(mode="json") for c in s.render.checks]
            if s.render
            else [],
            "verdetti": {
                step: [a.model_dump(mode="json") for a in s.attempts(step)]
                for step in ("maschera", "geometria", "materiale", "posa")
            },
            "sha256_risultato": s.render.final_sha256 if s.render else None,
            "costo_controllori_usd": round(s.cost_usd, 4),
        }

    def _trace(self, name: str, args: dict[str, Any]) -> None:
        """Traccia degli input grezzi ricevuti dagli agenti (audit)."""
        path = self.ws.root / "lavoro" / "tracce.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"tool": name, "args": args}, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- server MCP
    def mcp_server(self) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        def wrap(result: Result) -> dict[str, Any]:
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

        point = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
        poly = {"type": "array", "items": point, "minItems": 3}

        @tool("stato_pratica", "Stato della pratica, file disponibili e prossimo passo.", {})
        async def t_stato(args: dict[str, Any]) -> dict[str, Any]:
            return wrap(await self.stato_pratica())

        @tool(
            "proponi_maschera",
            "Propone la maschera del pavimento (coordinate in pixel di preview_griglia.jpg) e la "
            "fa giudicare a un controllore indipendente.",
            {
                "type": "object",
                "properties": {
                    "pavimento": {"type": "array", "items": poly},
                    "escludi": {"type": "array", "items": poly},
                    "rifinisci": {"type": "boolean"},
                },
                "required": ["pavimento"],
            },
        )
        async def t_maschera(args: dict[str, Any]) -> dict[str, Any]:
            self._trace("proponi_maschera", args)
            return wrap(
                await self.proponi_maschera(
                    args.get("pavimento"), args.get("escludi"), bool(args.get("rifinisci", True))
                )
            )

        @tool(
            "proponi_geometria",
            "Propone i 4 angoli del pavimento della stanza (pixel di preview_griglia.jpg, anche "
            "fuori immagine) e le misure; un controllore giudica la griglia proiettata.",
            {
                "type": "object",
                "properties": {
                    "angoli": {"type": "array", "items": point, "minItems": 4, "maxItems": 4},
                    "lunghezza_m": {"type": "number"},
                    "larghezza_m": {"type": "number"},
                },
                "required": ["angoli"],
            },
        )
        async def t_geometria(args: dict[str, Any]) -> dict[str, Any]:
            self._trace("proponi_geometria", args)
            return wrap(
                await self.proponi_geometria(
                    args.get("angoli"), args.get("lunghezza_m"), args.get("larghezza_m")
                )
            )

        @tool("catalogo", "Elenco degli articoli a catalogo e foglio delle miniature.", {})
        async def t_catalogo(args: dict[str, Any]) -> dict[str, Any]:
            return wrap(await self.catalogo())

        @tool(
            "scegli_materiale",
            "Sceglie l'articolo del catalogo da posare.",
            {
                "type": "object",
                "properties": {"codice": {"type": "string"}, "motivazione": {"type": "string"}},
                "required": ["codice"],
            },
        )
        async def t_materiale(args: dict[str, Any]) -> dict[str, Any]:
            return wrap(await self.scegli_materiale(args["codice"], args.get("motivazione", "")))

        @tool("esegui_posa", "Render deterministico + controllo integrità + revisione finale.", {})
        async def t_posa(args: dict[str, Any]) -> dict[str, Any]:
            return wrap(await self.esegui_posa())

        @tool("consegna", "Esporta il risultato se tutti i cancelli sono aperti.", {})
        async def t_consegna(args: dict[str, Any]) -> dict[str, Any]:
            return wrap(await self.consegna())

        @tool(
            "richiedi_operatore",
            "Sospende la pratica e la passa a un operatore umano.",
            {"motivo": str},
        )
        async def t_operatore(args: dict[str, Any]) -> dict[str, Any]:
            return wrap(await self.richiedi_operatore(args["motivo"]))

        return create_sdk_mcp_server(
            name="posa",
            version="1.0.0",
            tools=[
                t_stato,
                t_maschera,
                t_geometria,
                t_catalogo,
                t_materiale,
                t_posa,
                t_consegna,
                t_operatore,
            ],
        )
