"""Catalogo di un cliente gestito via API: articoli aggiunti caricando le foto delle
piastrelle (come farebbe `posa aggiungi-materiale`, ma con il riconoscimento automatico
della foto usato per i negozi Shopify).

Il file `catalog.json` è riscritto in modo atomico; un lucchetto evita che due richieste
contemporanee dello stesso processo si sovrascrivano.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.catalog.loader import SCHEMA_VERSION
from app.core.ingest import sha256_bytes
from app.core.models import SQRT3, MaterialSpec
from app.storage import catalog_import as ci
from app.storage import shopify

ARTICLE_ID = re.compile(r"^[a-z0-9][a-z0-9\-]{1,63}$")
MAX_PHOTOS = 8
MAX_ARTICLES = 5000
_lock = threading.Lock()


class ArticleError(ValueError):
    """Dati dell'articolo non validi: il messaggio è per chi integra, senza dati interni."""


def _read(catalog_dir: Path) -> dict[str, Any]:
    path = catalog_dir / "catalog.json"
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "materials": []}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write(catalog_dir: Path, doc: dict[str, Any]) -> None:
    catalog_dir.mkdir(parents=True, exist_ok=True)
    tmp = catalog_dir / "catalog.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, catalog_dir / "catalog.json")


def summary(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "codice": entry["id"],
        "nome": entry["name"],
        "formato_cm": [entry["tile_w_mm"] / 10, entry["tile_h_mm"] / 10],
        "fuga_mm": entry["joint_mm"],
        "posa": entry["pattern"]["type"],
        "finitura": entry.get("finish", "opaca"),
        "uso": entry.get("uses", ["pavimento", "rivestimento"]),
        "varianti": len(entry["variants"]),
    }


def list_articles(catalog_dir: Path) -> list[dict[str, Any]]:
    return [summary(m) for m in _read(catalog_dir)["materials"]]


def parse_rules(fields: dict[str, str]) -> shopify.Rules:
    """Regole di posa dai campi del modulo (testo): errori chiari, niente valori indovinati
    tranne quelli dichiarati facoltativi."""

    def num(key: str, default: float | None = None) -> float:
        raw = fields.get(key, "").strip().replace(",", ".")
        if not raw:
            if default is None:
                raise ArticleError(f"manca '{key}'")
            return default
        try:
            return float(raw)
        except ValueError:
            raise ArticleError(f"'{key}' non è un numero") from None

    pattern = fields.get("posa", "grid").strip() or "grid"
    if pattern not in ("grid", "running_bond", "hex"):
        raise ArticleError("'posa' deve essere grid, running_bond o hex")
    size = fields.get("formato", "").lower().replace(",", ".").replace(" ", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?", size)
    if not m:
        raise ArticleError("'formato' in cm, es. 60x60 (esagoni: larghezza punta-punta, es. 25)")
    w_cm = float(m.group(1))
    h_cm = float(m.group(2)) if m.group(2) else (w_cm * SQRT3 / 2 if pattern == "hex" else w_cm)
    if pattern == "hex":
        h_cm = w_cm * SQRT3 / 2
    if not (1 <= w_cm <= 400 and 1 <= h_cm <= 400):
        raise ArticleError("'formato' fuori misura (1-400 cm)")
    finish = fields.get("finitura", "opaca").strip() or "opaca"
    if finish not in ("opaca", "satinata", "lucida"):
        raise ArticleError("'finitura' deve essere opaca, satinata o lucida")
    rotation = int(num("rotazione", 0))
    if rotation not in (0, 45, 90):
        raise ArticleError("'rotazione' deve essere 0, 45 o 90")
    bond = num("sfalsamento", 0.5)
    if not 0 < bond < 1:
        raise ArticleError("'sfalsamento' tra 0 e 1 (es. 0.5)")
    joint = num("fuga", 0.0)
    if not 0 <= joint <= 30:
        raise ArticleError("'fuga' in mm, tra 0 e 30")
    color = fields.get("colore_fuga", "").strip() or None
    if color and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ArticleError("'colore_fuga' nel formato #rrggbb")
    uses = [u for u in re.split(r"[,\s]+", fields.get("uso", "").strip()) if u] or [
        "pavimento",
        "rivestimento",
    ]
    if any(u not in ("pavimento", "rivestimento") for u in uses):
        raise ArticleError("'uso': pavimento, rivestimento o entrambi (separati da virgola)")
    return shopify.Rules(
        w_cm=w_cm,
        h_cm=h_cm,
        pattern=pattern,
        bond_offset=bond,
        rotation=rotation,
        joint_mm=joint,
        joint_color=color,
        finish=finish,
        uses=sorted(set(uses)),
        wood=pattern == "running_bond" and joint == 0,
        mix=fields.get("misto", "").strip().lower() in ("1", "si", "sì", "true"),
    )


def put_article(
    catalog_dir: Path, article_id: str, name: str, rules: shopify.Rules, photos: list[bytes]
) -> tuple[dict[str, Any], list[str]]:
    """Aggiunge o sostituisce un articolo. Ritorna (riepilogo, note su ogni foto)."""
    if not ARTICLE_ID.match(article_id):
        raise ArticleError("codice articolo: minuscole, cifre e trattini (2-64 caratteri)")
    name = " ".join(name.split())[:120]
    if not name:
        raise ArticleError("manca 'nome'")
    if not 1 <= len(photos) <= MAX_PHOTOS:
        raise ArticleError(f"servono da 1 a {MAX_PHOTOS} foto della piastrella")
    notes: list[str] = []
    textures: list[Any] = []
    for k, data in enumerate(photos, 1):
        try:
            photo = shopify.load_photo(data)
            found, how = shopify.tile_textures(photo, rules, flatten=not rules.hexagon)
        except shopify.Unusable as exc:
            notes.append(f"foto {k} scartata: {exc}")
            continue
        except Exception:  # file rotto o formato non supportato
            notes.append(f"foto {k} scartata: file non leggibile")
            continue
        notes.append(f"foto {k}: {how}")
        textures.extend(found)
    textures = ci.distinct_tiles(textures, limit=MAX_PHOTOS)
    if not textures:
        raise ArticleError(
            "nessuna foto utilizzabile: serve la foto frontale di una singola piastrella "
            "(a tutto campo o su sfondo uniforme). " + "; ".join(notes)
        )
    tex_dir = catalog_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    blobs: list[tuple[str, bytes]] = []
    for k, tex in enumerate(textures, 1):
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(tex)).save(
            buf, format="JPEG", quality=94, subsampling=0, optimize=True
        )
        data = buf.getvalue()
        # il nome contiene l'hash: una foto nuova non sovrascrive quella in uso da una pratica
        fname = f"{article_id}-{sha256_bytes(data)[:10]}-v{k}.jpg"
        blobs.append((fname, data))
        variants.append({"texture_file": fname, "texture_sha256": sha256_bytes(data)})
    if rules.hexagon:
        rotations = [0, 60, 120, 180, 240, 300]
    elif abs(rules.w_cm - rules.h_cm) < 1e-6:
        rotations = [0, 90, 180, 270]
    else:
        rotations = [0, 180]
    joint_color = None
    if rules.joint_mm > 0:
        joint_color = rules.joint_color or shopify.default_joint_color(textures)
    entry: dict[str, Any] = {
        "id": article_id,
        "name": name,
        "version": "api",
        "tile_w_mm": round(rules.w_cm * 10, 3),
        "tile_h_mm": round(rules.h_cm * 10, 3),
        "joint_mm": rules.joint_mm,
        "joint_color": joint_color,
        "pattern": {
            "type": rules.pattern,
            "bond_offset": rules.bond_offset,
            "rotation_deg": rules.rotation,
        },
        "variants": variants,
        "variant_rotations": rotations,
        "texture_color_space": "srgb",
        "finish": rules.finish,
        "uses": rules.uses,
    }
    try:
        MaterialSpec.model_validate(entry)
    except ValueError as exc:
        first = str(exc).splitlines()[-1].strip() if str(exc) else "valori incoerenti"
        raise ArticleError(f"regole di posa non valide: {first}") from None
    with _lock:
        doc = _read(catalog_dir)
        others = [m for m in doc["materials"] if m["id"] != article_id]
        if len(others) >= MAX_ARTICLES:
            raise ArticleError(f"catalogo pieno ({MAX_ARTICLES} articoli)")
        for fname, data in blobs:
            (tex_dir / fname).write_bytes(data)
        doc["materials"] = [*others, entry]
        _write(catalog_dir, doc)
        _collect_garbage(tex_dir, doc)
    return summary(entry), notes


def delete_article(catalog_dir: Path, article_id: str) -> bool:
    with _lock:
        doc = _read(catalog_dir)
        keep = [m for m in doc["materials"] if m["id"] != article_id]
        if len(keep) == len(doc["materials"]):
            return False
        doc["materials"] = keep
        _write(catalog_dir, doc)
        _collect_garbage(catalog_dir / "textures", doc)
    return True


def _collect_garbage(tex_dir: Path, doc: dict[str, Any], grace_s: float = 3600.0) -> None:
    """Toglie le foto non più usate da nessun articolo, altrimenti chi sostituisce un
    articolo all'infinito riempie il disco. Quelle sostituite da meno di un'ora restano:
    una pratica in corso potrebbe ancora leggerle."""
    if not tex_dir.is_dir():
        return
    used = {v["texture_file"] for m in doc["materials"] for v in m["variants"]}
    limit = time.time() - grace_s
    for f in tex_dir.iterdir():
        try:
            if f.is_file() and f.name not in used and f.stat().st_mtime < limit:
                f.unlink()
        except OSError:
            continue


def article_uses(catalog_dir: Path, article_id: str) -> list[str] | None:
    """Superfici su cui l'articolo si può posare; None se l'articolo non esiste."""
    for m in _read(catalog_dir)["materials"]:
        if m["id"] == article_id:
            return list(m.get("uses", ["pavimento", "rivestimento"]))
    return None
