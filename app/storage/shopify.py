"""Catalogo dal negozio Shopify: gli articoli in vendita diventano materiali da posare.

Fonte: l'elenco pubblico degli articoli del negozio (`https://<negozio>/products.json`),
lo stesso che legge il sito. Non serve nessuna chiave: si vedono solo gli articoli
pubblicati, cioè quelli in vendita.

Per ogni articolo (o per ogni variante con una foto propria, es. i colori):
- FOTO della piastrella: quelle con "texture" nel testo alternativo (consigliato); se non ce
  ne sono, la foto prodotto in cui si riconosce una piastrella su sfondo uniforme, oppure la
  prima foto se è la superficie della piastrella. Le ambientazioni vengono scartate.
- FORMATO: tag `formato:60x60`, oppure l'opzione "Formato" della variante, oppure il titolo.
- POSA, FUGA, FINITURA: tag `posa:`, `fuga:`, `colore-fuga:`, `sfalsamento:`, `rotazione:`,
  `finitura:`; altrimenti valori ragionevoli per tipo (doghe sfalsate e senza fuga,
  esagoni, piastrelle a griglia con fuga di 2-3 mm; "lucido/lappato" -> finitura lucida).
- `render:no` tra i tag esclude l'articolo.

Ogni sincronizzazione riscrive `catalog.json` della cartella del negozio; gli articoli non
cambiati non vengono riscaricati, quelli tolti dal negozio spariscono dal catalogo.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from app.core.ingest import sha256_bytes
from app.core.models import SQRT3, MaterialSpec
from app.storage import catalog_import as ci

Fetch = Callable[[str], bytes]
USER_AGENT = "posa-catalogo/1.0 (+sincronizzazione catalogo rivestimenti)"
PAGE_LIMIT = 250
MAX_PAGES = 40
IMAGE_WIDTH = 2048  # le foto si scaricano già ridotte dal CDN di Shopify

WOOD = re.compile(r"laminat|parquet|legno|rovere|noce|listone|doga|\bspc\b|\blvt\b|vinilic", re.I)
HEX = re.compile(r"esagon|hexag", re.I)
GLOSSY = re.compile(r"lucid|lappat|levigat|glossy|polished|brillant", re.I)
SATIN = re.compile(r"satinat|semi.?lucid|satin|soft.?touch", re.I)
SIZE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[x×X*]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm)?", re.I)
ONE_SIZE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm)\b", re.I)
SIZE_OPTION = re.compile(r"formato|dimension|misur|size|taglia", re.I)
TEXTURE_ALT = re.compile(r"texture|render", re.I)


# ----------------------------------------------------------------------------- rete
def http_get(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:  # troppe richieste: aspettiamo
                time.sleep(2**attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("irraggiungibile")  # pragma: no cover


def store_url(store: str) -> str:
    """'negozio.myshopify.com', 'www.sito.it' o un URL completo -> 'https://host'."""
    s = store.strip().rstrip("/")
    if "://" not in s:
        s = "https://" + s
    u = urllib.parse.urlparse(s)
    host = u.hostname or ""
    if not host or "." not in host:
        raise ValueError(f"indirizzo del negozio non valido: {store!r}")
    return f"{u.scheme}://{u.netloc}"


def fetch_products(store: str, collection: str | None = None, fetch: Fetch = http_get) -> list[Any]:
    base = store_url(store)
    path = f"/collections/{collection}/products.json" if collection else "/products.json"
    out: list[Any] = []
    for page in range(1, MAX_PAGES + 1):
        data = json.loads(fetch(f"{base}{path}?limit={PAGE_LIMIT}&page={page}"))
        items = data.get("products", [])
        out.extend(items)
        if len(items) < PAGE_LIMIT:
            break
    return out


def image_url(src: str, width: int = IMAGE_WIDTH) -> str:
    if src.startswith("//"):
        src = "https:" + src
    sep = "&" if "?" in src else "?"
    return f"{src}{sep}width={width}"


# ------------------------------------------------------------------------- regole
def _tags(product: dict[str, Any]) -> list[str]:
    raw = product.get("tags") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    return [t.strip() for t in raw if t and t.strip()]


def _tag_values(tags: list[str]) -> dict[str, str]:
    out = {}
    for t in tags:
        if ":" in t:
            k, v = t.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def parse_size(text: str) -> tuple[float, float] | None:
    """'60x60', '20 x 120 cm', '600x600 mm' -> (larghezza, altezza) in cm."""
    m = SIZE.search(text or "")
    if not m:
        return None
    a, b = _num(m.group(1)), _num(m.group(2))
    unit = (m.group(3) or "").lower()
    if unit == "mm" or (not unit and max(a, b) > 200):
        a, b = a / 10, b / 10
    if not (1 <= a <= 400 and 1 <= b <= 400):
        return None
    return a, b


def parse_single(text: str) -> float | None:
    """'25 cm' -> 25 (per gli esagoni basta la misura punta-punta)."""
    m = ONE_SIZE.search(text or "")
    if not m:
        return None
    v = _num(m.group(1))
    return v / 10 if m.group(2).lower() == "mm" else v


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", unescape(text).lower()).strip("-")
    return s or "articolo"


def _plain(html: str | None) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", html or ""))


@dataclass
class Rules:
    """Regole di posa di un articolo, con la provenienza di ogni valore (per il report)."""

    w_cm: float
    h_cm: float
    pattern: str
    bond_offset: float
    rotation: int
    joint_mm: float
    joint_color: str | None
    finish: str
    notes: list[str] = field(default_factory=list)

    @property
    def hexagon(self) -> bool:
        return self.pattern == "hex"


def product_rules(product: dict[str, Any], size_text: str | None) -> Rules | str:
    """Regole dell'articolo, o il motivo per cui non si può posare."""
    tags = _tags(product)
    tv = _tag_values(tags)
    words = " ".join([product.get("title", ""), product.get("product_type", ""), *tags])
    notes: list[str] = []
    hexagon = tv.get("posa", "").lower() in {"hex", "esagoni", "esagonale"} or bool(
        HEX.search(words)
    )
    wood = bool(WOOD.search(words))

    size = None
    for source, text in (
        ("tag formato", tv.get("formato")),
        ("opzione della variante", size_text),
        ("titolo", product.get("title")),
    ):
        if text and (size := parse_size(text)):
            notes.append(f"formato dal {source}")
            break
    if hexagon:
        if size:
            w = max(size)  # 25x22: punta-punta x lato-lato
        elif (single := parse_single(tv.get("formato") or product.get("title", ""))) is not None:
            w = single
        else:
            return "formato non trovato: aggiungi il tag formato:25 (punta-punta in cm)"
        w_cm, h_cm = w, round(w * SQRT3 / 2, 3)
        pattern = "hex"
    else:
        if not size:
            return "formato non trovato: aggiungi il tag formato:60x60 (in cm)"
        w_cm, h_cm = size
        if w_cm < h_cm:  # le doghe si descrivono di solito "20x120": la lunghezza va su x
            w_cm, h_cm = h_cm, w_cm
        long_plank = w_cm >= 3 * h_cm
        pattern = tv.get("posa", "running_bond" if (wood or long_plank) else "grid").lower()
        pattern = {"griglia": "grid", "sfalsata": "running_bond", "correre": "running_bond"}.get(
            pattern, pattern
        )
        if pattern not in {"grid", "running_bond"}:
            return f"posa {tv.get('posa')!r} non supportata (usa grid, running_bond o esagoni)"

    bond = 0.33 if wood else 0.5
    if "sfalsamento" in tv:
        bond = _num(tv["sfalsamento"])
    rotation = int(_num(tv["rotazione"])) if "rotazione" in tv else 0

    if "fuga" in tv:
        joint = _num(tv["fuga"].lower().replace("mm", "").strip())
    else:
        joint = 0.0 if wood else (2.0 if max(w_cm, h_cm) >= 45 else 3.0)
        notes.append(f"fuga {joint:g} mm predefinita (tag fuga:N per cambiarla)")
    color = tv.get("colore-fuga")
    if color and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return f"colore-fuga {color!r} non valido: usa #rrggbb"

    text = words + " " + _plain(product.get("body_html"))
    finish = tv.get("finitura", "").lower()
    if finish not in {"opaca", "satinata", "lucida"}:
        finish = "lucida" if GLOSSY.search(text) else "satinata" if SATIN.search(text) else "opaca"
    return Rules(w_cm, h_cm, pattern, bond, rotation, joint, color, finish, notes)


# ------------------------------------------------------------------------- foto
class Unusable(Exception):
    """La foto non contiene una piastrella utilizzabile."""


def load_photo(data: bytes) -> NDArray[np.uint8]:
    img = Image.open(io.BytesIO(data))
    img.load()
    img = ImageOps.exif_transpose(img) or img
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def tile_texture(
    img: NDArray[np.uint8], rules: Rules, may_be_surface: bool, flatten: bool
) -> tuple[NDArray[np.uint8], str]:
    """Texture frontale della piastrella da una foto prodotto, e come è stata ottenuta."""
    if rules.hexagon:
        try:
            vertices = ci.detect_hexagon(img)
        except ValueError as exc:
            raise Unusable(f"esagono non riconosciuto ({exc})") from exc
        return ci.hexagon_texture(img, vertices, flatten=flatten), "esagono su sfondo raddrizzato"
    quad = ci.detect_quad(img)
    if quad is not None:
        return ci.warp_quad(img, quad, rules.w_cm, rules.h_cm, flatten), "piastrella su sfondo"
    if not may_be_surface:
        raise Unusable("non è una piastrella su sfondo uniforme (ambientazione?)")
    h, w = img.shape[:2]
    if abs((w / h) / (rules.w_cm / rules.h_cm) - 1) > 0.25 and rules.w_cm / rules.h_cm > 1.5:
        raise Unusable("proporzioni diverse dal formato: non è la superficie della doga")
    if not ci.looks_like_surface(img):
        raise Unusable("sembra un'ambientazione, non la superficie della piastrella")
    return ci.crop_to_aspect(img, rules.w_cm, rules.h_cm, flatten), "superficie della piastrella"


def default_joint_color(textures: list[NDArray[np.uint8]]) -> str:
    """Fuga in tinta: media delle piastrelle schiarita verso un grigio cemento."""
    mean = np.mean([t.reshape(-1, 3).mean(axis=0) for t in textures], axis=0)
    c = np.clip(0.55 * mean + 0.45 * np.array([200.0, 196.0, 188.0]), 0, 255).round().astype(int)
    return "#{:02x}{:02x}{:02x}".format(*c)


# ------------------------------------------------------------------------ articoli
@dataclass
class Plan:
    """Un articolo da posare: dall'articolo Shopify intero o da una sua variante."""

    article_id: str
    name: str
    rules: Rules
    images: list[dict[str, Any]]  # foto candidate, in ordine
    marked: bool  # foto indicate a mano con "texture" nel testo alternativo
    mixed: bool  # più foto = più varianti miste nella posa
    url: str


def plan_product(product: dict[str, Any], base_url: str) -> tuple[list[Plan], list[str]]:
    """Articoli da creare per un prodotto Shopify, e i motivi degli scarti."""
    tags = _tags(product)
    title = unescape(product.get("title", "articolo"))
    if any(t.lower() in {"render:no", "no-render"} for t in tags):
        return [], ["escluso dal tag render:no"]
    images: list[dict[str, Any]] = sorted(
        product.get("images") or [], key=lambda i: i.get("position", 0)
    )
    if not images:
        return [], ["nessuna foto"]
    by_id = {i.get("id"): i for i in images}
    marked = [i for i in images if TEXTURE_ALT.search(i.get("alt") or "")]
    handle = product.get("handle") or slug(title)
    url = f"{base_url}/products/{handle}"
    options = product.get("options") or []
    size_pos = next(
        (k for k, o in enumerate(options) if SIZE_OPTION.search(str(o.get("name", "")))), None
    )

    # varianti raggruppate per (formato, foto propria): ognuno diventa un articolo
    groups: dict[tuple[str | None, Any], list[dict[str, Any]]] = {}
    for v in product.get("variants") or [{}]:
        size_text = v.get(f"option{size_pos + 1}") if size_pos is not None else None
        img_id = v.get("image_id") if v.get("image_id") in by_id else None
        groups.setdefault((size_text, img_id), []).append(v)
    if len(groups) > 1 and all(img is None for _, img in groups) and size_pos is None:
        groups = {(None, None): [v for vs in groups.values() for v in vs]}

    plans: list[Plan] = []
    skipped: list[str] = []
    for (size_text, img_id), variants in groups.items():
        rules = product_rules(product, size_text)
        label = _variant_label(variants[0], options, len(groups) > 1)
        if isinstance(rules, str):
            skipped.append(f"{label or title}: {rules}")
            continue
        if img_id is not None and not (marked and len(groups) == 1):
            cand, is_marked = [by_id[img_id]], by_id[img_id] in marked
        elif marked:
            cand, is_marked = marked, True
        else:
            cand, is_marked = images, False
        aid = f"shop-{slug(handle)}" + (f"-{slug(label)}" if label else "")
        name = title + (f" — {label}" if label else "")
        mixed = is_marked or any(t.lower() in {"misto", "mix", "varianti"} for t in tags)
        plans.append(Plan(aid[:64].rstrip("-"), name, rules, cand, is_marked, mixed, url))
    return plans, skipped


def _variant_label(variant: dict[str, Any], options: list[Any], many: bool) -> str:
    if not many:
        return ""
    vals = [variant.get(f"option{k + 1}") for k in range(len(options))]
    return " ".join(str(v) for v in vals if v and v != "Default Title")


# ---------------------------------------------------------------------- sincronia
@dataclass
class ProductReport:
    title: str
    status: str  # importato | invariato | saltato | errore
    articles: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class SyncReport:
    store: str
    products: list[ProductReport] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def n_articles(self) -> int:
        return sum(len(p.articles) for p in self.products if p.status in {"importato", "invariato"})

    def summary(self) -> str:
        count: dict[str, int] = {}
        for p in self.products:
            count[p.status] = count.get(p.status, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(count.items())]
        return (
            f"{self.store}: {len(self.products)} articoli letti ({', '.join(parts) or 'nessuno'}), "
            f"{self.n_articles} pronti per il render"
            + (f", {len(self.removed)} tolti" if self.removed else "")
        )


def _fingerprint(product: dict[str, Any]) -> str:
    keys = ("title", "handle", "tags", "product_type", "options", "body_html")
    data = {k: product.get(k) for k in keys}
    data["images"] = [(i.get("id"), i.get("src"), i.get("alt")) for i in product.get("images", [])]
    data["variants"] = [
        (v.get("id"), v.get("image_id"), v.get("option1"), v.get("option2"), v.get("option3"))
        for v in product.get("variants", [])
    ]
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _build_article(
    plan: Plan, fetch: Fetch, tex_dir: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    notes = list(plan.rules.notes)
    rules = plan.rules
    flatten = not rules.hexagon
    textures: list[NDArray[np.uint8]] = []
    for k, image in enumerate(plan.images):
        if textures and not plan.mixed:
            break  # una sola foto basta, se l'articolo non è un misto
        try:
            photo = load_photo(fetch(image_url(str(image.get("src", "")))))
            may_be_surface = plan.marked or k == 0
            tex, how = tile_texture(photo, rules, may_be_surface, flatten)
        except Unusable as exc:
            notes.append(f"foto {image.get('position', k + 1)} ignorata: {exc}")
            continue
        except (OSError, ValueError, urllib.error.URLError) as exc:
            notes.append(f"foto {image.get('position', k + 1)} non scaricabile: {exc}")
            continue
        notes.append(f"foto {image.get('position', k + 1)}: {how}")
        textures.append(tex)
    if not textures:
        notes.append(
            "nessuna foto utilizzabile: scrivi 'texture' nel testo alternativo della foto "
            "della piastrella su Shopify"
        )
        return None, notes
    variants = []
    for k, tex in enumerate(textures):
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(tex)).save(buf, format="PNG")
        data = buf.getvalue()
        name = f"{plan.article_id}-v{k + 1}.png"
        (tex_dir / name).write_bytes(data)
        variants.append({"texture_file": name, "texture_sha256": sha256_bytes(data)})
    if rules.hexagon:
        rotations = [0, 60, 120, 180, 240, 300]
    elif abs(rules.w_cm - rules.h_cm) < 1e-6:
        rotations = [0, 90, 180, 270]
    else:
        rotations = [0, 180]
    joint_color = None
    if rules.joint_mm > 0:
        joint_color = rules.joint_color or default_joint_color(textures)
    entry = {
        "id": plan.article_id,
        "name": plan.name,
        "version": "shopify",
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
    }
    try:
        MaterialSpec.model_validate(entry)
    except ValueError as exc:
        notes.append(f"regole non valide: {exc}")
        return None, notes
    return entry, notes


def _unique(aid: str, taken: set[str], old_state: dict[str, Any], pid: str) -> str:
    """Codice libero; un codice di un altro articolo ancora nel negozio non si riusa."""
    others = {
        a for k, v in old_state.get("products", {}).items() if k != pid for a in v["articles"]
    }
    out, n = aid, 1
    while out in taken or out in others:
        n += 1
        out = f"{aid[:60]}-{n}"
    return out


def sync(
    store: str,
    out_dir: Path,
    collection: str | None = None,
    fetch: Fetch = http_get,
    log: Callable[[str], None] | None = None,
) -> SyncReport:
    """Scarica gli articoli del negozio e riscrive il catalogo in `out_dir`."""
    say = log or (lambda _m: None)
    base = store_url(store)
    products = fetch_products(base, collection, fetch)
    say(f"{len(products)} articoli nel negozio")
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = out_dir / "textures"
    tex_dir.mkdir(exist_ok=True)
    state_path = out_dir / "sync.json"
    old_state: dict[str, Any] = (
        json.loads(state_path.read_text()) if state_path.is_file() else {"products": {}}
    )
    old_catalog: dict[str, Any] = {}
    if (out_dir / "catalog.json").is_file():
        old_catalog = {
            m["id"]: m for m in json.loads((out_dir / "catalog.json").read_text())["materials"]
        }

    report = SyncReport(store=base)
    materials: list[dict[str, Any]] = []
    taken: set[str] = set()  # codici assegnati: articoli con lo stesso nome restano distinti
    new_state: dict[str, Any] = {}
    for product in products:
        pid = str(product.get("id"))
        title = unescape(product.get("title", pid))
        fp = _fingerprint(product)
        prev = old_state["products"].get(pid)
        if prev and prev.get("fingerprint") == fp:
            entries = [old_catalog[a] for a in prev.get("articles", []) if a in old_catalog]
            if all(
                (tex_dir / v["texture_file"]).is_file() for e in entries for v in e["variants"]
            ) and len(entries) == len(prev.get("articles", [])):
                materials.extend(entries)
                taken.update(e["id"] for e in entries)
                new_state[pid] = prev
                ids = [e["id"] for e in entries]
                status = "invariato" if ids else "saltato"
                report.products.append(ProductReport(title, status, ids, prev["notes"]))
                continue
        say(f"· {title}")
        plans, skipped = plan_product(product, base)
        pr = ProductReport(title, "saltato", notes=list(skipped))
        for plan in plans:
            plan.article_id = _unique(plan.article_id, taken, old_state, pid)
            taken.add(plan.article_id)
            entry, notes = _build_article(plan, fetch, tex_dir)
            pr.notes.extend(f"{plan.name}: {n}" for n in notes)
            if entry is not None:
                materials.append(entry)
                pr.articles.append(entry["id"])
        if pr.articles:
            pr.status = "importato"
        new_state[pid] = {"fingerprint": fp, "articles": pr.articles, "notes": pr.notes}
        report.products.append(pr)

    report.removed = sorted(set(old_catalog) - {m["id"] for m in materials})
    used = {v["texture_file"] for m in materials for v in m["variants"]}
    for f in tex_dir.glob("*.png"):
        if f.name not in used:
            f.unlink()
    doc = {"schema_version": "2.0", "materials": materials}
    tmp = out_dir / "catalog.json.tmp"
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_dir / "catalog.json")
    state_path.write_text(
        json.dumps(
            {"store": base, "collection": collection, "products": new_state},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return report


def configured_store(out_dir: Path) -> tuple[str | None, str | None]:
    """(negozio, collezione) da POSA_SHOPIFY_NEGOZIO / POSA_SHOPIFY_COLLEZIONE, oppure
    l'ultimo negozio sincronizzato in `out_dir`."""
    import os

    store = os.environ.get("POSA_SHOPIFY_NEGOZIO") or None
    collection = os.environ.get("POSA_SHOPIFY_COLLEZIONE") or None
    state = out_dir / "sync.json"
    if store is None and state.is_file():
        saved = json.loads(state.read_text())
        store = saved.get("store")
        collection = collection or saved.get("collection")
    return store, collection


def format_report(report: SyncReport) -> str:
    icons = {"importato": "✅", "invariato": "▫️", "saltato": "⚠️", "errore": "❌"}
    lines = [report.summary(), ""]
    for p in report.products:
        arts = f"  → {', '.join(p.articles)}" if p.articles else ""
        lines.append(f"{icons.get(p.status, '·')} {p.title} [{p.status}]{arts}")
        if p.status != "invariato":
            lines.extend(f"      {n}" for n in p.notes if not n.endswith("formato dal titolo"))
    if report.removed:
        lines.append(f"🗑️ tolti dal catalogo (non più nel negozio): {', '.join(report.removed)}")
    return "\n".join(lines)
