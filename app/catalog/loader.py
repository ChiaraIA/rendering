"""Catalogo materiali (regola 6: texture, formato e misure arrivano solo da qui)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from pydantic import BaseModel, ValidationError

from app.core.errors import CatalogError
from app.core.ingest import readonly, sha256_bytes
from app.core.models import MaterialSpec, TextureVariant

DEFAULT_CATALOG = Path(__file__).parent / "data" / "catalog.json"


def _fold(text: str) -> str:
    """Minuscole e senza accenti, per la ricerca."""
    import unicodedata

    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c)).replace(",", ".")


def shop_catalog_dir() -> Path:
    """Cartella del catalogo sincronizzato dal negozio Shopify (`posa shopify`)."""
    return Path(os.environ.get("POSA_SHOPIFY_DIR", "catalogo_shopify"))


SCHEMA_VERSION = "2.0"


class CatalogFile(BaseModel):
    schema_version: Literal["2.0"]
    materials: list[MaterialSpec]


class Catalog:
    def __init__(self, path: Path, doc: CatalogFile, sha256: str) -> None:
        self.path = path
        self.root = path.parent
        self.doc = doc
        self.sha256 = sha256
        self._by_id = {m.id: m for m in doc.materials}
        self._roots = {m.id: self.root for m in doc.materials}  # cartella delle texture

    @classmethod
    def load(cls, path: Path | None = None) -> Catalog:
        """Senza `path`: catalogo della repo + quello del negozio Shopify, se sincronizzato."""
        if path is not None:
            return cls._load_file(Path(path))
        main = cls._load_file(DEFAULT_CATALOG)
        shop = shop_catalog_dir() / "catalog.json"
        return main.merged(cls._load_file(shop)) if shop.is_file() else main

    def merged(self, other: Catalog) -> Catalog:
        """Catalogo unico: ogni articolo continua a leggere le texture dalla propria cartella."""
        clash = set(self._by_id) & set(other._by_id)
        if clash:
            raise CatalogError(f"codici articolo presenti in due cataloghi: {sorted(clash)}")
        doc = CatalogFile(schema_version="2.0", materials=self.materials + other.materials)
        out = Catalog(self.path, doc, sha256_bytes((self.sha256 + other.sha256).encode()))
        out._roots = {**self._roots, **other._roots}
        return out

    @classmethod
    def _load_file(cls, path: Path) -> Catalog:
        if not path.is_file():
            raise CatalogError(f"catalogo non trovato: {path}")
        raw = path.read_bytes()
        try:
            doc = CatalogFile.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise CatalogError(f"catalogo non valido: {exc}") from exc
        ids = [m.id for m in doc.materials]
        if len(ids) != len(set(ids)):
            raise CatalogError("id materiale duplicati nel catalogo")
        return cls(path, doc, sha256_bytes(raw))

    @property
    def materials(self) -> list[MaterialSpec]:
        return list(self.doc.materials)

    def search(self, query: str = "") -> list[MaterialSpec]:
        """Articoli che contengono tutte le parole cercate (nel codice o nel nome)."""
        words = [w for w in _fold(query).split() if w]
        if not words:
            return self.materials
        return [m for m in self.doc.materials if all(w in _fold(f"{m.id} {m.name}") for w in words)]

    def get(self, material_id: str) -> MaterialSpec:
        try:
            return self._by_id[material_id]
        except KeyError:
            raise CatalogError(f"materiale {material_id!r} non presente nel catalogo") from None

    def load_variants(self, spec: MaterialSpec) -> list[NDArray[np.uint8]]:
        """Texture RGB 8 bit sRGB di ogni variante, dopo la verifica dell'hash a catalogo."""
        return [self._load_texture(spec.id, v) for v in spec.variants]

    def _load_texture(self, material_id: str, variant: TextureVariant) -> NDArray[np.uint8]:
        path = self._roots.get(material_id, self.root) / "textures" / variant.texture_file
        if not path.is_file():
            raise CatalogError(f"texture mancante per {material_id}: {path.name}")
        data = path.read_bytes()
        if sha256_bytes(data) != variant.texture_sha256:
            raise CatalogError(
                f"hash della texture {variant.texture_file} di {material_id} "
                "non corrisponde al catalogo"
            )
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img) or img
        if img.mode not in {"RGB", "RGBA", "L"}:
            raise CatalogError(f"modalità texture non supportata: {img.mode}")
        return readonly(np.asarray(img.convert("RGB"), dtype=np.uint8))
