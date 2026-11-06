"""Catalogo materiali (regola 6: texture, formato e misure arrivano solo da qui)."""

from __future__ import annotations

import io
import json
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

    @classmethod
    def load(cls, path: Path | None = None) -> Catalog:
        path = Path(path or DEFAULT_CATALOG)
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

    def get(self, material_id: str) -> MaterialSpec:
        try:
            return self._by_id[material_id]
        except KeyError:
            raise CatalogError(f"materiale {material_id!r} non presente nel catalogo") from None

    def load_variants(self, spec: MaterialSpec) -> list[NDArray[np.uint8]]:
        """Texture RGB 8 bit sRGB di ogni variante, dopo la verifica dell'hash a catalogo."""
        return [self._load_texture(spec.id, v) for v in spec.variants]

    def _load_texture(self, material_id: str, variant: TextureVariant) -> NDArray[np.uint8]:
        path = self.root / "textures" / variant.texture_file
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
