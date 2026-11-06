"""Cartella di lavoro di una pratica: originale in sola lettura, immagini per gli agenti,
stato persistito. Unico punto (insieme all'exporter) che scrive immagini su disco.

pratica/
  originale/<file>        copia byte per byte, permessi 0o444
  normalizzata.png        originale normalizzato (riferimento per i controlli)
  originale.icc           profilo colore (se presente)
  preview.jpg             foto ridotta che gli agenti guardano
  preview_griglia.jpg     stessa foto con il righello delle coordinate
  lavoro/                 overlay di maschere, griglie, confronti
  consegna/               risultato finale (solo dopo tutti i cancelli)
  stato.json
"""

from __future__ import annotations

import io
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app import config
from app.agents.state import ClientRequest, JobState
from app.core import ingest, overlays
from app.core.errors import IntegrityError
from app.core.ingest import readonly, sha256_array, sha256_file
from app.core.models import SourceImage

PREVIEW_WIDTH = 1280


class Workspace:
    def __init__(self, root: Path, state: JobState, source: SourceImage) -> None:
        self.root = root
        self.state = state
        self.source = source
        self.listeners: list[Callable[[str], None]] = []
        self._masks: dict[str, NDArray[np.uint8]] = {}

    # ------------------------------------------------------------------ creazione
    @classmethod
    def create(cls, photo: Path, root: Path, request: ClientRequest) -> Workspace:
        photo, root = Path(photo), Path(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"la cartella della pratica non è vuota: {root}")
        source = ingest.load_image(photo)  # sola lettura sull'originale del cliente
        for sub in ("originale", "lavoro", "consegna"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        copy = root / "originale" / photo.name
        shutil.copyfile(photo, copy)
        os.chmod(copy, 0o444)
        if sha256_file(copy) != source.meta.source_sha256:
            raise IntegrityError("la copia dell'originale non corrisponde")
        _save(source.pixels, root / "normalizzata.png")
        if source.icc:
            (root / "originale.icc").write_bytes(source.icc)

        preview, scale = overlays.resize_to_width(source.pixels, PREVIEW_WIDTH)
        _save(preview, root / "preview.jpg")
        _save(overlays.coordinate_grid(preview), root / "preview_griglia.jpg")
        if request.sample_file:
            sample = Path(request.sample_file)
            dst = root / "originale" / f"campione{sample.suffix.lower() or '.jpg'}"
            shutil.copyfile(sample, dst)
            img = ingest.load_image(dst)
            small, _ = overlays.resize_to_width(img.pixels, 800)
            _save(small, root / "campione.jpg")
            request = request.model_copy(update={"sample_file": "campione.jpg"})

        state = JobState(
            job_id=uuid.uuid4().hex[:10],
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            image=source.meta,
            request=request,
            preview_scale=scale,
            preview_size=(preview.shape[1], preview.shape[0]),
        )
        ws = cls(root, state, source)
        ws.persist()
        return ws

    # --------------------------------------------------------------------- stato
    def persist(self) -> None:
        tmp = self.root / "stato.json.tmp"
        tmp.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, self.root / "stato.json")

    def log(self, text: str) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        self.state.events.append(f"{stamp} {text}")
        self.persist()
        for fn in self.listeners:
            fn(text)

    def verify_original(self) -> bool:
        """L'originale (copia e array normalizzato) non è cambiato."""
        copy = self.root / "originale" / self.state.image.source_name
        return (
            sha256_file(copy) == self.state.image.source_sha256
            and sha256_array(self.source.pixels) == self.state.image.normalized_sha256
            and not self.source.pixels.flags.writeable
        )

    # ------------------------------------------------------------------ immagini
    def to_full(self, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        s = self.state.preview_scale
        return [(x / s, y / s) for x, y in pts]

    def preview_of(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return overlays.resize_to_width(img, PREVIEW_WIDTH)[0]

    def write_image(self, img: NDArray[np.uint8], name: str) -> str:
        """Scrive un'immagine di lavoro in lavoro/ e ritorna il path relativo alla pratica."""
        rel = f"lavoro/{name}"
        _save(img, self.root / rel)
        return rel

    def keep_mask(self, mask: NDArray[np.uint8]) -> str:
        sha = sha256_array(mask)
        self._masks[sha] = readonly(mask)
        _save(mask, self.root / "lavoro" / f"maschera_{sha[:12]}.png")
        return sha

    def mask(self, sha: str) -> NDArray[np.uint8]:
        if sha not in self._masks:
            arr = np.asarray(Image.open(self.root / "lavoro" / f"maschera_{sha[:12]}.png"))
            if sha256_array(arr) != sha:
                raise IntegrityError("maschera su disco alterata")
            self._masks[sha] = readonly(arr)
        return self._masks[sha]


def _save(arr: NDArray[np.uint8], path: Path) -> None:
    img = Image.fromarray(np.ascontiguousarray(arr))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img.convert("RGB").save(path, format="JPEG", quality=88)
    else:
        img.save(path, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL)


def read_image(path: Path) -> NDArray[np.uint8]:
    img = Image.open(io.BytesIO(Path(path).read_bytes()))
    img.load()
    return np.asarray(img.convert("RGB"), dtype=np.uint8)
