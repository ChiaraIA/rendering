from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from app.catalog.loader import Catalog
from app.core import compositor, ingest, masking
from app.core.compositor import CompositeResult
from app.core.models import MaskRecord
from app.devtools import synthetic

MATERIAL = "gres-grigio-30x30"


@pytest.fixture(autouse=True)
def _no_segmentation_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """I test non scaricano il modello di segmentazione: usano poligoni o un segmentatore finto."""
    monkeypatch.setenv("POSA_SEGMENTAZIONE", "0")


@pytest.fixture(autouse=True)
def _no_shop_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """I test non leggono il catalogo Shopify eventualmente sincronizzato in locale."""
    monkeypatch.setenv("POSA_SHOPIFY_DIR", str(tmp_path_factory.mktemp("negozio-vuoto")))


@pytest.fixture(scope="session")
def catalog(tmp_path_factory: pytest.TempPathFactory) -> Catalog:
    """Catalogo di sola lettura condiviso; i test che lo alterano ne creano uno proprio."""
    return Catalog.load(synthetic.write_catalog(tmp_path_factory.mktemp("catalog")))


def confirmed(mask: NDArray[np.uint8], image_sha: str) -> MaskRecord:
    """Maschera approvata (come dopo il verdetto positivo di un controllore)."""
    rec = masking.make_record(mask, image_sha)
    return masking.approve(rec, mask, "controllore-di-test")


def forge(
    pixels: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    sha: str | None = None,
) -> CompositeResult:
    """Solo per i test: crea un CompositeResult arbitrario (simula un compositor difettoso)."""
    p = ingest.readonly(pixels)
    a = ingest.readonly(alpha)
    return CompositeResult(
        pixels=p,
        alpha=a,
        bbox=masking.bbox_of(a),
        final_sha256=sha or ingest.sha256_array(p),
        _seal=compositor._SEAL,
    )
