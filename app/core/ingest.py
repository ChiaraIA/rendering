"""Ingest: validazione, orientamento EXIF, profilo colore, hash.

Il file sorgente viene solo letto ('rb'). Il risultato è l'"originale normalizzato":
pixel dopo l'orientamento EXIF, 8 bit, RGB o RGBA, in un array in sola lettura.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pillow_heif
from numpy.typing import NDArray
from PIL import Image, ImageOps

from app import config
from app.core import icc as icc_gen
from app.core.colorspace import classify_icc, describe_icc
from app.core.errors import UnsupportedImageError
from app.core.models import ColorSpace, ImageMeta, SourceImage

pillow_heif.register_heif_opener()

_EXIF_ORIENTATION_TAG = 0x0112
_EXIF_IFD = 0x8769
_FOCAL_35MM = 0xA405
_MAKE, _MODEL = 0x010F, 0x0110
_NCLX_BT709, _NCLX_P3 = 1, 12
_SDR_TRANSFERS = {1, 6, 13}  # BT.709, BT.601, sRGB (esclusi PQ/HLG)

# modalità Pillow -> modalità normalizzata
_MODE_MAP = {
    "RGB": "RGB",
    "RGBA": "RGBA",
    "L": "RGB",
    "1": "RGB",
    "LA": "RGBA",
    "PA": "RGBA",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: NDArray[np.generic]) -> str:
    """Hash di contenuto + shape + dtype (stessi byte ma forma diversa => hash diverso)."""
    h = hashlib.sha256()
    h.update(f"{arr.shape}|{arr.dtype.str}\n".encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def readonly(arr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    out = np.ascontiguousarray(arr).copy()
    out.setflags(write=False)
    return out


def _normalized_mode(img: Image.Image) -> str:
    if img.mode == "P":
        return "RGBA" if "transparency" in img.info else "RGB"
    if img.mode in _MODE_MAP:
        return _MODE_MAP[img.mode]
    if img.mode.startswith("I") or img.mode in {"F", "RGB;16", "RGBA;16"}:
        raise UnsupportedImageError(
            f"modalità {img.mode!r}: le immagini a 16 bit/float sono previste in M3"
        )
    raise UnsupportedImageError(f"modalità colore non supportata: {img.mode!r}")


def load_image(path: Path) -> SourceImage:
    """Legge la foto in sola lettura e la normalizza."""
    path = Path(path)
    if not path.is_file():
        raise UnsupportedImageError(f"file non trovato: {path}")
    data = path.read_bytes()
    return decode_image(data, source_name=path.name)


def _exif_capture(img: Image.Image) -> tuple[float | None, str | None]:
    """Focale equivalente 35 mm e modello della fotocamera (se presenti)."""
    exif = img.getexif()
    f35 = exif.get_ifd(_EXIF_IFD).get(_FOCAL_35MM) or exif.get(_FOCAL_35MM)
    try:
        focal = float(f35) if f35 else None
    except (TypeError, ValueError):
        focal = None
    parts = [str(exif.get(t, "")).strip("\x00 ") for t in (_MAKE, _MODEL)]
    camera = " ".join(p for p in parts if p) or None
    return focal, camera


def _nclx_color(info: dict[str, object]) -> tuple[bytes | None, ColorSpace | None]:
    """HEIC senza ICC: il colore è dichiarato con nclx (primarie + TRC)."""
    nclx = info.get("nclx_profile")
    if not isinstance(nclx, dict):
        return None, None
    primaries = nclx.get("color_primaries")
    transfer = nclx.get("transfer_characteristics")
    if transfer not in _SDR_TRANSFERS:
        return None, ColorSpace.UNKNOWN
    if primaries == _NCLX_P3:
        return icc_gen.display_p3(), ColorSpace.DISPLAY_P3
    if primaries == _NCLX_BT709:
        return None, ColorSpace.SRGB
    return None, ColorSpace.UNKNOWN


def decode_image(data: bytes, source_name: str) -> SourceImage:
    source_sha = sha256_bytes(data)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise UnsupportedImageError(f"impossibile decodificare l'immagine: {exc}") from exc

    fmt = img.format or "?"
    if fmt not in config.ALLOWED_INPUT_FORMATS:
        raise UnsupportedImageError(f"formato {fmt!r} non supportato")
    if img.width * img.height > config.MAX_PIXELS:
        raise UnsupportedImageError(f"immagine troppo grande: {img.width}x{img.height}")
    n_frames = int(getattr(img, "n_frames", 1))
    # MPO = JPEG con immagini secondarie (Ultra HDR, foto Samsung): si usa il frame 0.
    if n_frames > 1 and fmt != "MPO":
        raise UnsupportedImageError("immagini multi-frame/animate non supportate")
    extra_images = (fmt == "MPO" and n_frames > 1) or bool(
        img.info.get("depth_images") or img.info.get("aux")
    )

    source_mode = img.mode
    target_mode = _normalized_mode(img)
    # pillow-heif applica già la rotazione HEIF e azzera l'EXIF: decidiamo SEMPRE in base
    # all'EXIF dell'immagine decodificata, così non si ruota mai due volte.
    exif_orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
    if not 1 <= exif_orientation <= 8:
        exif_orientation = 1
    orientation = int(img.info.get("original_orientation") or exif_orientation)

    icc: bytes | None = img.info.get("icc_profile") or None
    icc_dropped = False
    color_origin = "icc" if icc else "none"
    cs_override: ColorSpace | None = None
    if icc is not None and source_mode in {"L", "LA", "1"}:
        # Un profilo GRAY non è valido su un master RGB: lo scartiamo e lo dichiariamo.
        _, xcs = describe_icc(icc)
        if xcs is None or xcs.strip().upper() != "RGB":
            icc, icc_dropped, color_origin = None, True, "none"
    if icc is None and fmt == "HEIF":
        icc, cs_override = _nclx_color(img.info)
        if cs_override is not None:
            color_origin = "nclx"

    focal, camera = _exif_capture(img)
    oriented = ImageOps.exif_transpose(img) if exif_orientation != 1 else img
    assert oriented is not None
    if oriented.mode != target_mode:
        oriented = oriented.convert(target_mode)

    pixels = readonly(np.asarray(oriented, dtype=np.uint8))
    return from_array(
        pixels,
        icc=icc,
        source_name=source_name,
        source_sha256=source_sha,
        source_format=fmt,
        source_mode=source_mode,
        exif_orientation=orientation,
        icc_dropped=icc_dropped,
        color_space=cs_override,
        color_origin=color_origin,
        focal_35mm=focal,
        camera=camera,
        source_bit_depth=int(img.info.get("bit_depth") or 8),
        extra_images_ignored=extra_images,
    )


def from_array(
    pixels: NDArray[np.uint8],
    *,
    icc: bytes | None = None,
    source_name: str = "array",
    source_sha256: str | None = None,
    source_format: str = "array",
    source_mode: str | None = None,
    exif_orientation: int = 1,
    icc_dropped: bool = False,
    color_space: ColorSpace | None = None,
    color_origin: str | None = None,
    focal_35mm: float | None = None,
    camera: str | None = None,
    source_bit_depth: int = 8,
    extra_images_ignored: bool = False,
) -> SourceImage:
    """Costruisce un SourceImage da un array uint8 (H, W, 3|4)."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise UnsupportedImageError(
            f"atteso array uint8 (H, W, 3|4), trovato {pixels.dtype} {pixels.shape}"
        )
    pixels = pixels if not pixels.flags.writeable else readonly(pixels)
    norm_sha = sha256_array(pixels)
    cs, desc = classify_icc(icc)
    if color_space is not None:
        cs = color_space
    channels = int(pixels.shape[2])
    meta = ImageMeta(
        source_name=source_name,
        source_sha256=source_sha256 or norm_sha,
        normalized_sha256=norm_sha,
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        channels=3 if channels == 3 else 4,
        bit_depth=8,
        source_bit_depth=source_bit_depth,
        source_format=source_format,
        source_mode=source_mode or ("RGB" if channels == 3 else "RGBA"),
        exif_orientation=exif_orientation,
        icc_sha256=sha256_bytes(icc) if icc else None,
        icc_description=desc,
        icc_dropped=icc_dropped,
        color_space=cs,
        color_origin=color_origin or ("icc" if icc else "none"),
        focal_35mm=focal_35mm,
        camera=camera,
        extra_images_ignored=extra_images_ignored,
    )
    return SourceImage(pixels=pixels, icc=icc, meta=meta)


def colorspace_warnings(meta: ImageMeta) -> list[str]:
    out: list[str] = []
    if meta.color_space is ColorSpace.ASSUMED_SRGB:
        out.append("nessun profilo ICC nella foto: assunto sRGB")
    if meta.color_space is ColorSpace.UNKNOWN:
        out.append(f"profilo ICC non riconosciuto ({meta.icc_description!r}): usata TRC sRGB")
    if meta.icc_dropped:
        out.append("profilo ICC in scala di grigi scartato: immagine convertita in RGB")
    if meta.color_origin == "nclx" and meta.icc_sha256:
        out.append(f"colore dichiarato via nclx ({meta.color_space}): ICC equivalente generato")
    if meta.source_bit_depth > 8:
        out.append(f"sorgente a {meta.source_bit_depth} bit convertita a 8 bit")
    if meta.extra_images_ignored:
        out.append("immagini secondarie (gain map HDR / profondità) ignorate: usata la base SDR")
    return out
