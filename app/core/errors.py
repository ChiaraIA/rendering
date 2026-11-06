"""Eccezioni tipizzate. Ogni classe porta l'exit code usato dalla CLI."""

from __future__ import annotations


class PosaError(Exception):
    exit_code = 1


class InputError(PosaError):
    """Input non valido (foto, maschera, parametri)."""

    exit_code = 2


class UnsupportedImageError(InputError):
    pass


class MaskInvalidError(InputError):
    pass


class CatalogError(InputError):
    pass


class MaskNotConfirmedError(PosaError):
    """Maschera assente, non confermata o cambiata dopo la conferma (regola 7)."""

    exit_code = 3


class GeometryError(PosaError):
    exit_code = 4


class IntegrityError(PosaError):
    """Un controllo di integrità è fallito: il risultato non è valido (regola 9)."""

    exit_code = 5


class JobStateError(PosaError):
    exit_code = 2
