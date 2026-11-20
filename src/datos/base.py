"""Contrato común de los cargadores de rostros."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.modelos import ConjuntoRostros


class CargadorRostros(ABC):
    """Carga un conjunto público y lo deja en el contrato images / X / y."""

    @abstractmethod
    def cargar(self) -> ConjuntoRostros:
        """Devuelve fotografías, píxeles aplanados y la identidad de cada fila."""


def validar_conjunto(conjunto: ConjuntoRostros) -> None:
    """Comprueba formas alineadas y valores finitos antes de separar."""
    if conjunto.images.ndim != 3:
        raise ValueError("images debe tener forma (N, H, W).")
    n, alto, ancho = conjunto.images.shape
    if min(n, alto, ancho) < 1:
        raise ValueError("El conjunto debe contener imágenes no vacías.")
    if (alto, ancho) != (conjunto.alto, conjunto.ancho):
        raise ValueError("alto y ancho no coinciden con images.shape.")
    if conjunto.X.shape != (n, alto * ancho):
        raise ValueError(
            f"X debe tener forma {(n, alto * ancho)}; se recibió {conjunto.X.shape}."
        )
    if conjunto.y.shape != (n,):
        raise ValueError(f"y debe tener forma {(n,)}; se recibió {conjunto.y.shape}.")
    if not np.isfinite(conjunto.X).all():
        raise ValueError("X contiene valores no finitos.")
    if not np.array_equal(conjunto.images.reshape(n, -1), conjunto.X):
        raise ValueError("Cada fila de X debe coincidir con su imagen aplanada.")
    clases = np.unique(conjunto.y)
    if conjunto.y.dtype.kind not in "iu" or not np.array_equal(clases, np.arange(clases.size)):
        raise ValueError("y debe contener etiquetas enteras consecutivas desde cero.")
    if len(conjunto.nombres_identidades) != clases.size:
        raise ValueError("Debe haber un nombre por identidad.")
    ids = conjunto.ids_originales
    if ids and (len(ids) != clases.size or len(set(ids)) != clases.size):
        raise ValueError("Debe haber un identificador original distinto por identidad.")
