"""Construcción de PCA, LDA y Fisherfaces para el mismo protocolo."""

from __future__ import annotations

from src.config import DIMENSIONES, EMBEDDINGS
from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fisherfaces import ExtractorFisherfaces
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA


class FabricaExtractores:
    """Crea un extractor por nombre o la lista completa del experimento."""

    def crear(self, nombre: str, d: int, n_clases: int) -> ExtractorEmbedding:
        if nombre == "pca":
            return ExtractorPCA(n_components=d)
        if nombre == "lda":
            return ExtractorLDA(n_components=self.dimension_lda(d, n_clases))
        if nombre == "fisherfaces":
            return ExtractorFisherfaces(
                d_pca=d,
                d_lda=self.dimension_lda(d, n_clases),
            )
        raise ValueError(f"Embedding desconocido: {nombre}")

    def para_experimento(
        self, n_clases: int, forma_train: tuple[int, int] | None = None
    ) -> list[ExtractorEmbedding]:
        """Construye las combinaciones de la guía que cumplen los límites de TRAIN."""
        if forma_train is None:
            d_max = max(DIMENSIONES)
            forma_train = (n_clases + d_max, d_max)
        return [
            self.crear(nombre, d, n_clases)
            for nombre in EMBEDDINGS
            for d in self.dimensiones_validas(nombre, n_clases, forma_train)
        ]

    @staticmethod
    def dimensiones_validas(
        nombre: str, n_clases: int, forma_train: tuple[int, int]
    ) -> tuple[int, ...]:
        n, p = forma_train
        if nombre == "pca":
            limite = min(n - 1, p)
        elif nombre in {"lda", "fisherfaces"}:
            limite = min(n_clases - 1, p, n - n_clases)
        else:
            raise ValueError(f"Embedding desconocido: {nombre}")
        return tuple(d for d in DIMENSIONES if d <= limite)

    @staticmethod
    def dimension_lda(d: int, n_clases: int) -> int:
        """LDA no puede superar C − 1 ni pedir más coordenadas que las de entrada."""
        return max(1, min(d, n_clases - 1))
