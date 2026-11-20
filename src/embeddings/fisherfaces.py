"""Fisherfaces ajusta PCA y, sobre esas coordenadas, LDA."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y

from src.embeddings.base import ExtractorEmbedding
from src.embeddings.lda import ExtractorLDA
from src.embeddings.pca import ExtractorPCA
from src.excepciones import DimensionInvalida


class ExtractorFisherfaces(ExtractorEmbedding):
    """Dos proyecciones aprendidas en TRAIN con d_lda <= min(C-1, d_pca)."""

    nombre = "fisherfaces"

    def __init__(self, d_pca: int, d_lda: int) -> None:
        if any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in (d_pca, d_lda)):
            raise ValueError("Fisherfaces requiere d_pca y d_lda enteros positivos.")
        if d_lda > d_pca:
            raise ValueError(f"d_lda={d_lda} supera d_pca={d_pca}.")
        self.d_pca = int(d_pca)
        self.d_lda = int(d_lda)

    @property
    def dimension(self) -> int:
        return self.d_lda

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorFisherfaces:
        if y is None:
            raise ValueError("Fisherfaces necesita las etiquetas de TRAIN.")
        X, y = check_X_y(X, y, dtype=np.float64, ensure_min_samples=2)
        check_classification_targets(y)
        n_clases = np.unique(y).size
        maximo = min(n_clases - 1, self.d_pca, X.shape[0] - n_clases)
        if self.d_lda > maximo:
            raise DimensionInvalida(f"d_lda={self.d_lda} supera el máximo admisible {maximo}.")

        pca = ExtractorPCA(self.d_pca)
        Z_train = pca.fit_transform(X)
        lda = ExtractorLDA(self.d_lda).fit(Z_train, y)
        self.pca_, self.lda_ = pca, lda
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pca_"):
            raise NotFittedError("Ajuste Fisherfaces en TRAIN antes de transformar.")
        Z_pca = self.pca_.transform(X)
        return self.comprobar(self.lda_.transform(Z_pca), Z_pca.shape[0])
