"""LDA: usa las etiquetas para separar identidades."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_X_y

from src.embeddings.base import ExtractorEmbedding
from src.excepciones import DimensionInvalida


class ExtractorLDA(ExtractorEmbedding):
    """Análisis discriminante lineal. d no puede superar C − 1."""

    nombre = "lda"

    def __init__(self, n_components: int) -> None:
        if n_components < 1:
            raise ValueError("LDA requiere al menos una componente.")
        self.n_components = int(n_components)
        self._modelo = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            solver="svd",
        )

    @property
    def dimension(self) -> int:
        return self.n_components

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ExtractorLDA:
        if y is None:
            raise ValueError("LDA necesita y_train: aprende con las identidades.")
        X, y = check_X_y(X, y, dtype=np.float64)
        n_clases = int(np.unique(y).size)
        maximo = min(n_clases - 1, X.shape[1], X.shape[0] - n_clases)
        if self.n_components > maximo:
            raise DimensionInvalida(
                f"LDA con d={self.n_components} supera min(C-1, p, N_train-C)={maximo}."
            )
        self._modelo.fit(X, y)
        rango = self._modelo.scalings_.shape[1]
        if rango < self.n_components:
            raise DimensionInvalida(
                f"LDA con d={self.n_components} supera el rango efectivo {rango}."
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.comprobar(self._modelo.transform(X), X.shape[0])
