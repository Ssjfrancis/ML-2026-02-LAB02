"""Figuras del experimento: curva PCA, eigenfaces, confusión y mapas 2D."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE

from src.config import (
    DIMENSIONES,
    DIR_FIGURAS,
    NOMBRES_CLASIFICADORES,
    NOMBRES_DATASETS,
    NOMBRES_EMBEDDINGS,
    RANDOM_STATE,
    TSNE_PERPLEXITY,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
)
from src.modelos import ConjuntoRostros, Particion, ResultadoClasificacion


class Visualizador:
    """Guarda la evidencia gráfica de TRAIN y de la evaluación en TEST."""

    def __init__(self, directorio: Path = DIR_FIGURAS) -> None:
        self.directorio = directorio
        self.directorio.mkdir(parents=True, exist_ok=True)

    def curva_pca(self, curva: np.ndarray, dataset: str) -> Path:
        """Grafica la varianza acumulada aprendida en TRAIN."""
        ejes = np.arange(1, curva.size + 1)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(ejes, curva, color="#0076BA", linewidth=1.8)
        for d in DIMENSIONES:
            if d <= curva.size:
                ax.scatter([d], [curva[d - 1]], color="#B51700", zorder=3)
                ax.annotate(
                    f"d={d}\n{curva[d - 1]:.0%}",
                    (d, curva[d - 1]),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=8,
                )
        ax.set_xlabel("Componentes conservadas (d)")
        ax.set_ylabel("Varianza acumulada")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1.02)
        ax.set_title(f"PCA en TRAIN · {NOMBRES_DATASETS[dataset]}")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._guardar(fig, f"curva_pca_{dataset}.png")

    def eigenfaces(
        self,
        media: np.ndarray,
        componentes: np.ndarray,
        dataset: str,
        cantidad: int = 11,
    ) -> Path:
        """Rostro medio y las primeras componentes reordenadas a H×W."""
        cantidad = min(cantidad, componentes.shape[0])
        fig, ejes = plt.subplots(3, 4, figsize=(7.6, 6), layout="constrained")
        celdas = list(ejes.ravel())
        self._pintar_rostro(celdas[0], media, "Media de TRAIN")
        limite = np.abs(componentes[:cantidad]).max()
        for i in range(cantidad):
            ax = celdas[i + 1]
            pesos = ax.imshow(componentes[i], cmap="RdBu_r", vmin=-limite, vmax=limite)
            ax.set_title(f"PC{i + 1}", fontsize=9)
            ax.axis("off")
        for ax in celdas[cantidad + 1 :]:
            ax.axis("off")
        fig.colorbar(pesos, ax=celdas, shrink=0.8, label="Peso del píxel")
        fig.suptitle(f"Eigenfaces · {NOMBRES_DATASETS[dataset]}", fontsize=12)
        return self._guardar(fig, f"eigenfaces_{dataset}.png")

    def confusion(
        self,
        matriz: np.ndarray,
        dataset: str,
        embedding: str,
        modelo: str,
        d: int,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        imagen = ax.imshow(matriz, cmap="Blues", interpolation="nearest")
        marcas = np.arange(0, matriz.shape[0], 5)
        ax.set(xticks=marcas, yticks=marcas, xlabel="ID predicho", ylabel="ID real")
        ax.set_title(
            f"Confusión en TEST · {NOMBRES_DATASETS[dataset]}\n"
            f"{NOMBRES_EMBEDDINGS[embedding]} (d={d}) + {NOMBRES_CLASIFICADORES[modelo]}"
        )
        fig.colorbar(imagen, ax=ax, fraction=0.046, label="Fotografías")
        fig.tight_layout()
        nombre = f"confusion_{dataset}_{embedding}_{modelo}_d{d}.png"
        return self._guardar(fig, nombre)

    def mapa(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        metodo: str,
        dataset: str,
        embedding: str,
        d: int,
    ) -> Path:
        """Mapa 2D exploratorio de Z_train. No se usa para calcular accuracy."""
        coordenadas, detalle = self._proyectar_2d(Z, metodo)
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        c = np.unique(y).size
        colores = ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0.03, 0.97, c)))
        norma = BoundaryNorm(np.arange(c + 1) - 0.5, c)
        dispersion = ax.scatter(
            coordenadas[:, 0],
            coordenadas[:, 1],
            c=y,
            cmap=colores,
            norm=norma,
            s=11,
            alpha=0.85,
        )
        ax.set_xlabel("Eje 1")
        ax.set_ylabel("Eje 2")
        nombre_metodo = "t-SNE" if metodo == "tsne" else "UMAP"
        ax.set_title(
            f"{nombre_metodo} en TRAIN · {NOMBRES_DATASETS[dataset]}\n"
            f"{NOMBRES_EMBEDDINGS[embedding]} (d={d}) · {detalle}",
            fontsize=11,
        )
        fig.colorbar(
            dispersion, ax=ax, fraction=0.046, label="Código de identidad",
            ticks=np.arange(0, c, 5),
        )
        fig.tight_layout()
        nombre = f"{metodo}_{dataset}_{embedding}_d{d}.png"
        return self._guardar(fig, nombre)

    def _proyectar_2d(self, Z: np.ndarray, metodo: str) -> tuple[np.ndarray, str]:
        n = Z.shape[0]
        if metodo == "tsne":
            perplexity = min(TSNE_PERPLEXITY, (n - 1) / 3)
            modelo = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                max_iter=1000,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            detalle = f"perplexity={perplexity:.0f}, random_state={RANDOM_STATE}"
            return modelo.fit_transform(Z), detalle
        if metodo == "umap":
            import umap

            n_neighbors = min(UMAP_N_NEIGHBORS, n - 1)
            modelo = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=UMAP_MIN_DIST,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            detalle = (
                f"n_neighbors={n_neighbors}, min_dist={UMAP_MIN_DIST}, "
                f"random_state={RANDOM_STATE}"
            )
            return modelo.fit_transform(Z), detalle
        raise ValueError(f"Método de mapa desconocido: {metodo}")

    def reconstruccion(
        self,
        original: np.ndarray,
        reconstrucciones: dict[int, np.ndarray],
        errores: dict[int, float],
        dataset: str,
    ) -> Path:
        fig, ejes = plt.subplots(
            1, len(reconstrucciones) + 1, figsize=(9.2, 2.7), layout="constrained",
        )
        self._pintar_rostro(ejes[0], original, "Original de TEST")
        for ax, (k, imagen) in zip(ejes[1:], reconstrucciones.items()):
            self._pintar_rostro(ax, imagen, f"{k} componentes\nECM {errores[k]:.5f}")
        fig.suptitle(f"Reconstrucción PCA · {NOMBRES_DATASETS[dataset]}")
        return self._guardar(fig, f"reconstruccion_pca_{dataset}.png")

    def ejemplos(
        self,
        conjunto: ConjuntoRostros,
        particion: Particion,
        medicion: ResultadoClasificacion,
        embedding: str,
        modelo: str,
        d: int,
    ) -> Path:
        y_real, y_pred = particion.y_test, medicion.predicciones
        aciertos = np.flatnonzero(y_real == y_pred)[:2]
        errores = np.flatnonzero(y_real != y_pred)
        if errores.size:
            fuera_diagonal = medicion.matriz.copy()
            np.fill_diagonal(fuera_diagonal, 0)
            real, pred = np.unravel_index(fuera_diagonal.argmax(), fuera_diagonal.shape)
            del_par = errores[(y_real[errores] == real) & (y_pred[errores] == pred)]
            errores = np.concatenate((del_par, errores[~np.isin(errores, del_par)]))[:2]
        fig, ejes = plt.subplots(2, 2, figsize=(6.7, 6.2), layout="constrained")
        fig.suptitle(
            f"{NOMBRES_DATASETS[conjunto.nombre]} · {NOMBRES_EMBEDDINGS[embedding]} "
            f"(d={d}) + {NOMBRES_CLASIFICADORES[modelo]} · TEST",
            fontsize=12,
        )
        for fila, indices, etiqueta in zip(ejes, (aciertos, errores), ("Acierto", "Error")):
            for ax in fila:
                ax.axis("off")
            for ax, i in zip(fila, indices):
                nombre_real = fill(conjunto.nombres_identidades[y_real[i]], 24)
                nombre_pred = fill(conjunto.nombres_identidades[y_pred[i]], 24)
                indice = particion.indices_test[i]
                self._pintar_rostro(
                    ax, particion.images_test[i],
                    f"{etiqueta} · fila {indice}\nReal: {nombre_real}\nPred.: {nombre_pred}",
                )
        return self._guardar(fig, f"ejemplos_{conjunto.nombre}_{embedding}_{modelo}_d{d}.png")

    def comparar(self, datos: pd.DataFrame, dataset: str) -> None:
        metricas = {
            "accuracy": "Accuracy en TEST",
            "macro_f1": "Macro-F1 en TEST",
            "tiempo_s": "Ajuste + predicción del clasificador (s)",
        }
        estilos = {"svm": ("#125683", "o"), "knn": ("#b75111", "s"), "arbol": ("#39754d", "^")}
        for columna, titulo in metricas.items():
            fig, ejes = plt.subplots(1, 3, figsize=(9.6, 3.5), sharey=True, layout="constrained")
            fig.suptitle(f"{titulo} · {NOMBRES_DATASETS[dataset]}", fontsize=12)
            for ax, (embedding, nombre) in zip(ejes, NOMBRES_EMBEDDINGS.items()):
                for modelo, (color, marcador) in estilos.items():
                    filas = datos[(datos.embedding == embedding) & (datos.modelo == modelo)]
                    filas = filas.sort_values("d")
                    ax.plot(
                        filas.d, filas[columna], marker=marcador, color=color,
                        label=NOMBRES_CLASIFICADORES[modelo],
                    )
                ax.set(title=nombre, xticks=DIMENSIONES, xlabel="Dimensión (d)")
                ax.grid(axis="y", alpha=0.25)
            if columna == "tiempo_s":
                ejes[0].set_yscale("log")
                ejes[0].set_ylabel("Segundos (escala log)")
            else:
                ejes[0].set(ylim=(0, 1.03), ylabel="Proporción")
                ejes[0].yaxis.set_major_formatter(PercentFormatter(1))
            ejes[0].legend(loc="best", fontsize=8)
            self._guardar(fig, f"comparacion_{columna}_{dataset}.png")

    @staticmethod
    def _pintar_rostro(ax: plt.Axes, imagen: np.ndarray, titulo: str) -> None:
        ax.imshow(imagen, cmap="gray", vmin=0, vmax=1)
        ax.set_title(titulo, fontsize=9)
        ax.axis("off")

    def _guardar(self, fig: plt.Figure, nombre: str) -> Path:
        ruta = self.directorio / nombre
        fig.savefig(ruta, dpi=145, facecolor="white")
        plt.close(fig)
        return ruta
