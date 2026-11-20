"""Evidencia tabular del experimento y consolidación de las dos bases."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from platform import machine, python_version, system
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from filelock import FileLock

from src.config import DIMENSION_MAPAS, DIMENSIONES, DIR_RESULTADOS
from src.modelos import ConjuntoRostros, Particion, ResultadoExperimento


class RegistroExperimento:
    """Escribe las métricas, el split y los datos necesarios para auditarlos."""

    def __init__(self, dataset: str, directorio: Path = DIR_RESULTADOS) -> None:
        self.dataset = dataset
        self.directorio = directorio

    def escribir_tabla(self, nombre: str, tabla: pd.DataFrame) -> Path:
        ruta = self.directorio / f"{nombre}_{self.dataset}.csv"
        self._csv_atomico(tabla, ruta)
        return ruta

    def guardar_metricas(self, filas: list[ResultadoExperimento]) -> pd.DataFrame:
        nuevo = pd.DataFrame([asdict(fila) for fila in filas]).astype({"d_pca": "Int64"})
        self.escribir_tabla("resultados", nuevo)
        consolidado = self.directorio / "resultados_modelos.csv"
        with FileLock(str(consolidado) + ".lock"):
            if consolidado.exists():
                previo = pd.read_csv(consolidado, dtype={"d_pca": "Int64"})
                nuevo_total = pd.concat([previo[previo.dataset != self.dataset], nuevo], ignore_index=True)
            else:
                nuevo_total = nuevo
            nuevo_total = nuevo_total.sort_values(["dataset", "embedding", "d", "modelo"])
            self._csv_atomico(nuevo_total, consolidado)
        return nuevo

    def guardar_particion(
        self,
        conjunto: ConjuntoRostros,
        particion: Particion,
        semilla: int,
        test_size: float,
        huella_train: str,
        omitidas: list[dict],
    ) -> None:
        n = conjunto.y.size
        c = conjunto.n_clases
        etiquetas = np.full(n, "TRAIN", dtype="<U5")
        etiquetas[particion.indices_test] = "TEST"
        self.escribir_tabla(
            "particion",
            pd.DataFrame({"fila": np.arange(n), "id_real": conjunto.y, "particion": etiquetas}),
        )
        self.escribir_tabla(
            "identidades",
            pd.DataFrame({
                "id": np.arange(c),
                "id_origen": conjunto.ids_originales or list(range(c)),
                "nombre": conjunto.nombres_identidades,
                "total": np.bincount(conjunto.y, minlength=c),
                "train": np.bincount(particion.y_train, minlength=c),
                "test": np.bincount(particion.y_test, minlength=c),
            }),
        )
        metadata = {
            "dataset": self.dataset,
            "N": n,
            "C": c,
            "H": conjunto.alto,
            "W": conjunto.ancho,
            "n_train": particion.y_train.size,
            "n_test": particion.y_test.size,
            "intensidad": [float(conjunto.X.min()), float(conjunto.X.max())],
            "semilla_split": semilla,
            "test_size": test_size,
            "sha256_train": huella_train,
            "dimensiones": DIMENSIONES,
            "dimension_mapas": DIMENSION_MAPAS,
            "omitidas": omitidas,
            "sistema": f"{system()} {machine()}",
            "python": python_version(),
            "versiones": {
                paquete: version(paquete)
                for paquete in ("numpy", "scipy", "pandas", "scikit-learn", "umap-learn", "streamlit")
            },
        }
        ruta = self.directorio / f"protocolo_{self.dataset}.json"
        ruta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def _csv_atomico(tabla: pd.DataFrame, ruta: Path) -> None:
        with NamedTemporaryFile(dir=ruta.parent, suffix=".tmp", delete=False) as archivo:
            temporal = Path(archivo.name)
        try:
            tabla.to_csv(temporal, index=False)
            temporal.replace(ruta)
        finally:
            temporal.unlink(missing_ok=True)
