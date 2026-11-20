"""Caché local compartida por experimentos y Streamlit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from pickle import UnpicklingError
from tempfile import NamedTemporaryFile

import joblib
import sklearn

from src.config import DIR_MODELOS, ruta_modelo
from src.modelos import ConjuntoRostros, Particion


class AlmacenModelos:
    """Lee solo caché local generada por el proyecto y valida su procedencia."""

    def __init__(
        self,
        conjunto: ConjuntoRostros,
        particion: Particion,
        directorio: Path = DIR_MODELOS,
    ) -> None:
        self.directorio = directorio
        self.dataset = conjunto.nombre
        huella = sha256()
        for valores in (particion.indices_train, particion.y_train, particion.X_train):
            huella.update(str((valores.shape, valores.dtype.str)).encode())
            huella.update(valores.tobytes())
        self.huella_train = huella.hexdigest()
        self._base = {
            "formato": 1,
            "dataset": self.dataset,
            "sklearn": sklearn.__version__,
            "huella_train": self.huella_train,
            "nombres": conjunto.nombres_identidades,
            "alto": conjunto.alto,
            "ancho": conjunto.ancho,
        }

    def _metadatos(self, embedding: str, clasificador: str, d: int) -> dict:
        return dict(self._base, embedding=embedding, modelo=clasificador, d=d)

    def cargar(self, embedding: str, clasificador: str, d: int) -> dict | None:
        ruta = ruta_modelo(self.dataset, embedding, clasificador, d, self.directorio)
        try:
            bundle = joblib.load(ruta)
        except (OSError, EOFError, UnpicklingError, AttributeError, ImportError, ValueError, IndexError):
            return None
        esperado = self._metadatos(embedding, clasificador, d)
        if isinstance(bundle, dict) and all(bundle.get(k) == v for k, v in esperado.items()):
            if "extractor" in bundle and "clasificador" in bundle:
                return bundle
        return None

    def guardar(self, extractor, nombre_clasificador: str, clasificador) -> dict:
        metadatos = self._metadatos(extractor.nombre, nombre_clasificador, extractor.dimension)
        bundle = dict(metadatos, extractor=extractor, clasificador=clasificador)
        ruta = ruta_modelo(
            self.dataset, extractor.nombre, nombre_clasificador, extractor.dimension, self.directorio,
        )
        self.directorio.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=self.directorio, suffix=".tmp", delete=False) as archivo:
            temporal = Path(archivo.name)
        try:
            joblib.dump(bundle, temporal, compress=1)
            temporal.replace(ruta)
        finally:
            temporal.unlink(missing_ok=True)
        return bundle
