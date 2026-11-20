"""Carga y selección reproducible de identidades de LFW."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people

from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Selecciona las 50 identidades más frecuentes con al menos 20 fotos."""

    nombre = "lfw"

    def cargar(self) -> ConjuntoRostros:
        rostros = fetch_lfw_people(
            min_faces_per_person=20,
            resize=0.5,
            data_home=str(DIR_CACHE),
        )
        ids, frecuencias = np.unique(rostros.target, return_counts=True)
        elegibles = frecuencias >= 20
        ids, frecuencias = ids[elegibles], frecuencias[elegibles]
        if ids.size < 2:
            raise ValueError("LFW requiere al menos dos identidades con 20 fotos.")

        # Desempate por ID original del cargador. La máscara mantiene el orden de fotos.
        orden = np.lexsort((ids, -frecuencias))
        seleccion = np.isin(rostros.target, ids[orden[:50]])
        ids_originales, y = np.unique(rostros.target[seleccion], return_inverse=True)
        images = rostros.images[seleccion]
        _, alto, ancho = images.shape
        conjunto = ConjuntoRostros(
            nombre=self.nombre,
            images=images,
            X=np.asarray(rostros.data[seleccion], dtype=float),
            y=y,
            nombres_identidades=rostros.target_names[ids_originales].tolist(),
            alto=alto,
            ancho=ancho,
            ids_originales=ids_originales.tolist(),
        )
        validar_conjunto(conjunto)
        return conjunto
