"""Auditoría independiente del CSV, las predicciones y la evidencia gráfica."""

import json
from itertools import product
from unittest import TestCase, skipUnless

import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASIFICADORES, DIMENSIONES, DIR_FIGURAS, DIR_RESULTADOS, EMBEDDINGS

ARCHIVO = DIR_RESULTADOS / "resultados_modelos.csv"
BASES = {"olivetti": (400, 40, 64, 64, 300, 100), "lfw": (2773, 50, 62, 47, 2079, 694)}


@skipUnless(ARCHIVO.exists(), "Ejecuta python main.py experimentos para auditar las salidas.")
class EvidenciaTest(TestCase):
    def test_metricas_recalculadas_desde_todas_las_predicciones(self):
        resultados = pd.read_csv(ARCHIVO)
        self.assertEqual(len(resultados), 54)
        for dataset, (n, c, alto, ancho, train, test) in BASES.items():
            with self.subTest(dataset=dataset):
                protocolo = json.loads((DIR_RESULTADOS / f"protocolo_{dataset}.json").read_text())
                formas = tuple(protocolo[k] for k in ("N", "C", "H", "W", "n_train", "n_test"))
                self.assertEqual(formas, (n, c, alto, ancho, train, test))
                self.assertEqual(protocolo["semilla_split"], 42)
                self.assertEqual(protocolo["test_size"], 0.25)
                self.assertEqual(protocolo["omitidas"], [])
                split = pd.read_csv(DIR_RESULTADOS / f"particion_{dataset}.csv").set_index("fila")
                self.assertEqual(split.index.nunique(), n)
                self.assertEqual(split.particion.value_counts().to_dict(), {"TRAIN": train, "TEST": test})
                identidades = pd.read_csv(DIR_RESULTADOS / f"identidades_{dataset}.csv")
                self.assertTrue((identidades.total == identidades.train + identidades.test).all())
                self.assertTrue((identidades.total >= (10 if dataset == "olivetti" else 20)).all())
                ids_test = split[split.particion == "TEST"].id_real.sort_index()
                pred = pd.read_csv(DIR_RESULTADOS / f"predicciones_{dataset}.csv")
                self.assertEqual(len(pred), test * 27)
                grupo = pred.groupby(["embedding", "d", "modelo"])
                esperado = set(product(EMBEDDINGS, DIMENSIONES, CLASIFICADORES))
                self.assertEqual(set(grupo.groups), esperado)
                tablas = resultados[resultados.dataset == dataset].set_index(["embedding", "d", "modelo"])
                individual = pd.read_csv(DIR_RESULTADOS / f"resultados_{dataset}.csv")
                individual = individual.set_index(["embedding", "d", "modelo"])
                pd.testing.assert_frame_equal(individual.sort_index(), tablas.sort_index())
                for clave, muestra in grupo:
                    fila = tablas.loc[clave]
                    serie = muestra.set_index("fila").id_real.sort_index()
                    pd.testing.assert_series_equal(serie, ids_test)
                    matriz = np.zeros((c, c), dtype=int)
                    np.add.at(matriz, (muestra.id_real, muestra.id_predicho), 1)
                    tp = np.diag(matriz)
                    real, predicho = matriz.sum(axis=1), matriz.sum(axis=0)
                    np.testing.assert_allclose(
                        [fila.accuracy, fila.macro_precision, fila.macro_recall, fila.macro_f1],
                        [
                            tp.sum() / test,
                            np.mean(tp / np.maximum(predicho, 1)),
                            np.mean(tp / real),
                            np.mean(2 * tp / (real + predicho)),
                        ],
                        atol=1e-12,
                    )
                    self.assertGreater(fila.ajuste_s, 0)
                    self.assertGreater(fila.prediccion_s, 0)
                    self.assertAlmostEqual(fila.tiempo_s, fila.ajuste_s + fila.prediccion_s)
                    self.assertTrue(np.isnan(fila.d_pca) if clave[0] == "lda" else fila.d_pca == clave[1])

    def test_figuras_legibles_y_reconstrucciones_consistentes(self):
        for dataset in BASES:
            with self.subTest(dataset=dataset):
                rutas = [
                    DIR_FIGURAS / f"confusion_{dataset}_{e}_{m}_d{d}.png"
                    for e, d, m in product(EMBEDDINGS, DIMENSIONES, CLASIFICADORES)
                ]
                rutas += [
                    DIR_FIGURAS / f"{metodo}_{dataset}_{e}_d20.png"
                    for metodo, e in product(("tsne", "umap"), EMBEDDINGS)
                ]
                rutas += [
                    DIR_FIGURAS / f"ejemplos_{dataset}_{e}_{m}_d20.png"
                    for e, m in product(EMBEDDINGS, CLASIFICADORES)
                ]
                rutas += [
                    DIR_FIGURAS / f"{nombre}_{dataset}.png" for nombre in (
                        "curva_pca", "eigenfaces", "reconstruccion_pca", "comparacion_accuracy",
                        "comparacion_macro_f1", "comparacion_tiempo_s",
                    )
                ]
                for ruta in rutas:
                    with Image.open(ruta) as imagen:
                        self.assertGreater(imagen.width, 700)
                        imagen.verify()
                pca = pd.read_csv(DIR_RESULTADOS / f"varianza_pca_{dataset}.csv")
                rec = pd.read_csv(DIR_RESULTADOS / f"reconstruccion_pca_{dataset}.csv")
                self.assertTrue(pca.varianza_acumulada_train.is_monotonic_increasing)
                self.assertTrue(rec.ecm_ejemplo.is_monotonic_decreasing)
                self.assertTrue(rec.ecm_medio_test.is_monotonic_decreasing)
