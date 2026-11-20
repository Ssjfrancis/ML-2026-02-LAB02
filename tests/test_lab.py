"""Pruebas de contratos, ausencia de fuga y predicción de la interfaz."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

from src.almacen_modelos import AlmacenModelos
from src.clasificadores.fabrica import FabricaClasificadores
from src.config import ruta_modelo
from src.datos import CargadorLFW, DivisorEstratificado, validar_conjunto
from src.embeddings import ExtractorFisherfaces, ExtractorLDA, ExtractorPCA, FabricaExtractores
from src.evaluacion.registro import RegistroExperimento
from src.interfaz.servicio import ServicioReconocimiento
from src.modelos import ConjuntoRostros, ResultadoExperimento
from src.pipeline import PipelineExperimentos


def conjunto_prueba():
    y = np.repeat(np.arange(4), 12)
    X = np.random.default_rng(42).normal(size=(y.size, 8))
    X[:, 0] += y * 2
    return ConjuntoRostros("olivetti", X.reshape(-1, 2, 4), X, y, list("ABCD"), 2, 4)


class DatosTest(TestCase):
    def test_lfw_frecuencia_desempate_y_reindexacion(self):
        y = np.repeat(np.arange(53), [19, 30] + [20] * 51)
        X = np.arange(y.size * 6).reshape(-1, 6)
        orden = np.random.default_rng(5).permutation(y.size)
        datos = SimpleNamespace(
            target=y[orden], data=X[orden], images=X[orden].reshape(-1, 2, 3),
            target_names=np.array([f"Persona {i}" for i in range(53)]),
        )
        with patch("src.datos.lfw.fetch_lfw_people", return_value=datos) as fetch:
            conjunto = CargadorLFW().cargar()
        self.assertEqual(fetch.call_args.kwargs["min_faces_per_person"], 20)
        self.assertEqual(fetch.call_args.kwargs["resize"], 0.5)
        self.assertEqual(conjunto.ids_originales, list(range(1, 51)))
        self.assertEqual(conjunto.nombres_identidades, [f"Persona {i}" for i in range(1, 51)])
        np.testing.assert_array_equal(np.bincount(conjunto.y), [30] + [20] * 49)
        np.testing.assert_array_equal(conjunto.X, datos.data[np.isin(datos.target, range(1, 51))])
        self.assertEqual(conjunto.images.shape, (1010, 2, 3))

    def test_validador_rechaza_imagen_desalineada(self):
        conjunto = conjunto_prueba()
        conjunto.images = conjunto.images[::-1]
        with self.assertRaisesRegex(ValueError, "aplanada"):
            validar_conjunto(conjunto)

    def test_split_disjunto_repetible_y_estratificado(self):
        conjunto = conjunto_prueba()
        divisor = DivisorEstratificado(test_size=0.25, random_state=42)
        p = divisor.dividir(conjunto)
        q = divisor.dividir(conjunto)
        self.assertEqual(np.intersect1d(p.indices_train, p.indices_test).size, 0)
        np.testing.assert_array_equal(p.indices_test, q.indices_test)
        np.testing.assert_array_equal(np.bincount(p.y_test), [3] * 4)
        np.testing.assert_array_equal(np.bincount(p.y_train), [9] * 4)


class EmbeddingsTest(TestCase):
    def test_fisherfaces_aprende_solo_train_y_transform_no_reajusta(self):
        particion = DivisorEstratificado().dividir(conjunto_prueba())
        particion.X_test += 10_000
        extractor = ExtractorFisherfaces(5, 2)
        with (
            threadpool_limits(limits=1),
            patch.object(ExtractorPCA, "fit", autospec=True, side_effect=ExtractorPCA.fit) as fit_pca,
            patch.object(ExtractorLDA, "fit", autospec=True, side_effect=ExtractorLDA.fit) as fit_lda,
        ):
            Z_train, Z_test = PipelineExperimentos()._proyectar(extractor, particion)
            np.testing.assert_allclose(extractor.transform(particion.X_test), Z_test)
            fit_pca.assert_called_once()
            fit_lda.assert_called_once()
            np.testing.assert_array_equal(fit_pca.call_args.args[1], particion.X_train)
            np.testing.assert_allclose(fit_lda.call_args.args[1], extractor.pca_.transform(particion.X_train))
            np.testing.assert_array_equal(fit_lda.call_args.args[2], particion.y_train)
        np.testing.assert_allclose(extractor.pca_._modelo.mean_, particion.X_train.mean(axis=0))
        self.assertEqual(Z_train.shape, (36, 2))
        self.assertEqual(Z_test.shape, (12, 2))
        self.assertTrue(np.isfinite(Z_test).all())

    def test_fisherfaces_valida_dimensiones_y_valores(self):
        conjunto = conjunto_prueba()
        with self.assertRaises(ValueError):
            ExtractorFisherfaces(2, 3)
        with self.assertRaises(ValueError):
            ExtractorFisherfaces(4.5, 2)
        with self.assertRaises(NotFittedError):
            ExtractorFisherfaces(5, 2).transform(conjunto.X)
        with self.assertRaisesRegex(ValueError, "máximo admisible"):
            ExtractorFisherfaces(5, 4).fit(conjunto.X, conjunto.y)
        with self.assertRaisesRegex(ValueError, "rango numérico"):
            X_rango_uno = np.tile(np.arange(conjunto.y.size)[:, None], (1, 8))
            ExtractorFisherfaces(5, 2).fit(X_rango_uno, conjunto.y)
        with self.assertRaises(ValueError):
            ExtractorFisherfaces(5, 2).fit(conjunto.X * np.nan, conjunto.y)
        extractor = ExtractorFisherfaces(5, 2).fit(conjunto.X, conjunto.y)
        with self.assertRaises(ValueError):
            extractor.transform(conjunto.X[:, :2])

    def test_opciones_acotadas_a_train(self):
        fabrica = FabricaExtractores()
        self.assertEqual(fabrica.dimensiones_validas("pca", 15, (30, 12)), (10,))
        self.assertEqual(fabrica.dimensiones_validas("lda", 9, (40, 80)), ())
        self.assertEqual(fabrica.dimensiones_validas("fisherfaces", 35, (54, 80)), (10,))


class InterfazTest(TestCase):
    def test_prediccion_svm_y_su_probabilidad_pueden_no_ser_el_maximo(self):
        servicio = ServicioReconocimiento.__new__(ServicioReconocimiento)
        servicio.conjunto = conjunto_prueba()
        servicio.particion = DivisorEstratificado().dividir(servicio.conjunto)
        servicio.fabrica_clasificadores = FabricaClasificadores()
        modelo = SimpleNamespace(
            predict=lambda X: np.array([2]),
            predict_proba=lambda X: np.array([[0.7, 0.05, 0.2, 0.05]]),
            classes_=np.arange(4),
        )
        extractor = SimpleNamespace(transform=lambda X: X)
        with patch.object(servicio, "_obtener_bundle", return_value={
            "extractor": extractor, "clasificador": modelo,
        }):
            salida = servicio.predecir(0, "pca", "svm", 10)
        self.assertEqual(salida.identidad_predicha, 2)
        self.assertAlmostEqual(salida.probabilidad, 0.2)
        self.assertEqual(salida.top[0][0], "A")

    def test_cache_invalida_se_regenera_y_luego_se_reutiliza(self):
        conjunto = conjunto_prueba()
        with TemporaryDirectory() as tmp:
            directorio = Path(tmp)
            ruta = ruta_modelo("olivetti", "fisherfaces", "knn", 2, directorio)
            ruta.write_bytes(b"cache corrupta")
            with (
                patch("src.interfaz.servicio.CargadorOlivetti") as cargador,
                patch("src.interfaz.servicio.AlmacenModelos", side_effect=lambda c, p: AlmacenModelos(c, p, directorio)),
            ):
                cargador.return_value.cargar.return_value = conjunto
                primero = ServicioReconocimiento().predecir(0, "fisherfaces", "knn", 2)
                self.assertGreater(ruta.stat().st_size, 15)
                segundo_servicio = ServicioReconocimiento()
                with patch.object(segundo_servicio, "_entrenar") as entrenar:
                    segundo = segundo_servicio.predecir(0, "fisherfaces", "knn", 2)
                    entrenar.assert_not_called()
                self.assertEqual(primero, segundo)

            nueva_particion = DivisorEstratificado(random_state=17).dividir(conjunto)
            self.assertIsNone(AlmacenModelos(conjunto, nueva_particion, directorio).cargar("fisherfaces", "knn", 2))


class RegistroTest(TestCase):
    def test_actualizar_dataset_conserva_filas_del_otro(self):
        with TemporaryDirectory() as tmp:
            directorio = Path(tmp)
            for dataset, accuracy in (("lfw", 0.4), ("olivetti", 0.9), ("lfw", 0.5)):
                fila = ResultadoExperimento(
                    dataset=dataset, embedding="pca", d=10, modelo="svm", parametros="C=1",
                    accuracy=accuracy, macro_f1=0.5, tiempo_s=0.3, macro_precision=0.5,
                    macro_recall=0.5, ajuste_s=0.2, prediccion_s=0.1, d_pca=10,
                )
                RegistroExperimento(dataset, directorio).guardar_metricas([fila])
            actual = pd.read_csv(directorio / "resultados_modelos.csv").set_index("dataset")
            self.assertEqual(actual.accuracy.to_dict(), {"lfw": 0.5, "olivetti": 0.9})
