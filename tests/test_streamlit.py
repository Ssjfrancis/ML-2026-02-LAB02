"""Interacción con Streamlit sobre las bases y el split del laboratorio."""

from unittest import TestCase, skipUnless

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import DIR_CACHE, DIR_RESULTADOS, RAIZ

PREDICCIONES = DIR_RESULTADOS / "predicciones_lfw.csv"
CACHE_LISTA = (
    (DIR_CACHE / "olivetti_py3.pkz").is_file()
    and (DIR_CACHE / "lfw_home" / "lfw_funneled").is_dir()
)


@skipUnless(PREDICCIONES.exists() and CACHE_LISTA, "Ejecuta primero ambos experimentos.")
class StreamlitTest(TestCase):
    def test_foto_acierto_y_error_con_fisherfaces_en_lfw(self):
        app = AppTest.from_file(str(RAIZ / "app.py")).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("100 fotografías", app.caption[0].value)
        self.assertEqual(len(app.selectbox[1].options), 100)
        self.assertTrue(app.get("imgs"))

        app.radio[0].set_value("fisherfaces")
        app.radio[1].set_value("knn")
        app.select_slider[0].set_value(20)
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("k-NN con k=3" in elemento.value for elemento in app.caption))

        app.selectbox[0].set_value("lfw").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("694 fotografías", app.caption[0].value)
        self.assertEqual(len(app.selectbox[1].options), 694)
        self.assertEqual(app.select_slider[0].value, 20)
        csv = pd.read_csv(PREDICCIONES)
        filas = csv.query("embedding=='fisherfaces' and d==20 and modelo=='knn'").reset_index(drop=True)
        for acierto in (True, False):
            with self.subTest(acierto=acierto):
                indice = int(filas[filas.id_real.eq(filas.id_predicho) == acierto].index[0])
                app.selectbox[1].set_value(indice).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertEqual(bool(app.success), acierto)
                self.assertEqual(bool(app.error), not acierto)
                self.assertTrue(any("Identidad real" in e.value for e in app.markdown))
                self.assertTrue(any("Identidad predicha" in e.value for e in app.markdown))
                self.assertTrue(any("Probabilidad estimada" in e.value for e in app.markdown))
