"""Orquestación de carga, embeddings, clasificación y evidencia."""

from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.almacen_modelos import AlmacenModelos
from src.clasificadores.fabrica import FabricaClasificadores
from src.config import DIMENSION_MAPAS, DIMENSIONES_RECONSTRUCCION, asegurar_directorios
from src.datos.division import DivisorEstratificado
from src.datos.lfw import CargadorLFW
from src.datos.olivetti import CargadorOlivetti
from src.embeddings.base import ExtractorEmbedding
from src.embeddings.fabrica import FabricaExtractores
from src.embeddings.fisherfaces import ExtractorFisherfaces
from src.embeddings.pca import ExtractorPCA
from src.evaluacion.metricas import Evaluador
from src.evaluacion.registro import RegistroExperimento
from src.evaluacion.visualizacion import Visualizador
from src.excepciones import DimensionInvalida
from src.modelos import ConjuntoRostros, Particion, ResultadoExperimento


class PipelineExperimentos:
    """Ejecuta el mismo protocolo por separado para cada conjunto."""

    def __init__(
        self,
        divisor: DivisorEstratificado | None = None,
        fabrica_extractores: FabricaExtractores | None = None,
        fabrica_clasificadores: FabricaClasificadores | None = None,
        evaluador: Evaluador | None = None,
        visualizador: Visualizador | None = None,
    ) -> None:
        self.divisor = divisor or DivisorEstratificado()
        self.fabrica_extractores = fabrica_extractores or FabricaExtractores()
        self.fabrica_clasificadores = fabrica_clasificadores or FabricaClasificadores()
        self.evaluador = evaluador or Evaluador()
        self.visualizador = visualizador or Visualizador()

    def ejecutar_olivetti(self) -> list[ResultadoExperimento]:
        print("== Experimento: Olivetti ==")
        return self._ejecutar(CargadorOlivetti().cargar())

    def ejecutar_lfw(self) -> list[ResultadoExperimento]:
        print("== Experimento: LFW ==")
        return self._ejecutar(CargadorLFW().cargar())

    def ejecutar_experimentos(self) -> None:
        self.ejecutar_olivetti()
        self.ejecutar_lfw()

    @threadpool_limits.wrap(limits=1)
    def _ejecutar(self, conjunto: ConjuntoRostros) -> list[ResultadoExperimento]:
        asegurar_directorios()
        particion = self.divisor.dividir(conjunto)
        almacen = AlmacenModelos(conjunto, particion)
        registro = RegistroExperimento(conjunto.nombre)
        print(
            f"N={conjunto.X.shape[0]} C={conjunto.n_clases} "
            f"H={conjunto.alto} W={conjunto.ancho} "
            f"TRAIN={particion.y_train.size} TEST={particion.y_test.size}"
        )
        resultados: list[ResultadoExperimento] = []
        predicciones: list[pd.DataFrame] = []
        omitidas: list[dict] = []
        mapas: dict[str, np.ndarray] = {}

        for extractor in self.fabrica_extractores.para_experimento(
            conjunto.n_clases, particion.X_train.shape
        ):
            try:
                Z_train, Z_test = self._proyectar(extractor, particion)
            except DimensionInvalida as error:
                omitidas.append({"embedding": extractor.nombre, "d": extractor.dimension, "causa": str(error)})
                print(f"  {extractor.nombre} d={extractor.dimension} omitido: {error}")
                continue
            if extractor.dimension == DIMENSION_MAPAS:
                mapas[extractor.nombre] = Z_train
            filas, pred = self._evaluar_clasificadores(
                conjunto, extractor, particion, Z_train, Z_test, almacen,
            )
            resultados.extend(filas)
            predicciones.extend(pred)

        registro.guardar_particion(
            conjunto, particion, self.divisor.random_state, self.divisor.test_size,
            almacen.huella_train, omitidas,
        )
        if not resultados:
            raise ValueError(f"No hay combinaciones evaluables para {conjunto.nombre}.")
        registro.escribir_tabla("predicciones", pd.concat(predicciones, ignore_index=True))
        tabla = registro.guardar_metricas(resultados)
        self.visualizador.comparar(tabla, conjunto.nombre)
        self._figuras_pca(conjunto, particion, registro)
        for embedding, Z_train in mapas.items():
            for metodo in ("tsne", "umap"):
                ruta = self.visualizador.mapa(
                    Z_train, particion.y_train, metodo, conjunto.nombre, embedding, DIMENSION_MAPAS,
                )
                print(f"  mapa {ruta.name}")
        print(f"  {len(resultados)} combinaciones guardadas en {registro.directorio}")
        return resultados

    def _proyectar(
        self, extractor: ExtractorEmbedding, particion: Particion,
    ) -> tuple[np.ndarray, np.ndarray]:
        extractor.fit(particion.X_train, particion.y_train)
        return extractor.transform(particion.X_train), extractor.transform(particion.X_test)

    def _evaluar_clasificadores(
        self,
        conjunto: ConjuntoRostros,
        extractor: ExtractorEmbedding,
        particion: Particion,
        Z_train: np.ndarray,
        Z_test: np.ndarray,
        almacen: AlmacenModelos,
    ) -> tuple[list[ResultadoExperimento], list[pd.DataFrame]]:
        filas, predicciones = [], []
        d = extractor.dimension
        d_pca = {"pca": d, "lda": None}.get(extractor.nombre)
        if isinstance(extractor, ExtractorFisherfaces):
            d_pca = extractor.d_pca
        for nombre in self.fabrica_clasificadores.disponibles():
            modelo = self.fabrica_clasificadores.crear(nombre)
            medicion = self.evaluador.evaluar(
                modelo, Z_train, particion.y_train, Z_test, particion.y_test,
            )
            fila = ResultadoExperimento(
                dataset=conjunto.nombre,
                embedding=extractor.nombre,
                d=d,
                modelo=nombre,
                parametros=self.fabrica_clasificadores.parametros(nombre),
                accuracy=medicion.accuracy,
                macro_f1=medicion.macro_f1,
                tiempo_s=medicion.tiempo_s,
                macro_precision=medicion.macro_precision,
                macro_recall=medicion.macro_recall,
                ajuste_s=medicion.ajuste_s,
                prediccion_s=medicion.prediccion_s,
                d_pca=d_pca,
            )
            filas.append(fila)
            predicciones.append(pd.DataFrame({
                "dataset": conjunto.nombre,
                "embedding": extractor.nombre,
                "d": d,
                "modelo": nombre,
                "fila": particion.indices_test,
                "id_real": particion.y_test,
                "id_predicho": medicion.predicciones,
            }))
            almacen.guardar(extractor, nombre, modelo)
            ruta = self.visualizador.confusion(medicion.matriz, conjunto.nombre, extractor.nombre, nombre, d)
            if d == DIMENSION_MAPAS:
                self.visualizador.ejemplos(conjunto, particion, medicion, extractor.nombre, nombre, d)
            print(
                f"  {extractor.nombre:12} d={d:<3} {nombre:6} "
                f"acc={fila.accuracy:.3f}  macro-F1={fila.macro_f1:.3f}  "
                f"{fila.tiempo_s:.4f}s  -> {ruta.name}"
            )
        return filas, predicciones

    def _figuras_pca(
        self, conjunto: ConjuntoRostros, particion: Particion, registro: RegistroExperimento,
    ) -> None:
        d_ref = min(particion.X_train.shape[0] - 1, particion.X_train.shape[1], 120)
        referencia = ExtractorPCA(n_components=d_ref).fit(particion.X_train)
        curva = referencia.varianza_acumulada()
        registro.escribir_tabla("varianza_pca", pd.DataFrame({
            "d": np.arange(1, d_ref + 1),
            "varianza_acumulada_train": curva,
        }))
        self.visualizador.curva_pca(curva, conjunto.nombre)
        self.visualizador.eigenfaces(
            referencia.rostro_medio(conjunto.alto, conjunto.ancho),
            referencia.componentes_imagen(conjunto.alto, conjunto.ancho),
            conjunto.nombre,
        )

        imagenes, errores, filas = {}, {}, []
        for k in DIMENSIONES_RECONSTRUCCION:
            if k > d_ref:
                continue
            reconstruccion = referencia.reconstruir(particion.X_test, k)
            ecm = np.mean((particion.X_test - reconstruccion) ** 2, axis=1)
            imagenes[k] = reconstruccion[0].reshape(conjunto.alto, conjunto.ancho)
            errores[k] = float(ecm[0])
            filas.append({
                "d": k,
                "fila_ejemplo": particion.indices_test[0],
                "varianza_acumulada_train": curva[k - 1],
                "ecm_ejemplo": ecm[0],
                "ecm_medio_test": ecm.mean(),
            })
        registro.escribir_tabla("reconstruccion_pca", pd.DataFrame(filas))
        self.visualizador.reconstruccion(particion.images_test[0], imagenes, errores, conjunto.nombre)
