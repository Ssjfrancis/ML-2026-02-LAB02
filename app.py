#!/usr/bin/env python3
"""Interfaz web para revisar rostros de prueba con un modelo ya ajustado.

Uso (con el entorno conda activado):

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.config import NOMBRES_CLASIFICADORES, NOMBRES_DATASETS, NOMBRES_EMBEDDINGS
from src.interfaz.servicio import ServicioReconocimiento


st.set_page_config(page_title="Embeddings faciales", layout="wide")


@st.cache_resource
def obtener_servicio(dataset: str) -> ServicioReconocimiento:
    return ServicioReconocimiento(dataset)


def main() -> None:
    st.title("Reconocimiento de rostros con embeddings lineales")

    with st.sidebar:
        st.header("Modelo")
        dataset = st.selectbox(
            "Conjunto de datos", options=list(NOMBRES_DATASETS), format_func=NOMBRES_DATASETS.__getitem__,
        )
        try:
            servicio = obtener_servicio(dataset)
        except OSError as error:
            st.error(f"No se pudo cargar {NOMBRES_DATASETS[dataset]}. Revisa la descarga. {error}")
            st.stop()
        embedding = st.radio(
            "Representación", options=list(NOMBRES_EMBEDDINGS), format_func=NOMBRES_EMBEDDINGS.__getitem__,
        )
        clasificador = st.radio(
            "Clasificador",
            options=list(NOMBRES_CLASIFICADORES),
            format_func=NOMBRES_CLASIFICADORES.__getitem__,
        )
        d = st.select_slider("Componentes (d)", options=servicio.dimensiones_para(embedding))
        etiquetas = servicio.etiquetas_prueba()
        indice = st.selectbox(
            "Fotografía de prueba",
            options=range(len(etiquetas)),
            format_func=etiquetas.__getitem__,
            key=f"foto_{dataset}",
        )

    st.caption(
        f"{NOMBRES_DATASETS[dataset]} · {len(etiquetas)} fotografías reservadas para TEST. "
        "El modelo se ajusta con las fotografías de TRAIN."
    )
    with st.spinner("Cargando el modelo ajustado en TRAIN..."):
        prediccion = servicio.predecir(indice, embedding, clasificador, int(d))

    columna_foto, columna_resultado = st.columns([1, 1.4], gap="large")
    with columna_foto:
        st.image(
            servicio.imagen_prueba(indice),
            caption=prediccion.nombre_real,
            clamp=True,
            width=280,
        )
    with columna_resultado:
        if prediccion.acierto:
            st.success("El modelo identificó la cara.")
        else:
            st.error("El modelo no identificó la cara.")
        real, predicha, probabilidad = st.columns(3)
        real.markdown(f"**Identidad real**\n\n{prediccion.nombre_real}")
        predicha.markdown(f"**Identidad predicha**\n\n{prediccion.nombre_predicho}")
        probabilidad.markdown(f"**Probabilidad estimada**\n\n{prediccion.probabilidad:.1%}")
        st.caption(prediccion.nota_probabilidad)
        st.subheader("Cinco identidades más probables")
        tabla = pd.DataFrame(prediccion.top, columns=["identidad", "probabilidad"])
        st.bar_chart(
            tabla,
            x="identidad",
            y="probabilidad",
            horizontal=True,
            sort=False,
        )


main()
