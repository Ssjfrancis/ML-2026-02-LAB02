"""Entrada liviana de la aplicación del laboratorio en Streamlit Cloud."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"), run_name="__main__")
