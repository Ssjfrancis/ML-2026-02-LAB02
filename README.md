# Laboratorio 02 · Embeddings faciales lineales

Machine Learning · Universidad Católica del Norte · Segundo semestre de 2026
Docente Dr. Juan Bekios Calfa

Compara PCA, LDA y Fisherfaces con SVM lineal, k-NN y árbol de decisión. La grilla fija de dimensiones 10, 20 y 30 se evaluó en Olivetti y LFW. El código deriva de la revisión docente `4a25df2`. [Repositorio base del docente](https://github.com/jbekios/ML-2026-02-LAB02)

## Instalar y ejecutar

Desde la carpeta que contiene `environment.yml`, en una terminal con Conda disponible. En Windows se pueden ejecutar los mismos comandos desde Miniforge Prompt o Anaconda Prompt.

```bash
conda env create -f environment.yml
conda activate lab-embeddings-faciales
python main.py experimentos
streamlit run app.py
```

La primera carga requiere conexión para descargar las bases mediante scikit-learn. Olivetti y LFW pueden ejecutarse por separado. El CSV consolidado conserva las filas del otro conjunto cuando se vuelve a ejecutar uno.

```bash
python main.py olivetti
python main.py lfw
python -m unittest discover -s tests -v
jupyter lab Analisis_Laboratorio_02.ipynb
```

La aplicación muestra la dirección local en la terminal. Elige una base, un embedding, un clasificador, una dimensión y una foto de TEST. La identidad mostrada coincide con `predict()`. La probabilidad corresponde a esa identidad. Si el modelo falta o su huella difiere, se ajusta con TRAIN y se vuelve a guardar. Usa cachés `.joblib` generadas localmente por el proyecto.

El notebook se ejecuta de principio a fin desde la raíz. Lee la evidencia compatible que trae el ZIP y ejecuta el pipeline cuando falta. Para regenerar los experimentos, cambia `REGENERAR` a `True` en la primera celda de código. Las dos pruebas de evidencia esperan el CSV de ambos conjuntos. La prueba de interfaz usa también las bases descargadas.

## Protocolo experimental

- Split estratificado por identidad, semilla 42 y 25 % de TEST. Los índices de cada base se comparten entre las 27 combinaciones.
- LFW usa `min_faces_per_person=20`, `resize=0.5` y hasta 50 identidades. Los empates de frecuencia se resuelven por el ID de origen del cargador.
- PCA usa SVD completa. LDA usa `solver="svd"`. Fisherfaces ajusta PCA y luego LDA en TRAIN. Se conserva la configuración de la base docente con `d_pca=d_lda=d` en la grilla.
- SVM lineal usa `C=1` y `probability=True`. k-NN usa tres vecinos y voto uniforme. El árbol usa impureza Gini y profundidad máxima ocho.
- Se mide una corrida de ajuste y predicción por clasificador sobre sus embeddings. `tiempo_s` es la suma de `ajuste_s` y `prediccion_s`. El costo de obtener el embedding queda fuera de esta medida.
- t-SNE y UMAP se ajustan con las coordenadas de TRAIN a `d=20`. t-SNE usa perplejidad 30, inicio PCA y 1.000 iteraciones. UMAP usa 15 vecinos, distancia mínima 0,1 e inicio espectral. Ambos usan distancia euclidiana y semilla 42.

La configuración por defecto reproduce el experimento. `.env.example` documenta las variables para una ejecución distinta. Conserva los valores por defecto si quieres replicar las cifras del informe.

## Evidencia incluida

| Ruta | Contenido |
| --- | --- |
| `data/resultados/resultados_modelos.csv` | 54 combinaciones con configuración, accuracy, precisión macro, recall macro, macro-F1 y tiempos en segundos |
| `data/resultados/resultados_olivetti.csv` y `resultados_lfw.csv` | Tablas de 27 filas por conjunto |
| `data/resultados/predicciones_*.csv` | Foto de TEST, identidad real y predicción por combinación |
| `data/resultados/identidades_*.csv` y `particion_*.csv` | Equivalencia de IDs y nombres, conteos e índices TRAIN/TEST. `fila` identifica la posición tras la carga y el filtrado |
| `data/resultados/varianza_pca_*.csv` y `reconstruccion_pca_*.csv` | Varianza acumulada, ECM de la foto ilustrada y ECM medio en TEST |
| `data/resultados/protocolo_*.json` | Formas reales, huella de TRAIN, semilla y versiones |
| `figuras/` | Curvas PCA, eigenfaces, reconstrucciones, mapas, 54 confusiones, 18 paneles de ejemplos y comparaciones |
| `docs/` | Informe IEEE editable y compilado, auditoría y copia portable del historial Git |

El CSV usa coma como separador de campos y punto decimal. Los archivos por identidad permiten leer los códigos de las matrices de confusión. Las carpetas de datos, resultados, figuras y documentos se gestionan según el `.gitignore` docente. El ZIP entrega la evidencia regenerable junto al código.

## Compilar el informe

Con una distribución LaTeX que incluya IEEEtran, Babel en español, Latin Modern, booktabs y hyperref, desde la raíz del proyecto:

```bash
cd docs
pdflatex -halt-on-error Informe_Laboratorio_02.tex
pdflatex -halt-on-error Informe_Laboratorio_02.tex
```

En Overleaf se puede seleccionar `docs/Informe_Laboratorio_02.tex` como documento principal al subir la estructura del ZIP. Las figuras se leen desde `figuras/`.

## Historial y publicación del equipo

El archivo `docs/Repositorio_Lab02.bundle` contiene el historial docente y el commit de implementación. Para recuperar el repositorio con ese historial, ejecuta desde la raíz del ZIP:

```bash
git clone -b lab02-entrega docs/Repositorio_Lab02.bundle ../ML-2026-02-LAB02-repo
cd ../ML-2026-02-LAB02-repo
git log --oneline -3
```

Con un repositorio del equipo ya creado en GitHub y acceso de escritura, sustituye `USUARIO/REPOSITORIO` por su ruta real:

```bash
git remote set-url origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin lab02-entrega:main
```

El bloque de autoría del TeX puede completarse con los integrantes confirmados. El PDF y el README del equipo pueden incorporar la URL de entrega una vez publicado el repositorio.

## Datos y atribución

Las fotos de Olivetti provienen de AT&T Laboratories Cambridge. Las de LFW provienen de Huang et al. (UMass Amherst), versión *funneled*. Ambas se cargan vía scikit-learn. El análisis se limita al uso académico con esas dos bases. Los ejemplos visuales corresponden a fotografías de TEST. [Olivetti](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) · [LFW](https://vis-www.cs.umass.edu/lfw/) · [scikit-learn](https://scikit-learn.org/1.8/datasets/real_world.html)
