# Análisis de Clustering y Componentes Principales
Este repositorio contiene una App en Streamlit que realiza dinámicamente un análisis de clustering de diferntes países a partir de variables poblacionales.<br />
El código utiliza técnicas de análisis del K óptimo  en el algoritmo Kmeans .<br />
También propone la reducción interactiva de dimensionalidad del modelo , PCA , así como graficas en función de sus dos componentes principales . <br />
Estadistica descriptiva de los datos y su interpretación así como objetivo del análisis y sus conclusiones en el notebook .

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abeldata-clustering.streamlit.app/)

## Requisitos
Las librerías necesarias para ejecutar este proyecto son:

```bash
streamlit== 1.23.1
pandas== 2.0.2
numPy== 1.22.4
seaborn== 0.12.2
scipy== 1.11.3
matplotlib== 3.7.1
scikit-learn==1.2.2
yellowbrick== 1.5
plotly== 5.18.0
kaleido== 0.2.1
```
## Estructura del Repositorio
```bash
.
├── paises.csv                 # Archivo de datos poblacionales
├── datos_desbalanceados.py    # Código principal
├── requirements.txt           # Lista de librerías requeridas
└── README.md                  # Documentación del proyecto
```
## Uso
Elige los parámetros desde el panel lateral e interactua con ellos para evaluar los diferentes resultados del modelo .

## Créditos
Este proyecto utiliza las librerías de Python, incluyendo Streamlit, Pandas, NumPy, Matplotlib, Seaborn y Scikit-learn. <br /> Agradecimientos a la comunidad de desarrolladores de estas librerías.

