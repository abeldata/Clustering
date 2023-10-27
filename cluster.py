import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import streamlit as st
import scipy as sp
#import scipy.cluster.hierarchy as sch
print("matplotlib==", mpl.__version__)
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy as sch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer



df_paises = pd.read_csv('paises.csv' , sep=',')
#print(df_paises.head())
#dataframe sin paises
df_sinpaises = df_paises.drop(columns= ['pais'], axis=1)
#dataframes sin paises escalado
df_sinp = df_paises.copy()
y = df_sinp.pop('pais')
X = df_sinp

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


st.sidebar.header('Distribuciones')

distribuciones  = st.sidebar.selectbox ('Variables',('mort_inf','exportaciones','salud','importaciones','ingresos','inflacion','esp_vida','num_hijos','pib'))

st.sidebar.header('Correlaciones')

correlaciones = st.sidebar.selectbox ('Tipos' , ('pearson','spearman','kendall'))

st.sidebar.header('Número de Clusters')

clusters = st.sidebar.selectbox ('K' , ('Métodos', 'SilhouetteVisualizer','Dendograma' ))

st.sidebar.header('Entrenando Kmeans')

means = st.sidebar.slider('K',2,4,3)

st.sidebar.header('PCA')

pca =st.sidebar.slider('Porcentaje de reducción',0.68,0.997,0.95)







st.markdown("<h1 style='text-align: center'>Clustering</h1>", unsafe_allow_html=True)


#gráfica de la función de distribución de cada caracteristica

fig, ax = plt.subplots()
df_paises[distribuciones].hist(bins=50, figsize=(10,5))
plt.title("Distribuciones de las diferentes variables poblacionales")
plt.show()
st.pyplot(fig)


# Crear un mapa de calor de correlación
sns.heatmap(df_sinpaises.corr(method=correlaciones), annot=True, cmap="coolwarm")  # Mapa de calor

plt.title("Mapa de calor de correlación en df_paises")
plt.show()
st.pyplot(plt)

# visualizacíon de los mejores K

if clusters == 'Método_codo':
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X_scaled)
        inertia = kmeans.inertia_
        inertias.append(inertia)
    
    # Graficar la evolución de la inercia en función de K
    # Gráfica de la evolución de la inercia en función de K
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertias, '-o')
    ax.set_xlabel('Número de clusters (K)')
    ax.set_ylabel('Inercia')
    ax.set_title('Método del Codo para Selección de K en K-means')
    plt.show()
    st.pyplot(fig)

elif clusters == 'SilhouetteVisualizer':
    # Generamos una vista de los distintos Silhouette score en función de K
    # Genera un subplot por cada uno de los clústeres generados
    # Genera scores para 2, 3, 4 y 5 Ks

    fig, ax = plt.subplots(2, 2, figsize=(15,8))
    
    
    for i in [2,3,4,5]:
        # Creamos una instancia del modelo K-Means en función del valor de K
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        kmeans.fit(X_scaled)

        q, mod = divmod(i-1, 2)
        # Creamos una instancia de "SilhouetteVisualizer" con la instancia KMeans anterior
        # Alimentamos el visualizador
        visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(X_scaled)
    # Aumentar tamaño de fuente y hacerla negrita
    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    fig.suptitle("Vista de los distintos Silhouette score en función de K")
    
    st.pyplot(fig)  

elif clusters =='Dendograma':

    plt.figure(figsize=(15,6))
    plt.title('Dendrograma')
    plt.xlabel('Paises')
    plt.ylabel('Distancias euclídeas')
    #plt.grid(True)
    dendrogram = sch.dendrogram(sch.linkage(X_scaled, method = 'ward'))
    fig.suptitle("Dendograma")
    plt.show()
    st.pyplot(plt)


    # Crear modelo K-means con 3 clusters
kmeans = KMeans(n_clusters=means, random_state=42)

# Entrenar modelo con datos del dataset
kmeans.fit(X_scaled)

# Obtener las etiquetas de cluster asignadas a cada punto
labels = kmeans.labels_

# Obtener las coordenadas de los centroides de cada cluster
C = kmeans.cluster_centers_

df_paises['cluster'] = labels
st.markdown("<h1 style='text-align: center; font-size: 14px;'>Paises agrupados</h1>", unsafe_allow_html=True)
st.write(df_paises.set_index('pais'))

columnas=  ['mort_inf','exportaciones','salud','importaciones','ingresos','inflacion','esp_vida','num_hijos','pib']
centers = pd.DataFrame(C , columns=columnas)

f, axes = plt.subplots(means, 1, figsize=(5, 6), sharex=True)
for i, ax in enumerate(axes):
    center = centers.loc[i, :]
    maxPC = 1.01 * np.max(np.max(np.abs(center)))
    colors = ['C0' if l > 0 else 'C1' for l in center]
    ax.axhline(color='#888888')
    center.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'Cluster {i + 1}')
    ax.set_ylim(-maxPC, maxPC)

st.markdown("<h1 style='text-align: center; font-size: 14px;'>Interpretando los Clusters</h1>", unsafe_allow_html=True)
plt.tight_layout()
plt.show()
st.pyplot(plt)




pca_RF = PCA(n_components = pca)
X_reducida = pca_RF.fit(X_scaled)
# Crea un DataFrame para almacenar los valores de los componentes principales y la varianza explicada
df_pca = pd.DataFrame({'Componente Principal': range(1, pca_RF.n_components_ + 1),
                       'Varianza Explicada': pca_RF.explained_variance_ratio_})

# Crea el gráfico utilizando Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(df_pca['Componente Principal'], df_pca['Varianza Explicada'])
ax.set_xlabel('Componente Principal')
ax.set_ylabel('Varianza Explicada')
ax.set_title('Varianza Explicada por Componente Principal')

# Elimina los decimales del eje x
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Muestra el gráfico en Streamlit
st.pyplot(fig)


pca_2 = PCA(n_components = 2)
X_reducida2 = pca_2.fit_transform(X_scaled)

# Crear modelo K-means con 3 clusters
kmeans_pca2 = KMeans(n_clusters=means, random_state=42)

# Entrenar modelo con datos del dataset
kmeans_pca2.fit_transform(X_reducida2)

# Obtener las etiquetas de cluster asignadas a cada punto
labels_pca2 = kmeans_pca2.labels_

# Obtener las coordenadas de los centroides de cada cluster
centroids2 = kmeans_pca2.cluster_centers_

colors = sns.color_palette('bright', n_colors=4)

# Gráfica 1: Scatter plot de los puntos por cluster
fig1, ax1 = plt.subplots()
for label, color in zip(np.unique(labels_pca2), colors):
    ax1.scatter(X_reducida2[labels_pca2 == label, 0], X_reducida2[labels_pca2 == label, 1], color=color, label=label)
ax1.scatter(centroids2[:, 0], centroids2[:, 1], marker="*", color="black", s=150, label="Centroides")
ax1.set_title("Clusters en el espacio reducido por PCA")
ax1.set_xlabel("Componente principal 1")
ax1.set_ylabel("Componente principal 2")
ax1.legend()
plt.show()
st.pyplot(plt)





print("streamlit==", st.__version__)
print("pandas==", pd.__version__)
print("numPy==", np.__version__)
print("sklearn==",sklearn.__version__)
print("keras==",keras.__version__)
print("tensorflow==", tf.__version__)
