# tsne1



# Implementación t-SNE aislada del código original

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Paso 1: Preparar los datos (ejemplo genérico)
datos = pd.read_csv('tu_archivo.csv')  # Cargar datos
pred = datos["columna_objetivo"]  # Variable objetivo (opcional para colores)
datos = datos.drop('columna_objetivo', axis=1)  # Quitar la columna objetivo

# Paso 2: Estandarizar los datos
escalar = StandardScaler()
datos_escalados = escalar.fit_transform(datos)
datos_escalados = pd.DataFrame(datos_escalados)
datos_escalados.columns = datos.columns
datos_escalados.index = datos.index

# Paso 3: Aplicar t-SNE
tsne = TSNE(
    n_components=2,        # Número de componentes (2D para visualización)
    perplexity=25,         # Parámetro de perplejidad (ajustar según dataset)
    learning_rate='auto',  # Tasa de aprendizaje automática
    init='random'          # Inicialización aleatoria
)

# Transformar los datos
individuos = tsne.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)

# Paso 4: Visualizar los resultados
x = individuos.iloc[:, 0]  # Primera componente
y = individuos.iloc[:, 1]  # Segunda componente

fig, ax = plt.subplots(figsize=(10, 6))

# Si tienes variable objetivo para colorear por categorías
if 'pred' in locals():
    for cat in pred.unique():
        ax.scatter(x[pred == cat], y[pred == cat], label=cat)
    plt.legend()
else:
    # Sin categorías, todos los puntos del mismo color
    ax.scatter(x, y, color='steelblue')

# Agregar líneas de referencia
ax.axhline(y=0, color='dimgrey', linestyle='--')
ax.axvline(x=0, color='dimgrey', linestyle='--')

# Etiquetas de los ejes
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')

# Opcional: agregar etiquetas a los puntos
# for i in range(individuos.shape[0]):
#     ax.annotate(individuos.index[i], (x[i], y[i]))

plt.show()

# Explicación de parámetros clave de t-SNE:
"""
- n_components: Dimensiones de salida (típicamente 2 para visualización)
- perplexity: Controla cuántos vecinos considera cada punto
  * Valores típicos: 5-50
  * Datasets pequeños: valores menores (2-25)
  * Datasets grandes: valores mayores (25-89)
- learning_rate: Controla la velocidad de optimización
  * 'auto': ajuste automático basado en el tamaño del dataset
  * Valores típicos: 10-1000
- init: Método de inicialización
  * 'random': inicialización aleatoria
  * 'pca': inicialización usando PCA (a veces más estable)
"""
