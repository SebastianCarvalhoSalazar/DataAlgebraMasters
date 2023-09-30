# DataAlgebraMasters

## Tabla de Contenidos
- [Problema 1: Corrector de Palabras](#problema-1-corrector-de-palabras)
  - [Descripción del Problema (Problema 1)](#descripción-del-problema-problema-1)
  - [Etapas del proyecto (Problema 1)](#etapas-del-proyecto-problema-1)
- [Problema 2: GenderClassificationWithKNN](#problema-2-genderclassificationwithknn)
  - [Descripción del Problema (Problema 2)](#descripción-del-problema-problema-2)
  - [Etapas del proyecto (Problema 2)](#etapas-del-proyecto-problema-2)
  - [¿Qué distancias utilizamos y por qué?](#qué-distancias-utilizamos-y-por-qué)
- [Problema 3:](#problema-3)
  - [Descripción del Problema (Problema 3a)](#descripción-del-problema-problema-3a)
  - [Etapas del proyecto (Problema 3a)](#etapas-del-proyecto-problema-3a)
  - [Script (Problema 3a)](#script-problema-3a)
  - [Descripción del Problema (Problema 3b)](#descripción-del-problema-problema-3b)
  - [Etapas del proyecto (Problema 3b)](#etapas-del-proyecto-problema-3b)
- [Problema 5: Cálculo de Normas en una Matriz](#problema-5-cálculo-de-normas-en-una-matriz)
  - [Descripción del Problema (Problema 5)](#descripción-del-problema-problema-5)
  - [Script (Problema 5)](#script-problema-5)
  - [Etapas del proyecto (Problema 5)](#etapas-del-proyecto-problema-5)

## Problema 1: Corrector de Palabras

### Descripción del Problema (Problema 1)
El objetivo principal de este proyecto es corregir palabras en función de un texto previamente almacenado en un archivo. Cuando un usuario escribe una palabra, esta se consulta en el documento mencionado anteriormente para encontrar una sugerencia más adecuada.

### Etapas del proyecto (Problema 1)
1. Carga del archivo de entrada, que contiene las palabras base.
2. Corrección de la palabra ingresada por el usuario utilizando la información previamente almacenada.
3. La recomendación se basa en la información del documento y utiliza la distancia de Levenshtein entre dos cadenas de caracteres: la primera es la palabra ingresada por el usuario y la segunda es la que se encuentra en el diccionario o archivo cargado al principio. Es importante destacar que el resultado se determina por el número mínimo de ediciones (cambios) necesarios entre la palabra ingresada por el usuario y la base de datos (archivo) cargado previamente.

## Problema 2: GenderClassificationWithKNN

### Descripción del Problema (Problema 2)

El proyecto 'GenderClassificationWithKNN' se enfoca en la clasificación de género mediante el uso del algoritmo de vecinos más cercanos (KNN). El objetivo principal es entrenar un modelo capaz de analizar imágenes en escala de grises con una resolución de 400x600 píxeles y asignar una etiqueta que indique si la persona en la imagen es un hombre (1) o una mujer (0).

<p align="center">
  <img src="https://github.com/SebastianCarvalhoSalazar/DataAlgebraMasters/blob/master/GenderClassificationWithKNN/assets/Resultados.png" width="70%">
</p>

### Etapas del Proyecto (Problema 2)

1. **Carga de Datos:** Comienza con la carga de datos desde carpetas que contienen imágenes de hombres y mujeres. Los datos se obtuvieron del conjunto de datos disponible en Kaggle: [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset). Las imágenes se ajustan a la resolución deseada, se convierten a escala de grises y se pasan por 'equalize_hist()' para mejorar la calidad de la información visual.

2. **Aumento de Datos:** Se implementa un proceso de aumento de datos que genera variantes de las imágenes originales. Esto enriquece el conjunto de entrenamiento y permite al modelo aprender de manera más efectiva.

3. **Entrenamiento del Modelo:** Se utiliza un modelo KNN con búsqueda de hiperparámetros para encontrar la configuración óptima. El modelo se entrena utilizando los datos de entrenamiento y se evalúa su precisión en un conjunto de prueba.

4. **Visualización de Resultados:** Se presenta una visualización de la matriz de confusión para evaluar el rendimiento del modelo y se guarda el modelo óptimo en un archivo.

5. **Predicción de Género:** Finalmente, se proporciona una función para cargar el modelo entrenado y utilizarlo para predecir el género en nuevas imágenes. Se muestra un collage de imágenes de prueba con sus respectivas predicciones de género.

El éxito de este proyecto radica en la calidad de los datos de entrenamiento, la selección adecuada de hiperparámetros del modelo KNN y la capacidad de generalización del modelo en nuevas imágenes. Este enfoque de clasificación de género tiene aplicaciones potenciales en diversas áreas, como la identificación de género en fotografías, la segmentación demográfica en investigaciones y análisis de mercado, y más.

## ¿Qué distancias utilizamos y por qué?

La distancia coseno se centra en la orientación o dirección de los vectores de características de las imágenes, ignorando la magnitud. Esto la hace útil cuando se desea medir la similitud en términos de patrones o direcciones, lo que es común en la representación de imágenes como vectores.

En cambio, la distancia euclidiana mide la diferencia en términos de la magnitud y dirección de los vectores de características. Es más adecuada cuando se requiere tener en cuenta tanto la similitud en patrones como en magnitudes.

La elección entre una u otra depende de si las magnitudes de las características son importantes para el problema de clasificación de imágenes o si la orientación y patrones son suficientes para realizar la tarea de clasificación.

## Problema 3: 

### Descripción del Problema (Problema 3a)

Problema 3: a. Grafique en el plano R^2 la bola de radio r y centro en el origen respecto a la distancia de Minkowski de orden p, donde los parámetros r y p varían desde 0.1 hasta 2 y desde 1 hasta ∞.

### Etapas del Proyecto (Problema 3a)

1. Se define la función a utilizar, que en este caso es `minkowski_distance`.
2. Posteriormente, se calculan los radios en un par de bucles `for` anidados, calculando los ángulos cos(θ) y seno(θ).
3. Finalmente, se grafica usando `matplotlib`.
4. Se concluye que a menor radio, la distancia de Minkowski es menor.

### Script (Problema 3a)
```python
import matplotlib.pyplot as plt
import numpy as np
import random

# Definir la función de distancia de Minkowski
def minkowski_distance(x, y, p):
    return np.power(np.sum(np.abs(x - y)**p), 1/p)

# Rango de valores de r (radio)
# Me genera datos, iniciando en 0.1, hasta 2.0, generando 5 valores en dicho rango.
radios = np.linspace(0.1, 2.0, 5)

# Rango de valores de p para la distancia de Minkowski
p_values = np.linspace(1, 5, 100)

# Crear un gráfico
plt.figure(figsize=(10, 8))

# Calcular y graficar las circunferencias para cada valor de radio y p
for r in radios:
    distances = []
    for p in p_values:
        # Generar puntos en la circunferencia con radio r
        theta = np.linspace(0, 2 * np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Calcular la distancia de Minkowski desde el centro (0, 0)
        center = np.array([0, 0])
        distance = minkowski_distance(center, np.column_stack((x, y)), p)
        distances.append(distance)
    plt.plot(p_values, distances, label=f'Radio={r:.2f} (Minkowski = {minkowski_distance(x, y, p)})')

# Configurar el gráfico
plt.title('Circunferencias de Minkowski con centro en (0, 0)')
plt.xlabel('Valor de p (orden de la distancia de Minkowski)')
plt.ylabel('Distancia de Minkowski')
plt.legend()
plt.grid(True)
plt.show()
```

<p align="center">
  <img src="https://github.com/SebastianCarvalhoSalazar/DataAlgebraMasters/blob/master/Assets/cicunferencia_de_minkouski.jpg" width="70%">
</p>

### Descripción del Problema (Problema 3b)

Problema 3: b. Programe un algoritmo que dado n y p genere de forma aleatoria un vector de R^n cuya distancia de Minkowski de orden p al origen sea menor a 0.1.

### Etapas del Proyecto (Problema 3b)

1. Se define la función a utilizar, que en este caso es `minkowski_distance`.
2. Se generan valores aleatorios en un bucle `for` invocando las distancias.
3. Se concluye que a medida que aumenta el valor de n en R^n, el cálculo se vuelve más complejo, ya que la distancia es muy pequeña y el número de posibilidades aumenta.

## Problema 5: Cálculo de Normas en una Matriz

### Descripción del Problema (Problema 5)

El problema que aborda este script se refiere al cálculo de normas en una matriz. Específicamente, se busca determinar la norma 1 (||A||₁) de las columnas y la norma infinito (||A||∞) de las filas de una matriz dada. Las normas son medidas importantes en álgebra lineal y análisis numérico, y proporcionan información sobre la "tamaño" o magnitud de una matriz en diferentes aspectos.

### Script (Problema 5)

```python
import numpy as np

def calcular_normas(matriz):
    # Calcula la norma uno (||A||₁) - Columnas
    norm_1 = np.max(np.sum(np.abs(matriz), axis=0))
    
    # Calcula la norma infinito (||A||∞) - Filas
    norm_inf = np.max(np.sum(np.abs(matriz), axis=1))
    
    # Validar con NumPy
    norm_1_numpy = np.linalg.norm(matriz, 1)  # Norma 1
    norm_inf_numpy = np.linalg.norm(matriz, np.inf)  # Norma infinito
    
    # Crear un diccionario con los resultados
    resultados = {
        "Norma 1": norm_1,
        "Norma Infinito": norm_inf,
        "Norma 1 Numpy": norm_1_numpy,
        "Norma Infinito Numpy": norm_inf_numpy
    }
    
    return resultados

# Matriz de ejemplo
matriz_ejemplo = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

# Llamar a la función y obtener los resultados
resultados = calcular_normas(matriz_ejemplo)

# Mostrar resultados
for clave, valor in resultados.items():
    print(f"{clave} --> {valor}")
```

Este script de Python tiene como objetivo calcular la norma 1 y la norma infinito de una matriz. Las normas son medidas matemáticas que describen la magnitud de una matriz y se utilizan en diversas aplicaciones numéricas y algebraicas.

### Etapas del Proyecto (Problema 5)

El proyecto sigue las siguientes etapas:

1. **Definición de la Matriz**: Se crea una matriz de ejemplo, en este caso, la matriz `A`, que se utilizará para calcular las normas.

2. **Cálculo de la Norma 1**: Se realiza el cálculo de la norma 1 (||A||₁) de las columnas de la matriz. Esto implica sumar los valores absolutos de las columnas y encontrar el máximo de esas sumas.

3. **Cálculo de la Norma Infinito**: Se efectúa el cálculo de la norma infinito (||A||∞) de las filas de la matriz. Esto consiste en sumar los valores absolutos de las filas y encontrar el máximo de esas sumas.

4. **Mostrar Resultados**: Los resultados de las normas calculadas se imprimen en la consola para su visualización.

5. **Validación con NumPy**: Se valida el cálculo de las normas utilizando las funciones proporcionadas por la biblioteca NumPy, que es una herramienta ampliamente utilizada en computación científica y numérica.

Este script es útil para calcular normas en matrices y puede ser aplicado en diversos campos, como análisis de datos, procesamiento de señales, y más, donde medir la magnitud de los datos es esencial.

