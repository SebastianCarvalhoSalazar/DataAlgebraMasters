# DataAlgebraMasters

## Tabla de Contenidos
- [Problema 1: Corrector de Palabras](#problema-1-corrector-de-palabras)
  - [Descripción del Problema (Problema 1)](#descripción-del-problema-problema-1)
  - [Etapas del proyecto (Problema 1)](#etapas-del-proyecto-problema-1)
- [Problema 2: GenderClassificationWithKNN](#problema-2-genderclassificationwithknn)
  - [Descripción del Problema (Problema 2)](#descripción-del-problema-problema-2)
  - [Etapas del proyecto (Problema 2)](#etapas-del-proyecto-problema-2)

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
