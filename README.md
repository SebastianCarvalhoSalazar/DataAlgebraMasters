# DataAlgebraMasters

## Tabla de Contenidos
- [Problema 1: Corrector de Palabras](#problema-1-corrector-de-palabras)
  - [Descripción del Problema](#descripción-del-problema)
  - [Etapas del proyecto](#etapas-del-proyecto)
- [Problema 2: GenderClassificationWithKNN](#problema-2-genderclassificationwithknn)
  - [Descripción del Problema](#descripción-del-problema)
  - [Etapas del proyecto](#etapas-del-proyecto)

## Problema 1: Corrector de Palabras

### Descripción del Problema
El proyecto se focaliza en corregir las palabras, de acuerdo a un texto que se tiene previamente almacenado en un archivo, por ejemplo. De acuerdo a la palabra que la persona escribe, esta es consultada en el documento antes mencionado para buscar una palabra más recomendada.

### Etapas del proyecto

1. Carga del archivo insumo, el cual contiene las palabras base.
2. Corrección de la palabra digitada por el usuario, de acuerdo a la información que se tiene almacenada previamente. 
3. La recomendación se realización con base a la información que se tiene en el documento. Utilizando la distancia de Levanten entre dos string, el primero de ellos el ingresado por el usuario, y el segundo el que se encuentra en el diccionario u archivo cargado al inicio. Se debe tener en cuenta que el resultado se da por el numero mínimo de ediciones (cambios) que se deban realizar entre la palabra ingresada por el usuario Vs la base de datos (archivo) cargado previamente.

## Problema 2: GenderClassificationWithKNN

### Descripción del problema

El proyecto 'GenderClassificationWithKNN' se enfoca en la clasificación de género mediante el uso del algoritmo de vecinos más cercanos (KNN). El objetivo principal es entrenar un modelo capaz de analizar imágenes en escala de grises con una resolución de 400x600 píxeles y asignar una etiqueta que indique si la persona en la imagen es un hombre (1) o una mujer (0).

<p align="center">
  <img src="https://github.com/SebastianCarvalhoSalazar/DataAlgebraMasters/blob/master/GenderClassificationWithKNN/assets/Resultados.png" width="70%">
</p>

#### Etapas del Proyecto

1. **Carga de Datos:** Comienza con la carga de datos desde carpetas que contienen imágenes de hombres y mujeres. Los datos se obtuvieron del conjunto de datos disponible en Kaggle: [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset). Las imágenes se ajustan a la resolución deseada, se convierten a escala de grises y se pasan por 'equalize_hist()' para mejorar la calidad de la información visual.

2. **Aumento de Datos:** Se implementa un proceso de aumento de datos que genera variantes de las imágenes originales. Esto enriquece el conjunto de entrenamiento y permite al modelo aprender de manera más efectiva.

3. **Entrenamiento del Modelo:** Se utiliza un modelo KNN con búsqueda de hiperparámetros para encontrar la configuración óptima. El modelo se entrena utilizando los datos de entrenamiento y se evalúa su precisión en un conjunto de prueba.

4. **Visualización de Resultados:** Se presenta una visualización de la matriz de confusión para evaluar el rendimiento del modelo y se guarda el modelo óptimo en un archivo.

5. **Predicción de Género:** Finalmente, se proporciona una función para cargar el modelo entrenado y utilizarlo para predecir el género en nuevas imágenes. Se muestra un collage de imágenes de prueba con sus respectivas predicciones de género.

El éxito de este proyecto radica en la calidad de los datos de entrenamiento, la selección adecuada de hiperparámetros del modelo KNN y la capacidad de generalización del modelo en nuevas imágenes. Este enfoque de clasificación de género tiene aplicaciones potenciales en diversas áreas, como la identificación de género en fotografías, la segmentación demográfica en investigaciones y análisis de mercado, y más.
