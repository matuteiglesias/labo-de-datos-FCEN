# labo-de-datos-FCEN
Ejercicios practicos para las clases de la materia Laboratorio de Datos, de la Lic. en Cs. de Datos, FCEN, UBA.
<!-- Output copied to clipboard! -->

<!-- Yay, no errors, warnings, or alerts! -->




# Indice de clases

|Clase Numero|Capitulo                              |Titulo                                                     |
|------------|--------------------------------------|-----------------------------------------------------------|
|1           |Fundamentos de Datos                  |Fundamentos de almacenamiento y formatos de datos         |
|2           |Fundamentos de Datos                  |Introducción a bases de datos y SQL                        |
|3           |Fundamentos de Datos                  |Diseño y normalización de bases de datos                   |
|4           |Fundamentos de Datos                  |Manejo de datos no estructurados y web scraping            |
|5           |Transformación de Datos               |ETL y manipulación avanzada de datos con Pandas            |
|6           |Transformación de Datos               |Manipulación avanzada de datos con Pandas                  |
|7           |Transformación de Datos               |Agregación, resúmenes y análisis exploratorio avanzado     |
|8           |Transformación de Datos               |Transformación de datos en múltiples fuentes               |
|9           |Transformación de Datos               |Preparación de datos para modelado y Machine Learning      |
|10          |Visualización                         |Introducción a la visualización de datos                   |
|11          |Visualización                         |Visualizaciones avanzadas y técnicas interactivas          |
|E1          |Repaso de Estadistica            |Probabilidad y distribuciones estadísticas              |
|E2          |Repaso de Estadistica            |Pruebas de hipótesis e intervalos de confianza          |
|E3          |Repaso de Estadistica            |Análisis de regresión (OLS)                             |
|12          |Exploración y Comunicación            |Análisis exploratorio de datos y limpieza                  |
|13          |Exploración y Comunicación            |Introducción a la narración y comunicación de datos        |
|14          |Exploración y Comunicación            |Creación de presentaciones de datos e informes efectivos   |
|15          |Introducción a Machine Learning       |Fundamentos de ML y métodos de clasificación               |
|16          |Introducción a Machine Learning       |Evaluación y selección de modelos                          |
|17          |Introducción a Machine Learning       |Regresión: Modelos lineales y KNN                          |
|18          |Introducción a Machine Learning       |Modelos no supervisados: Clustering                        |
|19          |Machine Learning Avanzado             |Reducción de dimensionalidad y representaciones de datos   |
|20          |Machine Learning Avanzado             |Optimización y ajuste de modelos                           |
|21          |Machine Learning Avanzado             |Redes neuronales y arquitecturas modernas                  |
|22          |Machine Learning Avanzado             |ML explicable y causalidad                                 |
|23          |Producción y Automatización           |MLOps y automatización del ciclo de vida de modelos        |




# Guia de Contenidos por clase

## Fundamentos de Datos 

Clase 1: Fundamentos de almacenamiento y formatos de datos
   * Tipos de datos estructurados y semiestructurados (CSV, JSON, XML, Parquet).
   * Diferencias entre almacenamiento en filas vs. columnas (Parquet vs. CSV).
   * Introducción a **Pandas** para la carga y exploración de datos.
   * Buenas prácticas en almacenamiento y organización de datos.

Clase 2: Introducción a bases de datos y SQL
   * Modelo relacional: tablas, claves primarias/foráneas, relaciones.
   * Conceptos clave de SQL: SELECT, INSERT, UPDATE, DELETE.
   * Creación y consulta de bases de datos con **SQLite** y **PostgreSQL**.
   * Importar y exportar datos entre **Pandas** y SQL.

Clase 3: Diseño y normalización de bases de datos
   * Diferencias entre bases de datos **relacionales y NoSQL**.
   * Normalización: ¿por qué descomponer tablas? Primera, segunda y tercera forma normal.
   * Optimización de almacenamiento y rendimiento en bases de datos SQL.
   * Creación de modelos de datos eficientes con SQL y Pandas.

Clase 4: Manejo de datos no estructurados y web scraping
   * Extracción de datos desde la web con **requests** y **BeautifulSoup**.
   * Comprensión de HTML y CSS: conceptos básicos de HTML y CSS, incluidos los selectores y la sintaxis.
   * Introducción a las bibliotecas de web scraping: descripción general de las bibliotecas de web scraping más populares, como BeautifulSoup y Scrapy.
   * Introducción a **APIs** y procesamiento de datos semiestructurados.
   * Almacenamiento eficiente de datos extraídos en bases de datos.


## Transformación de Datos 

Clase 5: ETL y manipulación avanzada de datos con Pandas
   * Introducción a procesos ETL: Extract, Transform, Load.
   * Limpieza y transformación de datos con Pandas.
   * Conexión de Pandas con SQL y NoSQL para procesamiento de datos.
   * Creación de pipelines de transformación de datos reproducibles.

Clase 6: Manipulación avanzada de datos con Pandas
   * Técnicas avanzadas de indexado y selección de datos (`.loc`, `.iloc`, `.query`).
   * Aplicación de transformaciones vectorizadas con `.apply()`, `.map()`, `.transform()`.
   * Manejo de datos desordenados: reindexado, pivot tables y manejo de multi-index.
   * Creación de funciones personalizadas para procesamiento de datos.

Clase 7: Agregación, resúmenes y análisis exploratorio avanzado
   * Uso de `groupby()` para segmentación y análisis de datos.
   * Agregaciones avanzadas con `agg()`, `apply()`, y `transform()`.
   * Detección de patrones y generación de estadísticas descriptivas.
   * Análisis temporal: manejo de fechas, resampling y rolling windows.

Clase 8: Transformación de datos en múltiples fuentes
   * Integración de múltiples fuentes de datos (archivos CSV, SQL, APIs).
   * Cómo unir datasets: `merge()`, `concat()`, `join()`.
   * Identificación y resolución de duplicados y datos inconsistentes.
   * Creación de pipelines de transformación modulares.

Clase 9: Preparación de datos para modelado y Machine Learning
   * Codificación de variables categóricas (one-hot encoding, label encoding).
   * Transformaciones matemáticas: escalado, normalización y generación de features.
   * Manejo de valores faltantes: estrategias avanzadas de imputación.
   * Selección de variables relevantes para modelos de Machine Learning.


## Visualizacion

Herramientas para una buena visualizacion, clave para comunicar mensajes a traves de los datos.

Clase 10:

   * Introducción a la visualización de datos
   * Principios del diseño visual.
   * Selección de representaciones visuales apropiadas
   * Tipos de visualización comunes (por por ej. gráficos de barras, gráficos de líneas, diagramas de dispersión)

Clase 11:

   * Visualizaciones avanzadas (por por ej. mapas de calor, mapas de árboles, diagramas de red)
   * Visualización de datos multidimensionales
   * Interactividad en la visualización de datos
   * Las mejores prácticas para una visualización de datos eficaz


## Repaso de estadistica para analisis de datos

Contenidos minimos correlativos que preceden a los modelos y otros conceptos mas avanzados que ocupan el resto de la materia.

Clase E1: Probabilidad y distribuciones estadísticas

    * Introducción a la probabilidad y distribuciones de probabilidad
    * Tipos de distribuciones de probabilidad (por por ej. normal, binomial, Poisson)
    * Propiedades y aplicaciones de cada distribución
    * Teorema del límite central y su significado en estadística

Clase E2: Pruebas de hipótesis e intervalos de confianza

    * El concepto de prueba de hipótesis e hipótesis nulas/alternativas
    * Tipos de errores en la prueba de hipótesis (errores tipo I y tipo II)
    * Intervalos de confianza y su interpretación
    * Ejemplos prácticos de prueba de hipótesis e intervalos de confianza en ciencia de datos

Clase E3: Análisis de regresión

    * Introducción a la regresión lineal y sus suposiciones
    * Regresión lineal múltiple y sus extensiones
    * Selección de modelos y técnicas de regularización (por por ej. Lasso, regresión de Ridge)
    * Modelos de regresión no lineal (por por ej. regresión polinomial, regresión spline)


Recordar que tener intuicion de las bases de estadistica y la idea del TCL es clave para tomar decisiones en un proceso de analisis de informacion cuantitativa.


## EDA y Comunicacion

La exploracion es un paso crucial en cualquier proyecto de análisis de datos, ya que ayuda a comprender los datos e identificar posibles problemas.
    
Clase 12:

   * Descripción general del análisis exploratorio de datos
   * Técnicas para resumir y visualizar distribuciones de datos
   * Identificación de valores atípicos y valores perdidos
   * Limpieza y preprocesamiento de datos
    
Clase 13: Introducción a la narración y comunicación de datos

   * Importancia de la narración de datos y la comunicación efectiva en el análisis de datos
   * Comprender a la audiencia y adaptar la comunicación en consecuencia
   * Mejores prácticas para una comunicación clara y concisa
   * Importancia de crear una narrativa en la narración de datos
   * Cómo usar de manera efectiva la visualización de datos para comunicar ideas

Clase 14: Creación de presentaciones de datos e informes efectivos

   * Introducción a diferentes tipos de presentaciones de datos (ej. diapositivas, infografías, tableros)
   * Mejores prácticas para crear presentaciones de datos efectivas
   * Cómo estructurar un informe de datos para lograr el máximo impacto
   * Sugerencias para crear imágenes atractivas y usarlas e manera efectiva

## Introducción a Machine Learning

Clase 15: Fundamentos de ML y Métodos de Clasificación

   * ¿Qué es Machine Learning? Diferencia entre aprendizaje supervisado y no supervisado.
   * Componentes del pipeline de modelado de datos.
   * Introducción a modelos de clasificación: KNN, árboles de decisión, SVM.
   * Métricas básicas de desempeño en clasificación.
   * Implementación práctica con Scikit-learn.

Clase 16: Evaluación y Selección de Modelos

   * Importancia de la evaluación en ML.
   * Métricas clave: Accuracy, Precision, Recall, F1-score, AUC-ROC.
   * Validación cruzada y estrategias de partición de datos.
   * Diagnóstico de sobreajuste y subajuste.
   * Selección de modelos basada en datos.

Clase 17: Regresión: Modelos Lineales y KNN

   * Diferencias entre clasificación y regresión.
   * Regresión lineal simple y múltiple.
   * Introducción a regresión polinómica.
   * Uso de KNN para regresión y comparación con modelos lineales.
   * Evaluación de regresión con R² y MSE.

Clase 18: Modelos No Supervisados: Clustering

   * Introducción al aprendizaje no supervisado.
   * Algoritmos de clustering: K-Means, DBSCAN, clustering jerárquico.
   * Evaluación de calidad en agrupamientos (Silhouette Score, Davies-Bouldin).
   * Aplicaciones de clustering en datos reales.
   * Implementación con Scikit-learn.

## Machine Learning Avanzado

Clase 19: Reducción de Dimensionalidad y Representaciones de Datos

   * ¿Qué es la reducción de dimensionalidad y por qué es útil?
   * Métodos principales: PCA, t-SNE, UMAP.
   * Representaciones aprendidas vs. características manuales.
   * Embeddings y su impacto en modelos de ML.
   * Visualización de datos en espacios reducidos.

Clase 20: Optimización y Ajuste de Modelos

   * ¿Cómo mejorar el rendimiento de un modelo?
   * Búsqueda de hiperparámetros: Grid Search, Random Search, Bayesian Optimization.
   * Técnicas de regularización: L1, L2 y Elastic Net.
   * Early stopping y checkpointing.
   * Construcción de pipelines de ML eficientes.

Clase 21: Redes Neuronales y Arquitecturas Modernas

   * Fundamentos de redes neuronales artificiales.
   * Backpropagation y funciones de activación.
   * Introducción a CNNs (Redes Convolucionales) y su uso en visión por computadora.
   * RNNs y LSTMs para procesamiento de secuencias.
   * El impacto de los Transformers y la atención en NLP.

Clase 22: ML Explicable y Causalidad

   * ¿Por qué es importante la interpretabilidad en modelos de ML?
   * Herramientas de interpretabilidad: SHAP, LIME, PDP.
   * Causalidad en ML: Diferencias entre correlación y causalidad.
   * DAGs y métodos de inferencia causal.
   * Evaluación de impacto de cambios en modelos.

## Producción y Automatización  

Clase 23: MLOps y Automatización del Ciclo de Vida de Modelos

   * Introducción a MLOps y su importancia en producción.
   * Gestión del ciclo de vida de modelos: entrenamiento, versionado y monitoreo.
   * Automatización con CI/CD en ML.
   * Infraestructura escalable para ML en la nube.
   * Ejemplo de pipeline de ML en producción.


