# %% [markdown]
# ## Introducción a los datos no estructurados



# %% [markdown]
# ### 1. Cargar un archivo de texto y contar la cantidad de palabras.



# %% [markdown]
# #### a. Importar la librería os y utilizarla para obtener la ruta del archivo:



# %%
import os

ruta = os.path.join('ruta', 'del', 'archivo', 'texto.txt')



# %% [markdown]
# #### b. Abrir el archivo y leer su contenido:



# %%
with open(ruta, 'r', encoding='utf-8') as f:
    contenido = f.read()



# %% [markdown]
# #### c. Tokenizar el contenido del archivo en palabras:



# %%
import nltk

palabras = nltk.word_tokenize(contenido)



# %% [markdown]
# #### d. Contar la cantidad de palabras:



# %%
len(palabras)




# %% [markdown]
# ### 2. Cargar una imagen y mostrar su contenido.



# %% [markdown]
# #### a. Importar la librería Pillow y utilizarla para cargar la imagen:



# %%
from PIL import Image

ruta = os.path.join('ruta', 'del', 'archivo', 'imagen.jpg')

imagen = Image.open(ruta)



# %% [markdown]
# #### b. Mostrar la imagen en una ventana emergente:



# %%
imagen.show()





# %% [markdown]
# ### 1. Leer y visualizar una imagen utilizando Python.



# %% [markdown]
# #### a. Importar las bibliotecas necesarias:



# %%
import matplotlib.pyplot as plt
import cv2



# %% [markdown]
# #### b. Leer la imagen utilizando la función imread de OpenCV:



# %%
img = cv2.imread('ruta_de_la_imagen.jpg')



# %% [markdown]
# #### c. Visualizar la imagen utilizando la función imshow de Matplotlib:



# %%
plt.imshow(img)



# %% [markdown]
# ### 2. Leer y procesar un archivo de texto utilizando Python.



# %% [markdown]
# #### a. Leer el archivo de texto utilizando la función open de Python:



# %%
with open('ruta_del_archivo_de_texto.txt', 'r') as f:
    data = f.read()



# %% [markdown]
# #### b. Tokenizar el texto utilizando la función split de Python:



# %%
tokens = data.split()



# %% [markdown]
# #### c. Calcular la frecuencia de cada token utilizando un diccionario de Python:



# %%
freq = {}
for token in tokens:
    if token in freq:
        freq[token] += 1
    else:
        freq[token] = 1



# %% [markdown]
# #### d. Ordenar los tokens según su frecuencia en orden descendente:



# %%
sorted_freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}
print(sorted_freq)





# %% [markdown]
# ### 2. Utilizar la librería spaCy para procesar un texto y obtener información relevante.



# %% [markdown]
# #### a. Importar spaCy y cargar el modelo de lenguaje en español:



# %%
import spacy

nlp = spacy.load('es_core_news_sm')



# %% [markdown]
# #### b. Procesar un texto y obtener información relevante, como entidades nombradas y sus tipos:



# %%
text = "El presidente Alberto Fernández anunció un paquete de medidas económicas para impulsar el crecimiento del país."

doc = nlp(text)

# Print named entities and their labels
for ent in doc.ents:
    print(ent.text, ent.label_)



# %% [markdown]
# ### 3. Tokenizar un texto utilizando la librería NLTK.



# %%
import nltk

# Tokenize text
text = "Este es un ejemplo de texto para tokenizar."
tokens = nltk.word_tokenize(text)

print(tokens)


Ejercicios sobre Introducción a los datos no estructurados
Ejercicio 1: Descarga de imágenes de la web
a. Utilizar la librería requests de Python para descargar una imagen de la web y guardarla en su computadora.

python

import requests

url = 'https://www.example.com/image.jpg'
response = requests.get(url)

with open('image.jpg', 'wb') as f:
    f.write(response.content)

b. Utilizar la librería Pillow para abrir la imagen y visualizarla en su computadora.

python

from PIL import Image

image = Image.open('image.jpg')
image.show()

Ejercicio 2: Descarga de datos de texto
a. Utilizar la librería requests de Python para descargar un archivo de texto de la web y guardarlo en su computadora.

python

import requests

url = 'https://www.example.com/data.txt'
response = requests.get(url)

with open('data.txt', 'w') as f:
    f.write(response.text)

b. Leer el archivo de texto y realizar alguna operación de procesamiento de texto, como contar la frecuencia de las palabras o eliminar las palabras comunes.

python

with open('data.txt', 'r') as f:
    text = f.read()

# contar la frecuencia de las palabras
word_counts = {}
for word in text.split():
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
print(word_counts)

# eliminar las palabras comunes
common_words = ['el', 'la', 'los', 'las', 'de', 'en', 'a', 'y', 'o']
words = [word for word in text.split() if word not in common_words]
print(' '.join(words))




# %% [markdown]
# ## Procesamiento de datos de texto



# %% [markdown]
# ### 1. Tokenización de texto con NLTK



# %% [markdown]
# #### a. Instalar la librería NLTK (Natural Language Toolkit)



# %%
!pip install nltk



# %% [markdown]
# #### b. Importar NLTK y descargar los recursos necesarios



# %%
import nltk
nltk.download('punkt')



# %% [markdown]
# #### c. Tokenizar una oración utilizando la función word_tokenize de NLTK



# %%
from nltk.tokenize import word_tokenize

text = "El gato está en la alfombra."
tokens = word_tokenize(text)
print(tokens)



# %% [markdown]
# ### 2. Análisis de sentimientos con TextBlob



# %% [markdown]
# #### a. Instalar la librería TextBlob



# %%
!pip install textblob



# %% [markdown]
# #### b. Importar TextBlob y analizar el sentimiento de una oración



# %%
from textblob import TextBlob

text = "Este libro es excelente."
blob = TextBlob(text)
sentiment = blob.sentiment.polarity
print(sentiment)



# %% [markdown]
# ## Procesamiento de datos de imágenes



# %% [markdown]
# ### 1. Extracción de características con OpenCV



# %% [markdown]
# #### a. Instalar la librería OpenCV



# %%
!pip install opencv-python



# %% [markdown]
# #### b. Importar OpenCV y cargar una imagen



# %%
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('ruta_de_la_imagen.jpg')



# %% [markdown]
# #### c. Convertir la imagen a escala de grises y mostrarla



# %%
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')



# %% [markdown]
# ### 2. Redes neuronales convolucionales con TensorFlow



# %% [markdown]
# #### a. Instalar TensorFlow



# %%
!pip install tensorflow



# %% [markdown]
# #### b. Importar TensorFlow y cargar un modelo pre-entrenado



# %%
import tensorflow as tf

model = tf.keras.applications.MobileNetV2()



# %% [markdown]
# #### c. Cargar una imagen y pasarla por el modelo



# %%
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'ruta_de_la_imagen.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
predictions = model.predict(img_array)



# %% [markdown]
# #### d. Obtener las clases y las probabilidades de las predicciones



# %%
from tensorflow.keras.applications import imagenet_utils

results = imagenet_utils.decode_predictions(predictions)
for result in results[0]:
    print(result)




# %% [markdown]
# ## Introducción a los datos no estructurados



# %% [markdown]
# ### 1. Tokenización de texto



# %% [markdown]
# #### a. Importar NLTK y descargar los datos necesarios:



# %%
import nltk
nltk.download('punkt')



# %% [markdown]
# #### b. Tokenizar una oración en palabras individuales:



# %%
from nltk.tokenize import word_tokenize

sentence = "Hola a todos, ¿cómo están?"
tokens = word_tokenize(sentence)
print(tokens)



# %% [markdown]
# ### 2. Análisis de sentimientos



# %% [markdown]
# #### a. Importar TextBlob y analizar el sentimiento de una oración:



# %%
from textblob import TextBlob

sentence = "Amo este hermoso día"
blob = TextBlob(sentence)
sentiment = blob.sentiment.polarity
print(sentiment)



# %% [markdown]
# #### b. Analizar el sentimiento de un conjunto de comentarios en una lista:



# %%
comments = ['Estoy muy feliz hoy', 'Odio estar enfermo', 'Este restaurante es increíble']
sentiment_scores = []
for comment in comments:
    blob = TextBlob(comment)
    sentiment = blob.sentiment.polarity
    sentiment_scores.append(sentiment)
print(sentiment_scores)



# %% [markdown]
# ### 3. Extracción de características de imágenes



# %% [markdown]
# #### a. Importar OpenCV y cargar una imagen:



# %%
import cv2

image = cv2.imread('ruta_de_la_imagen.jpg')



# %% [markdown]
# #### b. Convertir la imagen a escala de grises y mostrarla:



# %%
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen en escala de grises', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# %% [markdown]
# ### 4. Redes neuronales convolucionales para clasificación de imágenes



# %% [markdown]
# #### a. Importar TensorFlow y Keras y cargar el conjunto de datos MNIST:



# %%
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()



# %% [markdown]
# #### b. Preprocesar los datos y construir el modelo de red neuronal convolucional:



# %%
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# %% [markdown]
# #### c. Entrenar y evaluar el modelo:



# %%
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print('Accuracy:', test_acc)


