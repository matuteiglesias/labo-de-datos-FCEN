# %% [markdown]
# ## Procesamiento de audio y video

# %% [markdown]
# ### 1. Utilizar la librería librosa de Python para cargar un archivo de audio y crear un espectrograma del mismo.

# %% [markdown]
# #### a. Descargar un archivo de audio en formato WAV desde alguna fuente en línea o utilizar uno propio.

# %%


# %% [markdown]
# #### b. Instalar la librería librosa de Python utilizando pip: pip install librosa.

# %%


# %% [markdown]
# #### c. Importar librosa y cargar el archivo de audio utilizando la función load:

# %%
```
import librosa
audio, sr = librosa.load('ruta_del_archivo_de_audio.wav')
```


# %% [markdown]
# #### d. Crear un espectrograma utilizando la función specshow de librosa:

# %%
```
import librosa.display
import matplotlib.pyplot as plt

spectrogram = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.show()
```

# %% [markdown]
# #### e. Ejecutar el código y verificar que el espectrograma se muestra correctamente en la pantalla. 
# 
# Experimentar con diferentes configuraciones de parámetros para la función specshow para ajustar el espectrograma según sus necesidades.

# %%


# %% [markdown]
# ### 2. Usar la librería OpenCV de Python para cargar un archivo de video y extraer los cuadros del mismo.

# %% [markdown]
# #### a. Descargar un archivo de video en algún formato compatible con OpenCV, como MP4 o AVI, desde alguna fuente en línea o utilizar uno propio.

# %%


# %% [markdown]
# #### b. Instalar la librería OpenCV de Python utilizando pip: pip install opencv-python.

# %%


# %% [markdown]
# #### c. Importar OpenCV y cargar el archivo de video utilizando la función VideoCapture:

# %%
```
import cv2

video = cv2.VideoCapture('ruta_del_archivo_de_video.mp4')
```


# %% [markdown]
# #### d. Extraer los cuadros del video utilizando un ciclo while y la función read:

# %%
```
success, image = video.read()
count = 0

while success:
    cv2.imwrite("frame%d.jpg" % count, image)
    success, image = video.read()
    count += 1
```


# %% [markdown]
# #### e. Ejecutar el código y verificar que se hayan generado correctamente los cuadros del video en formato JPG. 
# 
# Experimentar con diferentes configuraciones de parámetros para la función VideoCapture para ajustar la calidad y la velocidad de la extracción de cuadros según sus necesidades.

# %%


# %% [markdown]
# ### 3 .Escribir un programa en Python que tome un archivo de audio y determine si hay alguna palabra hablada en él utilizando técnicas de reconocimiento de voz.

# %% [markdown]
# #### a. Descargar un archivo de audio en formato WAV que contenga una o varias palabras habladas.

# %%


# %% [markdown]
# #### b. Instalar la librería SpeechRecognition de Python utilizando pip: pip install SpeechRecognition.

# %%


# %% [markdown]
# #### c. Importar la librería SpeechRecognition y cargar el archivo de audio utilizando la función AudioFile:

# %% [markdown]
# ```
# import speech_recognition as sr
# 
# recognizer = sr.Recognizer()
# audio_file = sr.AudioFile('ruta_del_archivo_de_audio.wav')
# ```
# 

# %% [markdown]
# #### d. Utilizar la función recognize_google de SpeechRecognition para transcribir el audio en texto:

# %%



# %% [markdown]
# ## Introducción a SQL
# %% [markdown]
# ### 1. Crear una base de datos en SQL
# %% [markdown]
# #### a. Instalar un motor de base de datos SQL, como SQLite o MySQL.
# %%
# %% [markdown]
# #### b. Crear una base de datos vacía utilizando la sintaxis SQL adecuada.
# %%
# %% [markdown]
# ### 2. Crear una tabla en la base de datos
# %% [markdown]
# #### a. Definir una estructura para la tabla, incluyendo los nombres y tipos de datos de las columnas.
# %%
# %% [markdown]
# #### b. Utilizar la sintaxis SQL adecuada para crear la tabla en la base de datos.
# %%
# %% [markdown]
# ### 3. Insertar datos en la tabla
# %% [markdown]
# #### a. Crear una serie de registros de datos para insertar en la tabla.
# %%
# %% [markdown]
# #### b. Utilizar la sintaxis SQL adecuada para insertar los registros en la tabla.
# %%
# %% [markdown]
# ### 4. Seleccionar datos de la tabla
# %% [markdown]
# #### a. Utilizar la sintaxis SQL adecuada para seleccionar todos los registros de la tabla.
# %%
# %% [markdown]
# #### b. Utilizar la sintaxis SQL adecuada para seleccionar registros específicos de la tabla.
# %%
# %% [markdown]
# ### 5. Actualizar registros en la tabla
# %% [markdown]
# #### a. Utilizar la sintaxis SQL adecuada para actualizar un registro específico de la tabla.
# %%
# %% [markdown]
# ### 6. Borrar registros de la tabla
# %% [markdown]
# #### a. Utilizar la sintaxis SQL adecuada para borrar un registro específico de la tabla.
# %%
# %% [markdown]
# #### b. Utilizar la sintaxis SQL adecuada para borrar todos los registros de la tabla.
# %%