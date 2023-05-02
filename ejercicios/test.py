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



