# calls_to_text

# Proyecto de Transcripción de Audio
Este proyecto utiliza la biblioteca whisperx para transcribir audio. El proyecto está diseñado para ser ejecutado en un contenedor Docker y expone una API FastAPI para interactuar con él.

## Requisitos
* Docker
* Python 3.10

## Instalación
1. Clona este repositorio en tu máquina local.
2. Navega hasta el directorio del proyecto.
3. Construye la imagen Docker con el siguiente comando:
```
docker build -t nombre_imagen .
```

Reemplaza nombre_imagen con el nombre que desees para la imagen Docker.

## Uso
Una vez que la imagen Docker está construida, puedes iniciar un contenedor con el siguiente comando:
```
docker run -p 7860:7860 nombre_imagen
```

Esto iniciará el servidor FastAPI en el puerto 7860.

La API tiene un endpoint en la ruta raíz (/) que devuelve un mensaje de bienvenida. También tiene un endpoint segment_and_transcribe_audio que acepta una ruta a un archivo de audio y devuelve las transcripciones para cada segmento de parlante.

## Desarrollo
Este proyecto utiliza varias bibliotecas de Python, incluyendo pandas, numpy, nltk, moviepy, pydub, transformers, pydantic, pytest, whisperx, fastapi y uvicorn. Estas dependencias se instalan automáticamente cuando se construye la imagen Docker.

El código fuente principal se encuentra en app.py. Este archivo define la aplicación FastAPI y los endpoints de la API.



Contribución
Las contribuciones a este proyecto son bienvenidas. Por favor, abre un issue para discutir los cambios propuestos antes de hacer un pull request.

Licencia
Este proyecto está licenciado bajo los términos de la licencia MIT.
