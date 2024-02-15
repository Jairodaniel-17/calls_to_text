# usar imagen base de Python 3.10
FROM python:3.10

# instalar dependencias de Python
RUN pip install pandas numpy nltk moviepy pydub transformers pydantic pytest whisperx fastapi uvicorn

# instalar paquete especial whisperx desde GitHub
RUN pip install git+https://github.com/m-bain/whisperx.git

# instalar paquetes de PyTorch para CUDA 12.1
RUN pip install torch==1.10.0+cu121 torchvision==0.11.1+cu121 torchaudio==0.10.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# instalar utilidades básicas
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    wget \
    zip \
    git-lfs \
    ffmpeg

# copiar archivos de la aplicación
COPY . /app
# copiar todos los archivos de la aplicación

COPY . .
# establecer directorio de trabajo
WORKDIR /app

# instalar dependencias de Python desde requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# cambiar permisos de archivos necesarios
RUN chmod -R 777 /app

# definir comando predeterminado para iniciar la aplicación
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
