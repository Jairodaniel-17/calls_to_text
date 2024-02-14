# crear instancia de python 3.10
FROM python:3.10
# instalar librerias
RUN pip install pandas numpy nltk moviepy pydub transformers pydantic pytest whisperx fastapi uvicorn
# instalar paquete especial
RUN pip install git+https://github.com/m-bain/whisperx.git
# instalar librerias para usar nucleos cuda
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    wget \
    zip \
    git-lfs 

# install ffmpeg
RUN apt-get install -y ffmpeg


COPY . .

RUN chmod -R 777 /app

# set the working directory in the container
WORKDIR /app

# run the command to start uWSGI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
