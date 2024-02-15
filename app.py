from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisperx
import torch
import json
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub.utils import mediainfo
import subprocess
from dotenv import load_dotenv

# cargar el archivo .env
load_dotenv()

# Inicializar la aplicación
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Carga de token y verificación de disponibilidad de GPU
hf_token = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tipo de cómputo #"float32" o "float16" si se desea utilizar la aceleración de la GPU o "float32" si se desea utilizar la CPU.
compute_type = "float32" if device == "cuda" else "float32"
modelo = "small"
# Transcripción
print("Descargando el modelo...")
model = whisperx.load_model(
    model=modelo,
    device=device,
    language="es",
    compute_type=compute_type,
    task="transcribe",
)
print("¡Modelo descargado con éxito!")


def transcribe_wav(audio):
    """
    Transcribe un archivo de audio WAV y devuelve el texto transrito.
    """
    resultado = model.transcribe(audio, language="es", batch_size=28)
    return "\n".join([segment["text"] for segment in resultado["segments"]])


def segment_and_transcribe_audio(audio_entrada: str) -> dict:
    """
    Segmenta y transcribe el archivo de audio de entrada.

    Parameters:
    :param audio_entrada: str: Ruta al archivo de audio de entrada.

    Returns:
    dict: Diccionario que contiene las transcripciones para cada segmento de parlante.
    """
    transcriptions = {}

    if not os.path.exists(audio_entrada):
        raise FileNotFoundError("El archivo de audio no existe.")

    audio_info = mediainfo(audio_entrada)
    duration_s = audio_info.get("duration")
    if duration_s is None:
        raise ValueError("No se pudo obtener la duración del archivo de audio.")

    duration_s = float(duration_s)
    print(f"Duración del audio: {duration_s} s")
    if duration_s < 10:
        raise ValueError(
            f"La duración del audio es demasiado corta para realizar una transcripción. Duración: {duration_s} s"
        )

    # Cargar el archivo de audio usando whisperx.load_audio
    audio = whisperx.load_audio(audio_entrada)

    # Realizar la transcripción completa del audio
    transcripcion_completa = transcribe_wav(audio)
    transcriptions["full"] = transcripcion_completa

    # Inicializar el pipeline de diarización de parlantes
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))

    # Realizar la diarización de los parlantes en el audio
    with open(audio_entrada, "rb") as f:
        diarization = pipeline(f)

    # Escribir los resultados de la diarización en un archivo de texto
    with open("audio.txt", "w") as lab:
        diarization.write_lab(lab)

    # Leer los resultados de la diarización desde el archivo de texto
    with open("audio.txt", "r") as file:
        lines = file.readlines()

    # Procesar los resultados de la diarización
    data = []
    for line in lines:
        parts = line.strip().split()
        data.append(
            {"start": float(parts[0]), "end": float(parts[1]), "speaker": parts[2]}
        )

    # Guardar los resultados de la diarización en un archivo JSON
    with open("audio.json", "w") as file:
        json.dump(data, file, indent=4)

    # Cargar el archivo de audio original usando AudioSegment
    audio_segment = AudioSegment.from_wav(audio_entrada)

    # Leer los resultados de la diarización desde el archivo JSON
    with open("audio.json", "r") as file:
        diarization = json.load(file)

    # Segmentar el audio en base a los resultados de la diarización
    segments = {}
    for segment in diarization:
        speaker = segment["speaker"]
        if speaker not in segments:
            segments[speaker] = []
        start = int(segment["start"] * 1000)
        end = int(segment["end"] * 1000)
        segments[speaker].append(audio_segment[start:end])

    # Transcribir cada segmento de parlante y guardar los resultados
    for speaker, segs in segments.items():
        result = sum(segs, AudioSegment.empty())
        result.export(f"{speaker}.wav", format="wav")
        transcriptions[speaker] = transcribe_wav(f"{speaker}.wav")

    # Eliminar los archivos temporales creados
    os.remove("audio.txt")
    os.remove(audio_entrada)
    os.remove("audio.json")
    for speaker in segments.keys():
        os.remove(f"{speaker}.wav")

    return transcriptions


@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a la API de transcripción de audio!"}


@app.post("/transcribe")
def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe un archivo de audio y devuelve el texto transcrito.

    Parameters:
    :param audio: UploadFile: Archivo de audio a transcribir.

    Returns:
    JSONResponse: Respuesta JSON que contiene el texto transcrito.
    """
    # Guardar el archivo de audio en wav con la libreria AudioSegment
    audio_segment = AudioSegment.from_file(audio.file)
    audio_segment.export("audio.wav", format="wav")
    return segment_and_transcribe_audio("audio.wav")


if __name__ == "__main__":
    subprocess.run(
        ["uvicorn", "app:app", "--host", "localhost", "--port", "7860", "--reload"]
    )
