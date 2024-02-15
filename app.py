import os
import json
import subprocess
import torch
import whisperx

from pydub import AudioSegment
from pyannote.audio import Pipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub.utils import mediainfo

# importar libreria para cargar el archivo .env
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

# Tipo de cómputo
compute_type = "float32"

# Transcripción
print("Descargando el modelo...")
model = whisperx.load_model(
    "large-v2", device, language="es", compute_type=compute_type
)
print("¡Modelo descargado con éxito!")


def transcribe_mp3(audio):
    """
    Transcribe un archivo de audio MP3 y devuelve el texto transrito.
    """
    resultado = model.transcribe(audio, language="es", batch_size=28)
    return "\n".join([segment["text"] for segment in resultado["segments"]])


def reparar_audio(input_audio, output_audio):
    """
    Repara un archivo de audio utilizando FFmpeg y convirtiéndolo a formato WAV.

    Parameters:
    :param: input_audio (str): Ruta al archivo de audio de entrada dañado.
    :param: output_audio (str): Ruta al archivo de audio de salida reparado.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_audio,
                "-vn",
                "-c:a",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                output_audio,
            ],
            check=True,
        )
        print(
            "El archivo de audio ha sido reparado y convertido a formato WAV exitosamente."
        )
    except subprocess.CalledProcessError as e:
        print("Ocurrió un error al intentar reparar el archivo de audio:", e)


def segment_and_transcribe_audio(audio_entrada) -> dict:
    """
    Segmenta y transcribe el archivo de audio de entrada.

    Parameters:
    :param: audio_entrada (str): Ruta al archivo de audio de entrada.

    Returns:
    dict
        Diccionario que contiene las transcripciones para cada segmento de parlante.
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
    # reparar audio si es necesario
    try:
        AudioSegment.from_file(audio_entrada)
    except Exception as e:
        print("El archivo de audio está dañado. Intentando reparar...")
        reparar_audio(audio_entrada, "repaired_audio.wav")
        audio_entrada = "repaired_audio.wav"

    transcripcion_completa = transcribe_mp3(audio_entrada)
    transcriptions["full"] = transcripcion_completa

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    pipeline.to(torch.device(device))
    diarization = pipeline(audio_entrada)

    with open("audio.txt", "w") as lab:
        diarization.write_lab(lab)

    with open("audio.txt", "r") as file:
        lines = file.readlines()

    data = []
    for line in lines:
        parts = line.strip().split()
        data.append(
            {"start": float(parts[0]), "end": float(parts[1]), "speaker": parts[2]}
        )

    with open("audio.json", "w") as file:
        json.dump(data, file, indent=4)

    audio = AudioSegment.from_file(audio_entrada)

    with open("audio.json", "r") as file:
        diarization = json.load(file)

    segments = {}
    for segment in diarization:
        speaker = segment["speaker"]
        if speaker not in segments:
            segments[speaker] = []
        start = int(segment["start"] * 1000)
        end = int(segment["end"] * 1000)
        segments[speaker].append(audio[start:end])

    for speaker, segs in segments.items():
        result = sum(segs, AudioSegment.empty())
        result.export(f"{speaker}.wav", format="wav")
        transcriptions[speaker] = transcribe_mp3(f"{speaker}.wav")

    os.remove("audio.txt")
    os.remove(audio_entrada)
    os.remove("audio.json")
    for speaker in segments.keys():
        os.remove(f"{speaker}.wav")
    print(
        f"Transcripción completada. {len(transcriptions)} segmentos transcritos. {transcriptions}"
    )
    return transcriptions


@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a la API de transcripción de audio!"}


@app.get("/transcribe")
def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe un archivo de audio y devuelve el texto transrito.
    """
    # Guardar el archivo de audio
    with open("audio.mp3", "wb") as file:
        file.write(audio.file.read())
    return segment_and_transcribe_audio("audio.mp3")


@app.post("/transcribe")
def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe un archivo de audio y devuelve el texto transrito.
    """
    # Guardar el archivo de audio
    with open("audio.mp3", "wb") as file:
        file.write(audio.file.read())
    # return segment_and_transcribe_audio("audio.mp3")
    result = segment_and_transcribe_audio("audio.mp3")
    return JSONResponse(content=result)


if __name__ == "__main__":
    # ejecutar el comando uvicorn app:app --reload
    import subprocess

    subprocess.run(
        ["uvicorn", "app:app", "--host", "localhost", "--port", "7860", "--reload"]
    )
