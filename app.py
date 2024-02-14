from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisperx
import torch
import json
import os

# from google.colab import userdata


hf_token = os.getenv("HF_TOKEN")
device = "cuda"
compute_type = "float16"


# Transcripción
print("Descargando el modelo")
model = whisperx.load_model(
    "large-v2", device, language="es", compute_type=compute_type
)
print("Se descargo el modelo ✓")


def transcribe_mp3(audio):
    resultado = model.transcribe(audio, language="es", batch_size=28)
    return "\n".join([segment["text"] for segment in resultado["segments"]])


def segment_and_transcribe_audio(audio_entrada) -> dict:
    """
    Segment and transcribe the input audio file.

    Parameters:
    audio_entrada: str
        Path to the input audio file.

    Returns:
    dict
        Dictionary containing transcriptions for each speaker segment.
    """
    hf_token_read = hf_token
    transcriptions = {}

    # Transcribe the main audio
    transcripcion_completa = transcribe_mp3(audio_entrada)

    # Add the transcription to the dictionary
    transcriptions = {"full": transcripcion_completa}

    # Rename audio file to audio.mp3
    audio = "audio.mp3"
    os.rename(audio_entrada, audio)

    # Export the segments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token_read
    )
    pipeline.to(torch.device(device))

    # Run the pipeline on an audio file
    diarization = pipeline(audio)

    # Write the diarization result to a text file
    with open("audio.txt", "w") as lab:
        diarization.write_lab(lab)

    # Read the text file
    with open("audio.txt", "r") as file:
        lines = file.readlines()

    # Process the lines
    data = []
    for line in lines:
        parts = line.strip().split()
        data.append(
            {"start": float(parts[0]), "end": float(parts[1]), "speaker": parts[2]}
        )

    # Write the JSON file
    with open("audio.json", "w") as file:
        json.dump(data, file, indent=4)

    # Load the audio file
    audio = AudioSegment.from_mp3(audio)

    # Load the diarization result
    with open("audio.json", "r") as file:
        diarization = json.load(file)

    # Segment the audio
    segments = {"SPEAKER_01": [], "SPEAKER_00": []}
    for segment in diarization:
        # pydub works in milliseconds
        start = int(segment["start"] * 1000)
        end = int(segment["end"] * 1000)
        speaker = segment["speaker"]
        segments[speaker].append(audio[start:end])

    # Transcribe each speaker segment
    for speaker, segs in segments.items():
        result = sum(segs, AudioSegment.empty())
        result.export(f"{speaker}.mp3", format="mp3")
        transcriptions[speaker] = transcribe_mp3(f"{speaker}.mp3")

    # Clean up temporary files
    os.remove("audio.txt")
    os.remove("audio.mp3")
    os.remove("audio.json")
    os.remove("SPEAKER_00.mp3")
    os.remove("SPEAKER_01.mp3")

    return transcriptions
