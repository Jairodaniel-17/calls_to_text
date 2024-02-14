import json
import os
from model.credentials import CredentialsHF
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment


class AudioSegmentation:
    def __init__(self, credentials: CredentialsHF, device=None):
        self.token = credentials.token
        self.model_name = credentials.model_name
        self.device = device if device else (0 if torch.cuda.is_available() else "cpu")
        self.pipeline = Pipeline.from_pretrained(
            self.model_name, use_auth_token=self.token
        )
        self.pipeline.to(torch.device(self.device))

    def rename_audio(self, audio_entrada):
        audio = "audio.mp3"
        os.rename(audio_entrada, audio)
        return audio

    def run_pipeline(self, audio):
        diarization = self.pipeline(audio)
        return diarization

    def save_to_lab(self, diarization, output_file):
        with open(output_file, "w") as lab:
            diarization.write_lab(lab)

    def process_lines(self, lines):
        data = []
        for line in lines:
            parts = line.strip().split()
            data.append(
                {"start": float(parts[0]), "end": float(parts[1]), "speaker": parts[2]}
            )
        return data

    def write_json(self, data, output_file):
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)

    def load_audio(self, audio_file):
        audio = AudioSegment.from_mp3(audio_file)
        return audio

    def segment_audio(self, audio, diarization):
        segments = {"SPEAKER_01": [], "SPEAKER_00": []}
        for segment in diarization:
            start = int(segment["start"] * 1000)
            end = int(segment["end"] * 1000)
            speaker = segment["speaker"]
            segments[speaker].append(audio[start:end])
        return segments

    def export_segments(self, segments):
        for speaker, segs in segments.items():
            result = sum(segs, AudioSegment.empty())
            result.export(f"{speaker}.mp3", format="mp3")

    def remove_files(self, *files):
        for file in files:
            os.remove(file)

    def segment(self, audio_entrada):
        audio = self.rename_audio(audio_entrada)
        diarization = self.run_pipeline(audio)
        self.save_to_lab(diarization, "audio.txt")

        with open("audio.txt", "r") as file:
            lines = file.readlines()

        data = self.process_lines(lines)
        self.write_json(data, "audio.json")

        audio = self.load_audio(audio)
        with open("audio.json", "r") as file:
            diarization = json.load(file)

        segments = self.segment_audio(audio, diarization)
        self.export_segments(segments)
        self.remove_files("audio.txt", "audio.mp3", "audio.json")

        return "SPEAKER_00.mp3 SPEAKER_01.mp3"
