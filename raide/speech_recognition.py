
# Internal import
import import_secret_key
from audio_inputer import AudioData

# External import
from openai import OpenAI
from io import BytesIO
from abc import ABC, abstractmethod
import wave
import numpy as np

class AutomaticSpeechRecognition(ABC):
    @abstractmethod
    def recognize(self, audio_data: AudioData) -> str:
        pass

class OpenAIWhiserASR(AutomaticSpeechRecognition):
    def __init__(self):
        self.client = OpenAI()
        self.transcriptor = self.client.audio.transcriptions

    def recognize(self, audio_data: AudioData) -> str:
        with self.__create_virtual_audio_file(audio_data) as audio_file:
            transcription = self.transcriptor.create(
                model = "whisper-1",
                file = ("audio.wav", audio_file, "audio/wav"),
                response_format="text"
            )

        return str(transcription)

    def __create_virtual_audio_file(self, audio_data: AudioData) -> BytesIO:
        audio_file = BytesIO()
        with wave.open(audio_file, "wb") as wav_file:
            wav_file.setnchannels(audio_data.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(audio_data.sample_rate)

            # convert float32 to int16
            audio_int16 = (audio_data.wave * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        audio_file.seek(0)
        return audio_file
