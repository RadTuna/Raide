

# Internal import
import import_secret_key

# External import
from openai import OpenAI
from abc import ABC, abstractmethod
import sounddevice
import numpy as np

class TextToSpeech(ABC):
    @abstractmethod
    def text_to_speech(self, text: str):
        pass

class OpenAITextToSpeech(TextToSpeech):
    def __init__(self):
        self.client = OpenAI()
        self.speech = self.client.audio.speech
        self.chunk_size = 2048

    def text_to_speech(self, text: str):
        response = self.speech.create(model="tts-1", voice="nova", input=text, response_format="pcm")
        print("TTS Response 받음")
        with sounddevice.OutputStream(
            channels=1,
            samplerate=24000,
            dtype=np.int16
        ) as stream:
            for chunk in response.iter_bytes(self.chunk_size):
                stream.write(np.frombuffer(chunk, dtype=np.int16))

            stream.stop()
