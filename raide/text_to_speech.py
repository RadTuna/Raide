

# Internal import
from pickle import load
import import_secret_key

# External import
from abc import ABC, abstractmethod
import sounddevice
import numpy as np
import sys
import torchaudio
import sounddevice as sd

class TextToSpeech(ABC):
    @abstractmethod
    def text_to_speech(self, text: str):
        pass


from openai import OpenAI

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


from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

class CosyVoiceTextToSpeech(TextToSpeech):
    def __init__(self):
        self.model = CosyVoice2(
            model_dir = "./models/cosyvoice2",
            load_jit = True,
            load_trt = False,
            load_vllm = False,
            fp16 = True
        )
        

    def text_to_speech(self, text: str):
        ref_voice = load_wav(wav = "./assets/Kafka_Voice_Sample.wav", target_sr = 16000)

        with sd.OutputStream(
            samplerate = self.model.sample_rate,
            channels = 1,
            dtype = np.float32
        ) as stream:
            for out in self.model.inference_cross_lingual(
                tts_text = text,
                prompt_speech_16k = ref_voice,
                stream=True,
                speed = 0.7
            ):
                audio = out['tts_speech'].numpy()
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio = audio[0]
                stream.write(audio)
