

# Internal import
from pickle import load
import import_secret_key

# External import
from abc import ABC, abstractmethod
import sounddevice
import numpy as np
import sys
import torchaudio

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
            model_dir = "models/cosyvoice2",
            load_jit = False,
            load_trt = False,
            load_vllm = False,
            fp16 = True
        )
        

    def text_to_speech(self, text: str):
        ref_voice = load_wav(wav = "assets/Kafka_Voice_Sample.wav", target_sr = 16000)

        instruct_text = "감정을 살려서 말하세요"
        for i, j in enumerate(self.model.inference_instruct2(
            tts_text = text,
            instruct_text = instruct_text,
            prompt_speech_16k = ref_voice,
            stream=False
        )):
            torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], self.model.sample_rate)
