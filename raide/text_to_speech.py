

# Internal import
from pickle import load
import import_secret_key

# External import
from typing import Optional
from abc import ABC, abstractmethod
import sounddevice
import numpy as np
import torch
import torchaudio
import time
import sounddevice as sd
from loguru import logger

class TextToSpeech(ABC):
    @abstractmethod
    def text_to_speech(self, text: str, prompt_text: Optional[str] = None):
        pass

    def create_speaker_profile(self, ref_voice_path: str):
        pass


from openai import OpenAI

class OpenAITextToSpeech(TextToSpeech):
    def __init__(self):
        self.client = OpenAI()
        self.speech = self.client.audio.speech
        self.chunk_size = 2048

    def text_to_speech(self, text: str, prompt_text: Optional[str] = None):
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


from fish_speech.models.dac import inference as DAC
from fish_speech.models.text2semantic import inference as T2C

class OpenAudioTextToSpeech(TextToSpeech):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dac_model = DAC.load_model(config_name="modded_dac_vq", checkpoint_path="./models/openaudio-s1-mini/codec.pth")
        self.text2semantic_model_path = "./models/openaudio-s1-mini"

        self.speaker_tokens = None
        self.semantic_tokens = []

    def text_to_speech(self, text: str, prompt_text: Optional[str] = None):
        if self.speaker_tokens is None:
            raise ValueError("Speaker profile not created. Call create_speaker_profile first.")

        logger.info(f"Generating semantic tokens for text: {text}")
        self._gen_semantic(text, prompt_text)

        logger.info("Generating audio from semantic tokens")
        audio = self._gen_audio()

        # play audio
        sd.play(audio, samplerate=self.dac_model.sample_rate, blocking = True)

    def create_speaker_profile(self, ref_voice_path: str):
        logger.info(f"Processing in-place reconstruction of {ref_voice_path}")

        # Load audio
        audio, sr = torchaudio.load(str(ref_voice_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, self.dac_model.sample_rate)

        audios = audio[None].to(self.device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / self.dac_model.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        indices, indices_lens = self.dac_model.encode(audios, audio_lengths)

        if indices.ndim == 3:
            indices = indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        self.speaker_tokens = indices.cpu().numpy()

    @torch.inference_mode()
    def _gen_semantic(self, text: str, prompt_text: Optional[str] = None):
        precision = torch.half if self.device != "cuda" else torch.bfloat16

        if prompt_text is not None and len(prompt_text) != len(self.speaker_tokens):
            raise ValueError(
                f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(self.speaker_tokens)}) should be the same"
            )

        logger.info("Loading model ...")

        t0 = time.time()
        model, decode_one_token = T2C.init_model(
            checkpoint_path = self.text2semantic_model_path,
            device = self.device,
            precision = precision,
            compile = True
        )

        with torch.device(self.device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        if self.speaker_tokens is not None:
            prompt_tokens = [torch.from_numpy(t) for t in self.speaker_tokens]

        #torch.manual_seed(seed)
        #if torch.cuda.is_available():
            #torch.cuda.manual_seed(seed)

        generator = T2C.generate_long(
            model = model,
            device = self.device,
            decode_one_token = decode_one_token,
            text = text,
            num_samples = 1,
            max_new_tokens = 0,
            top_p = 0.8,
            repetition_penalty = 1.1,
            temperature = 0.8,
            compile = True,
            iterative_prompt = True,
            chunk_length = 300,
            prompt_text = prompt_text,
            prompt_tokens = prompt_tokens,
        )

        idx = 0
        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                logger.info(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    self.semantic_tokens.append(torch.cat(codes, dim=1).cpu().numpy())
                    logger.info(f"Saved codes to self.semantic_tokens")
                logger.info(f"Next sample")
                codes = []
                idx += 1
            else:
                logger.error(f"Error: {response}")

    @torch.inference_mode()
    def _gen_audio(self):
        logger.info(f"Processing precomputed indices")

        indices = self.semantic_tokens[0]

        indices = torch.from_numpy(indices).to(self.device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        indices_lens = torch.tensor([indices.shape[1]], device=self.device, dtype=torch.long)

        audios, audio_lengths = self.dac_model.decode(indices, indices_lens)
        audio_time = audios.shape[-1] / self.dac_model.sample_rate

        logger.info(
            f"Generated audio of shape {audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
        )

        # return audio
        audio = audios[0, 0].detach().float().cpu().numpy()
        return audio
