# Internal import
from speaker_profile import profile_store
from fish_speech.models.dac import inference as DAC
from fish_speech.models.text2semantic import inference as T2C

# External import
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import torchaudio
import time
import sounddevice as sd
from loguru import logger


class TextToSpeech(ABC):
    @abstractmethod
    def text_to_speech(self, text: str):
        pass

    def warmpup(self):
        pass

@dataclass
class TextToSpeechConfig:
    model_path: str = "./models/openaudio-s1-mini"
    speaker_profile: Optional[str] = None
    seed: int = 42
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    top_p: float = 0.8

class OpenAudioTextToSpeech(TextToSpeech):
    def __init__(self, config: TextToSpeechConfig):
        self.config = config
        self.speaker_tokens = None
        self.semantic_tokens = []
        self.prompt_text = ""

        logger.info("Loading model ...")
        t0 = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.half if self.device != "cuda" else torch.bfloat16

        dac_path = os.path.join(config.model_path, "codec.pth")
        self.dac_model = DAC.load_model(config_name="modded_dac_vq", checkpoint_path=dac_path)

        self.t2c_model, self.decode_one_token = T2C.init_model(
            checkpoint_path=config.model_path,
            device=self.device,
            precision=precision,
            compile=True
        )

        with torch.device(self.device):
            self.t2c_model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.t2c_model.config.max_seq_len,
                dtype=next(self.t2c_model.parameters()).dtype,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

        if profile_store.is_profile_exists(config.speaker_profile):
            logger.info(f"Loading speaker profile: {config.speaker_profile}")
            speaker_profile = profile_store.get_profile(config.speaker_profile)
            self._create_speaker_profile(
                ref_voice_path=speaker_profile.voice_path,
                ref_voice_text=speaker_profile.prompt
            )

    def text_to_speech(self, text: str):
        if self.speaker_tokens is None:
            raise ValueError("Speaker profile not created. Call create_speaker_profile first.")

        logger.info(f"Generating semantic tokens for text: {text}")
        self._gen_semantic(text)

        logger.info("Generating audio from semantic tokens")
        audio = self._gen_audio()

        # play audio
        sd.play(audio, samplerate=self.dac_model.sample_rate, blocking=True)

    def warmpup(self):
        if self.speaker_tokens is None:
            raise ValueError("Speaker profile not created. Call create_speaker_profile first.")

        self._gen_semantic("warmpup")
        self._gen_audio()

    def _create_speaker_profile(self, ref_voice_path: str, ref_voice_text: str):
        logger.info(f"Processing in-place reconstruction of {ref_voice_path}")

        # Load audio
        audio, sr = torchaudio.load(str(ref_voice_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, self.dac_model.sample_rate)

        audios = audio[None].to(self.device)
        logger.info(f"Loaded audio with {audios.shape[2] / self.dac_model.sample_rate:.2f} seconds")

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        indices, indices_lens = self.dac_model.encode(audios, audio_lengths)

        if indices.ndim == 3:
            indices = indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        self.speaker_tokens = indices.detach().cpu()
        self.prompt_text = ref_voice_text

    @torch.inference_mode()
    def _gen_semantic(self, text: str):
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

        generator = T2C.generate_long(
            model=self.t2c_model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=0,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            compile=True,
            iterative_prompt=True,
            chunk_length=300,
            prompt_text=self.prompt_text,
            prompt_tokens=self.speaker_tokens,
        )

        idx = 0
        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                logger.info(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    self.semantic_tokens.append(torch.cat(codes, dim=1).detach().cpu())
                    logger.info("Saved codes to self.semantic_tokens")
                logger.info("Next sample")
                codes = []
                idx += 1
            else:
                logger.error(f"Error: {response}")

    @torch.inference_mode()
    def _gen_audio(self):
        logger.info("Processing precomputed indices")

        indices = self.semantic_tokens.pop(0)

        indices = indices.to(self.device).long()
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
