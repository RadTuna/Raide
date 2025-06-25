# Internal Imports
from llm import LanguageModelConfig
from text_to_speech import TextToSpeechConfig

# External Imports
from omegaconf import OmegaConf
import os


class Config:
    def __init__(self):
        self.config = OmegaConf.create({
            "asr": {
                "model_path": "./models/sensevoice_small"
            },
            "llm": {
                "model_path": "./models/gemma3/gemma-3-4b-it-q4_0.gguf",
                "context_window": 2048,
                "output_max_tokens": 512,
                "temperature": 1.0,
                "repeat_penalty": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "min_p": 0.01
            },
            "tts": {
                "model_path": "./models/openaudio-s1-mini",
                "speaker_profile": "kr_firefly",
                "seed": 42,
                "temperature": 0.8,
                "repetition_penalty": 1.1,
                "top_p": 0.8
            }
        })

    def __getattr__(self, name):
        return self.config.get(name)

    def load(self, config_dir: str):
        paths = {
            "asr": os.path.join(config_dir, "asr_config.yaml"),
            "llm": os.path.join(config_dir, "llm_config.yaml"),
            "tts": os.path.join(config_dir, "tts_config.yaml")
        }

        for section, path in paths.items():
            if os.path.exists(path):
                loaded_cfg = OmegaConf.load(path)
                self.config[section] = OmegaConf.merge(self.config.get(section, {}), loaded_cfg)

    def to_llm_config(self) -> LanguageModelConfig:
        return LanguageModelConfig(
            model_path=self.config.llm.model_path,
            speaker_profile=self.config.tts.speaker_profile,
            context_window=self.config.llm.context_window,
            output_max_tokens=self.config.llm.output_max_tokens,
            temperature=self.config.llm.temperature,
            repeat_penalty=self.config.llm.repeat_penalty,
            top_k=self.config.llm.top_k,
            top_p=self.config.llm.top_p,
            min_p=self.config.llm.min_p
        )
    
    def to_tts_config(self) -> TextToSpeechConfig:
        return TextToSpeechConfig(
            model_path=self.config.tts.model_path,
            speaker_profile=self.config.tts.speaker_profile,
            seed=self.config.tts.seed,
            temperature=self.config.tts.temperature,
            repetition_penalty=self.config.tts.repetition_penalty,
            top_p=self.config.tts.top_p
        )


# declare a global config instance
config = Config()
