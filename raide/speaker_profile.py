from omegaconf import OmegaConf
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SpeakerProfile:
    name: str
    lang: str
    voice_path: str
    voice_transcript: str
    system_prompt: str
    persona: str

class SpeakerProfileStore:
    LANG_TOKEN = "{{lang}}"

    def __init__(self):
        profile_files = list(Path("./speaker_profiles").glob("*.yaml"))

        self.profiles = {}
        for profile_file in profile_files:
            profile_data = OmegaConf.load(profile_file)

            profile = SpeakerProfile(
                name=profile_data.name,
                lang=profile_data.lang,
                voice_path=profile_data.voice_path,
                voice_transcript=profile_data.voice_transcript,
                system_prompt=profile_data.system_prompt,
                persona=profile_data.persona
            )

            profile = self._replace_special_tokens(profile)
            self.profiles[profile.name] = profile

    def get_profile(self, name: str) -> SpeakerProfile:
        if name in self.profiles:
            return self.profiles[name]
        else:
            raise ValueError(f"Profile '{name}' not found.")
    
    def is_profile_exists(self, name: str) -> bool:
        return name in self.profiles

    def _replace_special_tokens(self, profile: SpeakerProfile) -> SpeakerProfile:
        profile.system_prompt = profile.system_prompt.replace(self.LANG_TOKEN, profile.lang)
        profile.persona = profile.persona.replace(self.LANG_TOKEN, profile.lang)
        return profile

profile_store = SpeakerProfileStore()
