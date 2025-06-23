
from omegaconf import OmegaConf
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SpeakerProfile:
    name: str
    lang: str
    voice_path: str
    prompt: str

class SpeakerProfileStore:
    def __init__(self):
        profile_files = list(Path("./speaker_profiles").glob("*.json"))

        self.profiles = {}
        for profile_file in profile_files:
            profile_data = OmegaConf.load(profile_file)
            profile = SpeakerProfile(
                name=profile_data.name,
                lang=profile_data.lang,
                voice_path=profile_data.voice_path,
                prompt=profile_data.prompt
            )

            self.profiles[profile.name] = profile

    def get_profile(self, name: str) -> SpeakerProfile:
        if name in self.profiles:
            return self.profiles[name]
        else:
            raise ValueError(f"Profile '{name}' not found.")
        
    def is_profile_exists(self, name: str) -> bool:
        return name in self.profiles
            

profile_store = SpeakerProfileStore()
