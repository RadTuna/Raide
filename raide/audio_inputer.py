
import sys
import sounddevice
import numpy as np

from langchain_core.documents.base import Blob

class AudioData:
    def __init__(self, wave: np.ndarray, sample_rate: int, channels: int):
        self.wave: np.ndarray = wave
        self.sample_rate: int = sample_rate
        self.channels: int = channels


class AudioInputer:
    def __init__(self, vol_threshold: float = 0.1, sample_rate: int = 16000, channels: int = 1):
        self.vol_threshold: float = vol_threshold
        self.sample_rate: int = sample_rate
        self.channels: int = channels

        # temp limite
        self.silence_duration = 1.5

    def get_audio_from_mic(self) -> AudioData:
        recording = []
        silent_count = 0
        recording_active = False
        
        with sounddevice.InputStream(samplerate = self.sample_rate, channels = self.channels, dtype = np.float32) as stream:
            while True:
                audio_chunk, _ = stream.read(int(self.sample_rate * 0.1))
                volume = np.max(np.abs(audio_chunk))
                
                self.__print_volume(volume)

                if volume > self.vol_threshold:
                    if not recording_active:
                        print("🔴 녹음 시작!")
                        recording_active = True
                        recording = []
                        recording.append(np.zeros(shape=(int(self.sample_rate * self.silence_duration), 1), dtype=np.float32))
                    
                    silent_count = 0
                elif recording_active:
                    silent_count += 1
                    
                    if silent_count > self.silence_duration * 10:
                        print("🛑 녹음 종료!")
                        break

                if recording_active:
                    recording.append(audio_chunk)
        
        audio_date = np.array([])
        if recording:
            audio_data = np.concatenate(recording, axis=0)

        return AudioData(wave = audio_data, sample_rate = self.sample_rate, channels = self.channels)

    def __print_volume(self, volume: float):
        bar_length = 50
        filled_length = int(bar_length * volume)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        sys.stdout.write(f"\r🎤 볼륨: |{bar}| {volume:.2f}")
        sys.stdout.flush()
