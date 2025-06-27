from pydub import AudioSegment
import numpy as np
import io
import wave

def wav_bytes_to_pcm(
        wav_bytes: bytes,
        new_sample_rate: int = None,
        mono: bool = False) -> tuple[bytes, int, int, int]:
    audio = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")

    if new_sample_rate and new_sample_rate != audio.frame_rate:
        audio = audio.set_frame_rate(new_sample_rate)

    if mono and audio.channels > 1:
        audio = audio.set_channels(1)

    pcm_bytes = audio.raw_data
    return pcm_bytes, audio.sample_width, audio.frame_rate, audio.channels

def pcm_to_wav_bytes(
        pcm_bytes: bytes,
        sample_rate: int,
        sample_width: int,
        channels: int = 1) -> bytes:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_buffer.getvalue()

def sample_width_to_dtype(sample_width: int) -> np.dtype:
    if sample_width == 1:
        return np.uint8
    elif sample_width == 2:
        return np.int16
    elif sample_width == 3:
        return np.int32
    elif sample_width == 4:
        return np.float32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")