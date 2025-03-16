
import os
import sounddevice
from audio_inputer import AudioData

# open ai imports
from speech_recognition import OpenAIWhiserASR
from audio_inputer import AudioInputer

# realtime stt imports
import RealtimeSTT

def play_audio(audio_data: AudioData):
    sounddevice.play(audio_data.wave, samplerate=audio_data.sample_rate)
    sounddevice.wait()

def test_asr_openai_api():
    asr = OpenAIWhiserASR()

    audio_inputer = AudioInputer()
    audio_data = audio_inputer.get_audio_from_mic()

    play_audio(audio_data)

    result = asr.recognize(audio_data = audio_data)
    print(result)

def test_asr_funasr_local():
    model_path = "./models/sensevoice_small"
    # The default is to infer on the CPU
    model = SenseVoiceSmall(model_dir=model_path, batch_size=1, quantize=True)

    audio_inputer = AudioInputer()
    audio_data = audio_inputer.get_audio_from_mic()

    play_audio(audio_data)

    result = model(wav_content=audio_data.wave.flatten(), language="auto", textnorm="withitn")
    print([rich_transcription_postprocess(i) for i in result])


current_working_directory = os.getcwd()
print(f"CurWD: {current_working_directory}")

#test_asr_openai_api()
test_asr_funasr_local()
