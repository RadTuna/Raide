
# Internal imports
from text_to_speech import OpenAudioTextToSpeech
from llm import LocalLanguageModel, LanguageModelConfig
import RealtimeSTT

# External imports
import multiprocessing
import time

enable_tts = True

def main():
    tts = OpenAudioTextToSpeech()

    ref_voice_text = "날 도와줘서 정말 고마워. 그 덕분에 네게 이 낙원을 소개할 수 있었어. 내가 이곳에 속하지 않는다 해도 이 세계는 날 받아주려고 해. 그래서 내가 사랑하는 이 세계를… 다른 사람과 공유하고 싶었어."
    tts.create_speaker_profile(ref_voice_path="./assets/Firefly_KR.wav", ref_voice_text=ref_voice_text)
    tts.warmpup()

    asr = RealtimeSTT.AudioToTextRecorder(
            model_path="./models/sensevoice_small",
            silero_use_onnx=True,
            silero_deactivity_detection=True,
        )

    config = LanguageModelConfig() # default config
    config.temperature = 1.0
    config.top_k = 64
    llm = LocalLanguageModel(model_path="./models/gemma3/gemma-3-4b-it-Q4_K_M.gguf", config=config)

    while True:
        asr.wait_audio()
        recognized_text = asr.transcribe()

        print(f"User: {recognized_text}")

        chunk_list = []
        print("AI: ", end="")
        for chunk in llm.chat_sync(recognized_text):
            print(chunk, end="")
            chunk_list.append(chunk)
        print("")

        full_message = "".join(chunk_list)
        
        if len(full_message) > 0 and enable_tts:
            tts.text_to_speech(full_message)
            time.sleep(1)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
