
# Internal imports
from text_to_speech import OpenAudioTextToSpeech
from llm import LocalLanguageModel, LanguageModelConfig
import RealtimeSTT
import log

# External imports
import multiprocessing
import time
from loguru import logger


def main():
    log.init_logger()

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
    llm = LocalLanguageModel(model_path="./models/gemma3/gemma-3-4b-it-q4_0.gguf", config=config)

    while True:
        asr.wait_audio()
        recognized_text = asr.transcribe()

        logger.info(f"User: {recognized_text}")

        chunk_list = []
        for chunk in llm.chat_sync(recognized_text):
            chunk_list.append(chunk)

        full_message = "".join(chunk_list)
        logger.info(f"AI: {full_message}")
        
        if len(full_message) > 0:
            tts.text_to_speech(full_message)
            time.sleep(1)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
