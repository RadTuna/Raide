
# Internal imports
from text_to_speech import OpenAudioTextToSpeech, TextToSpeechConfig
from llm import LocalLanguageModel, LanguageModelConfig
import RealtimeSTT
import log
from config import config

# External imports
import multiprocessing
import time
from loguru import logger


def main():
    log.init_logger()
    config.load("./config")

    tts_config = config.to_tts_config()
    tts = OpenAudioTextToSpeech(config=tts_config)
    tts.warmpup()

    asr = RealtimeSTT.AudioToTextRecorder(
            model_path=config.asr.model_path,
            silero_use_onnx=True,
            silero_deactivity_detection=True,
        )

    llm_config = config.to_llm_config()
    llm = LocalLanguageModel(config=llm_config)

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
