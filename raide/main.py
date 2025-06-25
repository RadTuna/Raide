# Internal imports
from text_to_speech import OpenAudioTextToSpeech, TextToSpeechConfig
from llm import LocalLanguageModel, LanguageModelConfig
import RealtimeSTT
import log
from config import config
from frontend import VoiceChatFrontend

# External imports
import multiprocessing as mp
import time
from loguru import logger
import argparse


def main(use_websocket: bool = False):
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

def run_web_frontend():
    VoiceChatFrontend().run()

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Raide AI Assistant")
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the application: 'cli' or 'web'")

    args = parser.parse_args()
 
    if args.mode == "cli":
        use_websocket = False
    elif args.mode == "web":
        logger.info("Launching web interface...")
        mp.Process(target=run_web_frontend).start()
        use_websocket = True
        time.sleep(2)
    else:
        logger.error("Invalid mode specified. Use 'cli' or 'web'.")
        exit(1)

    main(use_websocket=use_websocket)
