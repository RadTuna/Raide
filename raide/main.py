
# Internal imports
from text_to_speech import OpenAITextToSpeech
from llm import LocalLanguageModel, LanguageModelConfig

# External imports
import RealtimeSTT
import multiprocessing

enable_tts = False

def main():
    asr = RealtimeSTT.AudioToTextRecorder(
            model_path="./models/RealtimeSTT/models/sensevoice_small",
            silero_use_onnx=True,
            silero_deactivity_detection=True,
        )

    config = LanguageModelConfig() # default config
    config.temperature = 1.0
    config.top_k = 64
    llm = LocalLanguageModel(model_path="./models/gemma3/gemma-3-4b-it-Q4_K_M.gguf", config=config)
    tts = OpenAITextToSpeech()

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

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
