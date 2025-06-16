
import os
import sounddevice
import multiprocessing

# realtime stt imports
import RealtimeSTT


def test_asr_realtime_stt():
    recorder = RealtimeSTT.AudioToTextRecorder(
        model_path="./third_party/RealtimeSTT/models/sensevoice_small",
        silero_use_onnx=True,
        silero_deactivity_detection=True,
    )

    while True:
       text = recorder.text()
       print(f"ASR: {text}")


if __name__ == '__main__':
    current_working_directory = os.getcwd()
    print(f"CurWD: {current_working_directory}")

    multiprocessing.freeze_support()  # Windows에서 필요

    test_asr_realtime_stt()