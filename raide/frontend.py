import gradio as gr
from dataclasses import dataclass
import io
import tempfile
from pydub import AudioSegment
import numpy as np

class VoiceChatFrontend:
    @dataclass
    class AppState:
        stream: np.ndarray | None = None
        sampling_rate: int = 0
        pause_detected: bool = False
        conversation: list = None
        started_talking: bool = False
        def __post_init__(self):
            if self.conversation is None:
                self.conversation = []

    def run(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    input_audio = gr.Audio(
                        label="Input Audio", sources="microphone", type="numpy"
                    )
                with gr.Column():
                    chatbot = gr.Chatbot(label="Conversation", type="messages")
                    output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
            state = gr.State(value=self.AppState())

            stream = input_audio.stream(
                self._process_audio,
                [input_audio, state],
                [input_audio, state],
                stream_every=0.5,
                time_limit=30,
            )
            respond = input_audio.stop_recording(
                self._response,
                [state],
                [output_audio, state]
            )
            respond.then(lambda s: s.conversation, [state], [chatbot])

            restart = output_audio.stop(
                self._start_recording_user,
                [state],
                [input_audio]
            )

        demo.launch()

    def _determine_pause(self, audio: np.ndarray, sampling_rate: int, state: 'VoiceChatFrontend.AppState') -> bool:
        window_size = int(sampling_rate * 0.5)
        if len(audio) < window_size:
            return False
        recent_audio = audio[-window_size:]
        energy = np.mean(np.abs(recent_audio))
        return energy < 0.1

    def _process_audio(self, audio: tuple, state: 'VoiceChatFrontend.AppState'):
        if state.stream is None:
            state.stream = audio[1]
            state.sampling_rate = audio[0]
            state.started_talking = True
        else:
            state.stream = np.concatenate((state.stream, audio[1]))

        pause_detected = self._determine_pause(state.stream, state.sampling_rate, state)
        state.pause_detected = pause_detected

        if state.pause_detected and state.started_talking:
            return gr.Audio(recording=False), state
        return None, state

    def _response(self, state: 'VoiceChatFrontend.AppState'):
        if not state.pause_detected and not state.started_talking:
            return None, self.AppState()
        
        audio_buffer = io.BytesIO()

        segment = AudioSegment(
            state.stream.tobytes(),
            frame_rate=state.sampling_rate,
            sample_width=state.stream.dtype.itemsize,
            channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
        )
        segment.export(audio_buffer, format="wav")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_buffer.getvalue())
        
        state.conversation.append({"role": "user",
                                    "content": {"path": f.name,
                                    "mime_type": "audio/wav"}})
        
        output_buffer = b""

        for mp3_bytes in self._speaking(audio_buffer.getvalue()):
            output_buffer += mp3_bytes
            yield mp3_bytes, state

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(output_buffer)
        
        state.conversation.append({"role": "assistant",
                        "content": {"path": f.name,
                                    "mime_type": "audio/mp3"}})
        yield None, self.AppState(conversation=state.conversation)

    def _start_recording_user(self, state: 'VoiceChatFrontend.AppState'):
        return gr.Audio(recording=True)

    def _speaking(self, wav_bytes: bytes):
        """
        입력된 wav 오디오를 무시하고, 1초짜리 무음 mp3 바이트를 chunk 단위로 yield하는 mock 함수입니다.
        실제 TTS/VC 엔진이 준비되기 전 테스트용입니다.
        """
        silence = AudioSegment.silent(duration=1000)  # 1초
        mp3_buffer = io.BytesIO()
        silence.export(mp3_buffer, format="mp3")
        mp3_bytes = mp3_buffer.getvalue()
        chunk_size = 1024
        for i in range(0, len(mp3_bytes), chunk_size):
            yield mp3_bytes[i:i+chunk_size]

