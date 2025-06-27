# Internal Imports
import utils

# External Imports
import gradio as gr
from dataclasses import dataclass
import io
import tempfile
from pydub import AudioSegment
import numpy as np
from websockets.asyncio import client
import json
from loguru import logger

class VoiceChatFrontend:
    DEFAULT_WEBSOCKET_MAX_SIZE = 100 * 1024 * 1024

    @dataclass
    class AppState:
        websocket: client.ClientConnection | None = None
        stream: np.ndarray | None = None
        sampling_rate: int = 0
        conversation: list = None
        started_talking: bool = False
        response_audio_buffer: bytes | None = None
        chunk_sent: bool = False

        def __post_init__(self):
            if self.conversation is None:
                self.conversation = []

    def run(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    input_audio = gr.Audio(label="Input Audio", sources="microphone", type="numpy", streaming=True)
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

    async def _process_audio(self, audio: tuple, state: AppState):
        if audio[1] is None or len(audio[1]) == 0:
            return None, state
        
        if state.stream is None:
            state.stream = audio[1]
            state.sampling_rate = audio[0]
            state.started_talking = True
        else:
            state.stream = np.concatenate((state.stream, audio[1]))
        
        if state.started_talking:
            current_chunk = audio[1]
            audio_buffer = io.BytesIO()
            sample_width = current_chunk.dtype.itemsize

            segment = AudioSegment(
                current_chunk.tobytes(),
                frame_rate=audio[0],
                sample_width=sample_width,
                channels=(1 if len(current_chunk.shape) == 1 else current_chunk.shape[1]),
            )
            
            segment.export(audio_buffer, format="wav")

            pcm_bytes, pcm_sample_width, pcm_sample_rate, _ = utils.wav_bytes_to_pcm(
                audio_buffer.getvalue(), new_sample_rate=16000, mono=True
            )

            await self._send_audio_chunk(pcm_bytes, pcm_sample_rate, pcm_sample_width, state)

        return None, state

    async def _send_audio_chunk(self, pcm_bytes: bytes, sample_rate: int, sample_width: int, state: AppState):
        try:
            websocket = await self._ensure_websocket(state)

            if not state.chunk_sent:
                header_dict = {
                    "type": "request_audio",
                    "sample_rate": sample_rate,
                    "sample_width": sample_width,
                }
                await websocket.send(json.dumps(header_dict))
                state.chunk_sent = True

            await websocket.send(pcm_bytes)
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")

    async def _response(self, state: AppState):
        if not state.started_talking:
            return None, self.AppState()

        state.started_talking = False
        request_segment = AudioSegment(
            state.stream.tobytes(),
            frame_rate=state.sampling_rate,
            sample_width=state.stream.dtype.itemsize,
            channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
        )

        request_buffer = io.BytesIO()
        request_segment.export(request_buffer, format="wav")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_file:
            in_file.write(request_buffer.getvalue())

        state.conversation.append({"role": "user",
                                    "content": {"path": in_file.name,
                                    "mime_type": "audio/wav"}})
        state.stream = None

        response_audio, response_sample_rate, response_sample_width = await self._receive_response(state)
        if utils.sample_width_to_dtype(response_sample_width) == np.float32:
            response_audio = (np.frombuffer(response_audio, dtype=np.float32) * 32767).astype(np.int16).tobytes()
            response_sample_width = 2

        response_buffer = io.BytesIO()
        if response_audio:
            response_segment = AudioSegment(
                response_audio,
                frame_rate=response_sample_rate,
                sample_width=response_sample_width,
                channels=1,
            )
            response_segment.export(response_buffer, format="wav")
        else:
            empty_segment = AudioSegment.silent(duration=500, frame_rate=16000)
            empty_segment.export(response_buffer, format="wav")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_file.write(response_buffer.getvalue())
        
        state.conversation.append({"role": "assistant",
                                    "content": {"path": out_file.name,
                                    "mime_type": "audio/wav"}})

        return None, state

    async def _receive_response(self, state: AppState):
        try:
            websocket = await self._ensure_websocket(state)
            current_sample_rate = 16000
            current_sample_width = 2
            while True:
                recv_data = await websocket.recv()
                if isinstance(recv_data, str):
                    recv_dict = json.loads(recv_data)
                    if recv_dict.get("type") == "response_audio":
                        if recv_dict.get("sample_rate"):
                            current_sample_rate = recv_dict["sample_rate"]

                        if recv_dict.get("sample_width"):
                            current_sample_width = recv_dict["sample_width"]

                        recv_audio_data = await websocket.recv()
                        if isinstance(recv_audio_data, bytes):
                            return recv_audio_data, current_sample_rate, current_sample_width

        except Exception as e:
            logger.error(f"Error receiving response: {e}")

        return None, None, None

    def _start_recording_user(self, state: AppState):
        return gr.Audio(recording=True)

    async def _ensure_websocket(self, state: AppState):
        if state.websocket is None:
            state.websocket = await client.connect("ws://localhost:8765", max_size=self.DEFAULT_WEBSOCKET_MAX_SIZE)
            logger.info("WebSocket connection established.")
        return state.websocket
