# Internal Imports
from text_to_speech import OpenAudioTextToSpeech, TextToSpeechConfig
from llm import LocalLanguageModel, LanguageModelConfig
import RealtimeSTT
import log
from config import config
import utils

# External Imports
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import asyncio
import sounddevice as sd
from contextlib import contextmanager
from websockets.asyncio import server
import json
import numpy as np


class SendPacketType(Enum):
    START_RESPONSE = "start_response"
    TRANSCRIPT = "transcript"
    RESPONSE_TEXT = "response_text"
    RESPONSE_AUDIO = "response_audio"
    END_RESPONSE = "end_response"
    ACK_STATUS = "ack_status"

@dataclass
class SendPacket:
    type: SendPacketType
    text: Optional[str] = None
    audio: Optional[bytes] = None

class RaideMode(Enum):
    STANDALONE = "standalone"
    WEBSOCKET = "websocket"

class Raide:
    DEFAULT_WEBSOCKET_MAX_SIZE = 100 * 1024 * 1024

    def __init__(self, mode: RaideMode, port: int, play_audio: bool = False):
        # read only
        self.mode = mode
        self.port = port
        self.play_audio = play_audio
        self.sample_rate = 16000
        self.asr = None
        self.llm = None
        self.tts = None

        # mutable
        self.send_queue: asyncio.Queue = asyncio.Queue()
        self.running: asyncio.Event = asyncio.Event()
        self.inferencing: asyncio.Event = asyncio.Event()

        self.running.set()
        self.inferencing.clear()

    def run(self):
        logger.info("Starting Raide...")
        self._init()
        asyncio.run(self._loop())

    def stop(self):
        logger.info("Stopping Raide...")
        self.running = False
        logger.info("Raide stopped.")

    def _init(self):
        log.init_logger()
        config.load("./config")

        tts_config = config.to_tts_config()
        self.tts = OpenAudioTextToSpeech(config=tts_config)
        self.sample_rate = self.tts.get_sample_rate()
        self.tts.warmup()

        self.asr = RealtimeSTT.AudioToTextRecorder(
            model_path=config.asr.model_path,
            silero_use_onnx=True,
            silero_deactivity_detection=True,
            use_microphone= True if self.mode == RaideMode.STANDALONE else False,
            allowed_latency_limit=30000,
            min_length_of_recording=-1.0
        )

        llm_config = config.to_llm_config()
        self.llm = LocalLanguageModel(config=llm_config)

    async def _loop(self):
        await asyncio.gather(
            self._network_run(),
            self._chat_loop()
        )

    async def _chat_loop(self):
        logger.info("Starting chat loop...")
        while self.running.is_set():
            await asyncio.to_thread(self.asr.wait_audio)

            self.inferencing.set()
            await self.send_queue.put(SendPacket(type=SendPacketType.START_RESPONSE))

            recognized_text = self.asr.transcribe()

            asr_send_packet = SendPacket(
                type=SendPacketType.TRANSCRIPT,
                text=recognized_text
            )
            await self.send_queue.put(asr_send_packet)

            logger.info(f"User: {recognized_text}")

            def get_llm_response():
                chunk_list = []
                for chunk in self.llm.chat(recognized_text):
                    chunk_list.append(chunk)
                return chunk_list
            chunk_list = await asyncio.to_thread(get_llm_response)

            full_message = "".join(chunk_list)
            logger.info(f"AI: {full_message}")

            llm_send_packet = SendPacket(
                type=SendPacketType.RESPONSE_TEXT,
                text=full_message
            )
            await self.send_queue.put(llm_send_packet)

            if len(full_message) > 0:
                audio = await asyncio.to_thread(self.tts.text_to_speech, text=full_message)

                if self.play_audio:
                    await asyncio.to_thread(sd.play, audio, samplerate=self.sample_rate, blocking=True)

                tts_send_packet = SendPacket(
                    type=SendPacketType.RESPONSE_AUDIO,
                    audio=audio.tobytes()
                )
                await self.send_queue.put(tts_send_packet)

            await self.send_queue.put(SendPacket(type=SendPacketType.END_RESPONSE))
            self.inferencing.clear()

    async def _network_run(self):
        logger.info("Starting WebSocket server...")
        async with server.serve(handler=self._network_loop, host="localhost", port=self.port, max_size=self.DEFAULT_WEBSOCKET_MAX_SIZE):
            logger.info(f"WebSocket server started on ws://localhost:{self.port}")
            await asyncio.Future()  # Run forever

    async def _network_loop(self, connection: server.ServerConnection):
        await asyncio.gather(
            self._recv_loop(connection),
            self._send_loop(connection)
        )

    async def _recv_loop(self, connection: server.ServerConnection):
        current_sample_rate = self.sample_rate
        current_sample_width = 2

        while self.running.is_set():
            recv_data = await connection.recv()

            if isinstance(recv_data, str):
                recv_text = str(recv_data)
                recv_dict = json.loads(recv_text)
                if recv_dict.get("type") == "request_audio":
                    current_sample_rate = recv_dict.get("sample_rate", self.sample_rate)
                    current_sample_width = recv_dict.get("sample_width", 2)
                elif recv_dict.get("type") == "query_status":
                    if self.inferencing.is_set():
                        send_json = { "type": "ack_status", "status": "busy" }
                        await connection.send(json.dumps(send_json))
                    else:
                        send_json = { "type": "ack_status", "status": "idle" }
                        await connection.send(json.dumps(send_json))

            elif isinstance(recv_data, bytes):
                recv_audio = bytes(recv_data)
                if not self.inferencing.is_set():
                    audio_dtype = utils.sample_width_to_dtype(current_sample_width)
                    recv_audio_np = np.frombuffer(recv_audio, dtype=audio_dtype)
                    self.asr.feed_audio(chunk=recv_audio_np, original_sample_rate=current_sample_rate)

    async def _send_loop(self, connection: server.ServerConnection):
        while self.running.is_set():
            send_packet: SendPacket = await self.send_queue.get()
            
            if send_packet.type == SendPacketType.START_RESPONSE:
                send_json = { "type": "start_response" }
            elif send_packet.type == SendPacketType.TRANSCRIPT:
                send_json = { "type": "transcript", "text": send_packet.text }
            elif send_packet.type == SendPacketType.RESPONSE_TEXT:
                send_json = { "type": "response_text", "text": send_packet.text }
            elif send_packet.type == SendPacketType.RESPONSE_AUDIO:
                send_json = { "type": "response_audio", "sample_rate": self.tts.get_sample_rate(), "sample_width": 4 }
            elif send_packet.type == SendPacketType.END_RESPONSE:
                send_json = { "type": "end_response" }
            elif send_packet.type == SendPacketType.ACK_STATUS:
                continue

            await connection.send(json.dumps(send_json))
            if send_packet.type == SendPacketType.RESPONSE_AUDIO:
                await connection.send(send_packet.audio)
