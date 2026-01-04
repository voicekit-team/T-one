"""Module contain simple website implementation for demo purposes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tone.pipeline import StreamingCTCPipeline
from tone.project import VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_BYTES_PER_SAMPLE = 2


@dataclass
class Settings:
    """Global website settings.

    Can be modified using environment variables.
    """

    cors_allow_all: bool = False
    load_from_folder: Path | None = field(default_factory=lambda: os.getenv("LOAD_FROM_FOLDER", None))


class SingletonPipeline:
    """Singleton object to store a single ASR pipeline."""

    pipeline: StreamingCTCPipeline | None = None

    def __new__(cls) -> None:
        """Ensure the class is never created."""
        raise RuntimeError("This is class is a singleton!")

    @classmethod
    def init(cls, settings: Settings) -> None:
        """Initialize singleton object using settings."""
        if settings.load_from_folder is None:
            cls.pipeline = StreamingCTCPipeline.from_hugging_face()
        else:
            cls.pipeline = StreamingCTCPipeline.from_local(settings.load_from_folder)

    @classmethod
    def process_chunk(
        cls,
        audio_chunk: StreamingCTCPipeline.InputType,
        state: StreamingCTCPipeline.StateType | None = None,
        *,
        is_last: bool = False,
    ) -> tuple[StreamingCTCPipeline.OutputType, StreamingCTCPipeline.StateType]:
        """Process audio chunk using ASR pipeline.

        See `StreamingCTCPipeline.forward` for more info.
        """
        if cls.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        return cls.pipeline.forward(audio_chunk, state, is_last=is_last)


router = APIRouter()


async def get_chunk_stream(ws: WebSocket) -> AsyncIterator[tuple[npt.NDArray[np.int16], bool]]:
    """Get audio chunks from websocket and return them as async iterator."""
    audio_data = bytearray()
    # See description of PADDING in StreamingCTCPipeline
    audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())

    is_last = False
    while True:
        await ws.send_json({"event": "ready"})
        recv_bytes = await ws.receive_bytes()
        if len(recv_bytes) == 0:  # Last chunk of audio
            is_last = True
            audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())
            fill_chunk_size = -(len(audio_data) // _BYTES_PER_SAMPLE) % StreamingCTCPipeline.CHUNK_SIZE
            audio_data.extend(np.zeros((fill_chunk_size,), dtype=np.int16).tobytes())
        else:
            audio_data.extend(recv_bytes)

        while len(audio_data) >= StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE:
            chunk = np.frombuffer(audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE], dtype=np.int16)
            del audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE]
            yield chunk, is_last and (len(audio_data) == 0)

        if len(recv_bytes) == 0:
            return


@router.websocket("/ws")
async def websocket_stt(ws: WebSocket) -> None:
    """Websocket endpoint for streaming audio processing."""
    await ws.accept()
    try:
        state: StreamingCTCPipeline.StateType | None = None
        speech_started = False
        async for audio_chunk, is_last in get_chunk_stream(ws):
            output, state = SingletonPipeline.process_chunk(audio_chunk.astype(np.int32), state, is_last=is_last)
            if not speech_started and state[1].in_speech:
                speech_started = True
                await ws.send_json(
                    {
                        "event": "speech_start"
                    }
                )
            for phrase in output:
                speech_started = False
                await ws.send_json(
                    {
                        "event": "transcript",
                        "phrase": {"text": phrase.text, "start_time": phrase.start_time, "end_time": phrase.end_time},
                    },
                )
    except WebSocketDisconnect:
        pass


def get_application() -> FastAPI:
    """Build and return FastAPI application."""
    app = FastAPI(title="T-one Streaming ASR", version=VERSION, docs_url=None, redoc_url=None)
    settings = Settings()
    if settings.cors_allow_all:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_event_handler("startup", lambda: SingletonPipeline.init(settings))

    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="Main website page")
    return app


app = get_application()
