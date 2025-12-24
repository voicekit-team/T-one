"""Audio I/O utilities for the streaming ASR pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from tone.pipeline import StreamingCTCPipeline

if TYPE_CHECKING:
    from collections.abc import Iterator


def read_example_audio(*, long_audio: bool = False) -> npt.NDArray[np.int16]:
    """Get one of two audio examples from the package."""
    audio_examples_dir = Path(__file__).parent / "audio_examples"
    if not long_audio:
        return read_audio(audio_examples_dir / "audio_short.flac")
    return read_audio(audio_examples_dir / "audio_long.flac")


def read_audio(path_to_file: Path | str) -> npt.NDArray[np.int16]:
    """Load a mono 8kHz audio file and return it as an int16 numpy array.

    Uses the `miniaudio` package for decoding. Resamples the audio file
    to mono 16-bit @ 8 kHz

    Args:
        path_to_file (Path | str): Path to the audio file to load.

    Returns:
        npt.NDArray[np.int16]: Audio samples as a 1D numpy array (dtype=int16).

    Raises:
        ModuleNotFoundError: If `miniaudio` is not installed.

    """
    try:
        import miniaudio
    except ImportError as e:
        raise ModuleNotFoundError(
            "Package 'miniaudio' not found.\n"
            "Install it with the following command:\n"
            "  poetry install -E demo   # using package extras\n",
        ) from e

    audio = miniaudio.decode_file(str(path_to_file), nchannels=1, sample_rate=8000)
    assert audio.sample_rate == 8000
    assert audio.nchannels == 1
    return np.asarray(audio.samples, dtype=np.int16)


def read_stream_example_audio(*, long_audio: bool = False) -> Iterator[StreamingCTCPipeline.InputType]:
    """Simple example of streaming audio source using example audio from the package."""
    chunk_size = StreamingCTCPipeline.CHUNK_SIZE
    audio = read_example_audio(long_audio=long_audio)
    # See description of PADDING in StreamingCTCPipeline
    audio = np.pad(audio, (StreamingCTCPipeline.PADDING, StreamingCTCPipeline.PADDING))

    for i in range(0, len(audio), chunk_size):
        audio_chunk = audio[i : i + chunk_size]
        audio_chunk = np.pad(audio_chunk, (0, -len(audio_chunk) % chunk_size))
        yield audio_chunk
