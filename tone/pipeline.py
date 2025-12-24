"""Module with a simple T-one pipeline CPU-only implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

import numpy as np
import numpy.typing as npt
from typing_extensions import Self, TypeAlias

from tone.decoder import BeamSearchCTCDecoder, DecoderType, GreedyCTCDecoder
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.onnx_wrapper import StreamingCTCModel


@dataclass
class TextPhrase:
    """Dataclass for the phrase from ASR pipeline.

    Attributes:
        text: decoded text
        start_time: phrase start time (in sec)
        end_time: phrase end time (in sec)

    """

    text: str
    start_time: float  # in seconds
    end_time: float  # in seconds


class StreamingCTCPipeline:
    """A streaming ASR pipeline for CTC-based models.

    This class orchestrates the entire speech recognition process. It takes raw
    audio chunks, passes them to a streaming acoustic model to get
    log-probabilities, splits the log-probabilities into sensible phrases based on
    silence, and decodes them into text.

    The pipeline is designed for low-latency streaming applications and operates
    on the CPU. For online streaming, use the `forward` method. For offline
    processing of a complete audio file, use the `forward_offline` method.
    """

    # Model was trained with left/right padding so it's necessary to add it to increase the recognition quality
    PADDING: int = 2400  # 300ms * 8KHz
    CHUNK_SIZE: int = StreamingCTCModel.AUDIO_CHUNK_SAMPLES

    InputType: TypeAlias = npt.NDArray[np.int16]
    OutputType: TypeAlias = "list[TextPhrase]"
    StateType: TypeAlias = tuple[npt.NDArray[np.float16], StreamingLogprobSplitter.StateType]

    @classmethod
    def from_hugging_face(cls, *, decoder_type: DecoderType = DecoderType.BEAM_SEARCH) -> Self:
        """Creates a pipeline instance by downloading artifacts from Hugging Face Hub.

        Args:
            decoder_type (DecoderType, optional): The decoding strategy to use.
                Defaults to `DecoderType.BEAM_SEARCH`.

        Returns:
            An initialized `StreamingCTCPipeline` instance.

        """
        model = StreamingCTCModel.from_hugging_face()
        logprob_splitter = StreamingLogprobSplitter()
        if decoder_type == DecoderType.GREEDY:
            decoder = GreedyCTCDecoder()
            return cls(model, logprob_splitter, decoder)
        if decoder_type == DecoderType.BEAM_SEARCH:
            decoder = BeamSearchCTCDecoder.from_hugging_face()
            return cls(model, logprob_splitter, decoder)
        raise ValueError("Unknown decoder type")

    @staticmethod
    def download_from_hugging_face(dir_path: str | Path, only_acoustic: bool = False) -> None:
        """Download all artifacts from HaggingFace to local folder."""
        dir_path = Path(dir_path)
        copyfile(StreamingCTCModel.download_from_hugging_face(), dir_path / "model.onnx")

        if not only_acoustic:
            copyfile(BeamSearchCTCDecoder.download_from_hugging_face(), dir_path / "kenlm.bin")

    @classmethod
    def from_local(cls, dir_path: str | Path, *, decoder_type: DecoderType = DecoderType.BEAM_SEARCH) -> Self:
        """Create StreamingCTCPipeline instance using artifacts from local folder."""
        dir_path = Path(dir_path)
        model = StreamingCTCModel.from_local(dir_path / "model.onnx")
        logprob_splitter = StreamingLogprobSplitter()
        if decoder_type == DecoderType.GREEDY:
            decoder = GreedyCTCDecoder()
            return cls(model, logprob_splitter, decoder)
        if decoder_type == DecoderType.BEAM_SEARCH:
            decoder = BeamSearchCTCDecoder.from_local(dir_path / "kenlm.bin")
            return cls(model, logprob_splitter, decoder)
        raise ValueError("Unknown decoder type")

    def __init__(
        self,
        model: StreamingCTCModel,
        logprob_splitter: StreamingLogprobSplitter,
        decoder: GreedyCTCDecoder | BeamSearchCTCDecoder,
    ) -> None:
        """Create StreamingCTCPipeline instance from model, logprob splitter and decoder."""
        self.model = model
        self.logprob_splitter = logprob_splitter
        self.decoder = decoder

    def forward(
        self,
        audio_chunk: InputType,
        state: StateType | None = None,
        *,
        is_last: bool = False,
    ) -> tuple[OutputType, StateType]:
        """Perform online (streaming) CTC decoding on a 300 ms audio chunk.

        Processes the given audio segment incrementally, updating and returning
        the state for low-latency streaming.

        Args:
            audio_chunk (InputType): A 300 ms slice of audio (2400 samples) to decode.
            state (StateType | None): Previous state, or None to initialize.
            is_last (bool): Whether this is the final chunk of the input stream.

        Returns:
            Tuple[OutputType, StateType]:
                - Decoded output for this chunk.
                - Updated state to pass into the next call.

        """
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError(f"Incorrect 'audio_chunk' type: expected np.ndarray, but got {type(audio_chunk)}")
        if audio_chunk.shape != (self.CHUNK_SIZE,):
            raise ValueError(f"Shape of 'audio_chunk' must be ({self.CHUNK_SIZE},), but got {audio_chunk.shape}")
        if not isinstance(state, (tuple, type(None))):
            raise TypeError(f"Incorrect 'state' type: expected tuple on None, but got {type(state)}")

        frame_size, time_bias = StreamingCTCModel.FRAME_SIZE, StreamingCTCModel.MEAN_TIME_BIAS

        model_state = state[0] if state is not None else None
        logprob_state = state[1] if state is not None else None

        logprobs, model_state_next = self.model.forward(audio_chunk[None, :, None], model_state)
        logprob_phrases, logprob_state_next = self.logprob_splitter.forward(logprobs[0], logprob_state, is_last=is_last)
        phrases: list[TextPhrase] = []
        for logprob_phrase in logprob_phrases:
            text = self.decoder.forward(logprob_phrase.logprobs)
            start_time = max(
                0,
                round(
                    logprob_phrase.start_frame * frame_size - time_bias - self.PADDING / StreamingCTCModel.SAMPLE_RATE,
                    2,
                ),
            )
            end_time = max(
                start_time,
                round(
                    logprob_phrase.end_frame * frame_size - time_bias - self.PADDING / StreamingCTCModel.SAMPLE_RATE,
                    2,
                ),
            )
            phrases.append(
                TextPhrase(
                    text=text,
                    start_time=start_time,
                    end_time=end_time,
                ),
            )
        return (phrases, (model_state_next, logprob_state_next))

    def forward_offline(self, audio: InputType) -> OutputType:
        """Performs offline CTC decoding on a complete audio segment.

        Processes the entire input in one shot without maintaining any state.

        Args:
            audio (InputType): The full audio waveform to decode.

        Returns:
            OutputType: The decoded output for the entire segment.

        """
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Incorrect 'audio' type: expected np.ndarray, but got {type(audio)}")
        if audio.ndim != 1:
            raise ValueError(f"Shape of 'audio' must be (L,), but got {audio.shape}")

        audio = np.pad(audio, (self.PADDING, self.PADDING))

        # Add padding to fill the last chunk and split into chunks of size self.CHUNK_SIZE
        audio = np.pad(audio, (0, -len(audio) % self.CHUNK_SIZE))
        audio_chunks = np.split(audio, len(audio) // self.CHUNK_SIZE)

        outputs: StreamingCTCPipeline.OutputType = []
        state: StreamingCTCPipeline.StateType | None = None
        for i, audio_chunk in enumerate(audio_chunks):
            output, state = self.forward(audio_chunk, state, is_last=i == len(audio_chunks) - 1)
            outputs.extend(output)

        return outputs

    def finalize(self, state: StateType | None) -> tuple[OutputType, StateType]:
        """Finalize the pipeline by sending an empty chunk and processing any remaining logprobs.

        Args:
            state (StateType | None): The current state of the pipeline.

        Returns:
            OutputType: The decoded output for the entire segment.
            StateType: The final state of the pipeline.

        """
        audio_chunk = np.zeros((StreamingCTCPipeline.CHUNK_SIZE,), dtype=np.int16)
        return self.forward(audio_chunk, state, is_last=True)
