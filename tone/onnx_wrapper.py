"""Module with wrapper for pretrained CTC acoustic model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from typing_extensions import Self, TypeAlias


class StreamingCTCModel:
    """Wrapper for a pretrained CTC acoustic model, running with ONNX Runtime.

    This class handles inference with a CTC-based acoustic model and supports
    batched streaming inputs. It provides factory methods to load the model
    from Hugging Face or from a local file, and exposes a forward method to compute
    log-probabilities from audio chunks.
    """

    InputType: TypeAlias = npt.NDArray[np.int16]
    OutputType: TypeAlias = npt.NDArray[np.float16]
    StateType: TypeAlias = npt.NDArray[np.float16]

    SAMPLE_RATE = 8000
    MEAN_TIME_BIAS = 0.33  # in seconds
    AUDIO_CHUNK_SAMPLES = 2400  # in audio samples
    FRAME_SIZE = 0.03  # in seconds
    STATE_SIZE = 219729

    _ort_sess: ort.InferenceSession

    @classmethod
    def from_hugging_face(cls) -> Self:
        """Load and initialize the model from Hugging Face Hub.

        Downloads the model if not present locally, and initializes
        an ONNX inference session.

        Returns:
            Self: An instance of StreamingCTCModel ready for inference.

        """
        model_path = cls.download_from_hugging_face()
        return cls.from_local(model_path)

    @classmethod
    def download_from_hugging_face(cls) -> str:
        """Download the model from Hugging Face Hub.

        Returns:
            str: Path to the downloaded ONNX model file.

        """
        return hf_hub_download(
            "t-tech/T-one",
            "model.onnx",
        )

    @classmethod
    def from_local(cls, model_path: str | Path) -> Self:
        """Initialize the model from a local ONNX file.

        Args:
            model_path (str | Path): Path to the ONNX model file.

        Returns:
            Self: An instance of StreamingCTCModel ready for inference.

        """
        ort_sess = ort.InferenceSession(model_path)
        return cls(ort_sess)

    def __init__(self, ort_sess: ort.InferenceSession) -> None:
        """Create instance of StreamingCTCModel from onnx session."""
        self._ort_sess = ort_sess

    def forward(self, audio_chunk: InputType, state: StateType | None = None) -> tuple[OutputType, StateType]:
        """Run the CTC acoustic model on a single audio chunk.

        Converts raw audio to frame-level log-probabilities using ONNX Runtime. Maintains
        model state for streaming.

        Args:
            audio_chunk (InputType): A batch or single audio chunk to process.
            state (StateType | None): Previous state, or None to initialize.

        Returns:
            Tuple[OutputType, StateType]:
                - OutputType: Model log-probabilities for each frame.
                - StateType: Updated state to pass into the next call.

        """
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError(f"Incorrect 'audio_chunk' type: expected np.ndarray, but got {type(audio_chunk)}")
        if audio_chunk.shape[1:] != (self.AUDIO_CHUNK_SAMPLES, 1):
            raise ValueError(
                f"Shape of 'audio_chunk' must be (B, {self.AUDIO_CHUNK_SAMPLES}, 1), but got {audio_chunk.shape}",
            )
        if audio_chunk.dtype != np.int16:
            raise ValueError(f"Incorrect dtype of 'audio_chunk': expected np.int16, but got {audio_chunk.dtype}")
        batch_size = audio_chunk.shape[0]
        if state is None:
            state = np.zeros((batch_size, self.STATE_SIZE), dtype=np.float16)  # Create empty initial states
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Incorrect 'state' type: expected np.ndarray or None, but got {type(state)}")
        if state.shape != (batch_size, self.STATE_SIZE):
            raise ValueError(f"Shape of 'state' must be ({batch_size}, {self.STATE_SIZE}), but got {state.shape}")
        if state.dtype != np.float16:
            raise ValueError(f"Incorrect dtype of 'state': expected np.int32, but got {state.dtype}")

        return self._ort_sess.run(None, {"signal": audio_chunk, "state": state})
