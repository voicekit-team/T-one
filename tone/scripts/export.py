"""Module that exports a T-one model to ONNX format."""

from __future__ import annotations

import argparse
import io
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from tone.nn.model import Tone

import onnx
import torch
from cloudpathlib import AnyPath

from tone.training.model_wrapper import ToneForCTC

_old_layer_norm = torch.nn.functional.layer_norm
DUMMY_BATCH_SIZE = 5
DUMMY_AUDIO_RANGE_MIN = -32767
DUMMY_AUDIO_RANGE_MAX = 32767


def layer_norm(
    inputs: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:  # pylint: disable=redefined-builtin
    r"""A custom implementation of Layer Normalization that supports both float16 and float32.

    This function is a workaround for a known issue in PyTorch's Layer Normalization
    implementation that prevents it from working with float16 data types.
    The workaround involves converting the input tensor to float32 before applying
    Layer Normalization and then converting the result back to float16.
    This ensures that the normalization is performed correctly while maintaining
    the desired precision.

    Args:
        inputs (torch.Tensor): The input tensor to be normalized.
        \*args: Variable length argument list to be passed to the original layer_norm function.
        \**kwargs: Arbitrary keyword arguments to be passed to the original layer_norm function.

    Returns:
        torch.Tensor: The normalized tensor, converted back to the original data type.

    """
    return _old_layer_norm(inputs.float(), *args, **kwargs)


class ModelToExport(torch.nn.Module):
    """A wrapper class for exporting a T-one model to ONNX format."""

    _model: Tone
    _state_shape: list[tuple[int, ...]]
    _state_place: list[tuple[int, int]]
    _signal_len: int

    @property
    def input_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Provides a dummy input tuple for model tracing, testing, or exporting.

        This property generates a correctly-shaped and typed sample input that
        the `forward` method expects.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A dummy input signal tensor of shape (batch, signal_len, 1).
                - An initial fused state tensor for the corresponding batch size.

        """
        signal = torch.randint(
            DUMMY_AUDIO_RANGE_MIN,
            DUMMY_AUDIO_RANGE_MAX,
            (DUMMY_BATCH_SIZE, self._signal_len, 1),
            dtype=torch.int16,
        )

        return signal, self.get_initial_state(DUMMY_BATCH_SIZE)

    def __init__(
        self,
        path_to_pretrained: Path | str,
        chunk_duration_ms: int,
    ) -> None:
        super().__init__()
        tone_ctc = ToneForCTC.from_pretrained(path_to_pretrained)
        self._model = tone_ctc.tone

        self._model.eval()
        self._signal_len = chunk_duration_ms * self._model.preprocessor.sample_rate // 1000

        state: tuple[torch.Tensor] = self._model.get_initial_state(
            batch_size=1,
            len_dtype=torch.int32,
            target="export",
        )
        state = (torch.zeros(1, 1),) + state[:3] + state[4:]
        self._state_shape = [tuple(i.shape[1:]) for i in state]
        state_size = [i.flatten(1).size(-1) for i in state]
        self._state_place = [(sum(state_size[:i]), sum(state_size[: i + 1])) for i in range(len(state_size))]

    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Generates the initial fused state tensor for a given batch size.

        This method creates a zero-filled tensor that serves as the starting
        `state` for a recurrent forward pass. The tensor's shape is designed to
        match the single, fused state representation expected by the `forward`
        method.

        The total dimension of the state is inferred from the `_state_place`
        attribute, which defines the layout of the concatenated sub-states.

        Args:
            batch_size (int): The number of parallel sequences to process; the
                batch size for the initial state.

        Returns:
            torch.Tensor: A zero-initialized tensor of dtype float16, representing
                the initial fused state with shape (batch_size, total_state_dim).

        """
        return torch.zeros(batch_size, self._state_place[-1][1], dtype=torch.float16)

    def _checkpoint_to_bytes(self, checkpoint: dict[str, Any]) -> IO:
        """Serializes a PyTorch checkpoint dictionary into an in-memory byte stream.

        This helper method takes a standard checkpoint dictionary and uses
        `torch.save` to write it to an in-memory buffer (`io.BytesIO`) instead of
        a file on disk.

        Args:
            checkpoint (dict[str, Any]): The checkpoint dictionary to serialize.

        Returns:
            IO: An in-memory, readable byte stream containing the serialized
                checkpoint data.

        """
        checkpoint_bytes = io.BytesIO()
        torch.save(checkpoint, checkpoint_bytes)
        checkpoint_bytes.seek(0)

        return checkpoint_bytes

    def forward(
        self,
        signal: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass using a fused state tensor for exported models.

        This method is designed for inference environments where passing multiple
        state tensors can be inefficient or unsupported. It manages a recurrent state
        that is "fused" into a single flat tensor.

        The process is as follows:
        1.  **Unpack State**: The input `state` tensor is deconstructed into its
            constituent parts based on a pre-defined layout.
        2.  **Model Execution**: The core model is called with the unpacked states
            using `torch.amp.autocast` for mixed-precision inference.
        3.  **Pack State**: The new state tensors returned by the core model are
            re-ordered and concatenated back into a single flat tensor to be
            used in the next step.

        Args:
            signal (torch.Tensor): The input data for the current time step.
            state (torch.Tensor): A single, fused tensor representing the model's
                recurrent state from the previous step.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The primary output of the model for the current step.
                - The new fused state tensor to be passed to the next call.

        """
        # Reconstruct the list of individual state tensors from the single, fused input state.
        # This operation unpacks the flat `state` tensor by slicing and reshaping it
        # according to the pre-defined layout in `_state_place` and `_state_shape`.
        state_mhsa_len, *state = [
            state[:, place[0] : place[1]].reshape(-1, *shape)
            for place, shape in zip(self._state_place, self._state_shape)
        ]
        with torch.amp.autocast(signal.device.type, dtype=torch.float16):
            res, *state_next = self._model.forward_for_export(
                signal,
                *state[:3],
                state_mhsa_len.int(),
                *state[3:],
            )
        return res, torch.cat(
            [state.flatten(1) for state in state_next[3:4] + state_next[:3] + state_next[4:]],
            dim=-1,
        ).half()


def _export_onnx(model: ModelToExport) -> bytes:
    output_sample = model(*model.input_sample)

    # Patch LayerNorm: repare for ONNX export. Convert to float since tf32 is not supported.
    torch.nn.functional.layer_norm = layer_norm

    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        model,
        model.input_sample,
        onnx_model_bytes,
        input_names=["signal", "state"],
        output_names=["logprobs", "state_next"],
        opset_version=17,
        dynamic_axes={
            k: {0: "batch_size"} for k in ["signal", "state", "logprobs", "state_next"]
        },
    )
    onnx_model_bytes.seek(0)
    onnx_model = onnx.load(onnx_model_bytes)
    onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = output_sample[0].size(1)
    onnx_model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = output_sample[0].size(2)
    onnx_model.graph.output[1].type.tensor_type.shape.dim[1].dim_value = output_sample[1].size(1)

    onnx_model_bytes = io.BytesIO()
    onnx.save(onnx_model, onnx_model_bytes)

    return onnx_model_bytes.getvalue()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-pretrained",
        type=str,
        default="t-tech/T-one",
        help="Path to huggingface pretrained checkpoint",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=300,
        help="Input audio chunk duration in ms",
    )
    parser.add_argument(
        "--output_path",
        type=AnyPath,
        required=True,
        help="Path to output model (on s3 or locally)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = ModelToExport(args.path_to_pretrained, args.chunk_duration_ms)
    model_bytes = _export_onnx(model)
    args.output_path.write_bytes(model_bytes)
