"""Module for splitting a stream of log-probabilities into phrases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class LogprobPhrase:
    """Class representing final phrase from splitter.

    Attributes:
        logprobs: logprobs corresponding to phrase
        start: phrase start (in logprobs timestamps)
        end: phrase end (in logprobs timestamps)

    """

    logprobs: npt.NDArray[np.float32]
    start_frame: int  # in acoustic frames
    end_frame: int  # in acoustic frames


@dataclass
class StreamingLogprobSplitterState:
    """Stores state for a StreaminglogprobSplitter."""

    past_logprobs: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 35), dtype=np.float32))
    offset: int = 0
    in_speech: bool = False


class StreamingLogprobSplitter:
    """Splits log-probabilities into semantically complete segments for decoding.

    This component receives log-probabilities outputs from a CTC-based acoustic model
    and identifies suitable segment boundaries (e.g., pauses or silence) for
    incremental decoding. It maintains internal state across chunks and is designed
    to operate in a streaming fashion.

    Note: Batching is not supported.
    """

    InputType: TypeAlias = npt.NDArray[np.float32]
    OutputType: TypeAlias = "list[LogprobPhrase]"
    StateType: TypeAlias = "StreamingLogprobSplitterState"

    SILENCE_THRESHOLD = 0.9  # probability (0 - 1)
    MIN_SILENCE_DURATION = 20  # in acoustic frames
    SPEECH_EXPAND_SIZE = 3  # in acoustic frames
    MAX_PHRASE_DURATION = 2000  # in acoustic frames

    def _iterate_over_phrases(
        self,
        is_speech: npt.NDArray[np.bool_],
        *,
        is_last: bool = False,
    ) -> Iterator[tuple[int, int]]:
        speech_len = len(is_speech)
        # Step 1. Guarantee the first silence (and the last if is_last is True) is a phrase separator
        is_speech = np.pad(is_speech, (self.MIN_SILENCE_DURATION, self.MIN_SILENCE_DURATION if is_last else 0))

        # Step 2. Get the indices of the start and end of the silence sequences, compute silence duration
        silence_changes = np.diff(np.pad(~is_speech, (1, 1)).astype(np.int32))  # -1 - end, 0 - no change, 1 - start
        silence_starts = (silence_changes == 1).nonzero()[0] - self.MIN_SILENCE_DURATION
        silence_ends = (silence_changes == -1).nonzero()[0] - self.MIN_SILENCE_DURATION
        silence_duration = silence_ends - silence_starts

        # Step 3. Keep only long enough silences
        silence_is_phrase_separator = silence_duration >= self.MIN_SILENCE_DURATION
        assert silence_is_phrase_separator[0], "First silence is guaranteed to be a phrase separator"
        silence_starts = silence_starts[silence_is_phrase_separator]
        silence_ends = silence_ends[silence_is_phrase_separator]

        # Step 4. Iterate through all the speeches between silances and construct phrases out of them
        speech_starts, speech_ends = silence_ends.tolist(), silence_starts.tolist()[1:] + [speech_len]
        for i, (speech_start, speech_end) in enumerate(zip(speech_starts, speech_ends)):
            while speech_end - speech_start >= self.MAX_PHRASE_DURATION:  # Split too long phrase by force
                yield speech_start, speech_start + self.MAX_PHRASE_DURATION
                speech_start += self.MAX_PHRASE_DURATION  # noqa: PLW2901
            if i < len(silence_ends) - 1:  # Do not yield last unfinished speech
                yield speech_start, speech_end

    def forward(
        self,
        logprobs: InputType,
        state: StateType | None = None,
        *,
        is_last: bool = False,
    ) -> tuple[OutputType, StateType]:
        """Process a chunk of log-probabilities and extracts complete segments.

        Analyzes log-probabilities to detect phrase or utterance boundaries suitable
        for decoding. Keeps track of context across multiple calls via `state`.

        Args:
            logprobs (InputType): Log-probabilities tensor from the acoustic model.
            state (StateType | None): Previous splitter state, or None to initialize.
            is_last (bool): Whether this is the final chunk of the input stream.

        Returns:
            Tuple[OutputType, StateType]:
                - A list of extracted segments ready for decoding.
                - Updated splitter state to be passed into the next call.

        """
        if not isinstance(logprobs, np.ndarray):
            raise TypeError(f"Incorrect 'logprobs' type: expected np.ndarray, but got {type(logprobs)}")
        if logprobs.shape[1:] != (35,):
            raise ValueError(f"Shape of 'logprobs' must be (L, 35), but got {logprobs.shape}")
        if logprobs.dtype != np.float32:
            raise ValueError(f"Incorrect dtype of 'logprobs': expected np.float32, but got {logprobs.dtype}")

        if state is None:
            state = StreamingLogprobSplitterState()  # Create empty initial states
        if not isinstance(state, StreamingLogprobSplitterState):
            raise TypeError(
                f"Incorrect 'state' type: expected StreamingLogprobSplitterState or None, but got {type(state)}",
            )

        speech_expand = self.SPEECH_EXPAND_SIZE

        # Step 1. Combine old logprobs (from state) with new ones
        logprobs = np.concatenate((state.past_logprobs, logprobs), axis=-2)

        # Step 2. If the probability of space + blank tokens is less than a threshold, consider it as speech
        is_speech = np.exp(logprobs[..., -2:]).sum(axis=-1) <= self.SILENCE_THRESHOLD

        # Step 3. Iterate through all the speeches and construct phrases out of them
        phrases: list[LogprobPhrase] = []
        last_phrase = 0
        for phrase_start, phrase_end in self._iterate_over_phrases(is_speech, is_last=is_last):
            phrase = LogprobPhrase(
                logprobs=logprobs[max(0, phrase_start - speech_expand) : phrase_end + speech_expand],
                start_frame=phrase_start + state.offset,
                end_frame=phrase_end + state.offset,
            )
            phrases.append(phrase)
            last_phrase = phrase_end

        # Step 4. Remove logprobs without speech saving the last 'speech_expand' logprobs
        speech_ids = np.nonzero(is_speech[last_phrase:])[0]
        if not len(speech_ids):
            last_phrase = max(last_phrase, len(logprobs) - speech_expand)
        next_offset = state.offset + last_phrase
        return (phrases, StreamingLogprobSplitterState(past_logprobs=logprobs[last_phrase:], offset=next_offset, in_speech=is_speech.any()))
