
"""Flair-based named entity recognition with document-wide character offsets."""

from __future__ import annotations
from bisect import bisect_left
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Protocol

from flair.data import Sentence
from flair.models import SequenceTagger

from ner_app.errors import NerError
from ner_app.models import Mention, ModelInfo


DEFAULT_MODEL_ID = "flair/ner-english-ontonotes-large"


class InferenceError(NerError):
    """An expected model-loading or inference failure."""


@dataclass(frozen=True)
class InferenceSettings:
    """Controls chunk sizes and overlap, measured in Python characters."""

    chunk_size: int = 10_000
    chunk_overlap: int = 256

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


class _TaggerLike(Protocol):
    def predict(self, sentences: Sentence | list[Sentence]) -> None: ...


_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, SequenceTagger] = {}


def _load_model(model_id: str) -> SequenceTagger:
    """Load and cache each model ID once per process."""
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(model_id)
        if model is not None:
            return model

        try:
            model = SequenceTagger.load(model_id)
        except Exception as exc:
            raise InferenceError(
                f"Could not load the NER model {model_id!r}."
            ) from exc

        _MODEL_CACHE[model_id] = model
        return model


def _flair_version() -> str:
    try:
        return version("flair")
    except PackageNotFoundError:
        return "unknown"


def _chunk_ranges(
    text: str,
    settings: InferenceSettings,
) -> list[tuple[int, int]]:
    """Return overlapping [start, end) character ranges."""
    if not text:
        return []

    ranges: list[tuple[int, int]] = []
    start = 0

    while start < len(text):
        end = min(start + settings.chunk_size, len(text))
        ranges.append((start, end))

        if end == len(text):
            break

        start = end - settings.chunk_overlap

    return ranges

def _resolve_overlaps(
    candidates: dict[tuple[int, int, str], float | None],
) -> list[tuple[tuple[int, int, str], float | None]]:
    """Keep only non-overlapping spans, preferring longer spans, then higher scores.

    Flair never returns overlapping spans within one chunk, so any overlap here comes
    from neighbouring chunks, typically one chunk's truncated view of an entity that
    was cut by its edge. Result is sorted by start offset.
    """

    def preference(item):
        (start, end, label), score = item
        return (-(end - start), -(-1.0 if score is None else score), start, label)

    kept_starts: list[int] = []
    kept: list[tuple[tuple[int, int, str], float | None]] = []
    for item in sorted(candidates.items(), key=preference):
        start, end, _ = item[0]
        i = bisect_left(kept_starts, start)
        if i > 0 and kept[i - 1][0][1] > start:  # previous kept span runs into this one
            continue
        if i < len(kept) and kept_starts[i] < end:  # next kept span starts inside this one
            continue
        kept_starts.insert(i, start)
        kept.insert(i, item)
    return kept

class FlairNER:
    """Run Flair NER and return mentions with offsets into the original text."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        settings: InferenceSettings | None = None,
        *,
        _tagger: _TaggerLike | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")

        self.model_id = model_id
        self.settings = settings or InferenceSettings()
        self._tagger = _tagger if _tagger is not None else _load_model(model_id)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            framework="Flair",
            framework_version=_flair_version(),
            score_meaning=(
                "Flair's predicted span score: a model confidence estimate, "
                "not a calibrated probability of correctness."
            ),
        )

    def predict(self, text: str) -> list[Mention]:
        """Predict entities and preserve exact document-wide character offsets."""
        if not text:
            return []

        ranges = _chunk_ranges(text, self.settings)
        # Keyed by exact document span and label to deduplicate overlap predictions.
        candidates: dict[tuple[int, int, str], float | None] = {}

        try:
            for chunk_start, chunk_end in ranges:
                chunk_text = text[chunk_start:chunk_end]
                sentence = Sentence(chunk_text)
                self._tagger.predict(sentence)

                for span in sentence.get_spans("ner"):
                    local_start = span.start_position
                    local_end = span.end_position

                    if (
                        not isinstance(local_start, int)
                        or not isinstance(local_end, int)
                        or local_start < 0
                        or local_end <= local_start
                        or local_end > len(chunk_text)
                    ):
                        raise InferenceError(
                            "The NER model returned an invalid entity span."
                        )

                    start = chunk_start + local_start
                    end = chunk_start + local_end

                    label = span.get_label("ner")
                    entity_label = label.value
                    score = label.score

                    if not isinstance(entity_label, str) or not entity_label:
                        raise InferenceError(
                            "The NER model returned an invalid entity label."
                        )

                    if score is not None and (
                        not isinstance(score, (int, float))
                        or not 0.0 <= score <= 1.0
                    ):
                        raise InferenceError(
                            "The NER model returned an invalid confidence score."
                        )

                    entity_text = text[start:end]
                    if not entity_text:
                        raise InferenceError(
                            "The NER model returned an empty entity span."
                        )

                    key = (start, end, entity_label)
                    previous_score = candidates.get(key)

                    # Prefer a scored prediction over an unscored one, and otherwise
                    # keep the higher score when the exact same span is predicted twice.
                    if key not in candidates or (
                        score is not None
                        and (previous_score is None or score > previous_score)
                    ):
                        candidates[key] = score

        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("NER inference failed.") from exc

        ordered = _resolve_overlaps(candidates)

        return [
            Mention(
                id=mention_id,
                text=text[start:end],
                label=label,
                start=start,
                end=end,
                score=score,
            )
            for mention_id, ((start, end, label), score) in enumerate(ordered)
        ]
