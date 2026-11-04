"""GLiNER-based named entity recognition: the caller chooses the labels."""

from __future__ import annotations

import json
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from ner_app.inference.flair_ner import (
    InferenceError,
    InferenceSettings,
    _chunk_ranges,
    _resolve_overlaps,
)
from ner_app.models import Mention, ModelInfo

DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
# GLiNER reads a limited number of tokens at once, so chunks must be much smaller
# than Flair's. 1000 characters is a cautious guess, not a measured value.
GLINER_CHUNK_SIZE = 1000
GLINER_CHUNK_OVERLAP = 200

# output label -> prompt text shown to the model
DEFAULT_LABELS: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Location",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Money",
    "PERCENT": "Percentage",
    "QUANTITY": "Quantity",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "LAW": "Law",
}

# Labels the fine-tuned model ("gliner-nuner" backend) was trained with. The prompts must stay
# exactly these lowercase strings, because the model learned them.
FINETUNED_LABELS: dict[str, str] = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "DATE": "date",
    "TIME": "time",
    "EVENT": "event",
    "PRODUCT": "product",
    "TECHNOLOGY": "technology",
    "QUANTITY": "quantity",
    "MONEY": "money",
}
FINETUNED_FALLBACK_THRESHOLD = 0.7
THRESHOLDS_FILENAME = "label_thresholds.json"

# Honorifics are not part of a name in OntoNotes-style annotation, and keeping them would
# split "Dr. Priya Desai" and "Priya Desai" into two different people when grouping.
_TITLE_RE = re.compile(
    r"^(?:(?:(?:dr|prof|mr|mrs|ms|mx|sir|shri|smt|sri|mme|mlle)\.?"
    r"|(?:rev|capt|col|gen|lt|sgt)\.)\s+)+",
    re.IGNORECASE,
)


def default_finetuned_path() -> str:
    """Folder of the fine-tuned model; override with the NER_NUNER_MODEL environment variable."""
    env = os.environ.get("NER_NUNER_MODEL")
    if env:
        return env
    return str(Path(__file__).resolve().parents[3] / "models" / "gliner-nuner-v1")


def load_label_thresholds(model_path: str | Path) -> tuple[float | None, dict[str, float]]:
    """Read label_thresholds.json from a model folder: (default, {OUTPUT_LABEL: threshold})."""
    path = Path(model_path) / THRESHOLDS_FILENAME
    if not path.is_file():
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        default = float(data["default"])
        by_prompt = {str(k).lower(): float(v) for k, v in data.get("labels", {}).items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InferenceError(f"Could not read the threshold file {path}.") from exc
    name_by_prompt = {prompt: name for name, prompt in FINETUNED_LABELS.items()}
    per_label = {name_by_prompt[k]: v for k, v in by_prompt.items() if k in name_by_prompt}
    for value in (default, *per_label.values()):
        if not 0.0 <= value <= 1.0:
            raise InferenceError(f"The threshold file {path} contains a value outside 0 to 1.")
    return default, per_label


class _ModelLike(Protocol):
    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]: ...


_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_id: str) -> Any:
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(model_id)
        if model is not None:
            return model
        try:
            from gliner import GLiNER

            model = GLiNER.from_pretrained(model_id)
        except Exception as exc:
            raise InferenceError(
                f"Could not load the GLiNER model {model_id!r}. "
                'Is it installed? Try: pip install -e ".[gliner]"'
            ) from exc
        _MODEL_CACHE[model_id] = model
        return model


def _gliner_version() -> str:
    try:
        return version("gliner")
    except PackageNotFoundError:
        return "unknown"


class GlinerNER:
    """Run GLiNER and return mentions with document-wide character offsets."""

    def __init__(
        self,
        model_id: str = DEFAULT_GLINER_MODEL,
        labels: dict[str, str] | None = None,
        threshold: float = 0.5,
        settings: InferenceSettings | None = None,
        *,
        label_thresholds: dict[str, float] | None = None,
        strip_titles: bool = False,
        _model: _ModelLike | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        labels = dict(labels or DEFAULT_LABELS)
        if not labels:
            raise ValueError("at least one label is required")
        prompts = [p.lower() for p in labels.values()]
        if len(set(prompts)) != len(prompts):
            raise ValueError("label prompts must be unique")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        label_thresholds = dict(label_thresholds or {})
        for name, value in label_thresholds.items():
            if name not in labels:
                raise ValueError(f"label threshold given for unknown label {name!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError("label thresholds must be between 0 and 1")

        self.model_id = model_id
        self.labels = labels
        self.threshold = threshold
        self.label_thresholds = label_thresholds
        self.strip_titles = strip_titles
        # The model is asked once at the lowest threshold; each label is then filtered
        # against its own threshold.
        self._query_threshold = min([threshold, *label_thresholds.values()])
        self.settings = settings or InferenceSettings(
            chunk_size=GLINER_CHUNK_SIZE, chunk_overlap=GLINER_CHUNK_OVERLAP
        )
        self._name_by_prompt = {p.lower(): n for n, p in labels.items()}
        self._model = _model if _model is not None else _load_model(model_id)

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_id=self.model_id,
            framework="GLiNER",
            framework_version=_gliner_version(),
            score_meaning=(
                "GLiNER's span score: a model confidence estimate, "
                "not a calibrated probability of correctness."
            ),
        )

    def predict(self, text: str) -> list[Mention]:
        if not text:
            return []

        prompts = list(self.labels.values())
        candidates: dict[tuple[int, int, str], float | None] = {}

        try:
            for chunk_start, chunk_end in _chunk_ranges(text, self.settings):
                chunk = text[chunk_start:chunk_end]
                for ent in self._model.predict_entities(
                    chunk, prompts, threshold=self._query_threshold
                ):
                    local_start, local_end = int(ent["start"]), int(ent["end"])
                    if not 0 <= local_start < local_end <= len(chunk):
                        raise InferenceError("The NER model returned an invalid entity span.")

                    name = self._name_by_prompt.get(str(ent["label"]).lower())
                    if name is None:
                        raise InferenceError("The NER model returned an unknown entity label.")

                    score = ent.get("score")
                    if score is not None:
                        score = float(score)
                        if not 0.0 <= score <= 1.0:
                            raise InferenceError("The NER model returned an invalid score.")
                        if score < self.label_thresholds.get(name, self.threshold):
                            continue

                    if self.strip_titles and name == "PERSON":
                        match = _TITLE_RE.match(chunk[local_start:local_end])
                        if match and match.end() < local_end - local_start:
                            local_start += match.end()

                    key = (chunk_start + local_start, chunk_start + local_end, name)
                    previous = candidates.get(key)
                    if key not in candidates or (
                        score is not None and (previous is None or score > previous)
                    ):
                        candidates[key] = score
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("NER inference failed.") from exc

        return [
            Mention(id=i, text=text[start:end], label=label, start=start, end=end, score=score)
            for i, ((start, end, label), score) in enumerate(_resolve_overlaps(candidates))
        ]