"""Build an NER backend from simple options. Backends are imported only when used."""

from __future__ import annotations

from pathlib import Path

from ner_app.inference.flair_ner import InferenceSettings

BACKENDS = ("flair", "gliner", "gliner-nuner")


def parse_labels(spec: str | None) -> dict[str, str] | None:
    """Parse 'PERSON=Person,COURSE_CODE=Course code'. 'marks' alone means marks=marks."""
    if spec is None or not spec.strip():
        return None
    labels: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, prompt = part.partition("=")
        name = name.strip()
        prompt = prompt.strip() if sep else name
        if not name or not prompt:
            raise ValueError(f"invalid label entry: {part!r}")
        labels[name] = prompt
    if not labels:
        raise ValueError("no labels given")
    return labels


def build_ner(
    backend: str,
    model_id: str | None = None,
    settings: InferenceSettings | None = None,
    labels: dict[str, str] | None = None,
    threshold: float = 0.5,
):
    if backend == "flair":
        from ner_app.inference.flair_ner import DEFAULT_MODEL_ID, FlairNER

        return FlairNER(model_id or DEFAULT_MODEL_ID, settings)
    if backend == "gliner":
        from ner_app.inference.gliner_ner import DEFAULT_GLINER_MODEL, GlinerNER

        return GlinerNER(model_id or DEFAULT_GLINER_MODEL, labels, threshold, settings)
    if backend == "gliner-nuner":
        from ner_app.inference.flair_ner import InferenceError
        from ner_app.inference.gliner_ner import (
            FINETUNED_FALLBACK_THRESHOLD,
            FINETUNED_LABELS,
            GlinerNER,
            default_finetuned_path,
            load_label_thresholds,
        )

        path = model_id or default_finetuned_path()
        if not Path(path).is_dir():
            raise InferenceError(
                f"The fine-tuned model folder was not found: {path}. "
                "Set NER_NUNER_MODEL to its location."
            )
        use_labels = labels or FINETUNED_LABELS
        default, per_label = load_label_thresholds(path)
        if threshold != 0.5:  # an explicit threshold replaces the tuned ones
            default, per_label = threshold, {}
        per_label = {k: v for k, v in per_label.items() if k in use_labels}
        return GlinerNER(
            path,
            use_labels,
            default if default is not None else FINETUNED_FALLBACK_THRESHOLD,
            settings,
            label_thresholds=per_label,
            strip_titles=True,
        )
    raise ValueError(f"unknown backend {backend!r}")