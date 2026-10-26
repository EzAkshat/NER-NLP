"""Document analysis pipeline: file -> ingestion -> NER -> AnalysisResult."""

from __future__ import annotations

from pathlib import Path

from ner_app.errors import NerError
from ner_app.inference.flair_ner import (
    DEFAULT_MODEL_ID,
    FlairNER,
    InferenceError,
    InferenceSettings,
)
from ner_app.ingestion import DEFAULT_LIMITS, ExtractedDocument, Limits, extract_document
from ner_app.models import AnalysisResult, EntityGroup, Mention

__all__ = ["analyze_document", "analyze_file"]


def _build_groups(mentions: list[Mention]) -> list[EntityGroup]:
    """Group mentions by exact (label, text), in order of first appearance.

    Every mention lands in exactly one group (singletons included).
    The original mentions are never modified or removed.
    """
    ids_by_key: dict[tuple[str, str], list[int]] = {}
    for mention in mentions:
        ids_by_key.setdefault((mention.label, mention.text), []).append(mention.id)
    return [
        EntityGroup(label=label, text=text, mention_ids=ids)
        for (label, text), ids in ids_by_key.items()
    ]


def _check_offsets(document: ExtractedDocument, mentions: list[Mention]) -> None:
    """Confirm each mention's text is really at its offsets in the document text.

    Mention only validates that len(text) == end - start; this checks the content.
    """
    for mention in mentions:
        if document.text[mention.start : mention.end] != mention.text:
            raise InferenceError(
                f"Mention {mention.id} ({mention.text!r}) does not match the "
                f"document text at offsets {mention.start}-{mention.end}."
            )


def analyze_document(document: ExtractedDocument, ner: FlairNER) -> AnalysisResult:
    """Run NER over an already-extracted document. Offsets refer to document.text."""
    mentions = ner.predict(document.text)
    _check_offsets(document, mentions)
    return AnalysisResult(
        document=document.to_document_info(),  # carries source name, format, warnings
        model=ner.model_info,
        mentions=mentions,
        groups=_build_groups(mentions),
    )


def analyze_file(
    path: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    settings: InferenceSettings | None = None,
    limits: Limits = DEFAULT_LIMITS,
    ner: FlairNER | None = None,
) -> AnalysisResult:
    """Analyze one file. Raises NerError subclasses for expected failures.

    Ingestion runs first, so a bad path or unsupported file fails fast
    without paying the cost of loading the model.
    """
    document = extract_document(path, limits)
    if ner is None:
        ner = FlairNER(model_id, settings)
    return analyze_document(document, ner)