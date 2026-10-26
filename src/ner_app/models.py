"""Structured result schema shared by the CLI, reports, exporters, and any future web layer.

Offset convention
-----------------
`start` and `end` are zero-based character offsets into the exact extracted text that
was passed to the NER pipeline. `start` is inclusive and `end` is exclusive, so
`text[start:end] == mention.text`. They are NOT positions in the original file
(PDF coordinates, DOCX bytes, etc.).
"""

from __future__ import annotations

from collections import Counter
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

SCHEMA_VERSION = "1.0"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExtractionWarning(_Frozen):
    code: str  # short stable identifier, e.g. "pdf_low_text"
    message: str  # human-readable explanation


class DocumentInfo(_Frozen):
    source_name: str  # file name only, never the full path
    detected_format: str  # "txt", "md", "pdf", "docx", "html"
    char_count: int = Field(ge=0)
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class ModelInfo(_Frozen):
    model_id: str
    framework: str
    framework_version: str
    score_meaning: str  # how to interpret Mention.score


class Mention(_Frozen):
    """One entity occurrence exactly as predicted by the model."""

    id: int = Field(ge=0)
    text: str
    label: str  # original model label, e.g. "ORG"; never renamed
    start: int = Field(ge=0)
    end: int
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # None = unavailable
    context: Optional[str] = None  # surrounding sentence, when available

    @model_validator(mode="after")
    def _check_span(self) -> "Mention":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("text length must equal end - start")
        return self


class EntityGroup(BaseModel):
    """Exact-text, same-label grouping of mentions. Never removes the originals."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    label: str
    text: str
    mention_ids: list[int] = Field(min_length=1)
    match_type: Literal["exact"] = "exact"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def count(self) -> int:
        return len(self.mention_ids)


class AnalysisResult(BaseModel):
    """The single result object produced by the analysis service."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: str = SCHEMA_VERSION
    document: DocumentInfo
    model: ModelInfo
    mentions: list[Mention] = Field(default_factory=list)
    groups: list[EntityGroup] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_mentions(self) -> int:
        return len(self.mentions)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts_by_label(self) -> dict[str, int]:
        return dict(sorted(Counter(m.label for m in self.mentions).items()))

    @model_validator(mode="after")
    def _check_consistency(self) -> "AnalysisResult":
        by_id = {m.id: m for m in self.mentions}
        if len(by_id) != len(self.mentions):
            raise ValueError("mention ids must be unique")
        seen: set[int] = set()
        for group in self.groups:
            for mention_id in group.mention_ids:
                mention = by_id.get(mention_id)
                if mention is None:
                    raise ValueError(f"group references unknown mention id {mention_id}")
                if mention_id in seen:
                    raise ValueError(f"mention id {mention_id} appears in more than one group")
                seen.add(mention_id)
                if mention.label != group.label or mention.text != group.text:
                    raise ValueError("group label/text must match its mentions exactly")
        return self