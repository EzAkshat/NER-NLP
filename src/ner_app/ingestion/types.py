from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ner_app.models import DocumentInfo, ExtractionWarning


@dataclass(frozen=True)
class Limits:
    """Resource limits. Defaults are starting points, not measured values."""

    max_file_bytes: int = 25 * 1024 * 1024
    max_chars: int = 5_000_000
    # PDF
    max_pdf_pages: int = 1000
    # DOCX (zip archive) safeguards
    max_uncompressed_bytes: int = 100 * 1024 * 1024
    max_archive_entries: int = 5000
    max_compression_ratio: int = 100  # only checked for entries over 1 MB

@dataclass(frozen=True)
class ExtractedDocument:
    """Consistent internal representation of an ingested document."""

    source_name: str
    detected_format: str
    text: str  # normalized; all entity offsets refer to this exact string
    warnings: tuple[ExtractionWarning, ...] = ()

    def to_document_info(self) -> DocumentInfo:
        return DocumentInfo(
            source_name=self.source_name,
            detected_format=self.detected_format,
            char_count=len(self.text),
            warnings=list(self.warnings),
        )


# An extractor turns raw file bytes into (unnormalized text, warnings).
RawExtraction = tuple[str, list[ExtractionWarning]]
Extractor = Callable[[bytes, Limits], RawExtraction]