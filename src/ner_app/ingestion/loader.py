"""Validate a file path, extract its text, and return an ExtractedDocument."""

from __future__ import annotations

from pathlib import Path

from ner_app.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    ResourceLimitError,
    UnreadableDocumentError,
    UnsupportedFormatError,
)
from ner_app.ingestion.html_extractor import extract_html
from ner_app.ingestion.normalize import normalize_text
from ner_app.ingestion.text_formats import extract_md, extract_txt
from ner_app.ingestion.types import ExtractedDocument, Extractor, Limits
from ner_app.ingestion.docx_extractor import extract_docx
from ner_app.ingestion.pdf_extractor import extract_pdf

# extension -> (format name, extractor). PDF and DOCX will be added in Phase 2b.
_REGISTRY: dict[str, tuple[str, Extractor]] = {
    ".txt": ("txt", extract_txt),
    ".md": ("md", extract_md),
    ".html": ("html", extract_html),
    ".htm": ("html", extract_html),
    ".pdf": ("pdf", extract_pdf),
    ".docx": ("docx", extract_docx),
}
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(_REGISTRY))
DEFAULT_LIMITS = Limits()


def _validate_path(path: Path, limits: Limits) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise DocumentNotFoundError(f"File not found: {path}") from exc
    except OSError as exc:
        raise UnreadableDocumentError(f"Could not access {path}: {exc.strerror or exc}") from exc

    if not resolved.is_file():
        raise DocumentNotFoundError(f"{path} is not a file (it may be a folder).")

    suffix = resolved.suffix.lower()
    if suffix not in _REGISTRY:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix or '(none)'}'. Supported types: {supported}"
        )

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise UnreadableDocumentError(f"Could not read the size of {path}: {exc}") from exc
    if size == 0:
        raise EmptyDocumentError(f"{path.name} is empty.")
    if size > limits.max_file_bytes:
        raise ResourceLimitError(
            f"{path.name} is {size / 1_048_576:.1f} MB, which is over the "
            f"{limits.max_file_bytes / 1_048_576:.0f} MB limit. "
            "The limit exists to keep processing safe and responsive."
        )
    return resolved


def extract_document(path: str | Path, limits: Limits = DEFAULT_LIMITS) -> ExtractedDocument:
    """Extract normalized text from a supported file. Raises NerError subclasses on failure."""
    original = Path(path)
    resolved = _validate_path(original, limits)
    format_name, extractor = _REGISTRY[resolved.suffix.lower()]

    try:
        data = resolved.read_bytes()
    except PermissionError as exc:
        raise UnreadableDocumentError(f"No permission to read {original.name}.") from exc
    except OSError as exc:
        raise UnreadableDocumentError(f"Could not read {original.name}: {exc}") from exc

    raw_text, warnings = extractor(data, limits)
    if len(raw_text) > limits.max_chars:
        raise ResourceLimitError(
            f"{original.name} contains {len(raw_text):,} characters of text, which is over "
            f"the {limits.max_chars:,} character limit."
        )

    text = normalize_text(raw_text)
    if not any(ch.isalnum() for ch in text):
        raise EmptyDocumentError(f"No readable text was found in {original.name}.")

    return ExtractedDocument(
        source_name=original.name,
        detected_format=format_name,
        text=text,
        warnings=tuple(warnings),
    )