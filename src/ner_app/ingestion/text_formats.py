"""Plain text and Markdown extractors."""

from __future__ import annotations

from markdown_it import MarkdownIt

from ner_app.ingestion.decoding import decode_bytes
from ner_app.ingestion.html_extractor import extract_html_text
from ner_app.ingestion.types import Limits, RawExtraction

_MARKDOWN = MarkdownIt("commonmark")


def extract_txt(data: bytes, limits: Limits) -> RawExtraction:
    return decode_bytes(data)


def extract_md(data: bytes, limits: Limits) -> RawExtraction:
    """Render Markdown to HTML, then reuse the safe HTML text extractor."""
    source, warnings = decode_bytes(data)
    plain, html_warnings = extract_html_text(_MARKDOWN.render(source))
    return plain, warnings + html_warnings