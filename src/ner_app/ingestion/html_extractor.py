"""Safe HTML-to-text extraction. Nothing is executed, rendered, or fetched."""

from __future__ import annotations

from html.parser import HTMLParser

from ner_app.errors import ExtractionError
from ner_app.ingestion.decoding import decode_html
from ner_app.ingestion.types import Limits, RawExtraction
from ner_app.models import ExtractionWarning

# Content of these elements is never part of the readable text.
_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "title",
     "iframe", "object", "canvas", "math"}
)
_PARAGRAPH_TAGS = frozenset(
    {"p", "div", "section", "article", "header", "footer", "main", "nav", "aside",
     "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "dl", "table", "blockquote",
     "pre", "figure", "figcaption", "form", "fieldset", "address", "details",
     "summary", "hr"}
)
_LINE_END_TAGS = frozenset({"li", "tr", "dt", "dd"})
_CELL_END_TAGS = frozenset({"td", "th"})


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_tag: str | None = None
        self._skip_depth = 0

    @property
    def ended_inside_skipped_element(self) -> bool:
        return self._skip_tag is not None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _SKIP_TAGS:
            self._skip_tag, self._skip_depth = tag, 1
        elif tag in _PARAGRAPH_TAGS:
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag in _PARAGRAPH_TAGS:
            self.parts.append("\n\n")
        elif tag in _LINE_END_TAGS:
            self.parts.append("\n")
        elif tag in _CELL_END_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_tag is None:
            self.parts.append(data)


def extract_html_text(html: str) -> RawExtraction:
    parser = _TextCollector()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - malformed input must become a clear error
        raise ExtractionError("The HTML could not be parsed. The file may be corrupted.") from exc

    warnings: list[ExtractionWarning] = []
    if parser.ended_inside_skipped_element:
        warnings.append(
            ExtractionWarning(
                code="html_unclosed_element",
                message=(
                    "The HTML ended inside an unclosed script, style, or similar element. "
                    "Some text after it may have been skipped."
                ),
            )
        )
    return "".join(parser.parts), warnings


def extract_html(data: bytes, limits: Limits) -> RawExtraction:
    html, decode_warnings = decode_html(data)
    text, html_warnings = extract_html_text(html)
    return text, decode_warnings + html_warnings