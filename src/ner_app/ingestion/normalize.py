"""Whitespace normalization that keeps the document's wording intact."""

from __future__ import annotations

import re
import unicodedata

# Invisible characters that never carry meaning in English text.
# (ZWJ/ZWNJ are deliberately kept: they matter in some scripts.)
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u2060\ufeff"), None)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize Unicode form, line endings, and whitespace.

    - Unicode NFC (composed accents), so "e" + combining accent becomes "é".
    - All line endings become "\\n"; form feeds and paragraph separators become blank lines.
    - Runs of spaces/tabs (including non-breaking spaces) collapse to one space.
    - Three or more newlines collapse to two; paragraphs are separated by one blank line.
    - Leading/trailing whitespace is removed. Words and punctuation are never changed.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2028", "\n").replace("\x85", "\n")
    text = text.replace("\u2029", "\n\n").replace("\x0c", "\n\n")
    text = _CONTROL_RE.sub("", text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()