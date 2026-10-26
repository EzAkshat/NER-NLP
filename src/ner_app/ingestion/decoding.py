"""Turn raw bytes into text without silently guessing."""

from __future__ import annotations

import codecs
import re

from ner_app.errors import ExtractionError
from ner_app.models import ExtractionWarning

# UTF-32 BOMs must be checked before UTF-16 (they share a prefix).
_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

_META_CHARSET = re.compile(
    rb"<meta[^>]{0,200}?charset\s*=\s*[\"']?\s*([A-Za-z0-9_.:\-]+)", re.IGNORECASE
)


def decode_bytes(data: bytes) -> tuple[str, list[ExtractionWarning]]:
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            try:
                return data.decode(encoding), []
            except UnicodeDecodeError as exc:
                raise ExtractionError(
                    f"The file declares {encoding.upper()} encoding but could not be decoded. "
                    "It may be corrupted."
                ) from exc

    if b"\x00" in data:
        raise ExtractionError(
            "The file contains binary data and does not look like a text document. "
            "Check that the file extension matches its real format."
        )

    try:
        return data.decode("utf-8"), []
    except UnicodeDecodeError:
        warning = ExtractionWarning(
            code="encoding_fallback",
            message=(
                "The file is not valid UTF-8, so it was read as Windows-1252. "
                "Some special characters may be wrong."
            ),
        )
        return data.decode("cp1252", errors="replace"), [warning]


def decode_html(data: bytes) -> tuple[str, list[ExtractionWarning]]:
    """Like decode_bytes, but honors a <meta charset> declaration when there is no BOM."""
    has_bom = any(data.startswith(bom) for bom, _ in _BOMS)
    if not has_bom:
        match = _META_CHARSET.search(data[:4096])
        if match:
            try:
                return data.decode(match.group(1).decode("ascii")), []
            except (LookupError, UnicodeDecodeError):
                pass  # declared charset is unknown or wrong; fall back with a warning if needed
    return decode_bytes(data)