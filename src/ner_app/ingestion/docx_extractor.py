"""DOCX extraction. Macros and embedded content are never executed."""

from __future__ import annotations

import io
import zipfile

from docx import Document

from ner_app.errors import ExtractionError, ResourceLimitError
from ner_app.ingestion.types import Limits, RawExtraction
from ner_app.models import ExtractionWarning

# Tiny XML parts often compress very well, so the ratio check only applies to big entries.
_RATIO_CHECK_MIN_BYTES = 1_000_000


def _check_archive(data: bytes, limits: Limits) -> list[ExtractionWarning]:
    """Inspect the zip container before any XML parsing happens."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtractionError(
            "This is not a valid .docx file. It may be corrupted, password-protected, "
            "or an older .doc file that was renamed."
        ) from exc

    warnings: list[ExtractionWarning] = []
    with archive:
        infos = archive.infolist()
        if len(infos) > limits.max_archive_entries:
            raise ResourceLimitError(
                f"The document contains {len(infos):,} internal files, which is over the "
                f"{limits.max_archive_entries:,} limit."
            )

        total = 0
        for info in infos:
            if info.flag_bits & 0x1:
                raise ExtractionError("The document is encrypted and cannot be read.")
            total += info.file_size
            if total > limits.max_uncompressed_bytes:
                raise ResourceLimitError(
                    "The document expands to more than "
                    f"{limits.max_uncompressed_bytes / 1_048_576:.0f} MB when unpacked, "
                    "which is over the safety limit."
                )
            if info.file_size > _RATIO_CHECK_MIN_BYTES:
                ratio = info.file_size / max(info.compress_size, 1)
                if ratio > limits.max_compression_ratio:
                    raise ResourceLimitError(
                        "The document contains data that is compressed suspiciously "
                        "tightly (possible archive bomb), so it was not processed."
                    )

        names = {info.filename.lower() for info in infos}
        if "word/document.xml" not in names:
            raise ExtractionError("The file is a zip archive but not a Word document.")
        if any(name.endswith("vbaproject.bin") for name in names):
            warnings.append(
                ExtractionWarning(
                    code="docx_macros_ignored",
                    message="The document contains macros. They were ignored and never executed.",
                )
            )
    return warnings


def _block_lines(document) -> list[str]:  # noqa: ANN001
    """Body paragraphs and tables in reading order. Each table cell gets its own line."""
    blocks: list[str] = []
    for block in document.iter_inner_content():
        if hasattr(block, "rows"):  # a table
            seen = set()
            cells: list[str] = []
            for row in block.rows:
                for cell in row.cells:
                    if cell._tc in seen:  # merged cells repeat the same XML cell
                        continue
                    seen.add(cell._tc)
                    cells.append(cell.text)
                    blocks.append("\n".join(cells))
        else:
            blocks.append(block.text)
    return blocks


def extract_docx(data: bytes, limits: Limits) -> RawExtraction:
    warnings = _check_archive(data, limits)
    try:
        document = Document(io.BytesIO(data))
        blocks = _block_lines(document)
    except Exception as exc:  # noqa: BLE001 - corrupted XML must become a clear error
        raise ExtractionError(
            "The Word document could not be read. The file may be corrupted."
        ) from exc
    return "\n\n".join(blocks), warnings