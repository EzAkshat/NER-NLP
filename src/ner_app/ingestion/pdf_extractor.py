"""Text-based PDF extraction. OCR is out of scope: scanned PDFs are detected and reported."""

from __future__ import annotations

import io

from pypdf import PdfReader

from ner_app.errors import EmptyDocumentError, ExtractionError, ResourceLimitError
from ner_app.ingestion.types import Limits, RawExtraction
from ner_app.models import ExtractionWarning


def extract_pdf(data: bytes, limits: Limits) -> RawExtraction:
    locked = False
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and int(reader.decrypt("")) == 0:
            locked = True
        page_count = 0 if locked else len(reader.pages)
    except Exception as exc:  # noqa: BLE001 - malformed PDFs must become a clear error
        raise ExtractionError(
            "The PDF could not be read. It may be corrupted, password-protected, "
            "or not really a PDF."
        ) from exc

    if locked:
        raise ExtractionError("The PDF is password-protected and cannot be read.")
    if page_count == 0:
        raise ExtractionError("The PDF has no readable pages.")
    if page_count > limits.max_pdf_pages:
        raise ResourceLimitError(
            f"The PDF has {page_count:,} pages, which is over the "
            f"{limits.max_pdf_pages:,} page limit."
        )

    page_texts: list[str] = []
    empty_pages = 0
    total_chars = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            raise ExtractionError(f"Text could not be extracted from page {number}.") from exc
        total_chars += len(text)
        if total_chars > limits.max_chars:
            raise ResourceLimitError(
                f"The PDF text is over the {limits.max_chars:,} character limit."
            )
        if any(ch.isalnum() for ch in text):
            page_texts.append(text)
        else:
            empty_pages += 1

    if not page_texts:
        raise EmptyDocumentError(
            "No text could be extracted. This PDF looks like scanned images, and "
            "scanned documents (OCR) are not supported in this version."
        )

    warnings: list[ExtractionWarning] = []
    if empty_pages:
        warnings.append(
            ExtractionWarning(
                code="pdf_pages_without_text",
                message=(
                    f"{empty_pages} of {page_count} pages had no extractable text "
                    "(possibly scanned images), so entities on those pages are missing."
                ),
            )
        )
    return "\n\n".join(page_texts), warnings