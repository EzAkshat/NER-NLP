import pytest

from ner_app.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    ExtractionError,
    ResourceLimitError,
    UnsupportedFormatError,
)
from ner_app.ingestion import SUPPORTED_EXTENSIONS, Limits, extract_document
from ner_app.ingestion.normalize import normalize_text


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return path


def codes(doc):
    return {w.code for w in doc.warnings}


# ---------- normalization ----------

def test_normalize_whitespace_and_line_endings():
    assert normalize_text("a  b\t c\u00a0d") == "a b c d"
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    assert normalize_text("one\n\n\n\n\ntwo") == "one\n\ntwo"
    assert normalize_text("  padded  \n") == "padded"


def test_normalize_unicode_form_and_invisible_chars():
    assert normalize_text("Cafe\u0301") == "Caf\u00e9"
    assert normalize_text("zero\u200bwidth") == "zerowidth"


def test_normalize_keeps_wording_and_punctuation():
    original = "Dr. Smith's \u201cquote\u201d \u2014 fine."
    assert normalize_text(original) == original


# ---------- plain text ----------

def test_txt_basic(tmp_path):
    doc = extract_document(write(tmp_path, "a.txt", "Jane  Smith\r\nvisited Berlin.\r\n\r\n\r\nShe left."))
    assert doc.text == "Jane Smith\nvisited Berlin.\n\nShe left."
    assert doc.detected_format == "txt"
    assert doc.source_name == "a.txt"
    assert doc.warnings == ()


def test_uppercase_extension_is_accepted(tmp_path):
    assert extract_document(write(tmp_path, "A.TXT", "Hello world")).text == "Hello world"


def test_utf16_with_bom(tmp_path):
    doc = extract_document(write(tmp_path, "u.txt", "Caf\u00e9 M\u00fcller".encode("utf-16")))
    assert doc.text == "Caf\u00e9 M\u00fcller"


def test_invalid_utf8_falls_back_with_warning(tmp_path):
    doc = extract_document(write(tmp_path, "l.txt", b"Caf\xe9 society"))
    assert doc.text == "Caf\u00e9 society"
    assert "encoding_fallback" in codes(doc)


def test_binary_content_rejected(tmp_path):
    with pytest.raises(ExtractionError):
        extract_document(write(tmp_path, "b.txt", b"abc\x00\x01def"))


def test_to_document_info(tmp_path):
    doc = extract_document(write(tmp_path, "a.txt", "Hello world"))
    info = doc.to_document_info()
    assert (info.source_name, info.detected_format, info.char_count) == ("a.txt", "txt", 11)


# ---------- validation and limits ----------

def test_missing_file(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        extract_document(tmp_path / "nope.txt")


def test_directory_rejected(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        extract_document(tmp_path)


def test_unsupported_extension(tmp_path):
    with pytest.raises(UnsupportedFormatError) as info:
        extract_document(write(tmp_path, "a.exe", "hello"))
    assert ".txt" in str(info.value)


@pytest.mark.parametrize("content", ["", "   \n\t  ", "... --- !!!"])
def test_empty_or_meaningless_documents(tmp_path, content):
    with pytest.raises(EmptyDocumentError):
        extract_document(write(tmp_path, "e.txt", content))


def test_file_size_limit(tmp_path):
    with pytest.raises(ResourceLimitError):
        extract_document(write(tmp_path, "big.txt", "hello world"), Limits(max_file_bytes=5))


def test_char_limit(tmp_path):
    with pytest.raises(ResourceLimitError):
        extract_document(write(tmp_path, "long.txt", "hello world"), Limits(max_chars=5))


def test_supported_extensions():
    assert set(SUPPORTED_EXTENSIONS) == {".txt", ".md", ".html", ".htm", ".pdf", ".docx"}

# ---------- HTML ----------

def test_html_strips_markup_keeps_paragraphs(tmp_path):
    html = (
        "<html><head><title>T</title><style>p{color:red}</style></head><body>"
        "<h1>Report</h1><p>Jane <b>Smith</b> joined &amp; left.</p>"
        "<script>alert('x')</script><p>Second.</p></body></html>"
    )
    doc = extract_document(write(tmp_path, "a.html", html))
    assert doc.text == "Report\n\nJane Smith joined & left.\n\nSecond."
    assert doc.detected_format == "html"


def test_html_inline_tags_do_not_split_words(tmp_path):
    assert extract_document(write(tmp_path, "a.htm", "<p>un<b>believ</b>able</p>")).text == "unbelievable"


def test_html_meta_charset_is_honored(tmp_path):
    raw = '<html><head><meta charset="iso-8859-1"></head><body><p>Caf\xe9</p></body></html>'
    doc = extract_document(write(tmp_path, "c.html", raw.encode("latin-1")))
    assert doc.text == "Caf\u00e9"
    assert doc.warnings == ()


def test_html_unclosed_script_warns_and_hides_script(tmp_path):
    doc = extract_document(write(tmp_path, "u.html", "<p>Hello world</p><script>var x = 1;"))
    assert "Hello world" in doc.text
    assert "var x" not in doc.text
    assert "html_unclosed_element" in codes(doc)


# ---------- Markdown ----------

def test_markdown_syntax_is_removed(tmp_path):
    md = "# Title\n\nSome **bold** text with [a link](http://x.example).\n\n- item one\n- item two\n"
    doc = extract_document(write(tmp_path, "a.md", md))
    assert "Title" in doc.text
    assert "Some bold text with a link." in doc.text
    assert "item one" in doc.text and "item two" in doc.text
    assert "**" not in doc.text
    assert "http://x.example" not in doc.text
    assert doc.detected_format == "md"