from pathlib import Path
from types import SimpleNamespace

import pytest

import ner_app.pipeline as pipeline
from ner_app.errors import DocumentError
from ner_app.inference.flair_ner import FlairNER, InferenceError
from ner_app.ingestion import extract_document
from ner_app.ingestion.types import ExtractedDocument
from ner_app.models import ExtractionWarning, Mention, ModelInfo
from ner_app.pipeline import analyze_document, analyze_file

SAMPLE_HTML = Path(__file__).parent.parent / "samples" / "sample.html"


class FakeSpan:
    def __init__(self, text, label, start, end, score=0.99):
        self.text = text
        self.start_position = start
        self.end_position = end
        self._label = SimpleNamespace(value=label, score=score)

    def get_label(self, typename):
        assert typename == "ner"
        return self._label

class FakeTagger:
    """Return the same configured spans for any chunk (tests use a single chunk)."""

    def __init__(self, spans):
        self.spans = spans

    def predict(self, sentence):
        sentence.get_spans = lambda label_name: self.spans


def span(text, needle, label, nth=0):
    """Build a FakeSpan for the nth occurrence of `needle` in `text`."""
    start = -1
    for _ in range(nth + 1):
        start = text.find(needle, start + 1)
        assert start != -1, f"{needle!r} not found in text"
    return FakeSpan(needle, label, start, start + len(needle))


def make_ner(text, spans):
    # Default chunk size (10,000) means short test texts form a single chunk.
    return FlairNER(_tagger=FakeTagger(spans))


def test_analyze_file_html_success():
    text = extract_document(SAMPLE_HTML).text
    ner = make_ner(
        text,
        [span(text, "Jane Smith", "PERSON"), span(text, "Berlin", "GPE")],
    )

    result = analyze_file(SAMPLE_HTML, ner=ner)

    assert result.document.source_name == "sample.html"
    assert result.document.detected_format == "html"
    assert result.document.char_count == len(text)
    assert result.model.framework == "Flair"
    assert [m.text for m in result.mentions] == ["Jane Smith", "Berlin"]
    assert result.total_mentions == 2
    assert result.counts_by_label == {"GPE": 1, "PERSON": 1}
    for mention in result.mentions:
        assert text[mention.start : mention.end] == mention.text


def test_repeated_entities_are_grouped_but_not_removed(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("Jane met Jane in Berlin.", encoding="utf-8")
    text = extract_document(path).text
    ner = make_ner(
        text,
        [
            span(text, "Jane", "PERSON", 0),
            span(text, "Jane", "PERSON", 1),
            span(text, "Berlin", "GPE"),
        ],
    )

    result = analyze_file(path, ner=ner)

    assert result.total_mentions == 3  # repeated names are preserved
    jane_ids = [m.id for m in result.mentions if m.text == "Jane"]
    groups = {(g.label, g.text): g for g in result.groups}
    assert groups[("PERSON", "Jane")].count == 2
    assert groups[("PERSON", "Jane")].mention_ids == jane_ids
    assert groups[("GPE", "Berlin")].count == 1


def test_no_entities_gives_empty_result(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("Nothing to see here.", encoding="utf-8")
    ner = FlairNER(_tagger=FakeTagger([]))  
    result = analyze_file(path, ner=ner)

    assert result.mentions == []
    assert result.groups == []
    assert result.total_mentions == 0


def test_ingestion_warnings_are_carried_through():
    warning = ExtractionWarning(code="test_warning", message="Something happened.")
    document = ExtractedDocument(
        source_name="x.txt",
        detected_format="txt",
        text="Jane Smith",
        warnings=(warning,),
    )
    ner = make_ner(document.text, [span(document.text, "Jane Smith", "PERSON")])

    result = analyze_document(document, ner)

    assert result.document.warnings == [warning]


def test_ingestion_failure_never_loads_a_model(tmp_path, monkeypatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("model should not be loaded when ingestion fails")

    monkeypatch.setattr(pipeline, "FlairNER", _must_not_be_called)

    with pytest.raises(DocumentError):
        analyze_file(tmp_path / "missing.txt")

    unsupported = tmp_path / "data.xyz"
    unsupported.write_text("hello", encoding="utf-8")
    with pytest.raises(DocumentError):
        analyze_file(unsupported)


def test_mention_that_does_not_match_document_text_is_rejected():
    class WrongOffsetsNER:
        model_info = ModelInfo(
            model_id="fake",
            framework="Fake",
            framework_version="0",
            score_meaning="n/a",
        )

        def predict(self, text):
            # Valid on its own (length 3 == 3 - 0), but the text at 0-3 is "Jan".
            return [Mention(id=0, text="Bob", label="PERSON", start=0, end=3)]
    document = ExtractedDocument(
        source_name="x.txt", detected_format="txt", text="Jane Smith"
    )

    with pytest.raises(InferenceError):
        analyze_document(document, WrongOffsetsNER())