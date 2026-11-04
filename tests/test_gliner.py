import pytest

from ner_app.inference.factory import parse_labels
from ner_app.inference.flair_ner import InferenceError, InferenceSettings
from ner_app.inference.gliner_ner import GlinerNER


class FakeGliner:
    """Find every whole occurrence of known names in each chunk."""

    def __init__(self, names):
        self.names = names  # {"Jane Smith": "Person"}
        self.calls = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.calls.append((labels, threshold))
        found = []
        for name, label in self.names.items():
            start = text.find(name)
            while start != -1:
                found.append({"start": start, "end": start + len(name),
                              "text": name, "label": label, "score": 0.9})
                start = text.find(name, start + 1)
        return found


def make(names, **kwargs):
    return GlinerNER(_model=FakeGliner(names), **kwargs)


def test_entity_seen_in_two_chunks_appears_once_with_global_offsets():
    text = "Hello Jane Smith went home"
    ner = make({"Jane Smith": "Person"}, settings=InferenceSettings(chunk_size=16, chunk_overlap=10))
    mentions = ner.predict(text)
    assert [(m.text, m.label, m.start, m.end) for m in mentions] == [("Jane Smith", "PERSON", 6, 16)]
    assert mentions[0].id == 0


def test_entity_in_a_later_chunk_gets_document_offsets():
    text = "a" * 30 + " Berlin rocks"
    ner = make({"Berlin": "Location"}, settings=InferenceSettings(chunk_size=20, chunk_overlap=5))
    mentions = ner.predict(text)
    assert [(m.text, m.label, m.start) for m in mentions] == [("Berlin", "GPE", 31)]
    assert text[mentions[0].start : mentions[0].end] == "Berlin"


def test_invalid_span_from_model_is_an_inference_error():
    class Bad(FakeGliner):
        def predict_entities(self, text, labels, threshold=0.5):
            return [{"start": 2, "end": 999, "text": "x", "label": "Person", "score": 0.9}]

    with pytest.raises(InferenceError):
        GlinerNER(_model=Bad({})).predict("short text")


def test_labels_and_threshold_reach_the_model():
    fake = FakeGliner({})
    GlinerNER(labels={"PERSON": "Person"}, threshold=0.3, _model=fake).predict("some text")
    assert fake.calls[0] == (["Person"], 0.3)


def test_parse_labels():
    assert parse_labels("PERSON=Person, COURSE_CODE=Course code, marks") == {
        "PERSON": "Person", "COURSE_CODE": "Course code", "marks": "marks"}
    assert parse_labels(None) is None
    with pytest.raises(ValueError):
        parse_labels("=oops")


def test_duplicate_prompts_are_rejected():
    with pytest.raises(ValueError):
        GlinerNER(labels={"A": "Person", "B": "person"}, _model=FakeGliner({}))