import json

import pytest

from ner_app.inference.flair_ner import InferenceError
from ner_app.inference.gliner_ner import FINETUNED_LABELS, GlinerNER, load_label_thresholds


class FakeModel:
    def __init__(self, entities):
        self.entities = entities
        self.thresholds = []

    def predict_entities(self, text, labels, threshold=0.5):
        self.thresholds.append(threshold)
        return [e for e in self.entities if e["score"] >= threshold]


def entity(text, label, start, score):
    return {"text": text, "label": label, "start": start, "end": start + len(text), "score": score}


def test_title_is_stripped_and_offsets_stay_correct():
    text = "Dr. Priya Desai joined the lab."
    fake = FakeModel([entity("Dr. Priya Desai", "person", 0, 0.9)])
    ner = GlinerNER("x", FINETUNED_LABELS, 0.5, strip_titles=True, _model=fake)
    [m] = ner.predict(text)
    assert (m.text, m.start, m.end) == ("Priya Desai", 4, 15)
    assert text[m.start:m.end] == m.text


def test_titles_are_kept_by_default():
    text = "Dr. Priya Desai joined the lab."
    fake = FakeModel([entity("Dr. Priya Desai", "person", 0, 0.9)])
    [m] = GlinerNER("x", FINETUNED_LABELS, 0.5, _model=fake).predict(text)
    assert m.text == "Dr. Priya Desai"


def test_per_label_threshold_is_applied():
    text = "Anna arrived at noon on Monday."
    fake = FakeModel([
        entity("noon", "time", text.index("noon"), 0.45),
        entity("Monday", "date", text.index("Monday"), 0.5),
    ])
    ner = GlinerNER("x", FINETUNED_LABELS, 0.7, label_thresholds={"TIME": 0.4}, _model=fake)
    assert [m.label for m in ner.predict(text)] == ["TIME"]
    assert fake.thresholds == [0.4]  # the model was asked at the lowest threshold


def test_model_is_queried_at_the_global_threshold_without_overrides():
    fake = FakeModel([])
    GlinerNER("x", FINETUNED_LABELS, 0.6, _model=fake).predict("Some text.")
    assert fake.thresholds == [0.6]


def test_invalid_label_thresholds_are_rejected():
    with pytest.raises(ValueError):
        GlinerNER("x", FINETUNED_LABELS, 0.5, label_thresholds={"NOPE": 0.5}, _model=FakeModel([]))
    with pytest.raises(ValueError):
        GlinerNER("x", FINETUNED_LABELS, 0.5, label_thresholds={"TIME": 1.5}, _model=FakeModel([]))


def test_load_label_thresholds(tmp_path):
    assert load_label_thresholds(tmp_path) == (None, {})
    (tmp_path / "label_thresholds.json").write_text(
        json.dumps({"default": 0.7, "labels": {"time": 0.45, "person": 0.6}}), encoding="utf-8"
    )
    assert load_label_thresholds(tmp_path) == (0.7, {"TIME": 0.45, "PERSON": 0.6})


def test_broken_threshold_file_is_a_clear_error(tmp_path):
    (tmp_path / "label_thresholds.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InferenceError):
        load_label_thresholds(tmp_path)