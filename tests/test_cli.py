import json
from types import SimpleNamespace

import pytest

import ner_app.cli as cli
import ner_app.pipeline as pipeline
from ner_app.inference.flair_ner import FlairNER, InferenceError
from ner_app.models import AnalysisResult

TEXT = "Jane Smith visited Berlin."


class FakeSpan:
    def __init__(self, text, label, start, end, score=0.99):
        self.text = text
        self.start_position = start
        self.end_position = end
        self._label = SimpleNamespace(value=label, score=score)

    def get_label(self, typename):
        return self._label


class FakeTagger:
    def __init__(self, spans):
        self.spans = spans

    def predict(self, sentence):
        sentence.get_spans = lambda label_name: self.spans


@pytest.fixture
def fake_model(monkeypatch):
    spans = [
        FakeSpan("Jane Smith", "PERSON", 0, 10),
        FakeSpan("Berlin", "GPE", 19, 25),
    ]
    monkeypatch.setattr(
        pipeline,
        "FlairNER",
        lambda model_id, settings: FlairNER(model_id, settings, _tagger=FakeTagger(spans)),
    )


@pytest.fixture
def doc(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


def test_text_output(fake_model, doc, capsys):
    assert cli.main(["analyze", str(doc)]) == 0
    out = capsys.readouterr().out
    assert "doc.txt" in out
    assert "PERSON" in out and "'Jane Smith'" in out and "[0:10]" in out
    assert "GPE" in out and "'Berlin'" in out
    assert "not probabilities" in out


def test_json_output_parses_and_round_trips(fake_model, doc, capsys):
    assert cli.main(["analyze", str(doc), "--format", "json"]) == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_mentions"] == 2
    assert [m["text"] for m in data["mentions"]] == ["Jane Smith", "Berlin"]
    assert AnalysisResult.model_validate_json(out).total_mentions == 2


def test_missing_file_exits_2_without_loading_model(tmp_path, monkeypatch, capsys):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("model should not be loaded")

    monkeypatch.setattr(pipeline, "FlairNER", _must_not_be_called)

    assert cli.main(["analyze", str(tmp_path / "nope.txt")]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert captured.out == ""


def test_inference_error_uses_its_exit_code(monkeypatch, doc, capsys):
    def boom(*args, **kwargs):
        raise InferenceError("model exploded")

    monkeypatch.setattr(cli, "analyze_file", boom)

    assert cli.main(["analyze", str(doc)]) == 1
    assert "model exploded" in capsys.readouterr().err


def test_invalid_chunk_settings_are_a_usage_error(doc, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["analyze", str(doc), "--chunk-size", "10", "--chunk-overlap", "10"])
    assert excinfo.value.code == 2
    assert "chunk_overlap" in capsys.readouterr().err


def test_subcommand_is_required(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2