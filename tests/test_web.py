from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ner_app.inference.flair_ner import FlairNER
from ner_app.web import create_app


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
def client():
    spans = [FakeSpan("Jane Smith", "PERSON", 0, 10), FakeSpan("Berlin", "GPE", 19, 25)]
    return TestClient(create_app(ner=FlairNER(_tagger=FakeTagger(spans))))


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ner-app" in response.text


def test_analyze_upload(client):
    files = {"file": ("notes.txt", b"Jane Smith visited Berlin.", "text/plain")}
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["result"]["document"]["source_name"] == "notes.txt"
    assert data["result"]["total_mentions"] == 2
    for m in data["result"]["mentions"]:
        assert data["text"][m["start"] : m["end"]] == m["text"]


def test_unsupported_format_is_422(client):
    response = client.post("/analyze", files={"file": ("a.xyz", b"hello", "text/plain")})
    assert response.status_code == 422


def test_empty_file_is_422(client):
    response = client.post("/analyze", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 422