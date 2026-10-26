from types import SimpleNamespace

import pytest

from ner_app.inference.flair_ner import (
    FlairNER,
    InferenceSettings,
    _chunk_ranges,
)


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
    """Return configured local spans for each chunk's exact text."""

    def __init__(self, spans_by_text):
        self.spans_by_text = spans_by_text

    def predict(self, sentences):
    # FlairNER passes one Sentence at a time in our current implementation.
        sentences.get_spans = lambda label_name: self.spans_by_text.get(
        sentences.to_original_text(),
        [],
    )


def test_chunk_ranges_cover_text_and_overlap():
    settings = InferenceSettings(chunk_size=10, chunk_overlap=2)

    assert _chunk_ranges("abcdefghijklmnop", settings) == [
        (0, 10),
        (8, 16),
    ]


def test_empty_text_returns_no_mentions():
    ner = FlairNER(_tagger=FakeTagger({}))

    assert ner.predict("") == []


def test_mention_text_matches_original_document_slice():
    text = "Jane Smith met officials."
    span = FakeSpan("Jane Smith", "PERSON", 0, 10)
    ner = FlairNER(
        settings=InferenceSettings(chunk_size=100, chunk_overlap=0),
        _tagger=FakeTagger({text: [span]}),
    )

    mentions = ner.predict(text)

    assert len(mentions) == 1
    assert mentions[0].text == "Jane Smith"
    assert text[mentions[0].start : mentions[0].end] == mentions[0].text
    assert mentions[0].label == "PERSON"


def test_identical_span_predicted_in_overlapping_chunks_is_deduplicated():
    text = "0123456789ABCDEFGHIJ"
    settings = InferenceSettings(chunk_size=12, chunk_overlap=4)
    chunks = _chunk_ranges(text, settings)

    # "ABCDEFGHIJ" starts at document offset 10, so it appears in both chunks.
    first_chunk = text[chunks[0][0] : chunks[0][1]]
    second_chunk = text[chunks[1][0] : chunks[1][1]]

    ner = FlairNER(
        settings=settings,
        _tagger=FakeTagger(
            {
                first_chunk: [FakeSpan("AB", "ORG", 10, 12, 0.8)],
                second_chunk: [FakeSpan("AB", "ORG", 2, 4, 0.95)],
            }
        ),
    )

    mentions = ner.predict(text)

    assert len(mentions) == 1
    assert (mentions[0].start, mentions[0].end) == (10, 12)
    assert mentions[0].text == "AB"
    assert mentions[0].score == pytest.approx(0.95)


def test_repeated_entity_text_at_different_offsets_is_preserved():
    text = "Jane met Jane."
    ner = FlairNER(
        settings=InferenceSettings(chunk_size=100, chunk_overlap=0),
        _tagger=FakeTagger(
            {
                text: [
                    FakeSpan("Jane", "PERSON", 0, 4),
                    FakeSpan("Jane", "PERSON", 9, 13),
                ]
            }
        ),
    )

    mentions = ner.predict(text)

    assert [(mention.text, mention.start, mention.end) for mention in mentions] == [
        ("Jane", 0, 4),
        ("Jane", 9, 13),
    ]


def test_entity_found_in_overlap_is_not_discarded_by_chunk_ownership():
    text = "abcdefghijKLMNOPQRST"
    settings = InferenceSettings(chunk_size=12, chunk_overlap=4)
    chunks = _chunk_ranges(text, settings)

    # The entity is only predicted by the second chunk.
    second_chunk = text[chunks[1][0] : chunks[1][1]]
    ner = FlairNER(
        settings=settings,
        _tagger=FakeTagger({
            second_chunk: [FakeSpan("KLM", "ORG", 2, 5)],            
        }),
    )

    mentions = ner.predict(text)

    assert len(mentions) == 1
    assert mentions[0].text == "KLM"
    assert (mentions[0].start, mentions[0].end) == (10, 13)

def test_invalid_chunk_settings_are_rejected():
    with pytest.raises(ValueError):
        InferenceSettings(chunk_size=10, chunk_overlap=10)


def test_model_info_documents_score_meaning():
    ner = FlairNER(_tagger=FakeTagger({}))

    assert ner.model_info.framework == "Flair"
    assert "not a calibrated probability" in ner.model_info.score_meaning
from ner_app.inference.flair_ner import _resolve_overlaps


def test_truncated_fragments_lose_to_the_full_span():
    candidates = {
        (76, 86, "PERSON"): 0.99,
        (81, 86, "PERSON"): 0.97,  # chunk that started mid-entity
        (90, 100, "ORG"): 0.95,  # chunk that ended mid-entity
        (90, 110, "ORG"): 0.99,
    }
    assert [key for key, _ in _resolve_overlaps(candidates)] == [
        (76, 86, "PERSON"),
        (90, 110, "ORG"),
    ]


def test_adjacent_and_separate_spans_are_all_kept():
    candidates = {(0, 4, "PERSON"): 0.9, (4, 8, "ORG"): None, (10, 12, "GPE"): 0.5}
    assert [key for key, _ in _resolve_overlaps(candidates)] == [
        (0, 4, "PERSON"),
        (4, 8, "ORG"),
        (10, 12, "GPE"),
    ]


def test_equal_length_overlap_prefers_the_higher_score():
    candidates = {(0, 5, "ORG"): 0.6, (2, 7, "PERSON"): 0.9}
    assert [key for key, _ in _resolve_overlaps(candidates)] == [(2, 7, "PERSON")]
    

class SearchTagger:
    """Tag every occurrence of known entities in a chunk, including truncated ones.

    A prefix or suffix of an entity cut by a chunk edge is tagged too, which
    mimics what a real model does at chunk boundaries.
    """

    def __init__(self, entities):
        self.entities = entities  # {"Jane Smith": "PERSON", ...}

    def predict(self, sentence):
        chunk = sentence.to_original_text()
        spans = []
        for name, label in self.entities.items():
            for start in range(len(chunk)):
                # whole entity
                if chunk.startswith(name, start):
                    spans.append(FakeSpan(name, label, start, start + len(name)))
                # entity cut by the end of the chunk
                tail = chunk[start:]
                if len(tail) < len(name) and name.startswith(tail) and len(tail) > 1:
                    spans.append(FakeSpan(tail, label, start, len(chunk)))
            # entity cut by the start of the chunk
            for cut in range(1, len(name)):
                suffix = name[cut:]
                if chunk.startswith(suffix) and len(suffix) > 1:
                    spans.append(FakeSpan(suffix, label, 0, len(suffix)))
        sentence.get_spans = lambda label_name: spans


def test_chunked_result_matches_unchunked_result():
    sentence = "Jane Smith of Example Technologies met officials in Berlin on 3 March 2021. "
    text = sentence * 12
    entities = {
        "Jane Smith": "PERSON",
        "Example Technologies": "ORG",
        "Berlin": "GPE",
        "3 March 2021": "DATE",
    }

    reference = FlairNER(
        settings=InferenceSettings(chunk_size=10_000, chunk_overlap=0),
        _tagger=SearchTagger(entities),
    ).predict(text)
    chunked = FlairNER(
        settings=InferenceSettings(chunk_size=100, chunk_overlap=20),
        _tagger=SearchTagger(entities),
    ).predict(text)

    spans = [(m.start, m.end, m.label) for m in chunked]
    assert spans == [(m.start, m.end, m.label) for m in reference]
    assert len(reference) == 48
    # no mention may overlap another
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))