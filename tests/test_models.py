import pytest
from pydantic import ValidationError

from ner_app.models import (
    AnalysisResult,
    DocumentInfo,
    EntityGroup,
    Mention,
    ModelInfo,
)


def mention(i: int, text: str, label: str, start: int, score=None) -> Mention:
    return Mention(id=i, text=text, label=label, start=start, end=start + len(text), score=score)


def make_result(mentions, groups=()) -> AnalysisResult:
    return AnalysisResult(
        document=DocumentInfo(source_name="a.txt", detected_format="txt", char_count=100),
        model=ModelInfo(
            model_id="test", framework="flair", framework_version="0", score_meaning="raw, uncalibrated"
        ),
        mentions=list(mentions),
        groups=list(groups),
    )


def test_valid_mention():
    m = mention(0, "Berlin", "GPE", 42, score=0.9)
    assert (m.start, m.end) == (42, 48)


def test_end_must_exceed_start():
    with pytest.raises(ValidationError):
        Mention(id=0, text="x", label="ORG", start=5, end=5)


def test_text_length_must_match_span():
    with pytest.raises(ValidationError):
        Mention(id=0, text="Berlin", label="GPE", start=0, end=3)


def test_score_range_and_none():
    assert mention(0, "BBC", "ORG", 0).score is None
    with pytest.raises(ValidationError):
        mention(0, "BBC", "ORG", 0, score=1.5)


def test_counts_by_label_are_sorted_and_correct():
    r = make_result([
        mention(0, "BBC", "ORG", 0),
        mention(1, "Berlin", "GPE", 10),
        mention(2, "BBC", "ORG", 20),
    ])
    assert r.total_mentions == 3
    assert list(r.counts_by_label.items()) == [("GPE", 1), ("ORG", 2)]


def test_group_keeps_all_mentions_and_counts():
    ms = [mention(0, "BBC", "ORG", 0), mention(1, "BBC", "ORG", 20)]
    r = make_result(ms, [EntityGroup(label="ORG", text="BBC", mention_ids=[0, 1])])
    assert r.groups[0].count == 2
    assert len(r.mentions) == 2


def test_group_with_unknown_mention_rejected():
    with pytest.raises(ValidationError):
        make_result([mention(0, "BBC", "ORG", 0)],
                    [EntityGroup(label="ORG", text="BBC", mention_ids=[7])])


def test_group_must_match_mention_text_and_label():
    with pytest.raises(ValidationError):
        make_result([mention(0, "BBC", "ORG", 0)],
                    [EntityGroup(label="GPE", text="BBC", mention_ids=[0])])


def test_mention_in_two_groups_rejected():
    ms = [mention(0, "BBC", "ORG", 0)]
    groups = [EntityGroup(label="ORG", text="BBC", mention_ids=[0]),
              EntityGroup(label="ORG", text="BBC", mention_ids=[0])]
    with pytest.raises(ValidationError):
        make_result(ms, groups)


def test_json_round_trip():
    r = make_result([mention(0, "Café Müller", "FAC", 3, score=0.9998)])
    assert AnalysisResult.model_validate_json(r.model_dump_json()) == r