import pytest

from evals.hallucination import detect_hallucinations, numbers_in
from evals.judge import validate_rubric, validate_verdict
from evals.metrics import (aggregate, composite_score, fuzzy_score, is_abstention, quality_from_ratings,
                           split_claims)
from evals.nli import LexicalNLI


@pytest.mark.parametrize("text,expected", [
    ("I don't know.", True), ("I do not know the answer", True), ("I don’t know", True),
    ("The documents do not mention this.", True), ("I cannot find that information.", True),
    ("The payload is 1,200 kg.", False), ("Atlas moves pallets.", False), ("", False),
])
def test_is_abstention(text, expected):
    assert is_abstention(text) is expected


def test_split_claims():
    ans = "Atlas carries 1,200 kg. It runs for 8 hours! Yes.\n- The price is $24,000 per unit.\nWhat else?"
    assert split_claims(ans) == ["Atlas carries 1,200 kg.", "It runs for 8 hours!", "The price is $24,000 per unit."]
    assert split_claims("I don't know.") == []
    assert split_claims("") == []
    assert split_claims("The price is $1.50 today. I don't know about tomorrow.") == ["The price is $1.50 today."]


def test_quality_normalisation():
    assert quality_from_ratings(1, 1, 1) == 0.0
    assert quality_from_ratings(5, 5, 5) == 1.0
    assert quality_from_ratings(3, 3, 3) == 0.5
    assert quality_from_ratings(4, 4, 5) == pytest.approx(0.8333, abs=1e-3)


def test_composite_formula():
    assert composite_score(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert composite_score(0.0, 1.0, 0.0) == pytest.approx(0.0)
    assert composite_score(0.5, 0.25, 0.8) == pytest.approx(0.4 * 0.5 + 0.4 * 0.75 + 0.2 * 0.8)


def test_fuzzy_score():
    assert fuzzy_score("1,200 kg", "1200 kg") == 1.0
    assert fuzzy_score("completely different", "24 days") < 0.5
    assert fuzzy_score("", "x") == 0.0


def _row(cat, correct, hall, quality, error=None):
    return {"category": cat, "correct": correct, "hallucinated": hall, "quality": quality, "error": error}


def test_aggregate():
    rows = [_row("a", True, False, 1.0), _row("a", False, True, 0.5), _row("b", True, False, 0.0, error="x"),
            _row("b", False, False, None)]
    m = aggregate(rows)
    assert m["n"] == 4 and m["accuracy"] == 0.5 and m["hallucination_rate"] == 0.25
    assert m["quality"] == pytest.approx(0.5)            # None excluded from the mean
    assert m["n_errors"] == 1 and m["error_rate"] == 0.25
    assert m["by_category"]["a"]["accuracy"] == 0.5
    assert m["composite"] == pytest.approx(composite_score(0.5, 0.25, 0.5))
    assert aggregate([])["n"] == 0


def test_numbers_in_normalises():
    assert numbers_in("$1,200 and 2.50 and 007") == {"1200", "2.5", "7"}


CTX = [{"doc_id": "d", "chunk_id": 0, "text": "Atlas can carry a payload of 1,200 kg and has a top speed of 2.0 m/s."}]


def test_hallucination_supported_vs_unsupported():
    nli = LexicalNLI()
    ok = detect_hallucinations("Atlas can carry a payload of 1,200 kg.", CTX, [], nli)
    assert ok["hallucinated"] is False and ok["n_claims"] == 1
    bad = detect_hallucinations("Atlas can carry 1,200 kg. Atlas was designed by astronauts in Antarctica.", CTX, [], nli)
    assert bad["hallucinated"] is True and bad["n_unsupported"] == 1
    assert "Antarctica" in bad["unsupported"][0]


def test_hallucination_no_context_means_unsupported_and_abstention_is_clean():
    nli = LexicalNLI()
    assert detect_hallucinations("The CFO is Jane Doe.", [], [], nli)["hallucinated"] is True
    assert detect_hallucinations("I don't know.", [], [], nli)["hallucinated"] is False


def test_hallucination_calculator_grounded_numbers():
    nli = LexicalNLI()
    ctx = [{"doc_id": "d", "chunk_id": 0, "text": "Atlas costs $24,000 per unit."}]
    calls = [{"tool": "calculator", "input": "24000 * 10", "output": "240000"}]
    r = detect_hallucinations("Ten units would total $240,000 in all.", ctx, calls, nli)
    assert r["hallucinated"] is False and r["verdicts"][0]["supported_by"] in ("tool", "nli")
    # a made-up number that no tool/context produced is still flagged
    r2 = detect_hallucinations("Ten units would total $999,000 in all.", ctx, calls, nli)
    assert r2["hallucinated"] is True


def test_judge_schema_validation():
    assert validate_verdict({"verdict": "Correct", "reason": "ok"})["verdict"] == "correct"
    for bad in ({"verdict": "maybe"}, {}, {"verdict": None}):
        with pytest.raises(ValueError):
            validate_verdict(bad)
    assert validate_rubric({"relevance": 5, "completeness": "4", "clarity": 3.0}) == \
        {"relevance": 5, "completeness": 4, "clarity": 3}
    for bad in ({"relevance": 6, "completeness": 3, "clarity": 3}, {"relevance": 3, "completeness": 3},
                {"relevance": "x", "completeness": 3, "clarity": 3}, {"relevance": 0, "completeness": 3, "clarity": 3}):
        with pytest.raises(ValueError):
            validate_rubric(bad)
