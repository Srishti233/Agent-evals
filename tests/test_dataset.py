import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_eval_set_shape_and_references():
    cases = json.loads((ROOT / "data/eval_set.json").read_text())
    assert len(cases) == 40
    assert len({c["id"] for c in cases}) == 40
    cats = Counter(c["category"] for c in cases)
    assert set(cats) == {"factual", "multi-hop", "math", "unanswerable"}
    assert cats["unanswerable"] == 8
    doc_ids = {p.stem for p in (ROOT / "data/docs").glob("*.md")}
    assert 15 <= len(doc_ids) <= 20
    for c in cases:
        assert set(c) >= {"question", "gold_answer", "supporting_doc_ids", "category"}
        assert set(c["supporting_doc_ids"]) <= doc_ids
        assert (c["category"] == "unanswerable") == (c["supporting_doc_ids"] == [])
        if c["category"] == "multi-hop":
            assert len(c["supporting_doc_ids"]) >= 1
