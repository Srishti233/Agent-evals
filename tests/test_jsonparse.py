import pytest

from common.jsonparse import JSONParseError, extract_json


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here you go: {"a": 1, "b": "x"} Hope that helps.', {"a": 1, "b": "x"}),
    ('{"a": 1,}', {"a": 1}),
    ('{"a": [1, 2,],}', {"a": [1, 2]}),
    ('noise {not json} then {"ok": true}', {"ok": True}),
    ('{"nested": {"k": "v"}}', {"nested": {"k": "v"}}),
])
def test_recovers_json(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "no json here", "[1, 2, 3]", '{"a": ', None])
def test_raises_on_garbage(text):
    with pytest.raises(JSONParseError):
        extract_json(text)
