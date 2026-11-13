from pvcrew.jsonl import read_jsonl, write_jsonl


def test_round_trip_with_unicode_and_nested_parent_dirs(tmp_path):
    rows = [{"a": 1, "text": "café – naïve"}, {"a": 2, "nested": {"x": [1, 2]}}]
    path = tmp_path / "deep" / "dir" / "f.jsonl"
    assert write_jsonl(path, rows) == 2
    assert list(read_jsonl(path)) == rows


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in read_jsonl(path)] == [1, 2]


def test_empty_write(tmp_path):
    path = tmp_path / "f.jsonl"
    assert write_jsonl(path, []) == 0
    assert list(read_jsonl(path)) == []
