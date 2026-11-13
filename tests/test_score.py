import json

import pytest

from pvcrew import score
from pvcrew.score import bucket, format_report, score_rows


def row(i, pred, truth):
    base = {"patient_age": 30, "patient_sex": "male", "suspect_drug": "ASPIRIN", "reactions": ["Rash"],
            "serious": False, "outcome": "recovered"}
    return {"id": i, "prediction": None if pred is None else {**base, **pred}, "truth": {**base, **truth}}


@pytest.mark.parametrize("field,p,t,want", [
    ("patient_age", 30, 30, "correct"), ("patient_age", None, 30, "missed"), ("patient_age", 31, 30, "wrong"),
    ("patient_age", 30, None, "spurious"), ("patient_age", None, None, "correct"),
    ("patient_sex", "unknown", "male", "missed"), ("patient_sex", "male", "female", "wrong"),
    ("patient_sex", "male", "unknown", "spurious"), ("patient_sex", "Male", "male", "correct"),
    ("outcome", "unknown", "fatal", "missed"), ("outcome", "fatal", "recovered", "wrong"),
    ("suspect_drug", "duragesic", "DURAGESIC-100", "correct"), ("suspect_drug", "ibuprofen", "ASPIRIN", "wrong"),
    ("serious", True, False, "wrong"), ("serious", False, False, "correct"),
])
def test_bucket(field, p, t, want):
    assert bucket(field, p, t) == want


def test_perfect_scores():
    s = score_rows([row("a", {}, {}), row("b", {}, {})])
    assert s["scored"] == 2 and all(c["accuracy"] == 1.0 for c in s["fields"].values()) and not s["mistakes"]
    assert s["reactions"] == {"exact_set_match": 1.0, "precision": 1.0, "recall": 1.0}


def test_counts_and_mistake_list():
    s = score_rows([row("a", {"patient_age": None}, {}), row("b", {}, {}), row("c", {"patient_sex": "female"}, {})])
    assert s["fields"]["patient_age"]["missed"] == 1 and s["fields"]["patient_age"]["accuracy"] == 0.667
    assert s["fields"]["patient_sex"]["wrong"] == 1
    assert ("a", "patient_age", "missed", None, 30) in s["mistakes"]


def test_invalid_rows_are_counted_but_not_scored():
    s = score_rows([row("a", {}, {}), row("b", None, {})])
    assert (s["total"], s["scored"], s["invalid"]) == (2, 1, 1)


def test_reaction_precision_recall_and_case_insensitive():
    s = score_rows([row("a", {"reactions": ["rash", "nausea"]}, {"reactions": ["Rash", "Dyspnoea"]})])
    assert s["reactions"] == {"exact_set_match": 0.0, "precision": 0.5, "recall": 0.5}
    assert ("a", "reactions", "differs", ["nausea"], ["dyspnoea"]) in s["mistakes"]


def test_empty_input_does_not_divide_by_zero():
    s = score_rows([])
    assert s["scored"] == 0 and s["reactions"]["recall"] == 0.0
    assert "0 scored" in format_report(s)


def test_report_text_and_show_limit():
    rows = [row(str(i), {"patient_age": None}, {}) for i in range(5)]
    text = format_report(score_rows(rows), show=2)
    assert "patient_age" in text and "first 2 of 5 mistakes" in text and text.count(" | missed | ") == 2
    assert "mistakes" not in format_report(score_rows(rows), show=0)


def test_cli_reads_a_file_and_reports_missing_file(tmp_path, capsys):
    p = tmp_path / "e.jsonl"
    p.write_text(json.dumps(row("a", {"outcome": "unknown"}, {})) + "\n", encoding="utf-8")
    score.main([str(p)])
    assert "1 scored" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        score.main([str(tmp_path / "nope.jsonl")])
    assert "not found" in str(e.value)


def test_precision_and_recall_differ_when_the_lists_differ_in_size():
    s = score_rows([row("a", {"reactions": ["rash"]}, {"reactions": ["rash", "dyspnoea", "nausea", "fever"]})])
    assert s["reactions"]["precision"] == 1.0 and s["reactions"]["recall"] == 0.25


def test_coded_rows_are_scored_separately_from_the_raw_extraction():
    r = row("a", {"reactions": ["the runs"]}, {"reactions": ["Diarrhoea"]})
    r["reactions_coded"] = ["Diarrhoea"]
    s = score_rows([r])
    assert s["reactions"]["exact_set_match"] == 0.0 and s["reactions_coded"]["exact_set_match"] == 1.0
    text = format_report(s)
    assert "as extracted" in text and "after coding" in text


def test_uncoded_files_have_no_coded_line():
    s = score_rows([row("a", {}, {})])
    assert "reactions_coded" not in s and "after coding" not in format_report(s)


def test_after_coding_only_the_remaining_reaction_mistakes_are_listed():
    fixed = row("a", {"reactions": ["the runs"]}, {"reactions": ["Diarrhoea"]})
    fixed["reactions_coded"] = ["Diarrhoea"]
    left = row("b", {"reactions": ["x"]}, {"reactions": ["Rash"]})
    left["reactions_coded"] = ["x"]
    kinds = [(m[0], m[1]) for m in score_rows([fixed, left])["mistakes"] if m[1].startswith("reactions")]
    assert kinds == [("b", "reactions_coded")]
