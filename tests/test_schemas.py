import pytest
from pydantic import ValidationError

from pvcrew.schemas import Case


def test_minimal_case_uses_defaults():
    c = Case(suspect_drug="Aspirin", reactions=["Nausea"], serious=False)
    assert c.patient_age is None
    assert c.patient_sex == "unknown"
    assert c.outcome == "unknown"


def test_reactions_are_stripped_and_deduplicated_case_insensitively():
    c = Case(suspect_drug="X", reactions=[" Rash ", "rash", "", "Nausea"], serious=False)
    assert c.reactions == ["Rash", "Nausea"]


def test_drug_name_is_stripped():
    assert Case(suspect_drug="  Aspirin ", reactions=["a"], serious=False).suspect_drug == "Aspirin"


@pytest.mark.parametrize(
    "bad",
    [
        {"suspect_drug": "", "reactions": ["a"], "serious": False},
        {"suspect_drug": "   ", "reactions": ["a"], "serious": False},
        {"suspect_drug": "X", "reactions": [], "serious": False},
        {"suspect_drug": "X", "reactions": ["  "], "serious": False},
        {"suspect_drug": "X", "reactions": ["a"]},  # serious is required
        {"suspect_drug": "X", "reactions": ["a"], "serious": False, "patient_age": 200},
        {"suspect_drug": "X", "reactions": ["a"], "serious": False, "patient_age": -1},
        {"suspect_drug": "X", "reactions": ["a"], "serious": False, "patient_sex": "other"},
        {"suspect_drug": "X", "reactions": ["a"], "serious": False, "outcome": "better"},
        {"suspect_drug": "X", "reactions": ["a"], "serious": False, "surprise": 1},
    ],
)
def test_invalid_cases_are_rejected(bad):
    with pytest.raises(ValidationError):
        Case(**bad)


def test_json_round_trip(case):
    assert Case.model_validate_json(case.model_dump_json()) == case


def test_json_schema_marks_required_fields_and_lists_enums():
    schema = Case.model_json_schema()
    assert {"suspect_drug", "reactions", "serious"} <= set(schema["required"])
    assert "outcome" in schema["properties"] and "patient_sex" in schema["properties"]
