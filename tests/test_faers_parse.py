import pytest

from pvcrew import faers


@pytest.mark.parametrize(
    "patient,expected",
    [
        ({"patientonsetage": "67", "patientonsetageunit": "801"}, 67),
        ({"patientonsetage": "2", "patientonsetageunit": "800"}, 20),  # decades
        ({"patientonsetage": "6", "patientonsetageunit": "802"}, 0),  # months
        ({"patientonsetage": "104", "patientonsetageunit": "803"}, 2),  # weeks
        ({"patientonsetage": "36.5", "patientonsetageunit": "801"}, 36),
        ({"patientonsetage": "67"}, None),  # no unit: refuse to guess
        ({"patientonsetage": "abc", "patientonsetageunit": "801"}, None),
        ({"patientonsetage": "200", "patientonsetageunit": "801"}, None),  # implausible
        ({"patientonsetageunit": "801"}, None),
        ({}, None),
    ],
)
def test_parse_age(patient, expected):
    assert faers.parse_age(patient) == expected


@pytest.mark.parametrize("code,expected", [("1", "male"), ("2", "female"), ("0", "unknown"), (None, "unknown"), ("9", "unknown")])
def test_parse_sex(code, expected):
    patient = {} if code is None else {"patientsex": code}
    assert faers.parse_sex(patient) == expected


def test_suspect_drug_ignores_concomitant_and_picks_first_suspect():
    patient = {
        "drug": [
            {"drugcharacterization": "2", "medicinalproduct": "PARACETAMOL"},
            {"drugcharacterization": "1", "medicinalproduct": " WARFARIN "},
            {"drugcharacterization": "1", "medicinalproduct": "ASPIRIN"},
        ]
    }
    assert faers.parse_suspect_drug(patient) == "WARFARIN"


def test_suspect_drug_none_when_only_concomitant_or_blank():
    assert faers.parse_suspect_drug({"drug": [{"drugcharacterization": "2", "medicinalproduct": "X"}]}) is None
    assert faers.parse_suspect_drug({"drug": [{"drugcharacterization": "1", "medicinalproduct": "  "}]}) is None
    assert faers.parse_suspect_drug({}) is None


def test_reactions_are_deduplicated_and_blanks_dropped():
    patient = {"reaction": [{"reactionmeddrapt": "Rash"}, {"reactionmeddrapt": "rash"}, {"reactionmeddrapt": ""}, {}]}
    assert faers.parse_reactions(patient) == ["Rash"]


@pytest.mark.parametrize(
    "codes,expected",
    [
        (["1", "5"], "fatal"),
        (["1", "3"], "not_recovered"),
        (["1", "4"], "recovered_with_sequelae"),
        (["1", "2"], "recovering"),
        (["1"], "recovered"),
        (["6", "6"], "unknown"),
        ([], "unknown"),
        (["1", "6"], "recovered"),  # unknown does not outrank a known outcome
    ],
)
def test_outcome_uses_worst_known_outcome(codes, expected):
    patient = {"reaction": [{"reactionmeddrapt": f"R{i}", "reactionoutcome": c} for i, c in enumerate(codes)]}
    assert faers.parse_outcome(patient) == expected


def test_record_to_case_full_record(sample_records):
    case = faers.record_to_case(sample_records[0])
    assert case.suspect_drug == "AMOXICILLIN"
    assert case.patient_age == 67 and case.patient_sex == "female"
    assert case.reactions == ["Dyspnoea", "Rash"]
    assert case.serious is True and case.outcome == "recovering"


def test_record_to_case_sparse_record(sample_records):
    case = faers.record_to_case(sample_records[4])  # SAMPLE-005
    assert case.patient_age is None and case.patient_sex == "unknown"
    assert case.outcome == "recovered_with_sequelae"


def test_record_to_case_infant_age_floors_to_zero(sample_records):
    assert faers.record_to_case(sample_records[3]).patient_age == 0


def test_record_without_suspect_drug_or_reactions_is_skipped(sample_records):
    assert faers.record_to_case(sample_records[5]) is None  # only a concomitant drug
    assert faers.record_to_case({"patient": {"drug": [{"drugcharacterization": "1", "medicinalproduct": "X"}]}}) is None
    assert faers.record_to_case({}) is None


def test_bundled_sample_has_six_records_and_five_usable(sample_records):
    assert len(sample_records) == 6
    assert sum(faers.record_to_case(r) is not None for r in sample_records) == 5
