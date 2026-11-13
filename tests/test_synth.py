import random

import pytest

from pvcrew import synth
from pvcrew.schemas import Case
from tests.conftest import FakeClient


@pytest.mark.parametrize("style", list(synth.STYLES))
def test_offline_report_mentions_drug_and_every_reaction(case, style):
    text = synth.generate_offline(case, style, random.Random(1)).lower()
    assert case.suspect_drug.lower() in text
    for reaction in case.reactions:
        assert reaction.lower() in text


@pytest.mark.parametrize("style", list(synth.STYLES))
def test_offline_report_is_deterministic_for_a_seed(case, style):
    a = synth.generate_offline(case, style, random.Random(7))
    b = synth.generate_offline(case, style, random.Random(7))
    assert a == b


def test_different_styles_give_different_text(case):
    texts = {synth.generate_offline(case, s, random.Random(0)) for s in synth.STYLES}
    assert len(texts) == len(synth.STYLES)


@pytest.mark.parametrize("style", list(synth.STYLES))
def test_missing_age_and_sex_are_omitted_not_printed_as_none(style):
    c = Case(suspect_drug="Aspirin", reactions=["Rash"], serious=False)
    text = synth.generate_offline(c, style, random.Random(0))
    assert "None" not in text and "unknown" not in text.lower()


def test_serious_flag_is_expressed(case):
    assert "hospital" in synth.generate_offline(case, "patient_email", random.Random(0))
    assert "Serious: Y" in synth.generate_offline(case, "call_note")
    mild = case.model_copy(update={"serious": False})
    assert "Serious: N" in synth.generate_offline(mild, "call_note")


def test_fatal_outcome_phrase(case):
    fatal = case.model_copy(update={"outcome": "fatal"})
    assert "passed away" in synth.generate_offline(fatal, "hcp_form")


def test_unknown_style_raises(case):
    with pytest.raises(ValueError):
        synth.generate_offline(case, "carrier_pigeon")


def test_facts_text_lists_all_facts(case):
    t = synth.facts_text(case)
    for piece in ["Amoxicillin", "Dyspnoea, Rash", "yes", "recovering", "67", "female"]:
        assert piece in t


@pytest.mark.parametrize(
    "report,ok",
    [
        ("My mum took Amoxicillin and got a nasty rash last week.", True),
        ("my mum took AMOXICILLIN and got a rash last week", True),
        ("too short", False),
        ('{"suspect_drug": "Amoxicillin", "reactions": ["rash"]}', False),
        ("My mum took some antibiotic and got a nasty rash last week.", False),
    ],
)
def test_looks_valid(case, report, ok):
    assert synth.looks_valid(report, case) is ok


def test_make_report_without_client_is_offline(case):
    text, source = synth.make_report(case, "call_note")
    assert source == "offline" and case.suspect_drug in text


def test_make_report_uses_llm_text_when_valid(case):
    llm_text = "Hi, my mum took Amoxicillin on Monday and ended up with a rash and wheezing, we went to A&E."
    client = FakeClient([llm_text])
    text, source = synth.make_report(case, "patient_email", client, model="m")
    assert (text, source) == (llm_text, "llm")
    call = client.calls[0]
    assert call["model"] == "m"
    assert "Amoxicillin" in call["messages"][1]["content"]
    assert synth.STYLES["patient_email"] in call["messages"][0]["content"]


def test_make_report_falls_back_when_llm_output_is_unusable(case):
    client = FakeClient(["Sure! Here you go."])
    text, source = synth.make_report(case, "patient_email", client)
    assert source == "offline_fallback" and case.suspect_drug in text


def test_build_pairs_skips_unusable_records_and_keeps_truth(sample_records):
    rows = synth.build_pairs(sample_records, client=None, seed=3)
    assert len(rows) == 5
    assert all(set(r) == {"id", "style", "source", "report", "truth"} for r in rows)
    assert rows[0]["id"] == "SAMPLE-001"
    assert Case.model_validate(rows[0]["truth"]).suspect_drug == "AMOXICILLIN"
    assert all(r["style"] in synth.STYLES for r in rows)


def test_build_pairs_is_deterministic_for_a_seed(sample_records):
    assert synth.build_pairs(sample_records, seed=5) == synth.build_pairs(sample_records, seed=5)


def test_build_pairs_calls_llm_once_per_usable_record(sample_records):
    client = FakeClient(["Patient took the medicine and had a reaction."])  # never names the drug
    rows = synth.build_pairs(sample_records, client=client)
    assert len(client.calls) == 5
    assert {r["source"] for r in rows} == {"offline_fallback"}
