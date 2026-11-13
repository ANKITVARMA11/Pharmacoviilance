"""Turn a ground-truth Case into a messy free-text safety report.

Two generators:
  * LLM  : realistic, varied text (needs a model).
  * offline templates : deterministic, no model needed (tests and fallback).

The Case stays attached as ground truth. Note: the LLM may describe reactions in
everyday words ("trouble breathing"); matching those to coded terms is Day 2 work.
"""
import random
from typing import Optional

from . import config
from .schemas import Case

STYLES = {
    "patient_email": "a worried patient or relative writing an informal email",
    "call_note": "a call-centre agent's terse typed note, with abbreviations",
    "hcp_form": "a physician filling in the free-text box of a reporting form",
    "text_message": "a short, casual text message from a patient",
}

OUTCOME_PHRASES = {
    "recovered": "has fully recovered",
    "recovering": "is recovering",
    "not_recovered": "has not recovered yet",
    "recovered_with_sequelae": "recovered but still has lasting problems",
    "fatal": "sadly passed away",
    "unknown": None,
}

SYSTEM_PROMPT = (
    "You write realistic, slightly messy adverse-event reports exactly as a real person would type them. "
    "Include every fact you are given, in everyday wording. Typos and informal phrasing are fine. "
    "Do not use JSON, bullet lists, headings or field labels. 40 to 120 words. Output only the report."
)


def facts_text(case: Case) -> str:
    lines = [
        f"Medicine suspected: {case.suspect_drug}",
        f"Reactions: {', '.join(case.reactions)}",
        f"Serious (life-threatening, disabling, fatal or hospital care): {'yes' if case.serious else 'no'}",
        f"Outcome: {case.outcome.replace('_', ' ')}",
    ]
    if case.patient_age is not None:
        lines.append(f"Patient age: {case.patient_age}")
    if case.patient_sex != "unknown":
        lines.append(f"Patient sex: {case.patient_sex}")
    return "\n".join(lines)


def generate_offline(case: Case, style: str, rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random(0)
    rx = ", ".join(case.reactions)
    age = f"{case.patient_age}" if case.patient_age is not None else None
    sex = case.patient_sex if case.patient_sex != "unknown" else None
    outcome = OUTCOME_PHRASES[case.outcome]
    hosp = "was admitted to hospital" if case.serious else "it was not severe and no hospital stay was needed"

    if style == "patient_email":
        who = " ".join(x for x in [f"{age}-year-old" if age else None, sex] if x) or "patient"
        opener = rng.choice(["Hello,", "Hi team,", "To whom it may concern,"])
        text = f"{opener} I am writing about {case.suspect_drug}. The {who} took it and then had {rx}. The patient {hosp}."
        if outcome:
            text += f" Update: the patient {outcome}."
        return text
    if style == "call_note":
        demo = " ".join(x for x in [f"{age}y" if age else None, sex[0].upper() if sex else None] if x)
        text = f"CALL NOTE. Caller rpts {case.suspect_drug}. Pt {demo}. AE: {rx}. Serious: {'Y' if case.serious else 'N'}."
        if outcome:
            text += f" Outcome: pt {outcome}."
        return text
    if style == "hcp_form":
        demo = ", ".join(x for x in [sex, f"{age} years old" if age else None] if x) or "demographics not given"
        text = f"Reporter is a physician. Patient: {demo}. Suspect product: {case.suspect_drug}. Events observed: {rx}. The patient {hosp}."
        if outcome:
            text += f" At last contact the patient {outcome}."
        return text
    if style == "text_message":
        text = f"took {case.suspect_drug} and got {rx}"
        if age:
            text = f"{age}{(sex or '')[:1]} " + text
        if case.serious:
            text += ", ended up in hospital"
        if outcome:
            text += f", {outcome}"
        return text + "."
    raise ValueError(f"unknown style: {style}")


def generate_llm(case: Case, style: str, client, model: str = config.MODEL) -> str:
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + f" Write as {STYLES[style]}."},
            {"role": "user", "content": facts_text(case)},
        ],
        options={"temperature": 0.8},
    )
    return resp["message"]["content"].strip()


def looks_valid(report: str, case: Case) -> bool:
    """Cheap guard: long enough, not JSON, and names the suspect drug."""
    if len(report) < 20 or report.lstrip().startswith(("{", "[")):
        return False
    return case.suspect_drug.lower() in report.lower()


def make_report(case: Case, style: str, client=None, model: str = config.MODEL, rng=None) -> tuple[str, str]:
    """Return (report_text, source) where source is 'llm', 'offline' or 'offline_fallback'."""
    if client is None:
        return generate_offline(case, style, rng), "offline"
    text = generate_llm(case, style, client, model)
    if looks_valid(text, case):
        return text, "llm"
    return generate_offline(case, style, rng), "offline_fallback"


def build_pairs(records: list[dict], client=None, model: str = config.MODEL, seed: int = 0) -> list[dict]:
    """Make {id, style, source, report, truth} rows from raw FAERS records."""
    from .faers import record_to_case

    rng = random.Random(seed)
    styles = list(STYLES)
    rows = []
    for i, rec in enumerate(records):
        case = record_to_case(rec)
        if case is None:
            continue
        style = rng.choice(styles)
        report, source = make_report(case, style, client, model, rng)
        rows.append(
            {
                "id": rec.get("safetyreportid", f"rec{i}"),
                "style": style,
                "source": source,
                "report": report,
                "truth": case.model_dump(),
            }
        )
    return rows
