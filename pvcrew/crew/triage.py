"""Demo triage rules. These are simple illustrations, NOT regulatory advice: real timelines depend on
expectedness, jurisdiction and company procedure."""
from ..schemas import Case


def triage(case: Case) -> dict:
    follow_up = []
    if case.patient_age is None:
        follow_up.append("age not stated")
    if case.patient_sex == "unknown":
        follow_up.append("sex not stated")
    if case.outcome == "unknown":
        follow_up.append("outcome not stated")
    if case.outcome == "fatal":
        level, why = "URGENT", "Fatal outcome: send to a safety physician straight away."
    elif case.serious:
        level, why = "HIGH", "Serious case: expedited medical review."
    else:
        level, why = "ROUTINE", "Non-serious: standard review queue."
    return {"priority": level, "why": why, "follow_up": follow_up}
