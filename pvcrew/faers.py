"""Fetch FAERS records from openFDA and convert them to ground-truth Case objects.

FAERS code reference (openFDA drug/event):
  patientsex: 1 male, 2 female, 0 unknown
  patientonsetageunit: 800 decade, 801 year, 802 month, 803 week, 804 day, 805 hour
  drugcharacterization: 1 suspect, 2 concomitant, 3 interacting
  reactionoutcome: 1 recovered, 2 recovering, 3 not recovered, 4 recovered with sequelae, 5 fatal, 6 unknown
"""
import time
from typing import Optional

import requests

from . import config
from .jsonl import read_jsonl, write_jsonl
from .schemas import Case

AGE_UNIT_TO_YEARS = {
    "800": 10.0,
    "801": 1.0,
    "802": 1 / 12,
    "803": 1 / 52,
    "804": 1 / 365,
    "805": 1 / 8766,
}
SEX = {"1": "male", "2": "female"}
OUTCOME = {
    "1": "recovered",
    "2": "recovering",
    "3": "not_recovered",
    "4": "recovered_with_sequelae",
    "5": "fatal",
    "6": "unknown",
}
# When reactions disagree, report the worst known outcome for the patient.
OUTCOME_PRIORITY = ["fatal", "not_recovered", "recovered_with_sequelae", "recovering", "recovered"]
MAX_SKIP = 25000  # openFDA paging limit


def parse_age(patient: dict) -> Optional[int]:
    """Age in whole years, or None if missing, unitless or implausible."""
    raw = patient.get("patientonsetage")
    factor = AGE_UNIT_TO_YEARS.get(str(patient.get("patientonsetageunit", "")))
    if raw in (None, "") or factor is None:
        return None
    try:
        years = int(float(raw) * factor)
    except (TypeError, ValueError):
        return None
    return years if 0 <= years <= 120 else None


def parse_sex(patient: dict) -> str:
    return SEX.get(str(patient.get("patientsex", "")), "unknown")


def parse_suspect_drug(patient: dict) -> Optional[str]:
    for drug in patient.get("drug", []):
        if str(drug.get("drugcharacterization")) == "1":
            name = (drug.get("medicinalproduct") or "").strip()
            if name:
                return name
    return None


def parse_reactions(patient: dict) -> list[str]:
    seen, out = set(), []
    for r in patient.get("reaction", []):
        term = (r.get("reactionmeddrapt") or "").strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


def parse_outcome(patient: dict) -> str:
    found = {OUTCOME.get(str(r.get("reactionoutcome")), "unknown") for r in patient.get("reaction", [])}
    for label in OUTCOME_PRIORITY:
        if label in found:
            return label
    return "unknown"


def record_to_case(rec: dict) -> Optional[Case]:
    """Convert one FAERS record to a Case, or None if it lacks a suspect drug or reactions."""
    patient = rec.get("patient") or {}
    drug = parse_suspect_drug(patient)
    reactions = parse_reactions(patient)
    if not drug or not reactions:
        return None
    return Case(
        patient_age=parse_age(patient),
        patient_sex=parse_sex(patient),
        suspect_drug=drug,
        reactions=reactions,
        serious=str(rec.get("serious")) == "1",
        outcome=parse_outcome(patient),
    )


def fetch_records(
    n: int,
    session: Optional[requests.Session] = None,
    api_key: Optional[str] = None,
    page_size: int = 100,
    start_skip: int = 0,
    base_url: str = config.OPENFDA_URL,
    pause: float = 0.3,
) -> list[dict]:
    """Page through openFDA and keep n records that convert cleanly to a Case."""
    session = session or requests.Session()
    out: list[dict] = []
    skip = start_skip
    while len(out) < n and skip <= MAX_SKIP:
        params = {"limit": min(page_size, 1000), "skip": skip}
        if api_key:
            params["api_key"] = api_key
        resp = session.get(base_url, params=params, timeout=30)
        if resp.status_code == 404:  # openFDA's "no matches"
            break
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for rec in results:
            if record_to_case(rec) is not None:
                out.append(rec)
                if len(out) >= n:
                    break
        skip += len(results)
        if pause and len(out) < n:
            time.sleep(pause)
    return out


def load_records(path) -> list[dict]:
    return list(read_jsonl(path))


def save_records(path, records: list[dict]) -> int:
    return write_jsonl(path, records)
