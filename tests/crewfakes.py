"""Fakes for the crew tests: a model that answers by role, and stand-ins for ModelSession."""
import json

GOOD_CASE = {"patient_age": 67, "patient_sex": "female", "suspect_drug": "Amoxicillin",
             "reactions": ["rash", "trouble breathing"], "serious": True, "outcome": "recovering"}
REPORT = "My mum (67) took Amoxicillin, got a rash and trouble breathing, was in hospital. Recovering now."
APPROVE = json.dumps({"verdict": "APPROVE", "issues": []})
REVISE = json.dumps({"verdict": "REVISE", "issues": ["The age is wrong.", "Trouble breathing is missing."]})
NORM = json.dumps({"items": [{"reported": "trouble breathing", "medical_term": "dyspnoea"}]})
NARRATIVE = "A 67-year-old woman developed a rash and trouble breathing after Amoxicillin. She was recovering."


class RoleClient:
    """Answers by role, read from the system prompt. Each answer is a string, a list (used in order, the last one
    repeats) or a callable(kwargs) -> str. Every call is recorded in `.calls` as (role, kwargs)."""

    ROLES = [("extract structured data", "extract"), ("convert informal", "normalize"),
             ("choose the one candidate", "choose"), ("quality reviewer", "critic"), ("narrative paragraph", "narrative")]

    def __init__(self, extract=None, normalize=NORM, choose='{"choices": []}', critic=APPROVE, narrative=NARRATIVE):
        self.answers = {"extract": json.dumps(GOOD_CASE) if extract is None else extract, "normalize": normalize,
                        "choose": choose, "critic": critic, "narrative": narrative}
        self.calls, self._used = [], {}

    def role_calls(self, role):
        return [kw for r, kw in self.calls if r == role]

    def chat(self, **kw):
        system = kw["messages"][0]["content"]
        role = next((r for needle, r in self.ROLES if needle in system), "other")
        self.calls.append((role, kw))
        a = self.answers.get(role, "{}")
        if isinstance(a, list):
            i = self._used.get(role, 0)
            self._used[role] = i + 1
            a = a[min(i, len(a) - 1)]
        if callable(a):
            a = a(kw)
        return {"message": {"content": a}}


class FakeSession:
    """Stands in for pipeline.ModelSession."""

    def __init__(self, model="fake/model"):
        self.model = model

    def call(self, fn):
        return fn(self.model)


class DeadSession(FakeSession):
    """A pinned model that stopped answering: ModelSession raises SystemExit in that case."""

    def call(self, fn):
        raise SystemExit("fake/model: too slow or erroring\nYour finished rows are saved.")
