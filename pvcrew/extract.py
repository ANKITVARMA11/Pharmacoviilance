"""Extract a validated Case from a free-text report using a local model."""
from pydantic import ValidationError

from . import config
from .schemas import Case

SYSTEM_PROMPT = (
    "You extract structured data from a free-text drug safety report. "
    "Use only what the report states. Do not guess. "
    "Use null for an unknown age, 'unknown' for an unknown sex or outcome, "
    "and list every reaction or symptom separately. "
    "Set serious to true if the report describes a life-threatening event, death, disability, or hospital care, "
    "or if the reporter says the event is serious or marks it as serious. "
    "Reply with JSON only."
)


class ExtractionError(Exception):
    """The model never produced schema-valid output."""


def build_messages(report: str, feedback: str | None = None) -> list[dict]:
    user = f"Report:\n{report}"
    if feedback:
        user += ("\n\nA reviewer found problems with an earlier attempt:\n" + feedback +
                 "\nExtract the case again. Fix these using only what the report states.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_case(report: str, client, model: str = config.MODEL, retries: int = 2, feedback: str | None = None) -> Case:
    """Ask the model for JSON matching the Case schema; retry with the validation error on failure."""
    messages = build_messages(report, feedback)
    last_error = None
    for _ in range(retries + 1):
        resp = client.chat(
            model=model,
            messages=messages,
            format=Case.model_json_schema(),
            options={"temperature": 0, "timeout": config.NIM_TIMEOUT},
        )
        content = resp["message"]["content"]
        try:
            return Case.model_validate_json(content)
        except ValidationError as e:
            last_error = e
            messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That output was invalid ({e.error_count()} errors: {str(e)[:300]}). "
                    "Return corrected JSON only.",
                },
            ]
    raise ExtractionError(f"no valid output after {retries + 1} attempts: {str(last_error)[:200]}")


EXAMPLE_REPORT = (
    "Hi, my mum (67) started taking Amoxicillin last week and was rushed to hospital "
    "with a rash and trouble breathing. She's recovering now."
)


def extraction_test(client):
    """(accept, cases): `accept(model)` is True when the model extracts EXAMPLE_REPORT into a valid Case.
    Used to pick a model by what it really does, not by whether it answers a one-word ping."""
    cases = {}

    def accept(model: str) -> bool:
        try:
            cases[model] = extract_case(EXAMPLE_REPORT, client, model, retries=1)
            return True
        except ExtractionError:
            return False

    return accept, cases
