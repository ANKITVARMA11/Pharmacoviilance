"""Shared test helpers.

The first lines keep the tests independent of YOUR real .env file and shell variables
(otherwise a filled-in key or PV_MODEL would change what the tests see).
"""
import os

os.environ["PV_ENV_FILE"] = os.path.join(os.path.dirname(__file__), "no-such.env")
for _name in ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MIN_INTERVAL", "PV_MODEL", "PV_PROVIDER",
              "PV_DATA", "OPENFDA_API_KEY", "OLLAMA_HOST"):
    os.environ.pop(_name, None)

import pytest
import requests

from pvcrew import config
from pvcrew.faers import load_records
from pvcrew.schemas import Case

GOOD_JSON = (
    '{"patient_age": 67, "patient_sex": "female", "suspect_drug": "Amoxicillin", '
    '"reactions": ["rash", "trouble breathing"], "serious": true, "outcome": "recovering"}'
)


class FakeClient:
    """Stands in for ollama.Client. Returns queued responses; the last one repeats."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if callable(r):
            r = r(kwargs)
        return {"message": {"content": r}}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves openFDA pages keyed by the `skip` parameter."""

    def __init__(self, pages=None, status_by_skip=None):
        self.pages = pages or {}
        self.status_by_skip = status_by_skip or {}
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append(dict(params or {}))
        skip = (params or {}).get("skip", 0)
        if skip in self.status_by_skip:
            return FakeResponse(self.status_by_skip[skip])
        return FakeResponse(200, {"results": self.pages.get(skip, [])})


class FakeAPIError(Exception):
    """Looks like an openai.APIStatusError: has a status_code."""

    def __init__(self, status_code, message="api error"):
        super().__init__(message)
        self.status_code = status_code


class APIConnectionError(Exception):
    """Same class name as openai's, which has no status code."""


class FakeOpenAI:
    """Stands in for openai.OpenAI. Outcomes are response strings or exceptions to raise; the last repeats."""

    def __init__(self, outcomes=("{}",), model_ids=(), by_model=None):
        from types import SimpleNamespace

        self.outcomes = list(outcomes)
        self.by_model = by_model or {}
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=i) for i in model_ids])

    def _create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        model = kwargs.get("model")
        if model in self.by_model:
            o = self.by_model[model]
        else:
            o = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=o))])


@pytest.fixture
def sample_records():
    return load_records(config.SAMPLE_FAERS)


@pytest.fixture
def case():
    return Case(
        patient_age=67,
        patient_sex="female",
        suspect_drug="Amoxicillin",
        reactions=["Dyspnoea", "Rash"],
        serious=True,
        outcome="recovering",
    )


@pytest.fixture
def good_json():
    return GOOD_JSON


@pytest.fixture(autouse=True)
def _no_real_pauses_between_openfda_pages(monkeypatch):
    """The vocabulary download pauses between pages; tests must never wait for real."""
    from pvcrew import coding
    monkeypatch.setattr(coding, "time", type("T", (), {"sleep": staticmethod(lambda s: None)}))
