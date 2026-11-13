import pytest

from pvcrew import check, config
from pvcrew.llm import NimClient
from tests.conftest import FakeAPIError, FakeOpenAI
from tests.fake_nim import GOOD

SECRET = "nvapi-SECRETSECRETSECRET1234"


@pytest.fixture(autouse=True)
def hosted_with_key(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "nim")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", SECRET)
    monkeypatch.delenv("PV_MODEL", raising=False)


def nim(**kw):
    return NimClient(FakeOpenAI(**kw), min_interval=0, sleep=lambda s: None)


def run(client=None):
    lines = []
    ok = check.run(client, say=lines.append)
    return ok, "\n".join(lines)


def test_success_reports_every_step_and_masks_the_key():
    ok, out = run(nim(outcomes=[GOOD], model_ids=[config.NIM_CANDIDATES[0]]))
    assert ok and "OK:" in out
    assert "models on your key: 1" in out and f"model chosen: {config.NIM_CANDIDATES[0]}" in out
    assert "suspect_drug" in out
    assert "nvapi...1234" in out and "SECRETSECRET" not in out


def test_missing_key_explains_where_to_put_it(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(config, "NIM_BASE_URL", config.NIM_HOSTED_URL)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    ok, out = run()
    assert not ok and "FAIL" in out and "NVIDIA_API_KEY" in out and ".env" in out and "(not set)" in out


def test_rejected_key_gives_plain_advice_and_never_prints_the_key():
    ok, out = run(nim(outcomes=[FakeAPIError(401)], model_ids=[config.NIM_CANDIDATES[0]]))
    assert not ok and "rejected the key" in out and "nvapi-" in out and ".env" in out
    assert "SECRETSECRET" not in out


def test_rate_limit_message():
    ok, out = run(nim(outcomes=[FakeAPIError(429)], model_ids=[config.NIM_CANDIDATES[0]]))
    assert not ok and "rate limited" in out


def test_no_usable_model_lists_every_model_tried_and_why():
    gone = {"x/one": FakeAPIError(410), "y/two": FakeAPIError(404)}
    ok, out = run(nim(outcomes=[GOOD], model_ids=list(gone), by_model=gone))
    assert not ok and "FAIL" in out and "x/one: retired or unavailable (HTTP 410)" in out
    assert "y/two: retired or unavailable (HTTP 404)" in out and "PV_MODEL" in out


def test_retired_candidates_are_skipped_and_the_next_one_used():
    first, second = config.NIM_CANDIDATES[:2]
    ok, out = run(nim(outcomes=[GOOD], model_ids=[first, second], by_model={first: FakeAPIError(410)}))
    assert ok and f"model chosen: {second}" in out


def test_model_that_never_returns_valid_json_is_reported():
    ok, out = run(nim(outcomes=["not json"], model_ids=[config.NIM_CANDIDATES[0]]))
    # (the 5-token probe answers "not json" happily; extraction then fails validation)
    assert not ok and "FAIL" in out and "failed the extraction test" in out


def test_pinned_model_from_the_env_is_used_without_a_probe(monkeypatch):
    monkeypatch.setenv("PV_MODEL", "my/pinned")
    client = nim(outcomes=[GOOD], model_ids=["a/b"])
    ok, out = run(client)
    assert ok and "model chosen: my/pinned" in out
    assert {c["model"] for c in client._client.calls} == {"my/pinned"} and len(client._client.calls) == 1


def test_ollama_provider_skips_the_key_line(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "ollama")

    class FakeOllama:
        def chat(self, **kw):
            return {"message": {"content": GOOD}}

    ok, out = run(FakeOllama())
    assert ok and "API key" not in out and f"model chosen: {config.DEFAULT_MODELS['ollama']}" in out


def test_next_model_is_tried_when_one_answers_but_cannot_extract():
    first, second = config.NIM_CANDIDATES[:2]
    ok, out = run(nim(outcomes=[GOOD], model_ids=[first, second], by_model={first: "not json", second: GOOD}))
    assert ok and f"model chosen: {second}" in out
    assert "answered, but failed the extraction test" in out


def test_other_chat_models_are_discovered_when_no_preferred_one_is_listed():
    ok, out = run(nim(outcomes=[GOOD], model_ids=["acme/shiny-12b-instruct", "acme/shiny-embed-1b"]))
    assert ok and "model chosen: acme/shiny-12b-instruct" in out


def test_tip_to_pin_the_model_appears_only_when_it_was_auto_picked(monkeypatch):
    ok, out = run(nim(outcomes=[GOOD], model_ids=[config.NIM_CANDIDATES[0]]))
    assert ok and f"PV_MODEL={config.NIM_CANDIDATES[0]}" in out and "Tip:" in out
    monkeypatch.setenv("PV_MODEL", "my/pinned")
    ok, out = run(nim(outcomes=[GOOD], model_ids=["a/b"]))
    assert ok and "Tip:" not in out


def test_search_progress_is_visible_to_the_user():
    first, second = config.NIM_CANDIDATES[:2]
    ok, out = run(nim(outcomes=[GOOD], model_ids=[first, second], by_model={first: FakeAPIError(410)}))
    assert f"trying {first}" in out and "skipped: retired or unavailable (HTTP 410)" in out and f"trying {second}" in out
