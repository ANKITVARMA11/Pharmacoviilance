import sys
from types import ModuleType, SimpleNamespace

import pytest

from pvcrew import config, llm
from pvcrew.llm import ModelUnavailableError, NimClient, _size_b, rank_models, resolve_model
from pvcrew.schemas import Case
from tests.conftest import APIConnectionError, FakeAPIError, FakeOpenAI

SCHEMA = Case.model_json_schema()
MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]


def make(outcomes=("OK",), **kw):
    sleeps = []
    kw.setdefault("min_interval", 0)
    client = NimClient(FakeOpenAI(outcomes), sleep=sleeps.append, **kw)
    return client, sleeps


# ---------- request / response translation ----------

def test_returns_ollama_shaped_response():
    client, _ = make(["hello there"])
    assert client.chat(model="m", messages=MSGS) == {"message": {"content": "hello there"}}


def test_schema_goes_in_nvext_guided_json_and_in_the_system_prompt():
    client, _ = make()
    client.chat(model="meta/x", messages=MSGS, format=SCHEMA, options={"temperature": 0})
    call = client._client.calls[0]
    assert call["model"] == "meta/x" and call["temperature"] == 0
    assert call["extra_body"] == {"nvext": {"guided_json": SCHEMA}}
    system = call["messages"][0]["content"]
    assert system.startswith("SYS") and "JSON schema" in system and "suspect_drug" in system
    assert call["messages"][1] == MSGS[1]


def test_callers_messages_are_not_mutated():
    client, _ = make()
    original = [dict(m) for m in MSGS]
    client.chat(model="m", messages=MSGS, format=SCHEMA)
    assert MSGS == original


def test_system_message_is_added_when_missing():
    client, _ = make()
    client.chat(model="m", messages=[{"role": "user", "content": "hi"}], format=SCHEMA)
    msgs = client._client.calls[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[1]["content"] == "hi"


def test_no_format_means_no_guided_decoding_and_untouched_messages():
    client, _ = make()
    client.chat(model="m", messages=MSGS, options={"temperature": 0.8})
    call = client._client.calls[0]
    assert "extra_body" not in call and call["messages"] == MSGS and call["temperature"] == 0.8


def test_temperature_defaults_to_zero():
    client, _ = make()
    client.chat(model="m", messages=MSGS)
    assert client._client.calls[0]["temperature"] == 0


def test_none_content_becomes_empty_string():
    client, _ = make([None])
    assert client.chat(model="m", messages=MSGS)["message"]["content"] == ""


def test_unknown_kwargs_are_ignored():
    client, _ = make()
    client.chat(model="m", messages=MSGS, keep_alive="5m")  # an Ollama-only argument


# ---------- retries ----------

def test_429_is_retried_with_exponential_backoff_then_succeeds():
    client, sleeps = make([FakeAPIError(429), FakeAPIError(429), "OK"])
    assert client.chat(model="m", messages=MSGS)["message"]["content"] == "OK"
    assert len(client._client.calls) == 3
    assert sleeps == [2.0, 4.0]


def test_5xx_and_connection_errors_are_retried():
    client, _ = make([FakeAPIError(503), APIConnectionError("net down"), "OK"])
    assert client.chat(model="m", messages=MSGS)["message"]["content"] == "OK"
    assert len(client._client.calls) == 3


def test_gives_up_after_max_retries_and_raises_the_last_error():
    client, _ = make([FakeAPIError(429, "slow down")], max_retries=2)
    with pytest.raises(FakeAPIError, match="slow down"):
        client.chat(model="m", messages=MSGS)
    assert len(client._client.calls) == 3


def test_backoff_is_capped():
    client, sleeps = make([FakeAPIError(429)], max_retries=8, backoff=10)
    with pytest.raises(FakeAPIError):
        client.chat(model="m", messages=MSGS)
    assert max(sleeps) == 30


@pytest.mark.parametrize("status", [401, 403, 422])
def test_client_errors_are_not_retried(status):
    client, _ = make([FakeAPIError(status)])
    with pytest.raises(FakeAPIError):
        client.chat(model="m", messages=MSGS, format=SCHEMA)
    assert len(client._client.calls) == 1


# ---------- guided_json fallback ----------

def test_400_on_guided_json_falls_back_to_prompt_only_and_remembers():
    client, _ = make([FakeAPIError(400, "nvext not supported"), "FIRST", "SECOND"])
    assert client.chat(model="m", messages=MSGS, format=SCHEMA)["message"]["content"] == "FIRST"
    calls = client._client.calls
    assert "extra_body" in calls[0] and "extra_body" not in calls[1]
    assert "JSON schema" in calls[1]["messages"][0]["content"]  # schema still in the prompt
    client.chat(model="m", messages=MSGS, format=SCHEMA)
    assert "extra_body" not in calls[2]  # does not retry guided mode again


def test_400_without_guided_json_is_a_real_error():
    client, _ = make([FakeAPIError(400, "bad model")])
    with pytest.raises(FakeAPIError):
        client.chat(model="m", messages=MSGS)
    assert len(client._client.calls) == 1


# ---------- pacing ----------

def test_calls_are_spaced_by_min_interval():
    now = [100.0]
    sleeps = []
    client = NimClient(FakeOpenAI(["a"]), min_interval=1.5, sleep=sleeps.append, clock=lambda: now[0])
    client.chat(model="m", messages=MSGS)
    assert sleeps == []  # first call goes straight out
    client.chat(model="m", messages=MSGS)
    assert sleeps == [1.5]
    now[0] += 10  # plenty of time passed: no wait needed
    client.chat(model="m", messages=MSGS)
    assert sleeps == [1.5]


# ---------- get_client ----------

def fake_openai_module(captured):
    mod = ModuleType("openai")
    mod.OpenAI = lambda **kw: captured.update(kw) or SimpleNamespace(**kw)
    return mod


def test_missing_key_on_hosted_endpoint_gives_a_helpful_message(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(config, "NIM_BASE_URL", config.NIM_HOSTED_URL)
    with pytest.raises(SystemExit) as e:
        llm.get_client("nim")
    assert "NVIDIA_API_KEY" in str(e.value) and "build.nvidia.com" in str(e.value)


def test_missing_key_message_points_at_the_env_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(config, "NIM_BASE_URL", config.NIM_HOSTED_URL)
    env_file = tmp_path / ".env"
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    with pytest.raises(SystemExit) as e:  # file does not exist yet
        llm.get_client("nim")
    assert "Create" in str(e.value) and ".env.example" in str(e.value) and str(env_file) in str(e.value)
    env_file.write_text("NVIDIA_API_KEY=\n")
    with pytest.raises(SystemExit) as e:  # file exists but the key is blank
        llm.get_client("nim")
    assert "Open" in str(e.value) and str(env_file) in str(e.value)


def test_hosted_client_uses_key_base_url_and_no_sdk_retries(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module(seen))
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setattr(config, "NIM_BASE_URL", config.NIM_HOSTED_URL)
    client = llm.get_client("nim")
    assert isinstance(client, NimClient)
    assert seen["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert seen["api_key"] == "nvapi-test" and seen["max_retries"] == 0


def test_self_hosted_nim_needs_no_key(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module(seen))
    monkeypatch.setattr(config, "NVIDIA_API_KEY", None)
    monkeypatch.setattr(config, "NIM_BASE_URL", "http://localhost:8000/v1")
    llm.get_client("nim")
    assert seen["base_url"] == "http://localhost:8000/v1" and seen["api_key"] == "not-used"


def test_provider_defaults_to_config(monkeypatch):
    monkeypatch.setattr(config, "PROVIDER", "ollama")
    fake = ModuleType("ollama")
    fake.Client = lambda host: ("ollama-client", host)
    monkeypatch.setitem(sys.modules, "ollama", fake)
    assert llm.get_client()[0] == "ollama-client"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        llm.get_client("carrier-pigeon")


# ---------- list models ----------

def test_list_models_sorted():
    client = NimClient(FakeOpenAI(model_ids=["b/model", "a/model"]))
    assert client.list_models() == ["a/model", "b/model"]


def test_models_cli_filters_by_substring(monkeypatch, capsys):
    monkeypatch.setattr(llm, "get_client", lambda: NimClient(FakeOpenAI(model_ids=["meta/llama-3.3-70b", "nvidia/nemotron", "meta/llama-4"])))
    llm.main(["llama"])
    assert capsys.readouterr().out.split() == ["meta/llama-3.3-70b", "meta/llama-4"]


def test_models_cli_refuses_non_nim_client(monkeypatch):
    monkeypatch.setattr(llm, "get_client", lambda: object())
    with pytest.raises(SystemExit):
        llm.main([])


# ---------- default model follows the provider ----------

def test_default_models_are_defined_for_both_providers():
    assert config.DEFAULT_MODELS["nim"].count("/") == 1  # owner/model slug
    assert ":" in config.DEFAULT_MODELS["ollama"]


# ---------- retired / unknown models ----------

@pytest.mark.parametrize("status", [404, 410])
def test_retired_or_unknown_model_gives_a_clear_error_and_is_not_retried(status):
    client, sleeps = make([FakeAPIError(status)])
    with pytest.raises(ModelUnavailableError) as e:
        client.chat(model="meta/gone-model", messages=MSGS)
    msg = str(e.value)
    assert "meta/gone-model" in msg and str(status) in msg
    assert "python -m pvcrew.llm" in msg and "PV_MODEL" in msg
    assert len(client._client.calls) == 1 and sleeps == []


def test_max_tokens_is_passed_through():
    client, _ = make()
    client.chat(model="m", messages=MSGS, options={"temperature": 0, "max_tokens": 5})
    assert client._client.calls[0]["max_tokens"] == 5
    client.chat(model="m", messages=MSGS)
    assert "max_tokens" not in client._client.calls[1]


# ---------- resolve_model ----------

CANDS = ["a/first", "b/second", "c/third"]


def nim(model_ids, by_model=None):
    return NimClient(FakeOpenAI(["OK"], model_ids=model_ids, by_model=by_model), min_interval=0, sleep=lambda s: None)


def test_explicit_model_is_returned_untouched_without_any_calls(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(CANDS)
    assert resolve_model(client, "my/model", CANDS) == "my/model"
    assert client._client.calls == []


def test_pv_model_env_is_honoured(monkeypatch):
    monkeypatch.setenv("PV_MODEL", "env/model")
    assert resolve_model(nim(CANDS), None, CANDS) == "env/model"


def test_picks_first_candidate_that_is_listed_and_answers(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(["b/second", "c/third", "z/other"])  # a/first is not on the list
    assert resolve_model(client, None, CANDS) == "b/second"
    assert len(client._client.calls) == 1
    probe = client._client.calls[0]
    assert probe["model"] == "b/second" and probe["max_tokens"] == 5 and probe["temperature"] == 0


def test_skips_a_model_that_is_listed_but_retired(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(CANDS, by_model={"a/first": FakeAPIError(410)})
    assert resolve_model(client, None, CANDS) == "b/second"
    assert [c["model"] for c in client._client.calls] == ["a/first", "b/second"]


def test_nothing_preferred_is_listed_so_other_chat_models_are_discovered(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    assert resolve_model(nim(["x/one", "y/two"]), None, CANDS) == "x/one"


def test_with_discovery_off_only_listed_preferred_models_are_tried(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    with pytest.raises(ModelUnavailableError) as e:
        resolve_model(nim(["x/one", "y/two"]), None, CANDS, discover=False)
    assert "no chat models found" in str(e.value) and "PV_MODEL" in str(e.value)


def test_error_when_every_candidate_is_retired(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    gone = {m: FakeAPIError(410) for m in CANDS}
    with pytest.raises(ModelUnavailableError):
        resolve_model(nim(CANDS, by_model=gone), None, CANDS)


def test_other_errors_during_the_probe_are_not_swallowed(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    with pytest.raises(FakeAPIError):
        resolve_model(nim(CANDS, by_model={"a/first": FakeAPIError(401)}), None, CANDS)


def test_non_nim_client_gets_the_ollama_default(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    assert resolve_model(object(), None, CANDS) == config.DEFAULT_MODELS["ollama"]


def test_shipped_candidates_are_non_empty_unique_owner_slash_name():
    c = config.NIM_CANDIDATES
    assert c and len(c) == len(set(c)) and all(m.count("/") == 1 for m in c)
    assert "meta/llama-3.3-70b-instruct" not in c  # retired 2026-08-26


# ---------- ranking and discovery ----------
# The 81 model IDs NVIDIA's hosted API listed on 2026-10-21 (several of them are retired but still listed).
LIVE_LIST = """01-ai/yi-large adept/fuyu-8b ai21labs/jamba-1.5-large-instruct aisingapore/sea-lion-7b-instruct bigcode/starcoder2-15b
databricks/dbrx-instruct deepseek-ai/deepseek-coder-6.7b-instruct deepseek-ai/deepseek-v4.1-flash google/codegemma-1.1-7b
google/codegemma-7b google/deplot google/diffusiongemma-26b-a4b-it google/gemma-2b google/gemma-3-12b-it google/gemma-3-4b-it
google/gemma-4-31b-it google/recurrentgemma-2b ibm/granite-3.0-3b-a800m-instruct ibm/granite-3.0-8b-instruct
ibm/granite-34b-code-instruct ibm/granite-8b-code-instruct meta/codellama-70b meta/llama-3.2-11b-vision-instruct
meta/llama-3.2-90b-vision-instruct meta/llama-guard-4-12b meta/llama2-70b meta/muse-glimmer-30b microsoft/kosmos-2
microsoft/phi-3-vision-128k-instruct microsoft/phi-3.5-moe-instruct mistralai/codestral-22b-instruct-v0.1
mistralai/mistral-7b-instruct-v0.3 mistralai/mistral-large mistralai/mistral-large-2-instruct mistralai/mixtral-8x22b-v0.1
moonshotai/kimi-k2.6 nv-mistralai/mistral-nemo-12b-instruct nvidia/ai-synthetic-video-detector nvidia/cosmos-reason2-8b
nvidia/embed-qa-4 nvidia/ising-calibration-1.5-31b nvidia/llama-3.1-nemoguard-8b-content-safety
nvidia/llama-3.1-nemoguard-8b-topic-control nvidia/llama-3.1-nemotron-51b-instruct nvidia/llama-3.1-nemotron-70b-instruct
nvidia/llama-3.1-nemotron-safety-guard-8b-v3 nvidia/llama-3.1-nemotron-ultra-253b-v1
nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1 nvidia/llama-3.2-nv-embedqa-1b-v1 nvidia/llama-nemotron-embed-vl-1b-v2
nvidia/llama3-chatqa-1.5-70b nvidia/mistral-nemo-minitron-8b-8k-instruct nvidia/nemotron-3-embed-1b
nvidia/nemotron-3-nano-omni-30b-a3b-reasoning nvidia/nemotron-3-super-120b-a12b nvidia/nemotron-3-ultra-550b-a55b
nvidia/nemotron-3.5-content-safety nvidia/nemotron-3.5-lightning-30b-a3b nvidia/nemotron-3.5-super-vl-120b-a12b
nvidia/nemotron-4-340b-instruct nvidia/nemotron-4-340b-reward nvidia/nemotron-nano-3-30b-a3b nvidia/nemotron-parse
nvidia/nemotron-parse-2.0 nvidia/neva-22b nvidia/nv-embedqa-mistral-7b-v2 nvidia/nvclip nvidia/riva-translate-4b-instruct
nvidia/riva-translate-4b-instruct-v2 nvidia/vila openai/gpt-oss-20b poolside/laguna-xs-2.1 snowflake/arctic-embed-l
stepfun-ai/step-5-preview writer/palmyra-creative-122b writer/palmyra-fin-70b-32k writer/palmyra-med-70b
writer/palmyra-med-70b-32k z-ai/glm-5.3 z-ai/glm-5.3-flash zyphra/zamba2-7b-instruct""".split()


def test_size_is_read_from_the_model_name():
    assert _size_b("google/gemma-4-31b-it") == 31.0
    assert _size_b("nvidia/nemotron-3-super-120b-a12b") == 120.0  # total size, not the active part
    assert _size_b("mistralai/mixtral-8x22b-v0.1") == 22.0
    assert _size_b("ibm/granite-3.0-3b-a800m-instruct") == 3.0
    assert _size_b("mistralai/mistral-large") is None


def test_listed_preferred_models_come_first_in_the_given_order():
    order = rank_models(["a/x", "b/y-it", "c/z"], ["c/z", "a/x", "missing/m"])
    assert order[:2] == ["c/z", "a/x"] and "missing/m" not in order


def test_embedding_vision_safety_code_and_reasoning_models_are_never_tried():
    ranked = rank_models(LIVE_LIST, [])
    for m in ["nvidia/nv-embedqa-mistral-7b-v2", "meta/llama-guard-4-12b", "google/codegemma-7b", "nvidia/nemotron-parse",
              "meta/llama-3.2-90b-vision-instruct", "nvidia/nvclip", "nvidia/riva-translate-4b-instruct",
              "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "google/diffusiongemma-26b-a4b-it",
              "mistralai/codestral-22b-instruct-v0.1", "snowflake/arctic-embed-l", "nvidia/nemotron-4-340b-reward"]:
        assert m not in ranked, m
    assert len(ranked) == len(set(ranked)) and "google/gemma-4-31b-it" in ranked


def test_instruction_tuned_models_come_first_and_nearest_30b_leads():
    ranked = rank_models(LIVE_LIST, [])
    assert ranked[:2] == ["google/gemma-4-31b-it", "nvidia/llama-3.1-nemotron-51b-instruct"]
    assert ranked.index("nvidia/nemotron-4-340b-instruct") < ranked.index("moonshotai/kimi-k2.6")  # tuned before unmarked


def test_ranking_edge_cases():
    assert rank_models([], ["a/b"]) == []
    assert rank_models(["a/embed-1b"], []) == []


def test_discovery_finds_a_working_model_when_every_preferred_one_is_retired(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(["old/one", "new/model-12b-instruct"], by_model={"old/one": FakeAPIError(410)})
    assert resolve_model(client, None, ["old/one"]) == "new/model-12b-instruct"
    assert [c["model"] for c in client._client.calls] == ["old/one", "new/model-12b-instruct"]


def test_error_explains_why_each_model_was_skipped(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    gone = {"x/one": FakeAPIError(410), "y/two": FakeAPIError(404), "z/three-it": FakeAPIError(400),
            "w/four": FakeAPIError(503)}
    with pytest.raises(ModelUnavailableError) as e:
        resolve_model(nim(list(gone), by_model=gone), None, [])
    msg = str(e.value)
    assert "x/one: retired or unavailable (HTTP 410)" in msg and "y/two: retired or unavailable (HTTP 404)" in msg
    assert "z/three-it: rejected the request (HTTP 400)" in msg and "w/four: too slow or erroring" in msg
    assert "PV_MODEL" in msg


@pytest.mark.parametrize("status", [403, 422])
def test_other_per_model_rejections_are_skipped_too(monkeypatch, status):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(["a/first-it", "b/second-it"], by_model={"a/first-it": FakeAPIError(status)})
    assert resolve_model(client, None, []) == "b/second-it"


@pytest.mark.parametrize("status", [401, 429])
def test_bad_key_or_throttling_stops_the_search_instead_of_skipping(monkeypatch, status):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(["a/first-it", "b/second-it"], by_model={"a/first-it": FakeAPIError(status)})
    with pytest.raises(FakeAPIError):
        resolve_model(client, None, [])
    assert "b/second-it" not in [c["model"] for c in client._client.calls]


def test_search_stops_after_max_tries_and_says_what_was_left(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    ids = [f"p/m{i}-it" for i in range(5)]
    client = nim(ids, by_model={m: FakeAPIError(410) for m in ids})
    with pytest.raises(ModelUnavailableError) as e:
        resolve_model(client, None, [], max_tries=2)
    assert "2 models tried" in str(e.value) and "3 more were not tried" in str(e.value)
    assert "NIM_MAX_MODEL_TRIES" in str(e.value) and len(client._client.calls) == 2


def test_max_tries_default_comes_from_config(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    monkeypatch.setattr(config, "NIM_MAX_MODEL_TRIES", 1)
    ids = ["p/m0-it", "p/m1-it"]
    client = nim(ids, by_model={"p/m0-it": FakeAPIError(410)})
    with pytest.raises(ModelUnavailableError):
        resolve_model(client, None, [])
    assert len(client._client.calls) == 1


def test_accept_callback_can_reject_a_model_that_answers(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    client = nim(["p/m0-it", "p/m1-it"])
    assert resolve_model(client, None, [], accept=lambda m: m != "p/m0-it") == "p/m1-it"


def test_progress_is_reported_through_say(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    lines = []
    client = nim(["a/first", "b/second"], by_model={"a/first": FakeAPIError(410)})
    resolve_model(client, None, ["a/first", "b/second"], say=lines.append)
    text = "\n".join(lines)
    assert "trying a/first" in text and "retired or unavailable" in text and "trying b/second" in text and "ok" in text


def test_explicit_model_is_silent_and_untouched(monkeypatch):
    lines = []
    assert resolve_model(nim(["a/b"]), "my/pick", say=lines.append) == "my/pick" and lines == []


# ---------- per-call timeout and retry overrides ----------

def test_timeout_option_is_passed_to_the_sdk():
    client, _ = make()
    client.chat(model="m", messages=MSGS, options={"timeout": 20})
    assert client._client.calls[0]["timeout"] == 20


def test_retries_option_overrides_the_default_for_one_call():
    client, _ = make([FakeAPIError(503), "OK"], max_retries=4)
    with pytest.raises(FakeAPIError):
        client.chat(model="m", messages=MSGS, options={"retries": 0})
    assert len(client._client.calls) == 1


def test_model_unavailable_error_carries_the_status():
    client, _ = make([FakeAPIError(410)])
    with pytest.raises(ModelUnavailableError) as e:
        client.chat(model="m", messages=MSGS)
    assert e.value.status == 410


# ---------- stalled models ----------
def _timeout():
    return type("APITimeoutError", (Exception,), {})("Request timed out.")


STALL_SCHEMA = {"type": "object"}


def test_first_timeout_retries_without_guided_json():
    client, _ = make([_timeout(), "OK"])
    assert client.chat(model="m", messages=MSGS, format=STALL_SCHEMA)["message"]["content"] == "OK"
    calls = client._client.calls
    assert "extra_body" in calls[0] and "extra_body" not in calls[1]
    assert client._guided is False  # remembered for later calls


def test_second_timeout_gives_up_instead_of_waiting_through_all_retries():
    client, sleeps = make([_timeout()], max_retries=4)
    with pytest.raises(Exception) as e:
        client.chat(model="m", messages=MSGS, format=STALL_SCHEMA)
    assert type(e.value).__name__ == "APITimeoutError" and len(client._client.calls) == 2


def test_timeout_without_guided_json_also_gets_exactly_one_retry():
    client, _ = make([_timeout(), "OK"])
    assert client.chat(model="m", messages=MSGS)["message"]["content"] == "OK"
    assert len(client._client.calls) == 2


def test_a_timeout_during_the_accept_test_skips_the_model_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)

    def accept(model):
        if model == "a/slow-it":
            raise _timeout()
        return True

    got = resolve_model(nim(["a/slow-it", "b/fast-it"]), None, [], accept=accept)
    assert got == "b/fast-it"
