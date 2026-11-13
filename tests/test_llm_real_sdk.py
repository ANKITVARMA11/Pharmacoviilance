"""NimClient driven through the real `openai` SDK against a local fake NIM server (no internet, no key).

Proves what goes over the wire: URL, auth header, `nvext` at the top level of the body, and that the
SDK's real exceptions are classified correctly. Skipped if the openai package is not installed.
"""
import pytest

openai = pytest.importorskip("openai")

from pvcrew.extract import extract_case  # noqa: E402
from pvcrew.llm import ModelUnavailableError, NimClient, resolve_model  # noqa: E402
from pvcrew.schemas import Case  # noqa: E402
from tests.fake_nim import GOOD, FakeNimServer, completion  # noqa: E402

SCHEMA = Case.model_json_schema()
USER = [{"role": "user", "content": "x"}]


@pytest.fixture
def nim(monkeypatch):
    """nim(script=..., models=...) -> (server, make_client)."""
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    servers = []

    def start(**kw):
        srv = FakeNimServer(**kw).start()
        servers.append(srv)

        def make_client(**ckw):
            sdk = openai.OpenAI(base_url=srv.base_url, api_key="nvapi-test", max_retries=0)
            return NimClient(sdk, min_interval=0, sleep=lambda s: None, **ckw)

        return srv, make_client

    yield start
    for s in servers:
        s.stop()


def test_wire_format_matches_nim_docs(nim):
    srv, make_client = nim(script=[(200, completion(GOOD))])
    out = make_client().chat(model="meta/x", messages=USER, format=SCHEMA, options={"temperature": 0})
    assert out["message"]["content"] == GOOD
    req = srv.chat_requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer nvapi-test"
    assert req["body"]["model"] == "meta/x" and req["body"]["temperature"] == 0
    assert req["body"]["nvext"] == {"guided_json": SCHEMA}  # top level, not nested under extra_body
    assert req["body"]["messages"][0]["role"] == "system"


def test_real_429_is_retried(nim):
    srv, make_client = nim(script=[(429, {"error": "slow"}), (429, {"error": "slow"}), (200, completion(GOOD))])
    assert make_client().chat(model="m", messages=USER)["message"]["content"] == GOOD
    assert len(srv.chat_requests) == 3


def test_real_500_is_retried(nim):
    srv, make_client = nim(script=[(500, {"error": "oops"}), (200, completion(GOOD))])
    assert make_client().chat(model="m", messages=USER)["message"]["content"] == GOOD
    assert len(srv.chat_requests) == 2


def test_real_400_on_nvext_falls_back_to_prompt_only(nim):
    srv, make_client = nim(script=[(400, {"error": "unknown field nvext"}), (200, completion(GOOD))])
    assert make_client().chat(model="m", messages=USER, format=SCHEMA)["message"]["content"] == GOOD
    first, second = srv.chat_requests
    assert "nvext" in first["body"] and "nvext" not in second["body"]
    assert "JSON schema" in second["body"]["messages"][0]["content"]


def test_real_401_is_raised_not_retried(nim):
    srv, make_client = nim(script=[(401, {"error": "bad key"})])
    with pytest.raises(openai.AuthenticationError):
        make_client().chat(model="m", messages=USER)
    assert len(srv.chat_requests) == 1


def test_real_dropped_connection_is_retried(nim):
    srv, make_client = nim(script=["drop", (200, completion(GOOD))])
    assert make_client().chat(model="m", messages=USER)["message"]["content"] == GOOD
    assert len(srv.chat_requests) == 2


def test_real_persistent_429_raises_after_retries(nim):
    srv, make_client = nim(script=[(429, {"error": "slow"})])
    with pytest.raises(openai.RateLimitError):
        make_client(max_retries=2).chat(model="m", messages=USER)
    assert len(srv.chat_requests) == 3


def test_real_410_gone_for_a_retired_model_raises_model_unavailable(nim):
    body = {"type": "about:blank", "title": "Gone", "status": 410,
            "detail": "The model 'meta/llama-3.3-70b-instruct' has reached its end of life and is no longer available."}
    srv, make_client = nim(script=[(410, body)])
    with pytest.raises(ModelUnavailableError, match="meta/llama-3.3-70b-instruct"):
        make_client().chat(model="meta/llama-3.3-70b-instruct", messages=USER, format=SCHEMA)
    assert len(srv.chat_requests) == 1  # not retried


def test_real_list_models_returns_sorted_ids(nim):
    srv, make_client = nim(models=["nvidia/zeta", "a/first", "b/second"])
    assert make_client().list_models() == ["a/first", "b/second", "nvidia/zeta"]
    assert srv.requests[0]["path"] == "/v1/models"


def test_real_resolve_model_skips_retired_and_uses_live_model(nim, monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    srv, make_client = nim(models=["a/first", "b/second"], retired=["a/first"])
    assert resolve_model(make_client(), None, ["a/first", "b/second"]) == "b/second"
    assert srv.requests[0]["path"] == "/v1/models" and srv.requests[0]["auth"] == "Bearer nvapi-test"
    assert [r["body"]["model"] for r in srv.chat_requests] == ["a/first", "b/second"]


def test_extraction_end_to_end_through_the_sdk(nim):
    srv, make_client = nim()
    case = extract_case("Took amoxicillin, got a rash.", make_client(), model="m")
    assert case.suspect_drug == "Amoxicillin" and case.reactions == ["rash"]
