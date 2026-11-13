import pytest

from pvcrew import extract
from pvcrew.extract import ExtractionError, extract_case
from pvcrew.schemas import Case
from tests.conftest import FakeClient


def test_returns_validated_case(good_json):
    client = FakeClient([good_json])
    c = extract_case("some report", client, model="m")
    assert isinstance(c, Case)
    assert c.suspect_drug == "Amoxicillin" and c.serious is True and c.patient_age == 67


def test_request_carries_schema_zero_temperature_and_the_report(good_json):
    client = FakeClient([good_json])
    extract_case("my report text", client, model="tiny-model")
    call = client.calls[0]
    assert call["model"] == "tiny-model"
    assert call["format"] == Case.model_json_schema()
    assert call["options"]["temperature"] == 0
    assert "my report text" in call["messages"][-1]["content"]
    assert call["messages"][0]["role"] == "system"


def test_retries_after_invalid_json_and_tells_the_model_what_was_wrong(good_json):
    client = FakeClient(["not json at all", good_json])
    assert extract_case("r", client).suspect_drug == "Amoxicillin"
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "invalid" in retry_messages[-1]["content"]


def test_retries_after_schema_violation(good_json):
    empty_reactions = '{"suspect_drug": "X", "reactions": [], "serious": false}'
    client = FakeClient([empty_reactions, good_json])
    assert extract_case("r", client).reactions == ["rash", "trouble breathing"]
    assert len(client.calls) == 2


def test_gives_up_after_retries_exhausted():
    client = FakeClient(["nope"])
    with pytest.raises(ExtractionError):
        extract_case("r", client, retries=2)
    assert len(client.calls) == 3


def test_zero_retries_means_one_attempt():
    client = FakeClient(["nope"])
    with pytest.raises(ExtractionError):
        extract_case("r", client, retries=0)
    assert len(client.calls) == 1


def test_build_messages_shape():
    msgs = extract.build_messages("hello")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"].endswith("hello")


def test_connection_errors_are_not_swallowed():
    class Boom:
        def chat(self, **kw):
            raise ConnectionError("ollama is not running")

    with pytest.raises(ConnectionError):
        extract_case("r", Boom())
