import pytest
import requests

from pvcrew import faers
from tests.conftest import FakeSession


def test_collects_requested_count_across_pages_and_skips_unusable(sample_records):
    r = sample_records
    session = FakeSession(pages={0: [r[0], r[5]], 2: [r[1], r[2]], 4: [r[3]]})  # r[5] is unusable
    got = faers.fetch_records(3, session=session, page_size=2, pause=0)
    assert [g["safetyreportid"] for g in got] == ["SAMPLE-001", "SAMPLE-002", "SAMPLE-003"]
    assert [q["skip"] for q in session.requests] == [0, 2]  # stopped as soon as it had enough


def test_stops_when_results_run_out(sample_records):
    session = FakeSession(pages={0: sample_records[:2]})
    got = faers.fetch_records(50, session=session, page_size=2, pause=0)
    assert len(got) == 2


def test_404_means_no_matches_and_is_not_an_error():
    session = FakeSession(status_by_skip={0: 404})
    assert faers.fetch_records(5, session=session, pause=0) == []


def test_server_error_is_raised():
    session = FakeSession(status_by_skip={0: 500})
    with pytest.raises(requests.HTTPError):
        faers.fetch_records(5, session=session, pause=0)


def test_api_key_and_start_skip_are_sent(sample_records):
    session = FakeSession(pages={100: sample_records[:1]})
    faers.fetch_records(1, session=session, api_key="KEY", start_skip=100, pause=0)
    assert session.requests[0]["api_key"] == "KEY"
    assert session.requests[0]["skip"] == 100


def test_no_api_key_param_when_not_given(sample_records):
    session = FakeSession(pages={0: sample_records[:1]})
    faers.fetch_records(1, session=session, pause=0)
    assert "api_key" not in session.requests[0]


def test_page_size_is_capped_at_openfda_limit():
    session = FakeSession()
    faers.fetch_records(1, session=session, page_size=5000, pause=0)
    assert session.requests[0]["limit"] == 1000


def test_does_not_page_beyond_openfda_skip_limit(sample_records):
    session = FakeSession(pages={faers.MAX_SKIP + 1: sample_records[:1]})
    assert faers.fetch_records(1, session=session, start_skip=faers.MAX_SKIP + 1, pause=0) == []
    assert session.requests == []


def test_save_and_load_records(tmp_path, sample_records):
    path = tmp_path / "raw.jsonl"
    assert faers.save_records(path, sample_records) == 6
    assert faers.load_records(path) == sample_records
