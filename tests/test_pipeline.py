import pytest

from pvcrew import pipeline
from pvcrew.llm import NimClient
from pvcrew.jsonl import read_jsonl
from tests.conftest import FakeAPIError, FakeClient, FakeOpenAI, FakeSession


def test_fetch_sample_copies_bundled_records(tmp_path):
    out = tmp_path / "raw.jsonl"
    assert pipeline.cmd_fetch(10, out, sample=True) == 6
    assert len(list(read_jsonl(out))) == 6


def test_fetch_live_path_writes_what_the_session_returns(tmp_path, sample_records):
    session = FakeSession(pages={0: sample_records[:3]})
    out = tmp_path / "raw.jsonl"
    assert pipeline.cmd_fetch(3, out, session=session) == 3


def test_synth_offline_writes_pairs(tmp_path):
    raw, pairs = tmp_path / "raw.jsonl", tmp_path / "pairs.jsonl"
    pipeline.cmd_fetch(10, raw, sample=True)
    assert pipeline.cmd_synth(raw, pairs, offline=True) == 5
    rows = list(read_jsonl(pairs))
    assert all(r["source"] == "offline" and r["report"] and r["truth"] for r in rows)


def test_synth_with_llm_client(tmp_path):
    raw, pairs = tmp_path / "raw.jsonl", tmp_path / "pairs.jsonl"
    pipeline.cmd_fetch(10, raw, sample=True)
    client = FakeClient(["Report: patient on AMOXICILLIN AMOXICILLIN had issues, details to follow soon."])
    pipeline.cmd_synth(raw, pairs, offline=False, client=client)
    assert len(client.calls) == 5


def _make_pairs(tmp_path):
    raw, pairs = tmp_path / "raw.jsonl", tmp_path / "pairs.jsonl"
    pipeline.cmd_fetch(10, raw, sample=True)
    pipeline.cmd_synth(raw, pairs, offline=True)
    return pairs


def test_extract_summarises_and_keeps_truth_next_to_prediction(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"
    summary = pipeline.cmd_extract(pairs, out, client=FakeClient([good_json]))
    assert summary == {"total": 5, "valid": 5, "failed": 0, "validity_rate": 1.0}
    rows = list(read_jsonl(out))
    assert rows[0]["prediction"]["suspect_drug"] == "Amoxicillin"
    assert rows[0]["truth"]["suspect_drug"] == "AMOXICILLIN"
    assert rows[0]["error"] is None


def test_extract_records_failures_instead_of_crashing(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"
    # first record burns all 3 attempts on junk, the other four succeed
    client = FakeClient(["junk", "junk", "junk", good_json])
    summary = pipeline.cmd_extract(pairs, out, client=client)
    assert (summary["valid"], summary["failed"]) == (4, 1)
    rows = list(read_jsonl(out))
    assert rows[0]["prediction"] is None and "no valid output" in rows[0]["error"]


def test_extract_saves_each_row_immediately_so_a_crash_loses_nothing(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"

    class DiesOnThirdCall:
        def __init__(self):
            self.n = 0

        def chat(self, **kw):
            self.n += 1
            if self.n == 3:
                raise ConnectionError("network dropped")
            return {"message": {"content": good_json}}

    with pytest.raises(ConnectionError):
        pipeline.cmd_extract(pairs, out, client=DiesOnThirdCall())
    assert len(list(read_jsonl(out))) == 2


def test_resume_continues_without_redoing_finished_ids(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"
    pipeline.cmd_extract(pairs, out, limit=2, client=FakeClient([good_json]))
    client = FakeClient([good_json])
    summary = pipeline.cmd_extract(pairs, out, client=client, resume=True)
    assert len(client.calls) == 3  # only the 3 new ones
    assert summary["total"] == 5
    ids = [r["id"] for r in read_jsonl(out)]
    assert len(ids) == len(set(ids)) == 5


def test_without_resume_the_output_is_overwritten(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"
    pipeline.cmd_extract(pairs, out, client=FakeClient([good_json]))
    pipeline.cmd_extract(pairs, out, limit=1, client=FakeClient([good_json]))
    assert len(list(read_jsonl(out))) == 1


def test_resume_flag_on_cli(tmp_path, good_json, monkeypatch):
    client = FakeClient([good_json])
    monkeypatch.setattr(pipeline, "get_client", lambda: client)
    d = ["--data-dir", str(tmp_path)]
    pipeline.main(d + ["fetch", "--sample"])
    pipeline.main(d + ["synth", "--offline"])
    pipeline.main(d + ["extract", "--limit", "2"])
    pipeline.main(d + ["extract", "--resume"])
    assert len(client.calls) == 5
    assert len(list(read_jsonl(tmp_path / "extractions.jsonl"))) == 5


def test_extract_limit(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"
    assert pipeline.cmd_extract(pairs, out, limit=2, client=FakeClient([good_json]))["total"] == 2


def test_extract_on_empty_input_has_zero_rate(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("")
    assert pipeline.cmd_extract(pairs, tmp_path / "o.jsonl", client=FakeClient(["x"]))["validity_rate"] == 0.0


def test_cli_all_runs_end_to_end(tmp_path, good_json, monkeypatch):
    monkeypatch.setattr(pipeline, "get_client", lambda: FakeClient([good_json]))
    pipeline.main(["--data-dir", str(tmp_path), "all", "--sample", "--offline"])
    for name in ["faers_raw.jsonl", "pairs.jsonl", "extractions.jsonl"]:
        assert (tmp_path / name).exists()
    assert len(list(read_jsonl(tmp_path / "extractions.jsonl"))) == 5


def test_cli_steps_run_separately(tmp_path, good_json, monkeypatch):
    monkeypatch.setattr(pipeline, "get_client", lambda: FakeClient([good_json]))
    d = ["--data-dir", str(tmp_path)]
    pipeline.main(d + ["fetch", "--sample"])
    pipeline.main(d + ["synth", "--offline", "--seed", "4"])
    pipeline.main(d + ["extract", "--limit", "3"])
    assert len(list(read_jsonl(tmp_path / "extractions.jsonl"))) == 3


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        pipeline.main([])


def test_extract_auto_picks_a_live_model_and_uses_it_for_every_call(tmp_path, good_json, monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    from pvcrew import config

    first, second = config.NIM_CANDIDATES[:2]
    openai_fake = FakeOpenAI([good_json], model_ids=[first, second], by_model={first: FakeAPIError(410)})
    client = NimClient(openai_fake, min_interval=0, sleep=lambda s: None)
    pipeline.cmd_extract(_make_pairs(tmp_path), tmp_path / "ext.jsonl", limit=2, client=client)
    used = [c["model"] for c in openai_fake.calls]
    assert used[0] == first            # probe of the retired model
    assert set(used[1:]) == {second}   # probe + both extractions use the working one


def test_explicit_model_flag_skips_the_probe(tmp_path, good_json, monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    openai_fake = FakeOpenAI([good_json], model_ids=["x/y"])
    client = NimClient(openai_fake, min_interval=0, sleep=lambda s: None)
    pipeline.cmd_extract(_make_pairs(tmp_path), tmp_path / "ext.jsonl", model="my/choice", limit=2, client=client)
    assert {c["model"] for c in openai_fake.calls} == {"my/choice"} and len(openai_fake.calls) == 2


def test_retired_model_stops_the_run_with_a_clear_error_and_keeps_finished_rows(tmp_path, good_json):
    pairs, out = _make_pairs(tmp_path), tmp_path / "ext.jsonl"
    openai_fake = FakeOpenAI([good_json, good_json, FakeAPIError(410)])
    client = NimClient(openai_fake, min_interval=0, sleep=lambda s: None)
    with pytest.raises(SystemExit) as e:
        pipeline.cmd_extract(pairs, out, model="m", client=client)
    assert "HTTP 410" in str(e.value) and "--resume" in str(e.value)
    assert len(list(read_jsonl(out))) == 2


# ---------- a model that passes the small probe but 404s on real requests ----------
from tests.fake_nim import GOOD  # noqa: E402


class ProbeOnlyOpenAI(FakeOpenAI):
    """`flaky` answers the 5-token probe. Its real requests succeed `ok_real` times, then fail `fail_count`
    times (default: forever) with `error`. Other models always answer."""

    def __init__(self, flaky, ok_real=0, fail_count=10**6, error=None, **kw):
        super().__init__(outcomes=[GOOD], **kw)
        self.flaky, self.ok_real, self.fail_count = flaky, ok_real, fail_count
        self.error = error or FakeAPIError(404)

    def _create(self, **kwargs):
        if kwargs.get("model") == self.flaky and kwargs.get("max_tokens") != 5:
            if self.ok_real > 0:
                self.ok_real -= 1
            elif self.fail_count > 0:
                self.fail_count -= 1
                self.calls.append(kwargs)
                raise self.error
        return super()._create(**kwargs)


def _client(fake):
    return NimClient(fake, min_interval=0, sleep=lambda s: None)


@pytest.fixture
def unpinned(monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)


def test_model_that_fails_the_extraction_test_is_not_chosen(tmp_path, unpinned):
    fake = ProbeOnlyOpenAI("a/flaky-it", model_ids=["a/flaky-it", "b/steady-it"])
    s = pipeline.cmd_extract(_make_pairs(tmp_path), tmp_path / "e.jsonl", client=_client(fake), limit=2)
    assert s["valid"] == 2 and {c["model"] for c in fake.calls[-2:]} == {"b/steady-it"}


def test_unpinned_run_fails_over_mid_run_and_finishes(tmp_path, unpinned, capsys):
    fake = ProbeOnlyOpenAI("a/flaky-it", ok_real=2, model_ids=["a/flaky-it", "b/steady-it"])  # test + report 1
    s = pipeline.cmd_extract(_make_pairs(tmp_path), tmp_path / "e.jsonl", client=_client(fake), limit=3)
    assert s["valid"] == 3 and s["failed"] == 0
    assert "stopped answering real requests" in capsys.readouterr().out
    assert {c["model"] for c in fake.calls[-2:]} == {"b/steady-it"}


def test_a_stalled_model_triggers_failover_too(tmp_path, unpinned, capsys):
    stalled = type("APITimeoutError", (Exception,), {})("Request timed out.")
    fake = ProbeOnlyOpenAI("a/flaky-it", ok_real=2, error=stalled, model_ids=["a/flaky-it", "b/steady-it"])
    s = pipeline.cmd_extract(_make_pairs(tmp_path), tmp_path / "e.jsonl", client=_client(fake), limit=3)
    out = capsys.readouterr().out
    assert s["valid"] == 3 and "too slow or erroring" in out and "b/steady-it" in {c["model"] for c in fake.calls}


def test_one_stray_404_is_retried_on_the_same_model(tmp_path, unpinned, capsys):
    fake = ProbeOnlyOpenAI("a/flaky-it", ok_real=2, fail_count=1, model_ids=["a/flaky-it", "b/steady-it"])
    s = pipeline.cmd_extract(_make_pairs(tmp_path), tmp_path / "e.jsonl", client=_client(fake), limit=3)
    assert s["valid"] == 3 and "stopped answering" not in capsys.readouterr().out
    assert "b/steady-it" not in {c["model"] for c in fake.calls}


def test_pinned_model_is_never_swapped_and_finished_rows_are_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MODEL", "a/flaky-it")
    fake = ProbeOnlyOpenAI("a/flaky-it", ok_real=1, model_ids=["a/flaky-it", "b/steady-it"])  # test + report 1
    out = tmp_path / "e.jsonl"
    with pytest.raises(SystemExit) as e:
        pipeline.cmd_extract(_make_pairs(tmp_path), out, client=_client(fake), limit=3)
    assert "PV_MODEL" in str(e.value) and "--resume" in str(e.value)
    assert "b/steady-it" not in {c["model"] for c in fake.calls}


def test_resolve_model_can_exclude_models():
    from pvcrew.llm import resolve_model
    fake = FakeOpenAI(outcomes=["OK"], model_ids=["a/one-it", "b/two-it"])
    assert resolve_model(_client(fake), None, [], exclude=["a/one-it"]) == "b/two-it"


# ---------- terms and code commands ----------
import json as _json  # noqa: E402

from pvcrew import config as _config  # noqa: E402


def _extractions(tmp_path, reactions_per_row):
    rows = []
    for i, reacts in enumerate(reactions_per_row):
        rows.append({"id": str(i), "report": "r", "truth": {"reactions": ["Diarrhoea"]},
                     "prediction": None if reacts is None else {"reactions": reacts}, "error": None})
    p = tmp_path / "ext.jsonl"
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_terms_sample_copies_the_bundled_list(tmp_path):
    out = tmp_path / "terms.jsonl"
    n = pipeline.cmd_terms(out=out, sample=True)
    assert n > 50 and len(list(read_jsonl(out))) == n


def test_terms_live_path_writes_what_openfda_returns(tmp_path):
    from tests.test_coding import CountSession
    out = tmp_path / "terms.jsonl"
    assert pipeline.cmd_terms(10, out, session=CountSession([{"term": "Rash", "count": 2}])) == 1
    assert list(read_jsonl(out)) == [{"term": "Rash", "count": 2}]


def test_code_needs_the_terms_file_and_says_how_to_make_it(tmp_path):
    with pytest.raises(SystemExit) as e:
        pipeline.cmd_code(_extractions(tmp_path, [["x"]]), tmp_path / "c.jsonl", tmp_path / "nope.jsonl",
                          client=FakeClient(["{}"]))
    assert "terms" in str(e.value)


def test_code_writes_coded_rows_and_keeps_failed_rows(tmp_path, monkeypatch):
    monkeypatch.delenv("PV_MODEL", raising=False)
    pipeline.cmd_terms(out=tmp_path / "terms.jsonl", sample=True)
    ext = _extractions(tmp_path, [["the runs", "Nausea"], None, ["Rash"]])
    fake = FakeOpenAI(outcomes=[GOOD], model_ids=["a/b-it"])
    # the model answers: probe ok, extraction test ok, then the normalising call
    fake.outcomes = [GOOD, GOOD, '{"items":[{"reported":"the runs","medical_term":"diarrhoea"}]}']
    out = tmp_path / "c.jsonl"
    s = pipeline.cmd_code(ext, out, tmp_path / "terms.jsonl", client=_client(fake))
    rows = list(read_jsonl(out))
    assert len(rows) == 3 and rows[1]["prediction"] is None and "coded" not in rows[1]
    assert rows[0]["reactions_coded"] == ["Diarrhoea", "Nausea"] and rows[2]["reactions_coded"] == ["Rash"]
    assert s["exact"] == 3 and s["reactions"] == 3


def test_code_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MODEL", "m")
    pipeline.cmd_terms(out=tmp_path / "terms.jsonl", sample=True)
    ext = _extractions(tmp_path, [["Rash"], ["Rash"], ["Rash"]])
    s = pipeline.cmd_code(ext, tmp_path / "c.jsonl", tmp_path / "terms.jsonl", limit=2, client=_client(FakeOpenAI()))
    assert s["rows"] == 2


def test_model_session_is_reusable_for_any_call(tmp_path, unpinned):
    fake = ProbeOnlyOpenAI("a/flaky-it", ok_real=1, model_ids=["a/flaky-it", "b/steady-it"])
    session = pipeline.ModelSession(_client(fake), say=lambda *_: None)
    assert session.model == "a/flaky-it"
    seen = []

    def fn(m):
        seen.append(m)
        if m == "a/flaky-it":
            from pvcrew.llm import ModelUnavailableError
            raise ModelUnavailableError("gone", status=404)
        return "done"

    assert session.call(fn) == "done" and seen == ["a/flaky-it", "a/flaky-it", "b/steady-it"]
    assert session.model == "b/steady-it"


def test_code_warns_when_the_vocabulary_is_small(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PV_MODEL", "m")
    pipeline.cmd_terms(out=tmp_path / "terms.jsonl", sample=True)  # ~100 terms
    pipeline.cmd_code(_extractions(tmp_path, [["Rash"]]), tmp_path / "c.jsonl", tmp_path / "terms.jsonl",
                      client=_client(FakeOpenAI()))
    assert "only 99 terms" in capsys.readouterr().out


def test_code_resume_skips_rows_already_coded_and_appends_the_rest(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MODEL", "m")
    pipeline.cmd_terms(out=tmp_path / "terms.jsonl", sample=True)
    ext = _extractions(tmp_path, [["Rash"], ["Nausea"], ["Vomiting"]])
    out = tmp_path / "c.jsonl"
    pipeline.cmd_code(ext, out, tmp_path / "terms.jsonl", limit=1, client=_client(FakeOpenAI()))
    assert len(list(read_jsonl(out))) == 1
    s = pipeline.cmd_code(ext, out, tmp_path / "terms.jsonl", client=_client(FakeOpenAI()), resume=True)
    rows = list(read_jsonl(out))
    assert [r["id"] for r in rows] == ["0", "1", "2"] and s["rows"] == 3 and s["reactions"] == 2


def test_pinned_model_failure_while_coding_keeps_finished_rows_and_resume_works(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_MODEL", "m")
    pipeline.cmd_terms(out=tmp_path / "terms.jsonl", sample=True)
    ext = _extractions(tmp_path, [["Rash"], ["the runs"], ["Nausea"]])
    stalled = type("APITimeoutError", (Exception,), {})("Request timed out.")
    fake = FakeOpenAI(outcomes=[stalled])
    out = tmp_path / "c.jsonl"
    with pytest.raises(SystemExit) as e:
        pipeline.cmd_code(ext, out, tmp_path / "terms.jsonl", client=_client(fake))
    assert "--resume" in str(e.value) and len(list(read_jsonl(out))) == 1
    fake2 = FakeOpenAI(outcomes=['{"items":[{"reported":"the runs","medical_term":"diarrhoea"}]}'])
    pipeline.cmd_code(ext, out, tmp_path / "terms.jsonl", client=_client(fake2), resume=True)
    assert [r["reactions_coded"] for r in read_jsonl(out)] == [["Rash"], ["Diarrhoea"], ["Nausea"]]


def test_fetch_starts_at_the_requested_skip(tmp_path, sample_records):
    session = FakeSession(pages={700: sample_records[:2]})
    assert pipeline.cmd_fetch(2, tmp_path / "raw.jsonl", session=session, start_skip=700) == 2
    assert session.requests[0]["skip"] == 700


def test_skip_flag_on_cli(tmp_path, sample_records, monkeypatch):
    asked = {}

    def fake_fetch(n, session=None, api_key=None, start_skip=0):
        asked.update(n=n, start_skip=start_skip)
        return sample_records[:n]

    monkeypatch.setattr(pipeline, "fetch_records", fake_fetch)
    pipeline.main(["--data-dir", str(tmp_path), "fetch", "--n", "3", "--skip", "2000"])
    assert asked == {"n": 3, "start_skip": 2000}
    pipeline.main(["--data-dir", str(tmp_path), "fetch", "--n", "2"])
    assert asked == {"n": 2, "start_skip": 0}  # default: from the start
