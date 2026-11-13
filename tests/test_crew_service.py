import json

import pytest

pytest.importorskip("autogen_agentchat")

from pvcrew import config  # noqa: E402
from pvcrew.crew import service  # noqa: E402
from pvcrew.crew.agents import CrewError  # noqa: E402
from pvcrew.crew.events import EventBus  # noqa: E402
from pvcrew.crew.service import Runtime, _changes, load_samples, read_audit, read_reviews  # noqa: E402
from tests.crewfakes import GOOD_CASE, NARRATIVE, REPORT, DeadSession, FakeSession, RoleClient  # noqa: E402


@pytest.fixture
def rt(tmp_path, monkeypatch):
    """A runtime with a fake model, the bundled demo vocabulary and every file under tmp_path."""
    monkeypatch.setattr(config, "REVIEWS_PATH", tmp_path / "reviews.jsonl")
    monkeypatch.setattr(config, "AUDIT_PATH", tmp_path / "audit.jsonl")
    return Runtime(session=FakeSession(), client=RoleClient(), terms_path=tmp_path / "no-terms.jsonl",
                   audit_path=tmp_path / "audit.jsonl")


# ---------- running ----------

def test_run_returns_the_crew_result_and_writes_the_audit_file(rt, tmp_path):
    res = rt.run(REPORT, "rep-1")
    assert res["status"] == "done" and res["report_id"] == "rep-1" and res["case"]["suspect_drug"] == "Amoxicillin"
    rows = read_audit(path=tmp_path / "audit.jsonl")
    assert {r["run"] for r in rows if r["kind"] == "message"} == {res["run"]}


def test_an_empty_report_is_refused_before_anything_is_called(rt):
    for blank in ("", "   \n ", None):
        with pytest.raises(CrewError, match="empty"):
            rt.run(blank)
    assert rt.bus.since(0) == []


def test_the_report_is_trimmed(rt):
    rt.run("  " + REPORT + "\n\n")
    start = [e for e in rt.bus.since(0) if e["kind"] == "run_start"][0]
    assert start["report"] == REPORT


def test_missing_vocabulary_falls_back_to_the_demo_list_and_says_so(rt):
    rt.run(REPORT)
    assert "small demo vocabulary" in rt.terms_note
    notes = [e["text"] for e in rt.bus.since(0) if e["kind"] == "system"]
    assert any("small demo vocabulary" in n for n in notes)


def test_a_real_vocabulary_is_used_when_it_exists(rt, tmp_path):
    terms = tmp_path / "terms.jsonl"
    terms.write_text('{"term": "Rash", "count": 50}\n{"term": "Dyspnoea", "count": 40}\n', encoding="utf-8")
    rt2 = Runtime(session=FakeSession(), client=RoleClient(), terms_path=terms, audit_path=tmp_path / "a2.jsonl")
    res = rt2.run(REPORT)
    assert rt2.terms_note == "" and [c["term"] for c in res["coded"]] == ["Rash", "Dyspnoea"]
    assert any("Vocabulary: 2 standard terms" in e.get("text", "") for e in rt2.bus.since(0))


def test_the_revision_limit_comes_from_config(rt, monkeypatch):
    monkeypatch.setattr(config, "CREW_MAX_REVISIONS", 0)
    from tests.crewfakes import REVISE
    rt.client = RoleClient(critic=REVISE)
    res = rt.run(REPORT)
    assert res["attempts"] == 1 and res["verdict"] == "ESCALATE"


def test_setup_failure_is_raised_and_shown_in_the_monitor(rt, monkeypatch):
    rt.session = rt.client = None
    from pvcrew import llm
    monkeypatch.setattr(llm, "get_client", lambda: (_ for _ in ()).throw(SystemExit("NVIDIA_API_KEY is not set.")))
    with pytest.raises(CrewError, match="NVIDIA_API_KEY"):
        rt.run(REPORT)
    err = [e for e in rt.bus.since(0) if e["kind"] == "error"]
    assert err and err[0]["agent"] == "setup" and "NVIDIA_API_KEY" in err[0]["error"]


def test_no_available_model_is_a_crew_error_not_a_crash(rt, monkeypatch):
    rt.session = rt.client = None
    from pvcrew import llm, pipeline

    def no_model(*a, **k):
        raise llm.ModelUnavailableError("no model answered")
    monkeypatch.setattr(llm, "get_client", lambda: object())
    monkeypatch.setattr(pipeline, "ModelSession", no_model)
    with pytest.raises(CrewError, match="no model answered"):
        rt.run(REPORT)


def test_a_dead_pinned_model_gives_a_failed_result_the_page_can_show(rt):
    rt.session = DeadSession()
    res = rt.run(REPORT)
    assert res["status"] == "failed" and "too slow" in res["error"]


def test_runs_are_serialised(rt):
    import threading
    import time
    order, gate = [], threading.Event()

    class Slow(RoleClient):
        def chat(self, **kw):
            order.append("call")
            gate.wait(0.3)
            return super().chat(**kw)

    rt.client = Slow()
    out = []
    t1 = threading.Thread(target=lambda: out.append(rt.run(REPORT, "a")))
    t2 = threading.Thread(target=lambda: out.append(rt.run(REPORT, "b")))
    t1.start(); time.sleep(0.05); t2.start(); gate.set(); t1.join(10); t2.join(10)
    assert len(out) == 2 and out[0]["run"] != out[1]["run"]
    # no interleaving: all of one run's messages come before the other run's first
    runs = [e["run"] for e in rt.bus.since(0) if e["kind"] == "message"]
    assert runs == sorted(runs, key=runs.index) and runs.count(runs[0]) == 4


# ---------- the monitor on/off switches ----------

def test_monitor_can_be_switched_off(rt, monkeypatch):
    monkeypatch.setattr(config, "MONITOR_ENABLED", False)
    assert rt.start_monitor() is None and rt._monitor is None


def test_monitor_starts_once_and_opens_a_window_only_when_asked(rt, monkeypatch):
    opened = []
    monkeypatch.setattr(config, "MONITOR_ENABLED", True)
    monkeypatch.setattr(config, "MONITOR_AUTO_OPEN", True)
    monkeypatch.setattr(config, "MONITOR_PORT", 0)
    monkeypatch.setattr(service, "open_window", lambda url: opened.append(url))
    monkeypatch.setattr(service, "MonitorServer", lambda bus: __import__("pvcrew.crew.monitor", fromlist=["x"]).MonitorServer(bus, port=0))
    try:
        url = rt.start_monitor()
        assert url.startswith("http://127.0.0.1:") and opened == [url]
        assert rt.start_monitor() == url and opened == [url]  # second call: same server, no second window
    finally:
        rt._monitor.stop()


def test_auto_open_off_still_serves_the_monitor(rt, monkeypatch):
    opened = []
    monkeypatch.setattr(config, "MONITOR_ENABLED", True)
    monkeypatch.setattr(config, "MONITOR_AUTO_OPEN", False)
    monkeypatch.setattr(service, "open_window", lambda url: opened.append(url))
    monkeypatch.setattr(service, "MonitorServer", lambda bus: __import__("pvcrew.crew.monitor", fromlist=["x"]).MonitorServer(bus, port=0))
    try:
        assert rt.start_monitor() and opened == []
    finally:
        rt._monitor.stop()


# ---------- the human reviewer ----------

def _final(result, **case_changes):
    case = dict(result["case"], **case_changes)
    return {"case": case, "coded": [{"reported": c["reported"], "term": c["term"]} for c in result["coded"]],
            "narrative": result["narrative"]}


def test_approving_as_proposed_is_recorded_without_changes(rt, tmp_path):
    res = rt.run(REPORT, "rep-1")
    rec = rt.review(res, "approved", _final(res), reviewer="Dr Rao", comment="ok", path=tmp_path / "rv.jsonl")
    assert rec["decision"] == "approved" and rec["changes"] == [] and rec["reviewer"] == "Dr Rao"
    saved = read_reviews(tmp_path / "rv.jsonl")
    assert len(saved) == 1 and saved[0]["report_id"] == "rep-1" and saved[0]["run"] == res["run"]
    assert saved[0]["final"]["case"]["suspect_drug"] == "Amoxicillin" and saved[0]["comment"] == "ok"
    assert saved[0]["triage"]["priority"] == "HIGH" and saved[0]["verdict"] == "APPROVE"


def test_edits_make_it_approved_with_edits_and_list_every_change(rt, tmp_path):
    res = rt.run(REPORT)
    final = _final(res, patient_age=68, outcome="recovered")
    final["coded"][0]["term"] = "Urticaria"
    final["narrative"] = "Changed by hand."
    rec = rt.review(res, "approved", final, path=tmp_path / "rv.jsonl")
    assert rec["decision"] == "approved_with_edits"
    assert "age 67 → 68" in rec["changes"] and "outcome 'recovering' → 'recovered'" in rec["changes"]
    assert any(c.startswith("term for 'rash'") and "'Urticaria'" in c for c in rec["changes"])
    assert "narrative edited" in rec["changes"]
    assert rec["reviewer"] == "reviewer"  # default when no name is given


def test_rejecting_stays_rejected_even_with_edits(rt, tmp_path):
    res = rt.run(REPORT)
    rec = rt.review(res, "rejected", _final(res, patient_age=1), path=tmp_path / "rv.jsonl")
    assert rec["decision"] == "rejected" and rec["changes"]


def test_unknown_decision_is_refused_and_nothing_is_written(rt, tmp_path):
    res = rt.run(REPORT)
    with pytest.raises(ValueError):
        rt.review(res, "maybe", _final(res), path=tmp_path / "rv.jsonl")
    assert not (tmp_path / "rv.jsonl").exists()


def test_the_decision_appears_in_the_group_chat_and_the_audit_log(rt, tmp_path):
    res = rt.run(REPORT)
    rt.review(res, "approved", _final(res, serious=False), comment="not serious", path=tmp_path / "rv.jsonl")
    ev = [e for e in rt.bus.since(0) if e["kind"] == "review"][0]
    assert ev["decision"] == "approved_with_edits" and "serious True → False" in ev["text"] and "not serious" in ev["text"]
    assert any(r["kind"] == "review" for r in read_audit(path=tmp_path / "audit.jsonl"))


def test_review_defaults_to_the_configured_reviews_file(rt, tmp_path):
    res = rt.run(REPORT)
    rt.review(res, "approved", _final(res))
    assert len(read_reviews()) == 1 and (tmp_path / "reviews.jsonl").exists()


def test_reviews_of_a_failed_run_can_still_be_recorded(rt, tmp_path):
    rt.session = DeadSession()
    res = rt.run(REPORT)
    rec = rt.review(res, "rejected", {"case": None, "coded": [], "narrative": ""}, path=tmp_path / "rv.jsonl")
    assert rec["decision"] == "rejected" and rec["proposed"]["case"] is None


def test_changes_helper_detects_reaction_list_edits_and_ignores_identical_input():
    case = dict(GOOD_CASE)
    proposed = {"case": case, "coded": [{"reported": "rash", "term": "Rash"}], "narrative": NARRATIVE}
    assert _changes(proposed, json.loads(json.dumps(proposed))) == []
    final = {"case": dict(case, reactions=["rash"]), "coded": [{"reported": "rash", "term": "Rash"}], "narrative": NARRATIVE + " "}
    assert _changes(proposed, final) == ["reactions list edited"]  # trailing space in the narrative is not an edit
    added = {"case": dict(case, reactions=["rash", "trouble breathing", "fever"]),
             "coded": [{"reported": "rash", "term": "Rash"}, {"reported": "fever", "term": "Pyrexia"}], "narrative": NARRATIVE}
    assert _changes(proposed, added) == ["reactions list edited"]  # a new reaction is a list edit, not a term change


# ---------- reading back ----------

def test_read_reviews_and_audit_cope_with_missing_files(tmp_path):
    assert read_reviews(tmp_path / "none.jsonl") == [] and read_audit(path=tmp_path / "none.jsonl") == []


def test_read_audit_filters_by_run_and_limits(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("".join(json.dumps({"run": r, "n": i}) + "\n" for i, r in enumerate("aabab")), encoding="utf-8")
    assert [r["n"] for r in read_audit("a", p)] == [0, 1, 3]
    assert [r["n"] for r in read_audit(None, p, limit=2)] == [3, 4]


def test_samples_fall_back_to_bundled_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PAIRS_PATH", tmp_path / "data" / "pairs.jsonl")
    samples = load_samples()
    assert samples and all({"id", "report", "source"} <= s.keys() for s in samples)
    assert {s["source"] for s in samples} == {"bundled sample"}


def test_samples_come_from_every_pairs_file_without_duplicates(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data_hard").mkdir()
    row = lambda i, t: json.dumps({"id": i, "report": t}) + "\n"  # noqa: E731
    (tmp_path / "data" / "pairs.jsonl").write_text(row("1", "easy one") + row("2", "easy two"), encoding="utf-8")
    (tmp_path / "data_hard" / "pairs.jsonl").write_text(row("1", "easy one") + row("3", "hard three"), encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "PAIRS_PATH", tmp_path / "data" / "pairs.jsonl")
    got = load_samples()
    assert [(s["id"], s["source"]) for s in got] == [("1", "data"), ("2", "data"), ("3", "data_hard")]
    assert len(load_samples(limit=2)) == 2


def test_the_process_wide_runtime_is_shared_and_swappable():
    try:
        a = service.get_runtime()
        assert service.get_runtime() is a
        mine = Runtime(session=FakeSession(), client=RoleClient())
        service.set_runtime(mine)
        assert service.get_runtime() is mine
    finally:
        service.set_runtime(None)
