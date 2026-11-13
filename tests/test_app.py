"""The Streamlit page, driven headlessly with a fake model (no browser, no network)."""
import json
from pathlib import Path

import pyarrow as pa
import pytest

pytest.importorskip("autogen_agentchat")
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pvcrew import config  # noqa: E402
from pvcrew.crew import service  # noqa: E402
from pvcrew.crew.agents import CrewError  # noqa: E402
from tests.crewfakes import GOOD_CASE, REPORT, REVISE, DeadSession, FakeSession, RoleClient  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SAMPLES = [{"id": "s1", "report": REPORT, "source": "test"}, {"id": "s2", "report": "Second report text.", "source": "test"}]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REVIEWS_PATH", tmp_path / "reviews.jsonl")
    monkeypatch.setattr(config, "AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")  # where the page looks for model comparisons
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MONITOR_ENABLED", False)
    monkeypatch.setattr(service, "load_samples", lambda limit=200: SAMPLES)
    holder = {}

    def make(client=None, session=None):
        holder["rt"] = service.Runtime(session=session or FakeSession(), client=client or RoleClient(),
                                       terms_path=tmp_path / "none.jsonl", audit_path=tmp_path / "audit.jsonl")
        service.set_runtime(holder["rt"])
        return holder["rt"]
    make()
    yield type("Env", (), {"make": staticmethod(make), "reviews": tmp_path / "reviews.jsonl", "audit": tmp_path / "audit.jsonl"})
    service.set_runtime(None)


def start():
    return AppTest.from_file(APP, default_timeout=30).run()


def button(at, text):
    return next(b for b in at.button if text in b.label)


# NOTE: everything inside the review form (drug, age, terms, tick boxes, extra reactions, comment) only reaches the
# app when the submit button is pressed, so the tests set those values and click Approve/Reject without a .run() between.
def run_crew(at):
    button(at, "Run the crew").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def field(elements, label):
    return next(e for e in elements if e.label == label)


def saved(env):
    return [json.loads(l) for l in env.reviews.read_text(encoding="utf-8").splitlines()]


# ---------- loading ----------

def test_the_page_loads_clean_and_nothing_runs_until_asked(env):
    at = start()
    assert not at.exception and [t.label for t in at.tabs] == ["Review", "Past reviews", "Audit log", "Model comparison"]
    assert any("Monitor is switched off" in c.value for c in at.sidebar.caption)
    assert any("fake/model" in c.value for c in at.sidebar.caption)
    assert env.make and not env.audit.exists()  # opening the page calls no agent
    assert "result" not in at.session_state


def test_the_monitor_url_is_offered_when_it_is_on(env, monkeypatch):
    from pvcrew.crew.monitor import MonitorServer
    rt = env.make()
    monkeypatch.setattr(config, "MONITOR_ENABLED", True)
    monkeypatch.setattr(service, "MonitorServer", lambda bus: MonitorServer(bus, port=0))
    opened = []
    monkeypatch.setattr(service, "open_window", lambda u: opened.append(u))
    try:
        at = start()
        assert not at.exception and rt.monitor_url and opened == [rt.monitor_url]
        assert any(rt.monitor_url in c.value for c in at.sidebar.caption)
        start()  # a second page load must not open a second window
        assert opened == [rt.monitor_url]
    finally:
        rt._monitor.stop()


def test_the_run_button_needs_a_report(env):
    at = start()
    assert button(at, "Run the crew").disabled is False  # a sample is preselected
    at.radio[0].set_value("Paste your own").run()
    assert button(at, "Run the crew").disabled is True
    field(at.text_area, "Report text").set_value("   ").run()
    assert button(at, "Run the crew").disabled is True
    field(at.text_area, "Report text").set_value(REPORT).run()
    assert button(at, "Run the crew").disabled is False


# ---------- running ----------

def test_running_a_sample_shows_the_proposal(env):
    at = run_crew(start())
    res = at.session_state["result"]
    assert res["report_id"] == "s1" and res["status"] == "done"
    assert any("HIGH" in m.value for m in at.markdown) and any("Critic:** APPROVE" in m.value for m in at.markdown)
    assert any(REPORT in i.value for i in at.info)  # the source text sits beside the form
    assert field(at.text_input, "Suspect drug").value == "Amoxicillin"
    assert field(at.number_input, "Age (years)").value == 67
    assert field(at.selectbox, "Sex").value == "female" and field(at.selectbox, "Outcome").value == "recovering"
    assert field(at.checkbox, "Serious").value is True
    assert field(at.text_input, "rash").value == "Rash" and field(at.text_input, "trouble breathing").value == "Dyspnoea"
    assert "Narrative" in [t.label for t in at.text_area]


def test_pasted_reports_are_run_too(env):
    at = start()
    at.radio[0].set_value("Paste your own").run()
    field(at.text_area, "Report text").set_value(REPORT).run()
    run_crew(at)
    assert at.session_state["result"]["report_id"] is None and at.session_state["result"]["case"]["patient_age"] == 67


def test_an_escalated_case_shows_what_the_critic_found(env):
    env.make(RoleClient(critic=REVISE))
    at = run_crew(start())
    warnings = [w.value for w in at.warning]
    assert "The age is wrong." in warnings and "Trouble breathing is missing." in warnings
    assert any("ESCALATE" in m.value and "revisions: 1" in m.value for m in at.markdown)


def test_a_missing_age_is_shown_as_not_stated(env):
    env.make(RoleClient(extract=json.dumps(dict(GOOD_CASE, patient_age=None))))
    at = run_crew(start())
    assert field(at.checkbox, "Age not stated").value is True
    button(at, "Approve").click().run()
    assert saved(env)[0]["final"]["case"]["patient_age"] is None


def test_a_setup_problem_is_shown_in_plain_words(env, monkeypatch):
    rt = env.make()
    monkeypatch.setattr(rt, "run", lambda *a: (_ for _ in ()).throw(CrewError("NVIDIA_API_KEY is not set.")))
    at = start()
    button(at, "Run the crew").click().run()
    assert not at.exception and "NVIDIA_API_KEY is not set." in [e.value for e in at.error]
    assert "result" not in at.session_state
    # the status box is collapsed: a message inside it would be hidden from the reviewer (found with a real browser)
    assert at.status[0].state == "error" and list(at.status[0].error) == []


def test_a_run_that_failed_midway_is_shown_not_hidden(env):
    env.make(session=DeadSession())
    at = run_crew(start())
    assert any("The run failed" in e.value and "too slow" in e.value for e in at.error)
    assert not any("Traceback" in e.value for e in at.error)
    assert at.session_state["result"]["status"] == "failed"


def test_a_new_run_replaces_the_old_result(env):
    at = run_crew(start())
    first = at.session_state["result"]["run"]
    at.selectbox[0].set_value(1).run()
    run_crew(at)
    assert at.session_state["result"]["run"] != first and at.session_state["result"]["report_id"] == "s2"


# ---------- the reviewer's decision ----------

def test_approve_saves_the_record(env):
    at = run_crew(start())
    field(at.text_input, "Your name (for the record)").set_value("Dr Rao").run()
    field(at.text_input, "Comment (optional)").set_value("looks right")
    button(at, "Approve").click().run()
    assert not at.exception
    [rec] = saved(env)
    assert rec["decision"] == "approved" and rec["reviewer"] == "Dr Rao" and rec["comment"] == "looks right"
    assert rec["final"]["case"] == dict(GOOD_CASE, reactions=["rash", "trouble breathing"], outcome="recovering")
    assert [c["term"] for c in rec["final"]["coded"]] == ["Rash", "Dyspnoea"]
    assert any(s.value.startswith("Approved.") for s in at.success)


def test_editing_a_term_makes_it_approved_with_edits(env):
    at = run_crew(start())
    field(at.text_input, "rash").set_value("Urticaria")
    field(at.number_input, "Age (years)").set_value(68)
    button(at, "Approve").click().run()
    [rec] = saved(env)
    assert rec["decision"] == "approved_with_edits" and rec["final"]["case"]["patient_age"] == 68
    assert [c["term"] for c in rec["final"]["coded"]] == ["Urticaria", "Dyspnoea"]
    assert any("Approved with edits" in s.value for s in at.success)
    assert any("Your edits" in c.value and "age 67 → 68" in c.value for c in at.caption)


def test_unticking_a_reaction_removes_it(env):
    at = run_crew(start())
    run = at.session_state["result"]["run"]
    at.checkbox(key=f"keep0-{run}").uncheck()
    button(at, "Approve").click().run()
    [rec] = saved(env)
    assert rec["final"]["case"]["reactions"] == ["trouble breathing"] and rec["decision"] == "approved_with_edits"


def test_missed_reactions_can_be_added_one_per_line(env):
    at = run_crew(start())
    run = at.session_state["result"]["run"]
    at.text_area(key=f"extra-{run}").set_value("fever\n\n  chills  \n")
    button(at, "Approve").click().run()
    [rec] = saved(env)
    assert rec["final"]["case"]["reactions"] == ["rash", "trouble breathing", "fever", "chills"]
    assert rec["final"]["coded"][-2:] == [{"reported": "fever", "term": None}, {"reported": "chills", "term": None}]


def test_a_cleared_term_is_saved_as_no_term(env):
    at = run_crew(start())
    field(at.text_input, "rash").set_value("   ")
    button(at, "Approve").click().run()
    assert saved(env)[0]["final"]["coded"][0] == {"reported": "rash", "term": None}


def test_reject_is_saved_and_reported(env):
    at = run_crew(start())
    button(at, "Reject").click().run()
    [rec] = saved(env)
    assert rec["decision"] == "rejected" and any(s.value.startswith("Rejected.") for s in at.success)


def test_approving_without_a_drug_is_refused(env):
    at = run_crew(start())
    field(at.text_input, "Suspect drug").set_value("  ")
    button(at, "Approve").click().run()
    assert any("suspect drug and at least one reaction" in e.value for e in at.error)
    assert not env.reviews.exists() and not at.success


def test_approving_with_no_reactions_is_refused_but_rejecting_is_allowed(env):
    at = run_crew(start())
    run = at.session_state["result"]["run"]
    at.checkbox(key=f"keep0-{run}").uncheck()
    at.checkbox(key=f"keep1-{run}").uncheck()
    button(at, "Approve").click().run()
    assert any("at least one reaction" in e.value for e in at.error) and not env.reviews.exists()
    button(at, "Reject").click().run()
    assert saved(env)[0]["decision"] == "rejected"


def test_the_saved_message_belongs_to_the_run_on_screen(env):
    at = run_crew(start())
    button(at, "Approve").click().run()
    assert at.success
    run_crew(at)  # a new run: the old "Approved" banner must be gone
    assert not at.success


# ---------- history and audit ----------

def test_past_reviews_and_audit_tabs_fill_in(env):
    at = start()
    assert any("No reviews yet" in i.value for i in at.info)
    assert any("Nothing recorded yet" in i.value for i in at.info)
    run_crew(at)
    button(at, "Approve").click().run()
    assert not at.exception and not any("No reviews yet" in i.value for i in at.info)
    rows = at.dataframe[0].value
    assert list(rows["decision"]) == ["approved"] and list(rows["report"]) == ["s1"] and list(rows["triage"]) == ["HIGH"]
    audit = at.dataframe[1].value
    assert {"run_start", "called", "llm", "message", "manager", "run_end", "review"} <= set(audit["event"])
    assert list(audit[audit["event"] == "message"]["agent"]) == ["Intake", "Coding", "Critic", "Narrative"]


# ---------- model comparison ----------

def write_comparison(folder, models=("vendor/perfect", "other/half"), incomplete=()):
    out = folder / "compare"
    out.mkdir(parents=True)
    rows = [{"model": m, "complete": m not in incomplete, "items": {"value": 1.0 - 0.25 * i, "low": 0.5, "high": 1.0}}
            for i, m in enumerate(models)]
    (out / "compare.json").write_text(json.dumps({"made": "2026-11-06T10:00:00+00:00", "reports": 50, "models": rows}), encoding="utf-8")
    (out / "compare.md").write_text("# Model comparison\n\n## Overall\n\n| # | Model |\n|---|---|\n" +
                                    "".join(f"| {i} | `{m}` |\n" for i, m in enumerate(models, 1)), encoding="utf-8")


def test_comparison_tab_says_how_to_make_one_when_there_is_none(env):
    at = start()
    assert not at.exception
    assert any("python -m pvcrew.compare" in i.value for i in at.info)


def test_comparison_tab_shows_the_table_and_a_chart(env, tmp_path):
    write_comparison(tmp_path / "data")
    at = start()
    assert not at.exception, [e.value for e in at.exception]
    assert not any("python -m pvcrew.compare" in i.value for i in at.info)
    text = " ".join(m.value for m in at.markdown)
    assert "`vendor/perfect`" in text and "`other/half`" in text
    assert any("from 50 reports" in c.value for c in at.caption)
    assert len(at.get("vega_lite_chart")) == 1  # the bar chart of items correct
    assert not any(m.value.startswith("# Model comparison") for m in at.markdown)  # no second big title under the tab
    assert any(m.value.lstrip().startswith("### Overall") or "\n### Overall" in m.value for m in at.markdown)
    assert not any(s.label == "Comparison" for s in at.selectbox)  # one comparison: nothing to pick


def test_comparison_tab_lets_you_pick_between_several(env, tmp_path):
    write_comparison(tmp_path / "data")
    write_comparison(tmp_path / "a_eval", models=("only/one",))
    at = start()
    assert not at.exception
    picker = next(s for s in at.selectbox if s.label == "Comparison")
    assert picker.options == ["data", "a_eval"]  # the data folder first, then the others
    picker.select("a_eval").run()
    text = " ".join(m.value for m in at.markdown)
    assert "`only/one`" in text and "`vendor/perfect`" not in text


def test_a_damaged_comparison_file_does_not_break_the_page(env, tmp_path):
    write_comparison(tmp_path / "data")
    (tmp_path / "data" / "compare" / "compare.json").write_text("{half written", encoding="utf-8")
    at = start()
    assert not at.exception
    assert any("python -m pvcrew.compare" in i.value for i in at.info)


def test_the_chart_is_best_first_with_short_names_and_flags_unfinished_models(env, tmp_path):
    write_comparison(tmp_path / "data", models=("vendor/perfect", "other/half"), incomplete=("other/half",))
    at = start()
    assert not at.exception
    proto = at.get("vega_lite_chart")[0].proto
    spec = json.loads(proto.spec)
    data = pa.ipc.open_stream(proto.data.data).read_all().to_pydict()
    assert list(zip(data["model"], data["finished"])) == [("perfect", "yes"), ("half", "no")]  # publisher dropped, flag kept
    assert data["label"] == ["100.0%", "75.0%"]
    layers = {layer["mark"]["type"]: layer for layer in spec["layer"]}
    assert layers["bar"]["encoding"]["y"]["sort"] == ["perfect", "half"]  # best model on top
    assert layers["bar"]["encoding"]["x"]["scale"]["domain"] == [0, 100]  # a full-width bar must mean 100%, not "the best so far"
