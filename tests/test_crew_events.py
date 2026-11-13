import json
import threading

from pvcrew.crew.events import EventBus
from pvcrew.crew.triage import triage
from pvcrew.schemas import Case


def test_events_get_increasing_ids_and_come_back_with_since():
    bus = EventBus()
    a, b, c = bus.emit("system", text="a"), bus.emit("system", text="b"), bus.emit("system", text="c")
    assert [a["id"], b["id"], c["id"]] == [1, 2, 3]
    assert [e["text"] for e in bus.since(0)] == ["a", "b", "c"]
    assert [e["text"] for e in bus.since(2)] == ["c"] and bus.since(3) == [] and bus.last_id == 3


def test_events_carry_the_run_and_a_clock():
    bus = EventBus()
    assert bus.emit("system", text="x")["run"] is None
    run = bus.new_run()
    e = bus.emit("message", agent="Intake", text="hi")
    assert e["run"] == run and e["kind"] == "message" and len(e["clock"]) == 8 and bus.running
    bus.end_run()
    assert not bus.running and bus.state()["runs"] == 1 and bus.state()["run"] == run


def test_runs_get_distinct_ids():
    bus = EventBus()
    assert bus.new_run() != bus.new_run()


def test_every_event_is_appended_to_the_audit_file(tmp_path):
    path = tmp_path / "sub" / "audit.jsonl"
    bus = EventBus(path)
    bus.new_run()
    bus.emit("called", agent="Critic", received=[{"source": "Coding", "preview": "x"}], total=3)
    bus.emit("message", agent="Critic", text="héllo — ünïcode")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert [r["kind"] for r in rows] == ["called", "message"] and rows[1]["text"] == "héllo — ünïcode"
    assert rows[0]["received"][0]["source"] == "Coding" and rows[0]["run"] == bus.run_id


def test_an_unwritable_audit_file_never_breaks_a_run(tmp_path):
    bus = EventBus(tmp_path)  # a directory: opening it for append fails
    assert bus.emit("system", text="still works")["text"] == "still works"


def test_old_events_are_dropped_but_ids_stay_stable(monkeypatch):
    from pvcrew.crew import events
    monkeypatch.setattr(events, "MAX_EVENTS", 5)
    bus = EventBus()
    for i in range(12):
        bus.emit("system", text=str(i))
    assert bus.last_id == 12 and [e["text"] for e in bus.since(0)] == ["7", "8", "9", "10", "11"]
    assert [e["text"] for e in bus.since(10)] == ["10", "11"] and bus.since(12) == []


def test_the_epoch_differs_between_buses():
    assert EventBus().epoch != EventBus().epoch


def test_emit_is_thread_safe():
    bus = EventBus()
    threads = [threading.Thread(target=lambda: [bus.emit("system", text="x") for _ in range(200)]) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    ids = [e["id"] for e in bus.since(0)]
    assert ids == list(range(1, 1601))


def case(**kw):
    base = dict(patient_age=40, patient_sex="male", suspect_drug="X", reactions=["a"], serious=False, outcome="recovered")
    return Case(**{**base, **kw})


def test_triage_levels():
    assert triage(case(outcome="fatal", serious=True))["priority"] == "URGENT"
    assert triage(case(outcome="fatal", serious=False))["priority"] == "URGENT"
    assert triage(case(serious=True))["priority"] == "HIGH"
    assert triage(case())["priority"] == "ROUTINE"


def test_triage_lists_what_is_missing():
    t = triage(case(patient_age=None, patient_sex="unknown", outcome="unknown"))
    assert t["follow_up"] == ["age not stated", "sex not stated", "outcome not stated"]
    assert triage(case())["follow_up"] == []
