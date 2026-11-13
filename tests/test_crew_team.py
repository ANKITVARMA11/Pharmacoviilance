import json

import pytest

pytest.importorskip("autogen_agentchat")

from pvcrew import config  # noqa: E402
from pvcrew.coding import load_term_counts  # noqa: E402
from pvcrew.crew.agents import AGENT_NAMES, CrewTools, RunState  # noqa: E402
from pvcrew.crew.events import EventBus  # noqa: E402
from pvcrew.crew.team import build_team, route, run_crew  # noqa: E402
from tests.crewfakes import (APPROVE, GOOD_CASE, NARRATIVE, REPORT, REVISE, DeadSession, FakeSession,  # noqa: E402
                             RoleClient)

COUNTS = load_term_counts(config.SAMPLE_TERMS)


def crew(client=None, max_revisions=1, session=None, bus=None):
    bus = bus or EventBus()
    client = client or RoleClient()
    tools = CrewTools(bus=bus, session=session or FakeSession(), client=client, terms=list(COUNTS), counts=COUNTS,
                      max_revisions=max_revisions)
    return bus, client, tools


def kinds(bus, kind, **match):
    return [e for e in bus.since(0) if e["kind"] == kind and all(e.get(k) == v for k, v in match.items())]


# ---------- routing ----------

def test_route_table():
    assert route("Intake", "x")[0] == "Coding" and route("Coding", "x")[0] == "Critic"
    assert route("Critic", "VERDICT: REVISE\n- age")[0] == "Intake"
    assert route("Critic", "VERDICT: APPROVE")[0] == "Narrative"
    assert route("Critic", "VERDICT: ESCALATE")[0] == "Narrative"
    nxt, why = route("Narrative", "anything")
    assert nxt is None and "complete" in why


# ---------- the happy path ----------

def test_the_agents_speak_in_order_and_the_case_comes_out():
    bus, client, tools = crew()
    res = run_crew(REPORT, tools, "r1")
    speakers = [e["agent"] for e in kinds(bus, "message")]
    assert speakers == ["Intake", "Coding", "Critic", "Narrative"]
    assert res["status"] == "done" and res["verdict"] == "APPROVE" and res["revisions"] == 0 and res["attempts"] == 1
    assert res["case"]["patient_age"] == 67 and res["case"]["suspect_drug"] == "Amoxicillin"
    assert [c["term"] for c in res["coded"]] == ["Rash", "Dyspnoea"]
    assert res["narrative"] == NARRATIVE and res["triage"]["priority"] == "HIGH"
    assert res["report_id"] == "r1" and res["run"].startswith("run-")


def test_events_tell_the_whole_story_in_a_sensible_order():
    bus, _, tools = crew()
    run_crew(REPORT, tools)
    seq = [e["kind"] for e in bus.since(0)]
    assert seq[0] == "run_start" and seq[1] == "manager" and seq[-1] == "run_end"
    assert seq.count("called") == 4 and seq.count("typing") == 4 and seq.count("llm") == 4
    for agent in AGENT_NAMES:  # called -> typing -> llm -> message, per agent
        mine = [e["kind"] for e in bus.since(0) if e.get("agent") == agent and e["kind"] != "autogen"]
        assert mine == ["called", "typing", "llm", "message"], agent
    managers = [e["next"] for e in kinds(bus, "manager")]
    assert managers == ["Intake", "Coding", "Critic", "Narrative", None]


def test_the_raw_autogen_stream_is_recorded():
    bus, _, tools = crew()
    run_crew(REPORT, tools)
    raw = kinds(bus, "autogen")
    assert raw[0]["source"] == "user" and raw[0]["type"] == "TextMessage" and REPORT[:20] in raw[0]["preview"]
    assert [r["source"] for r in raw[1:]] == AGENT_NAMES


def test_run_end_event_and_bus_state():
    bus, _, tools = crew()
    res = run_crew(REPORT, tools)
    end = kinds(bus, "run_end")[0]
    assert end["status"] == "done" and end["seconds"] >= 0 and not bus.running
    assert "complete" in (res["stop_reason"] or "").lower()


def test_each_agent_is_called_with_only_the_new_messages_and_remembers_the_rest():
    bus, _, tools = crew()
    run_crew(REPORT, tools)
    called = {e["agent"]: e for e in kinds(bus, "called")}
    assert [r["source"] for r in called["Intake"]["received"]] == ["user"]
    got = {a: [r["source"] for r in called[a]["received"]] for a in AGENT_NAMES}
    assert got == {"Intake": ["user"], "Coding": ["user", "Intake"], "Critic": ["user", "Intake", "Coding"],
                   "Narrative": ["user", "Intake", "Coding", "Critic"]}
    assert [called[a]["total"] for a in AGENT_NAMES] == [1, 2, 3, 4]


# ---------- the revise loop ----------

def test_the_critic_can_send_the_case_back_once_and_intake_sees_the_feedback():
    wrong = dict(GOOD_CASE, patient_age=61)
    client = RoleClient(extract=[json.dumps(wrong), json.dumps(GOOD_CASE)], critic=[REVISE, APPROVE])
    bus, _, tools = crew(client)
    res = run_crew(REPORT, tools)
    assert [e["agent"] for e in kinds(bus, "message")] == ["Intake", "Coding", "Critic", "Intake", "Coding", "Critic", "Narrative"]
    assert res["attempts"] == 2 and res["revisions"] == 1 and res["verdict"] == "APPROVE"
    assert res["case"]["patient_age"] == 67
    second = client.role_calls("extract")[1]["messages"][-1]["content"]
    assert "The age is wrong." in second and "Trouble breathing is missing." in second
    first = client.role_calls("extract")[0]["messages"][-1]["content"]
    assert "reviewer found problems" not in first
    assert any("REVISE" in e["reason"] for e in kinds(bus, "manager", next="Intake")[1:])


def test_intake_remembers_what_it_saw_across_the_loop():
    client = RoleClient(critic=[REVISE, APPROVE])
    bus, _, tools = crew(client)
    run_crew(REPORT, tools)
    intake_calls = kinds(bus, "called", agent="Intake")
    assert [c["total"] for c in intake_calls] == [1, 3]  # report, then Coding + Critic joined its memory
    assert [r["source"] for r in intake_calls[1]["received"]] == ["Coding", "Critic"]


def test_after_the_revision_limit_the_case_is_escalated_not_looped_forever():
    client = RoleClient(critic=REVISE)  # always unhappy
    bus, _, tools = crew(client, max_revisions=1)
    res = run_crew(REPORT, tools)
    assert res["revisions"] == 1 and res["attempts"] == 2 and res["verdict"] == "ESCALATE"
    assert "The age is wrong." in res["issues"] and res["status"] == "done"
    assert [e["agent"] for e in kinds(bus, "message")].count("Intake") == 2
    assert kinds(bus, "message")[-1]["agent"] == "Narrative"


def test_zero_revisions_allowed_means_escalate_straight_away():
    bus, _, tools = crew(RoleClient(critic=REVISE), max_revisions=0)
    res = run_crew(REPORT, tools)
    assert res["attempts"] == 1 and res["verdict"] == "ESCALATE"


def test_two_revisions_allowed():
    client = RoleClient(critic=[REVISE, REVISE, APPROVE])
    _, _, tools = crew(client, max_revisions=2)
    res = run_crew(REPORT, tools)
    assert res["attempts"] == 3 and res["revisions"] == 2 and res["verdict"] == "APPROVE"


# ---------- the Critic's own checks ----------

def test_a_fatal_outcome_not_marked_serious_is_always_flagged_even_if_the_model_approves():
    bad = dict(GOOD_CASE, outcome="fatal", serious=False)
    client = RoleClient(extract=[json.dumps(bad), json.dumps(dict(bad, serious=True))], critic=APPROVE)
    bus, _, tools = crew(client)
    res = run_crew(REPORT, tools)
    assert res["attempts"] == 2  # the hard check forced a revision
    first_verdict = kinds(bus, "message", agent="Critic")[0]["card"]
    assert first_verdict["verdict"] == "REVISE" and "fatal" in first_verdict["issues"][0]
    assert res["case"]["serious"] is True and res["triage"]["priority"] == "URGENT"


def test_uncoded_reactions_are_noted_but_do_not_block_approval():
    odd = dict(GOOD_CASE, reactions=["zzzqqq xyzzy"])
    client = RoleClient(extract=json.dumps(odd), normalize=json.dumps({"items": [{"reported": "zzzqqq xyzzy", "medical_term": "zzz"}]}))
    bus, _, tools = crew(client)
    res = run_crew(REPORT, tools)
    assert res["verdict"] == "APPROVE" and res["coded"][0]["term"] is None
    assert any("Not coded to a standard term: zzzqqq xyzzy" in n for n in res["notes"])
    card = kinds(bus, "message", agent="Critic")[0]["card"]
    assert card["notes"] == res["notes"]


def test_an_unreadable_critic_answer_is_said_out_loud():
    bus, _, tools = crew(RoleClient(critic="not json"))
    res = run_crew(REPORT, tools)
    assert res["verdict"] == "APPROVE" and any("not independently checked" in n for n in res["notes"])


def test_the_critic_card_and_text_agree():
    client = RoleClient(critic=[REVISE, APPROVE])
    bus, _, tools = crew(client)
    run_crew(REPORT, tools)
    first = kinds(bus, "message", agent="Critic")[0]
    assert first["text"].startswith("VERDICT: REVISE") and first["card"]["revision"] == 1
    assert kinds(bus, "message", agent="Critic")[1]["text"].startswith("VERDICT: APPROVE")


# ---------- failures ----------

def test_when_intake_cannot_extract_the_report_goes_to_a_person_without_asking_the_critic_model():
    client = RoleClient(extract="not json at all")
    bus, _, tools = crew(client)
    res = run_crew(REPORT, tools)
    assert res["status"] == "done" and res["case"] is None and res["extraction_error"]
    assert res["verdict"] == "ESCALATE" and res["triage"]["priority"] == "REVIEW" and res["narrative"] == ""
    assert client.role_calls("critic") == [] and client.role_calls("narrative") == []
    assert [e["agent"] for e in kinds(bus, "message")] == ["Intake", "Coding", "Critic", "Narrative"]
    assert kinds(bus, "message", agent="Intake")[0]["card"]["type"] == "error"


def test_a_pinned_model_that_stops_answering_fails_the_run_cleanly():
    bus, _, tools = crew(session=DeadSession())
    res = run_crew(REPORT, tools)
    assert res["status"] == "failed" and "too slow" in res["error"]
    assert "Traceback" not in res["error"] and res["error"].count("\n") <= 1  # the reviewer gets a message, not a stack
    assert kinds(bus, "error", agent="Intake") and kinds(bus, "error", agent="crew")
    assert "Traceback" in kinds(bus, "error", agent="crew")[0]["detail"]  # but the audit trail keeps it
    assert kinds(bus, "run_end")[0]["status"] == "failed" and not bus.running


def test_a_crash_inside_an_agent_is_reported_not_raised():
    class Boom(RoleClient):
        def chat(self, **kw):
            raise ValueError("boom")
    bus, _, tools = crew(Boom())
    res = run_crew(REPORT, tools)
    assert res["status"] == "failed" and "boom" in res["error"] and "Traceback" not in res["error"]


def test_brief_keeps_the_message_and_drops_the_stack():
    from pvcrew.crew.team import brief
    assert brief("CrewError: Intake: model too slow\nSaved.\nTraceback:\nTraceback (most recent call last):\n  x") == "Intake: model too slow\nSaved."
    assert brief("plain message") == "plain message"
    assert brief("\nTraceback (most recent call last): only a stack") == "Traceback (most recent call last): only a stack"
    assert brief("ValueError: boom") == "ValueError: boom"


def test_the_next_run_works_after_a_failed_one():
    bus = EventBus()
    _, _, bad = crew(session=DeadSession(), bus=bus)
    assert run_crew(REPORT, bad)["status"] == "failed"
    _, _, good = crew(bus=bus)
    res = run_crew(REPORT, good)
    assert res["status"] == "done" and bus.state()["runs"] == 2


def test_two_runs_are_kept_apart_in_the_bus():
    bus = EventBus()
    _, _, t1 = crew(bus=bus)
    r1 = run_crew(REPORT, t1)
    _, _, t2 = crew(bus=bus)
    r2 = run_crew(REPORT, t2)
    assert r1["run"] != r2["run"]
    assert {e["run"] for e in bus.since(0)} >= {r1["run"], r2["run"]}
    assert len(kinds(bus, "message", run=r1["run"])) == len(kinds(bus, "message", run=r2["run"])) == 4


# ---------- the graph itself ----------

def test_the_graph_has_the_expected_edges():
    _, _, tools = crew()
    team = build_team(RunState(report="x"), tools)
    graph = team._graph
    edges = {(n, e.target) for n, node in graph.nodes.items() for e in node.edges}
    assert edges == {("Intake", "Coding"), ("Coding", "Critic"), ("Critic", "Intake"), ("Critic", "Narrative")}
    cond = {(n, e.target): e.condition_function for n, node in graph.nodes.items() for e in node.edges
            if e.condition_function is not None}
    assert set(cond) == {("Critic", "Intake"), ("Critic", "Narrative")}
    from autogen_agentchat.messages import TextMessage
    def say(text):
        return TextMessage(content=text, source="Critic")
    # the real conditions, not just their presence: REVISE loops back, anything else moves on
    assert cond[("Critic", "Intake")](say("VERDICT: REVISE\n- age")) is True
    assert cond[("Critic", "Narrative")](say("VERDICT: REVISE\n- age")) is False
    for ok in ("VERDICT: APPROVE", "VERDICT: ESCALATE\n- x"):
        assert cond[("Critic", "Intake")](say(ok)) is False and cond[("Critic", "Narrative")](say(ok)) is True


def test_the_turn_limit_is_a_safety_net_above_the_normal_path():
    _, _, tools = crew(max_revisions=1)
    team = build_team(RunState(report="x"), tools)
    assert team._max_turns >= 4 * 2  # a full run with one revision is 7 turns


def test_agents_forget_their_memory_on_reset():
    import asyncio
    _, _, tools = crew()
    team = build_team(RunState(report="x"), tools)
    agent = team._participants[0]
    agent.seen.append("x")
    asyncio.run(agent.on_reset(None))
    assert agent.seen == []


def test_every_model_call_is_logged_with_agent_purpose_and_model():
    bus, _, tools = crew(session=FakeSession("acme/m-1"))
    run_crew(REPORT, tools)
    llm = kinds(bus, "llm")
    assert [(e["agent"], e["model"]) for e in llm] == [(a, "acme/m-1") for a in AGENT_NAMES]
    assert all(e["ms"] >= 0 and e["purpose"] for e in llm)


def test_the_audit_log_holds_every_agent_input_and_output(tmp_path):
    bus = EventBus(tmp_path / "audit.jsonl")
    _, _, tools = crew(bus=bus)
    run_crew(REPORT, tools)
    rows = [json.loads(l) for l in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["kind"] for r in rows} >= {"run_start", "called", "llm", "message", "manager", "autogen", "run_end"}
    assert [r["agent"] for r in rows if r["kind"] == "message"] == AGENT_NAMES
    assert any(r["kind"] == "run_start" and r["report"] == REPORT for r in rows)  # the input
    assert any(r["kind"] == "message" and r["agent"] == "Narrative" and NARRATIVE in r["text"] for r in rows)  # an output
