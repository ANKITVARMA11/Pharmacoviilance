"""The AutoGen workflow. `GraphFlow` is AutoGen's deterministic group chat: the agents are nodes, and edges say who
speaks after whom. An edge can carry a condition on the last message, which is how the Critic can send the case back:

    Intake ──▶ Coding ──▶ Critic ──(VERDICT: REVISE)──▶ Intake      (at most CREW_MAX_REVISIONS times)
                             └─────(otherwise)────────▶ Narrative ──▶ end

AutoGen delivers every message to every participant; the manager only decides who answers next.
"""
import asyncio
import time

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow

from .agents import AGENT_NAMES, Coding, Critic, CrewError, CrewTools, Intake, Narrative, RunState


def route(source: str, text: str, revisions_left_ok: bool = True):
    """Who speaks after `source`, and why. This one function is used by the graph's edge conditions and by the
    monitor's explanations, so what is shown is what AutoGen actually does."""
    if source == "Intake":
        return "Coding", "Intake always hands over to Coding."
    if source == "Coding":
        return "Critic", "Coding always hands over to the Critic."
    if source == "Critic":
        if text.startswith("VERDICT: REVISE"):
            return "Intake", "The Critic said REVISE, so the case goes back to Intake."
        return "Narrative", "The Critic did not ask for a revision, so Narrative goes next."
    return None, "Narrative has no outgoing edge: the workflow is complete."


def build_team(state: RunState, tools: CrewTools):
    intake, coding = Intake(state, tools), Coding(state, tools)
    critic, narrative = Critic(state, tools), Narrative(state, tools)
    b = DiGraphBuilder()
    for a in (intake, coding, critic, narrative):
        b.add_node(a)
    b.add_edge(intake, coding)
    b.add_edge(coding, critic)
    b.add_edge(critic, intake, condition=lambda m: route("Critic", m.to_text())[0] == "Intake")
    b.add_edge(critic, narrative, condition=lambda m: route("Critic", m.to_text())[0] == "Narrative")
    b.set_entry_point(intake)
    turns = 4 * (tools.max_revisions + 1) + 2  # a safety stop; the graph normally ends by itself
    return GraphFlow([intake, coding, critic, narrative], graph=b.build(), max_turns=turns)


def brief(message: str) -> str:
    """AutoGen wraps an agent's failure together with its whole traceback. The reviewer needs the first part only."""
    head = message.split("\nTraceback", 1)[0].strip()
    if head.startswith("CrewError: "):
        head = head[len("CrewError: "):]
    return head or message.strip().splitlines()[0]


async def _run(state: RunState, tools: CrewTools) -> dict:
    bus = tools.bus
    t0 = time.time()
    status, stop_reason, error = "done", "", None
    bus.emit("manager", text="Task delivered to every participant. Entry point: Intake.", next="Intake",
             reason="The graph's entry point is Intake.")
    try:
        team = build_team(state, tools)
        async for item in team.run_stream(task=state.report):
            if isinstance(item, TaskResult):
                stop_reason = item.stop_reason or ""
                continue
            text = item.to_text() if hasattr(item, "to_text") else str(item)
            bus.emit("autogen", type=type(item).__name__, source=getattr(item, "source", ""),
                     preview=" ".join(text.split())[:140])
            if isinstance(item, TextMessage) and item.source in AGENT_NAMES:
                nxt, why = route(item.source, text)
                bus.emit("manager", text=(f"Manager: {item.source} spoke → {nxt} is next." if nxt
                                          else "Manager: nobody is next; the workflow ends."),
                         next=nxt, reason=why)
    except Exception as e:  # noqa: BLE001 - report any failure as a failed run, never as a crash
        detail = str(e) or type(e).__name__
        status, error = "failed", brief(detail)
        bus.emit("error", agent="crew", error=error, detail=detail)  # the audit log keeps the full traceback
    return {"status": status, "error": error, "stop_reason": stop_reason, "seconds": round(time.time() - t0, 1)}


def run_crew(report: str, tools: CrewTools, report_id: str | None = None) -> dict:
    """Run the crew on one report and return everything the reviewer screen needs."""
    state = RunState(report=report, report_id=report_id)
    run = tools.bus.new_run()
    tools.bus.emit("run_start", run=run, report=report, report_id=report_id)
    try:
        info = asyncio.run(_run(state, tools))
    finally:
        tools.bus.end_run()
    tools.bus.emit("run_end", status=info["status"], seconds=info["seconds"], stop_reason=info["stop_reason"],
                   error=info["error"])
    case = state.case
    return {
        "run": run, "report": report, "report_id": report_id, **info,
        "case": case.model_dump() if case else None,
        "coded": state.coded, "verdict": state.verdict, "issues": state.issues, "notes": state.notes,
        "revisions": state.revisions, "attempts": state.attempts, "narrative": state.narrative,
        "triage": state.triage, "extraction_error": state.extraction_error,
    }
