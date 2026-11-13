"""The four crew members. Each is an AutoGen `BaseChatAgent`: AutoGen calls `on_messages(new_messages)` when it is
that agent's turn, and the agent answers with a `TextMessage` that AutoGen shows to the whole group.

Model calls go through the project's own client and `ModelSession` (pacing, retries, switching to another model
if one stalls), not AutoGen's built-in OpenAI client. The blocking calls run in a worker thread so AutoGen's event
loop stays free.

  Intake     reads the report and fills in the case fields
  Coding     turns the reaction wording into standard terms
  Critic     compares the case with the report: APPROVE, or REVISE with concrete problems
  Narrative  writes the short case narrative and the triage recommendation
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from pydantic import BaseModel

from ..coding import ask_json, code_reactions
from ..extract import ExtractionError, extract_case
from ..schemas import Case
from .events import EventBus
from .triage import triage

AGENT_NAMES = ["Intake", "Coding", "Critic", "Narrative"]


class CrewError(RuntimeError):
    """The crew cannot continue (for example a pinned model stopped answering)."""


@dataclass
class RunState:
    """What the agents share besides the chat text: the structured results of each step."""
    report: str
    report_id: str | None = None
    case: Case | None = None
    coded: list[dict] = field(default_factory=list)
    verdict: str = ""  # APPROVE | REVISE | ESCALATE
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    revisions: int = 0
    attempts: int = 0
    extraction_error: str | None = None
    narrative: str = ""
    triage: dict | None = None


@dataclass
class CrewTools:
    bus: EventBus
    session: Any  # pipeline.ModelSession
    client: Any
    terms: list[str]
    counts: dict
    max_revisions: int = 1


def _preview(msg, n=90) -> str:
    text = " ".join(msg.to_text().split())
    return text if len(text) <= n else text[: n - 1] + "…"


class CrewAgent(BaseChatAgent):
    def __init__(self, name: str, description: str, state: RunState, tools: CrewTools):
        super().__init__(name, description)
        self.state, self.tools, self.seen = state, tools, []

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token) -> Response:
        bus = self.tools.bus
        self.seen.extend(messages)  # an agent keeps its own memory of everything it was shown
        bus.emit("called", agent=self.name, total=len(self.seen),
                 received=[{"source": m.source, "preview": _preview(m)} for m in messages])
        bus.emit("typing", agent=self.name)
        try:
            text, card = await asyncio.to_thread(self.work)
        except BaseException as e:  # noqa: BLE001 - SystemExit from a pinned model must not leak into AutoGen
            bus.emit("error", agent=self.name, error=str(e) or type(e).__name__)
            raise CrewError(f"{self.name}: {e}") from e
        bus.emit("message", agent=self.name, text=text, card=card)
        return Response(chat_message=TextMessage(content=text, source=self.name))

    async def on_reset(self, cancellation_token) -> None:
        self.seen = []

    def work(self) -> tuple[str, dict | None]:
        raise NotImplementedError

    def llm(self, purpose: str, fn):
        """Run fn(model) with retry/failover, and put a line in the log saying what it cost."""
        t0 = time.time()
        try:
            result = self.tools.session.call(fn)
        except SystemExit as e:
            raise CrewError(str(e)) from e
        self.tools.bus.emit("llm", agent=self.name, purpose=purpose, model=self.tools.session.model,
                            ms=int((time.time() - t0) * 1000))
        return result


# ---------- Intake ----------

class Intake(CrewAgent):
    def __init__(self, state, tools):
        super().__init__("Intake", "Reads the report and fills in the case fields.", state, tools)

    def work(self):
        st = self.state
        st.attempts += 1
        feedback = "\n".join(f"- {i}" for i in st.issues) if st.verdict == "REVISE" else None
        try:
            case = self.llm("extract the case fields",
                            lambda m: extract_case(st.report, self.tools.client, m, feedback=feedback))
        except ExtractionError as e:
            st.case, st.extraction_error = None, str(e)
            return (f"I could not get valid fields out of this report ({st.extraction_error[:120]}).",
                    {"type": "error", "text": "No valid case could be extracted."})
        st.case, st.extraction_error = case, None
        lead = "Here is the case." if feedback is None else f"Revised after the Critic's feedback (attempt {st.attempts})."
        return f"{lead}\n{_case_line(case)}", {"type": "case", "attempt": st.attempts, "fields": case.model_dump()}


def _case_line(case: Case) -> str:
    age = "age not stated" if case.patient_age is None else f"{case.patient_age} years"
    return (f"{age}, {case.patient_sex}; drug: {case.suspect_drug}; reactions: {', '.join(case.reactions)}; "
            f"serious: {'yes' if case.serious else 'no'}; outcome: {case.outcome.replace('_', ' ')}.")


# ---------- Coding ----------

class Coding(CrewAgent):
    def __init__(self, state, tools):
        super().__init__("Coding", "Maps reaction wording to standard terms.", state, tools)

    def work(self):
        st, t = self.state, self.tools
        if st.case is None:
            st.coded = []
            return "Nothing to code: Intake has no valid case.", {"type": "coding", "rows": []}
        st.coded = self.llm("code the reactions",
                            lambda m: code_reactions(st.case.reactions, t.client, m, t.terms, t.counts))
        rows = [{"reported": c["reported"], "term": c["term"], "method": c["method"], "note": c.get("note", "")}
                for c in st.coded]
        lines = [f"{r['reported']} → {r['term']}" if r["term"] else f"{r['reported']} → (no standard term found)"
                 for r in rows]
        done = sum(1 for r in rows if r["term"])
        return f"Coded {done} of {len(rows)} reactions.\n" + "\n".join(lines), {"type": "coding", "rows": rows}


# ---------- Critic ----------

class _Verdict(BaseModel):
    verdict: Literal["APPROVE", "REVISE"]
    issues: list[str] = []


CRITIC_SYSTEM = (
    "You are a pharmacovigilance quality reviewer. Compare the extracted case with the source report. "
    "List only concrete problems: a value that contradicts the report, a fact the report states that the case "
    "misses (age, sex, drug, a reaction, seriousness, outcome), or a value the report does not support. "
    "Informal reaction wording is fine. Do not nitpick style. "
    "If the case matches the report, answer APPROVE with an empty issue list. Reply with JSON only."
)


class Critic(CrewAgent):
    def __init__(self, state, tools):
        super().__init__("Critic", "Checks the case against the report.", state, tools)

    def work(self):
        st, t = self.state, self.tools
        if st.case is None:
            st.verdict, st.issues, st.notes = "ESCALATE", ["Intake could not produce a valid case."], []
            return self._say("Nothing to check. A person needs to read this report.")
        case = st.case
        issues = []
        if case.outcome == "fatal" and not case.serious:
            issues.append("The outcome is fatal but the case is not marked serious.")
        user = (f"Report:\n{st.report}\n\nExtracted case:\n{json.dumps(case.model_dump(), ensure_ascii=False)}\n\n"
                f"Reactions coded to standard terms:\n"
                + "\n".join(f"- {c['reported']} -> {c['term'] or 'no term found'}" for c in st.coded))
        reply = self.llm("check the case against the report",
                         lambda m: ask_json(t.client, m, CRITIC_SYSTEM, user, _Verdict))
        if reply is None:
            st.notes = ["The Critic's answer could not be read, so this case is not independently checked."]
        else:
            st.notes = []
            issues += [i.strip() for i in reply.issues if i.strip()]
        uncoded = [c["reported"] for c in st.coded if not c["term"]]
        if uncoded:
            st.notes.append("Not coded to a standard term: " + ", ".join(uncoded) + ".")
        if issues and st.revisions < t.max_revisions:
            st.revisions += 1
            st.verdict, st.issues = "REVISE", issues
            return self._say(f"Sending the case back to Intake (revision {st.revisions} of {t.max_revisions}).")
        st.verdict, st.issues = ("ESCALATE", issues) if issues else ("APPROVE", [])
        return self._say("Passing to Narrative; a person should look at the open points." if issues else
                         "The case matches the report. Passing to Narrative.")

    def _say(self, closing: str):
        st = self.state
        lines = [f"VERDICT: {st.verdict}"] + [f"- {i}" for i in st.issues] + [f"Note: {n}" for n in st.notes] + [closing]
        card = {"type": "verdict", "verdict": st.verdict, "issues": st.issues, "notes": st.notes,
                "revision": st.revisions, "max": self.tools.max_revisions}
        return "\n".join(lines), card


# ---------- Narrative ----------

NARRATIVE_SYSTEM = (
    "You write the narrative paragraph of a pharmacovigilance case file. Three to four plain sentences, neutral "
    "tone, past tense. Use only the facts given. Do not add causes, diagnoses, doses or dates that are not given. "
    "If a fact is not stated, say it was not stated. Reply with the paragraph only."
)


class Narrative(CrewAgent):
    def __init__(self, state, tools):
        super().__init__("Narrative", "Writes the case narrative and the triage recommendation.", state, tools)

    def work(self):
        st, t = self.state, self.tools
        if st.case is None:
            st.triage = {"priority": "REVIEW", "why": "Automatic extraction failed: a person must read the report.",
                         "follow_up": []}
            st.narrative = ""
            return ("No narrative: the report could not be turned into a case.\nTriage: REVIEW. "
                    + st.triage["why"]), {"type": "narrative", "text": "", "triage": st.triage}
        st.triage = triage(st.case)
        facts = {"case": st.case.model_dump(), "standard_terms": [c["term"] or c["reported"] for c in st.coded],
                 "open_points": st.issues + st.notes}
        messages = [{"role": "system", "content": NARRATIVE_SYSTEM},
                    {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]
        st.narrative = self.llm("write the narrative", lambda m: t.client.chat(
            model=m, messages=messages, options={"temperature": 0, "timeout": 90})["message"]["content"].strip())
        follow = f" Follow-up needed: {', '.join(st.triage['follow_up'])}." if st.triage["follow_up"] else ""
        text = f"{st.narrative}\n\nTriage: {st.triage['priority']}. {st.triage['why']}{follow}"
        return text, {"type": "narrative", "text": st.narrative, "triage": st.triage}
