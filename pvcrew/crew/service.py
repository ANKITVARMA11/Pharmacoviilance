"""What the Streamlit page talks to: one process-wide runtime holding the event bus, the monitor server and the
model session, plus the reviewer's decisions."""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..coding import load_term_counts
from ..jsonl import read_jsonl
from .agents import CrewError, CrewTools
from .events import EventBus
from .monitor import MonitorServer, open_window
from .team import run_crew

_lock = threading.Lock()
_runtime = None


class Runtime:
    def __init__(self, bus: EventBus | None = None, session=None, client=None, terms_path=None, audit_path=None):
        self.bus = bus or EventBus(audit_path if audit_path is not None else config.AUDIT_PATH)
        self.session, self.client = session, client
        self.terms_path = Path(terms_path) if terms_path else config.TERMS_PATH
        self.monitor_url = None
        self._monitor = None
        self.terms_note = ""
        self._run_lock = threading.Lock()  # one crew run at a time: the monitor shows a single conversation

    # ---- monitor window ----
    def start_monitor(self, open_browser: bool = True) -> str | None:
        if self._monitor is None and config.MONITOR_ENABLED:
            self._monitor = MonitorServer(self.bus)
            self.monitor_url = self._monitor.start()
            if open_browser and config.MONITOR_AUTO_OPEN:
                open_window(self.monitor_url)
        return self.monitor_url

    # ---- model + vocabulary ----
    def ensure_ready(self) -> None:
        if self.session is None:
            from ..llm import ModelUnavailableError, get_client
            from ..pipeline import ModelSession

            say = lambda *a: self.bus.emit("system", text=" ".join(str(x) for x in a).strip())  # noqa: E731
            try:
                self.client = self.client or get_client()
                self.session = ModelSession(self.client, say=say)
            except SystemExit as e:
                raise CrewError(str(e)) from e
            except ModelUnavailableError as e:
                raise CrewError(str(e)) from e
        if not hasattr(self, "_terms"):
            path = self.terms_path if self.terms_path.exists() else config.SAMPLE_TERMS
            self._counts = load_term_counts(path)
            self._terms = list(self._counts)
            self.terms_note = ("" if path == self.terms_path else
                               "Using the small demo vocabulary. Run `python -m pvcrew.pipeline terms` for the full one.")
            self.bus.emit("system", text=self.terms_note or f"Vocabulary: {len(self._terms)} standard terms.")

    # ---- running ----
    def run(self, report: str, report_id: str | None = None) -> dict:
        report = (report or "").strip()
        if not report:
            raise CrewError("The report is empty.")
        with self._run_lock:
            return self._run(report, report_id)

    def _run(self, report: str, report_id) -> dict:
        try:
            self.ensure_ready()
        except CrewError as e:
            self.bus.emit("error", agent="setup", error=str(e))
            raise
        tools = CrewTools(bus=self.bus, session=self.session, client=self.client, terms=self._terms,
                          counts=self._counts, max_revisions=config.CREW_MAX_REVISIONS)
        return run_crew(report, tools, report_id)

    # ---- the human reviewer ----
    def review(self, result: dict, decision: str, final: dict, reviewer: str = "", comment: str = "",
               path=None) -> dict:
        """Save the reviewer's decision. `final` holds the (possibly edited) case, coded terms and narrative."""
        if decision not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        proposed = {"case": result.get("case"), "coded": [{"reported": c["reported"], "term": c["term"]} for c in result.get("coded", [])],
                    "narrative": result.get("narrative", "")}
        changes = _changes(proposed, final)
        label = decision if not (decision == "approved" and changes) else "approved_with_edits"
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run": result.get("run"),
                  "report_id": result.get("report_id"), "decision": label, "reviewer": reviewer or "reviewer",
                  "comment": comment, "changes": changes, "proposed": proposed, "final": final,
                  "triage": result.get("triage"), "verdict": result.get("verdict")}
        target = Path(path) if path else config.REVIEWS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        text = {"approved": "Approved.", "approved_with_edits": "Approved with edits: " + "; ".join(changes),
                "rejected": "Rejected."}[label]
        if comment:
            text += f" “{comment}”"
        self.bus.emit("review", decision=label, reviewer=record["reviewer"], text=text,
                      card={"type": "review", "decision": label, "changes": changes})
        return record


def _changes(proposed: dict, final: dict) -> list[str]:
    out = []
    pc, fc = proposed.get("case") or {}, final.get("case") or {}
    for k in ("patient_age", "patient_sex", "suspect_drug", "serious", "outcome"):
        if pc.get(k) != fc.get(k):
            out.append(f"{k.replace('patient_', '')} {pc.get(k)!r} → {fc.get(k)!r}")
    old = {c["reported"]: c["term"] for c in proposed.get("coded", [])}
    for c in final.get("coded", []):
        if old.get(c["reported"], c["term"]) != c["term"]:
            out.append(f"term for {c['reported']!r}: {old.get(c['reported'])!r} → {c['term']!r}")
    if [r for r in pc.get("reactions", [])] != [r for r in fc.get("reactions", [])]:
        out.append("reactions list edited")
    if (proposed.get("narrative") or "").strip() != (final.get("narrative") or "").strip():
        out.append("narrative edited")
    return out


def get_runtime() -> Runtime:
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = Runtime()
        return _runtime


def set_runtime(rt: Runtime | None) -> None:
    """For tests: swap in a runtime with a fake model."""
    global _runtime
    with _lock:
        _runtime = rt


def read_reviews(path=None) -> list[dict]:
    p = Path(path) if path else config.REVIEWS_PATH
    return list(read_jsonl(p)) if p.exists() else []


def load_samples(limit: int = 200) -> list[dict]:
    """Reports to pick from in the page: any pairs.jsonl found under the project (data/, data_hard/, ...);
    if there is none, template reports made from the bundled FAERS sample. Each: {id, report, source}."""
    seen, out = set(), []
    files = sorted({p for p in [config.PAIRS_PATH, *config.ROOT.glob("*/pairs.jsonl"), *config.DATA_DIR.glob("*/pairs.jsonl")]
                    if Path(p).exists()}, key=lambda p: (str(p) != str(config.PAIRS_PATH), str(p)))
    for p in files:
        for row in read_jsonl(p):
            key = (row["id"], row["report"])
            if key not in seen:
                seen.add(key)
                out.append({"id": row["id"], "report": row["report"], "source": Path(p).parent.name})
    if not out:
        from ..faers import load_records
        from ..synth import build_pairs
        for row in build_pairs(load_records(config.SAMPLE_FAERS), client=None, model=None, seed=0):
            out.append({"id": row["id"], "report": row["report"], "source": "bundled sample"})
    return out[:limit]


def read_comparisons() -> dict:
    """Model comparisons made by `python -m pvcrew.compare`, found under the data folder and the project folders.
    {folder name: (results dict from compare.json, text of compare.md)}"""
    paths = {Path(p) for p in [config.DATA_DIR / "compare" / "compare.json", *config.ROOT.glob("*/compare/compare.json")]
             if Path(p).exists()}
    out = {}
    for p in sorted(paths, key=lambda p: (p.parent.parent != config.DATA_DIR, str(p))):
        text = p.with_name("compare.md")
        try:
            out[p.parent.parent.name] = (json.loads(p.read_text(encoding="utf-8")),
                                         text.read_text(encoding="utf-8") if text.exists() else "")
        except (OSError, ValueError):
            continue  # half-written or damaged: skip it rather than break the page
    return out


def read_audit(run: str | None = None, path=None, limit: int = 2000) -> list[dict]:
    p = Path(path) if path else config.AUDIT_PATH
    if not p.exists():
        return []
    rows = [r for r in read_jsonl(p) if run is None or r.get("run") == run]
    return rows[-limit:]
