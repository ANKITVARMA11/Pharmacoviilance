"""One place every crew event goes through.

The agents, the AutoGen workflow and the human reviewer all call `bus.emit(kind, **fields)`. The bus
  - keeps the events in memory, so the monitor window can poll them (`since`), and
  - appends each one to the audit log file (JSON lines), so every agent input and output is on record.

Event kinds (the monitor draws each one differently):
  run_start    a report entered the crew                  {run, report, report_id}
  system       something happened around the agents        {text}   (model chosen, model switched, ...)
  manager      the workflow manager routed the conversation {text, next, reason}
  called       AutoGen called an agent with new messages   {agent, received: [{source, preview}], total}
  typing       the agent started working                   {agent}
  llm          one model call                               {agent, purpose, model, ms}
  message      the agent said something in the group chat  {agent, text, card}
  autogen      a raw AutoGen stream object                 {type, source, preview}
  run_end      the workflow finished                        {status, seconds, stop_reason}
  review       the human reviewer decided                   {decision, reviewer, text, card}
  error        something failed                              {agent, error}
"""
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

MAX_EVENTS = 20000


class EventBus:
    def __init__(self, audit_path=None):
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._offset = 0  # how many old events were dropped, so event ids stay stable
        self.epoch = uuid.uuid4().hex[:8]  # changes when the server restarts; the monitor then reloads from zero
        self.audit_path = Path(audit_path) if audit_path else None
        self.run_id = None
        self.running = False
        self.runs = 0

    def new_run(self) -> str:
        with self._lock:
            self.runs += 1
            self.run_id = f"run-{self.runs:03d}-{uuid.uuid4().hex[:4]}"
            self.running = True
            return self.run_id

    def end_run(self) -> None:
        with self._lock:
            self.running = False

    def emit(self, kind: str, **fields) -> dict:
        now = time.time()
        with self._lock:
            ev = {"id": self._offset + len(self._events) + 1, "t": round(now, 3),
                  "clock": datetime.fromtimestamp(now, timezone.utc).astimezone().strftime("%H:%M:%S"),
                  "run": self.run_id, "kind": kind, **fields}
            self._events.append(ev)
            if len(self._events) > MAX_EVENTS:
                drop = len(self._events) - MAX_EVENTS
                del self._events[:drop]
                self._offset += drop
            self._audit(ev)
        return ev

    def _audit(self, ev: dict) -> None:
        if not self.audit_path:
            return
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # never let a full disk or a locked file break a crew run

    def since(self, last_id: int = 0) -> list[dict]:
        with self._lock:
            start = max(0, last_id - self._offset)
            return list(self._events[start:])

    @property
    def last_id(self) -> int:
        with self._lock:
            return self._offset + len(self._events)

    def state(self) -> dict:
        return {"epoch": self.epoch, "last": self.last_id, "running": self.running, "run": self.run_id, "runs": self.runs}
