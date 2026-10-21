"""The live diagnosis in a background thread, and the guard that decides live or replay mode."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from faulttwin.agent import Trace
from faulttwin.config import get_secret
from faulttwin.llm import BudgetExceeded, SpendLedger
from faulttwin.monitor import Event
from faulttwin.runtime import run_diagnosis
from faulttwin.scenarios import Scenario
from faulttwin.tools import PlantSession

ESTIMATED_RUN_USD = 0.05   # one diagnosis with its work order, rounded up (measured: $0.008-0.025)
ESTIMATED_RUN_CREDITS = 3  # Tavily credits of one diagnosis, rounded up (measured: 1-2)


def outcome(trace: Trace) -> str:
    """done, or budget / failed when the run was stopped by a cap or an outage rather than by the evidence."""
    if trace.budget_exhausted or (trace.stop_reason or "").startswith("budget cap reached") or \
            any("BudgetExceeded" in e for e in trace.service_errors):     # the Tavily credit cap, mid-run
        return "budget"
    if (trace.stop_reason or "").startswith("model call failed"):
        return "failed"
    if (trace.final or {}).get("label") == "inconclusive" and trace.service_errors:
        return "failed"
    return "done"


def friendly_budget_reason(exc: Exception | str) -> str:
    """A visitor-facing sentence for a spend-cap stop; the details stay in the trace."""
    text = str(exc)
    what = "live web search" if "Tavily" in text else "live AI"
    if "daily" in text:
        return f"Today's demo budget for {what} is used up, so recorded diagnoses are shown."
    if "total" in text:
        return f"The demo's total budget for {what} is used up, so recorded diagnoses are shown."
    return "Live AI is paused, so recorded diagnoses are shown."


class LiveRun:
    """Runs triage, the agent and the work order in a thread; the UI reads the trace as it grows."""

    def __init__(self, session: PlantSession, cfg: dict[str, Any], event: Event, scenario: Scenario | None,
                 attempt: int = 0) -> None:
        self.attempt = attempt           # which inject or mystery this run belongs to (scoreboard, refresh)
        self.session = session
        self.cfg = cfg
        self.event = event
        self.scenario = scenario
        self.trace = Trace(scenario=scenario.name if scenario else None)
        if scenario is not None:
            self.trace.hidden_truth = {"scenario": scenario.name, "truth": scenario.truth.value, "title": scenario.title}
        self.work_order: Any | None = None
        self.status = "running"          # running / done / budget / failed
        self.error: str | None = None
        self.started = time.time()
        self._thread = threading.Thread(target=self._work, name="faulttwin-agent", daemon=True)

    def start(self) -> LiveRun:
        self._thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.status == "running"

    def _work(self) -> None:
        try:
            _, self.work_order = run_diagnosis(self.session, self.event, self.cfg, trace=self.trace,
                                               scenario=self.scenario)
            self.status = outcome(self.trace)
        except BudgetExceeded as exc:
            self.status, self.error = "budget", str(exc)
        except Exception as exc:  # noqa: BLE001 - shown to the user as a friendly message, never a stack trace
            self.status, self.error = "failed", f"{type(exc).__name__}: {str(exc)[:200]}"

    def snapshot(self) -> dict[str, Any]:
        """The trace so far, in the same shape as a replay's partial trace."""
        t = self.trace
        done = self.status != "running"
        out = {"scenario": t.scenario, "started_at": t.started_at, "event": t.event, "triage": t.triage,
               "nameplate": t.nameplate, "steps": list(t.steps), "final": t.final, "overlays": t.overlays,
               "actions": t.actions, "sources": t.sources, "totals": t.totals,
               "validation_notes": list(t.validation_notes), "stop_reason": t.stop_reason,
               "hidden_truth": t.hidden_truth if done else None, "finished": done}
        wo = None
        if self.work_order is not None:
            wo = self.work_order.as_dict()   # picks up Ultra's late "senior review" once it has arrived
        out["work_order"] = wo
        return out


@dataclass
class Guard:
    """Decides whether a live diagnosis may run, or the app shows recorded ones instead."""

    ledger: SpendLedger
    runs_per_session: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Guard:
        return cls(SpendLedger.from_config(cfg), int(cfg["budget"]["live_runs_per_session"]))

    @staticmethod
    def configured() -> bool:
        return bool(get_secret("NEBIUS_API_KEY") and get_secret("TAVILY_API_KEY"))

    def live_allowed(self, runs_used: int, api_failed: bool) -> tuple[bool, str]:
        ok, why, _ = self.decide(runs_used, api_failed)
        return ok, why

    def decide(self, runs_used: int, api_failed: bool) -> tuple[bool, str, str]:
        """(live allowed, visitor-facing reason, kind): kind is ok / no_keys / api / session / budget."""
        if not self.configured():
            return False, "Live AI is not configured on this deployment, so recorded diagnoses are shown.", "no_keys"
        if api_failed:
            return False, "The live AI service did not answer, so recorded diagnoses are shown.", "api"
        if runs_used >= self.runs_per_session:
            return False, (f"This visit has used its {self.runs_per_session} live diagnoses, so recorded "
                           "diagnoses are shown."), "session"
        try:
            self.ledger.check(ESTIMATED_RUN_USD)
            self.ledger.check_credits(ESTIMATED_RUN_CREDITS)
        except BudgetExceeded as exc:
            return False, friendly_budget_reason(exc), "budget"
        return True, "", "ok"

    def remaining_usd(self) -> float | None:
        try:
            return self.ledger.remaining_usd()
        except BudgetExceeded:
            return None
