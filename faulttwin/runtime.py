"""Assemble a plant session from config.yaml, drive a scenario to its first event, and run a diagnosis."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable

from faulttwin.config import REPO_ROOT
from faulttwin.llm import SpendLedger, TokenFactory
from faulttwin.monitor import EdgeMonitor, Event, MonitorThresholds
from faulttwin.plant import InverterPlant
from faulttwin.scenarios import Scenario
from faulttwin.tools import Nameplate, PlantSession
from faulttwin.twin import DigitalTwin
from faulttwin.web import TavilyWeb


def nameplate_from_config(cfg: dict[str, Any]) -> Nameplate:
    return Nameplate(**{k: v for k, v in cfg["nameplate"].items() if k in Nameplate.model_fields})


def build_session(cfg: dict[str, Any], *, seed: int | None, llm: TokenFactory | None = None,
                  web: TavilyWeb | None = None, realistic: bool = True) -> PlantSession:
    plant = InverterPlant.from_config(cfg, seed=seed, realistic=realistic)
    monitor = EdgeMonitor(MonitorThresholds.from_config(cfg), plant.params.w1_rad_s)
    twin = DigitalTwin.from_config(cfg)
    llm = llm or TokenFactory(cfg, ledger=SpendLedger.from_config(cfg))
    web = web or TavilyWeb(cfg)
    return PlantSession(cfg, plant, monitor, twin, web, llm, nameplate_from_config(cfg), seed=seed)


TRACE_DIR = REPO_ROOT / "data" / "traces"
PLANT_TICK_S = 0.8          # the app advances the plant this often ...
PLANT_CYCLES_PER_TICK = 2   # ... by this many fundamental cycles


@contextmanager
def plant_clock(session: PlantSession, period_s: float = PLANT_TICK_S, cycles: int = PLANT_CYCLES_PER_TICK):
    """Keep the plant running in real time while a diagnosis runs, as the app does on every UI update.

    Used when recording replays from the command line, so the recorded plant keeps moving while the
    models think, exactly as it does in the app.
    """
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(period_s):
            session.advance(cycles)

    thread = threading.Thread(target=run, name="faulttwin-plant-clock", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def run_diagnosis(session: PlantSession, event: Event, cfg: dict[str, Any], *, trace: Any | None = None,
                  scenario: Scenario | None = None, on_step: Callable[[dict[str, Any]], None] | None = None,
                  wait_for_ultra: bool = False, save: bool = True, after_cycles: int = 8) -> tuple[Any, Any]:
    """Triage, agent and work order for one event. Returns (trace, work_order) and saves the trace."""
    from faulttwin.agent import DiagnosisAgent, Trace   # local import: agent imports this module's siblings
    from faulttwin.report import write_work_order

    t0 = time.perf_counter()
    cost0 = session.llm.spent_usd()
    trace = trace or Trace(scenario=scenario.name if scenario else None)
    if scenario is not None:
        trace.hidden_truth = {"scenario": scenario.name, "truth": scenario.truth.value, "title": scenario.title}
    agent = DiagnosisAgent(session, session.llm, cfg, on_step=on_step)
    agent.run(event, trace)
    session.advance(after_cycles)               # the plant keeps running after the actions (derated, stopped)
    trace.plant_record = list(session.record)
    asset = session.nameplate.inverter_rating or "two-level inverter"
    outage = (trace.stop_reason or "").startswith("model call failed")   # do not wait on the models again
    work_order = write_work_order(session.llm, cfg, trace.final or {}, session, asset, wait_for_ultra=wait_for_ultra,
                                  use_models=not outage)
    trace.work_order = work_order.as_dict()
    trace.totals["llm_cost_usd_incl_report"] = round(session.llm.spent_usd() - cost0, 5)
    trace.totals["wall_s_incl_report"] = round(time.perf_counter() - t0, 2)
    if save:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (TRACE_DIR / f"{stamp}_{trace.scenario or 'live'}.json").write_text(trace.to_json(), encoding="utf-8")
    return trace, work_order


def run_to_event(session: PlantSession, scenario: Scenario, *, healthy_cycles: int = 10,
                 max_cycles: int = 12) -> Event | None:
    """Run healthy, inject the scenario, and return the first monitor event of medium or high severity."""
    with session.lock:
        session.plant.set_modulation_index(scenario.m)
        session.advance(healthy_cycles)
        session.events.clear()
        session.plant.set_condition(scenario.condition)
        for _ in range(max_cycles):
            for status in session.advance(1):
                for event in status.new_events:
                    if event.severity in ("medium", "high"):
                        return event
    return None
