"""Assemble a plant session from config.yaml, and drive a scenario up to the first monitor event."""

from __future__ import annotations

from typing import Any

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
