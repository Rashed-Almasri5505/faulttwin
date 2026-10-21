"""What the plant panels show, built either from the live session or from a recorded trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from faulttwin.features import compute_features
from faulttwin.tools import PlantSession

PLOT_CYCLES = 3
SAMPLES_PER_CYCLE = 100   # plots and recordings keep every 10th of the 1000 samples per cycle


@dataclass
class PlantView:
    cycle: int
    m: float
    stopped: bool
    t_s: np.ndarray                      # last PLOT_CYCLES cycles
    i_A: np.ndarray                      # measured phase currents, (n, 3)
    features: dict[str, Any] | None      # compact features of the monitor's 2-cycle window
    active_checks: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)   # newest last
    derated_m: float | None = None


def view_from_session(s: PlantSession, max_events: int = 8) -> PlantView:
    with s.lock:
        chunks = list(s.history)[-PLOT_CYCLES:]
        if chunks:
            step = max(1, len(chunks[0].t_s) // SAMPLES_PER_CYCLE)
            t = np.concatenate([c.t_s[::step] for c in chunks])
            i = np.concatenate([c.i_meas_A[::step] for c in chunks])
        else:
            t, i = np.zeros(0), np.zeros((0, 3))
        status = s.statuses[-1] if s.statuses else None
        features = status.features.compact() if status and status.features else None
        if features is None and len(s.history) >= 2:
            last = list(s.history)[-2:]
            features = compute_features(np.concatenate([c.t_s for c in last]), np.concatenate([c.i_meas_A for c in last]),
                                        s.plant.params.w1_rad_s).compact()
        events = [{"cycle": e.cycle_index, "type": e.type, "severity": e.severity, "message": e.message}
                  for e in s.events[-max_events:]]
        return PlantView(cycle=s.cycle, m=s.plant.m, stopped=s.plant.stopped, t_s=t, i_A=i, features=features,
                         active_checks=[k for k, v in (status.active.items() if status else []) if v],
                         events=events, derated_m=s.derated_m)


def view_from_record(record: list[dict[str, Any]], upto: int, dt_s: float, max_events: int = 8) -> PlantView | None:
    """The plant as it was after the first `upto` recorded cycles."""
    shown = record[:upto]
    if not shown:
        return None
    last = shown[-PLOT_CYCLES:]
    step_s = dt_s * 10
    t = np.concatenate([r["t0_s"] + step_s * np.arange(len(r["i_meas_A"])) for r in last])
    i = np.concatenate([np.asarray(r["i_meas_A"], dtype=float) for r in last])
    events = [dict(e, cycle=r["cycle"]) for r in shown for e in r.get("events", [])][-max_events:]
    head = shown[-1]
    return PlantView(cycle=head["cycle"] + 1, m=head["m"], stopped=head["stopped"], t_s=t, i_A=i,
                     features=head.get("features"), active_checks=head.get("active_checks", []), events=events,
                     derated_m=head.get("derated_m"))
