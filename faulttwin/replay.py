"""Recorded diagnoses: save them, list them, and play them back step by step in the same UI.

A replay is a saved trace (data/replays/<name>.json) that also holds the plant record: per
cycle, the measured currents, the monitor's features and events. The player interleaves the
plant cycles and the agent steps in their original order and pace, with a minimum time per
item so short bursts (an active test, quick tool calls) stay visible. A file that does not have
everything the UI needs is skipped, so one bad recording never breaks replay mode.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from faulttwin.config import REPO_ROOT

REPLAY_DIR = REPO_ROOT / "data" / "replays"
LEAD_IN_CYCLES = 8          # plant cycles shown before the event
LEAD_IN_TOTAL_S = 2.8       # healthy plant on screen before the event, spread over the lead-in cycles
CYCLE_S = (0.15, 0.6)       # min / max time on screen per plant cycle
STEP_S = (0.6, 2.5)         # min / max time on screen per agent step
RECORD_KEYS = ("cycle", "t0_s", "m", "stopped", "i_meas_A")
SEVERITIES = ("low", "medium", "high")
STEP_KEYS = {"tool": ("tool", "result_summary"), "llm": ("prompt_tokens", "completion_tokens", "latency_s", "cost_usd"),
             "triage": ("result",)}


def _check_features(f: Any) -> None:
    """The compact features the plant panel shows."""
    ok = (isinstance(f, dict) and all(isinstance(f.get(k), (int, float)) for k in ("i_s_A", "r_kcl_dc", "r_kcl_w1"))
          and all(isinstance(f.get(k), list) and len(f[k]) == 3 for k in ("mu", "amp_ratio", "rms_A"))
          and isinstance(f.get("spectrum_A"), dict)
          and all(isinstance(v, list) and len(v) == 3 for v in f["spectrum_A"].values()))
    if not ok:
        raise ValueError("recorded features are incomplete")


def save_replay(trace_json: str, name: str, directory: Path | None = None) -> Path:
    data = json.loads(trace_json)
    if not data.get("plant_record"):
        raise ValueError("this trace has no plant record and cannot be replayed")
    check_replay(data)
    directory = directory or REPLAY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")              # a running app never sees a half-written replay
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    return path


def list_replays(directory: Path | None = None) -> dict[str, Path]:
    directory = directory or REPLAY_DIR
    return {p.stem: p for p in sorted(directory.glob("*.json"))} if directory.exists() else {}


def check_replay(data: Any) -> None:
    """Raise ValueError unless the trace has everything the UI shows during and after a replay."""
    if not isinstance(data, dict):
        raise ValueError("a replay must be a JSON object")
    record = data.get("plant_record")
    if not isinstance(record, list) or not record:
        raise ValueError("no plant record")
    for r in record:
        if not isinstance(r, dict) or any(k not in r for k in RECORD_KEYS):
            raise ValueError(f"a plant record entry lacks one of {RECORD_KEYS}")
        rows = r["i_meas_A"]
        if not isinstance(rows, list) or not rows or any(not isinstance(x, list) or len(x) != 3 for x in rows):
            raise ValueError("measured currents must be rows of three phases")
        if r.get("features") is not None:
            _check_features(r["features"])
        for e in r.get("events", []):
            if not isinstance(e, dict) or e.get("severity") not in SEVERITIES or not isinstance(e.get("message"), str):
                raise ValueError("an event lacks a severity or a message")
    steps = data.get("steps")
    if not isinstance(steps, list) or any(not isinstance(s, dict) or "kind" not in s for s in steps):
        raise ValueError("steps must be a list of step objects")
    for s in steps:
        need = STEP_KEYS.get(s["kind"], ())
        if any(k not in s for k in need):
            raise ValueError(f"a {s['kind']} step lacks one of {need}")
        if s["kind"] == "triage" and not all(k in (s.get("result") or {}) for k in ("family", "urgency")):
            raise ValueError("a triage step lacks its result")
    final = data.get("final")
    if not isinstance(final, dict) or any(k not in final for k in ("label", "confidence", "summary")):
        raise ValueError("the final diagnosis lacks label, confidence or summary")
    if not isinstance(final["confidence"], (int, float)) or not isinstance(final["summary"], str):
        raise ValueError("the final diagnosis has a malformed confidence or summary")
    if not isinstance(data.get("work_order"), dict) or not isinstance(data["work_order"].get("markdown"), str):
        raise ValueError("no work order")
    truth = data.get("hidden_truth")
    if not isinstance(truth, dict) or any(k not in truth for k in ("scenario", "truth", "title")):
        raise ValueError("no scenario truth")
    ReplayPlayer(data, start=0.0)                     # the schedule must build


def load_replay(path: Path) -> dict[str, Any] | None:
    """The recording, or None if it cannot be read or lacks something the UI needs."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        check_replay(data)
    except Exception:  # noqa: BLE001 - unreadable, not JSON, wrong shape: skip this file
        return None
    return data


def _derates(steps: list[dict[str, Any]]) -> list[tuple[float, float | None]]:
    """(wall time, derated m) for each approved derate or resume, from the recorded tool results."""
    out = []
    for s in steps:
        verdict = (s.get("result") or {}).get("interlock") if s.get("tool") == "apply_action" else None
        if verdict and verdict.get("approved") and "fault" in (verdict.get("applied") or {}):
            out.append((float(s.get("t_wall", 0.0)), verdict["applied"].get("m")))
    return out


class ReplayPlayer:
    """Plays a recorded trace against the wall clock."""

    def __init__(self, trace: dict[str, Any], *, speed: float = 1.0, start: float | None = None,
                 attempt: int = 0) -> None:
        self.trace = trace
        self.attempt = attempt
        self.start = time.time() if start is None else start
        record = trace["plant_record"]
        event_cycle = (trace.get("event") or {}).get("cycle", record[0]["cycle"])
        first = next((i for i, r in enumerate(record) if r["cycle"] >= event_cycle - LEAD_IN_CYCLES), 0)
        # Start where the monitor already has features (no "warming up" and no empty spectrum on screen).
        first = max(first, next((i for i, r in enumerate(record) if r.get("features")), first))
        steps = trace.get("steps", [])
        derates = _derates(steps)
        self.record = []
        for r in record[first:]:
            if "derated_m" not in r:        # recordings made before the plant record kept the derate
                applied = [m for t, m in derates if t <= r.get("t_wall", 0.0)]
                r = dict(r, derated_m=applied[-1] if applied else None)
            self.record.append(r)
        items: list[tuple[float, int, str, int]] = []
        for i, r in enumerate(self.record):
            items.append((r.get("t_wall", i * 0.2), 0, "cycle", i))
        for j, s in enumerate(steps):
            items.append((s.get("t_wall", self.record[-1].get("t_wall", 0) + j), 1, "step", j))
        items.sort()
        self.schedule: list[tuple[float, str, int]] = []
        t = 0.0
        prev: float | None = None
        n_lead = sum(1 for r in self.record if r["cycle"] < event_cycle)
        lead_s = min(max(LEAD_IN_TOTAL_S / max(1, n_lead), CYCLE_S[0]), 0.6)
        for t_wall, _, kind, idx in items:
            lo, hi = CYCLE_S if kind == "cycle" else STEP_S
            if kind == "cycle" and self.record[idx]["cycle"] < event_cycle:
                lo = hi = lead_s
            gap = 0.0 if prev is None else t_wall - prev
            t += min(max(gap, lo), hi) / speed
            self.schedule.append((t, kind, idx))
            prev = t_wall
        self.duration_s = t + 1.0
        sims = [j for j, s in enumerate(steps) if s.get("tool") == "simulate_hypotheses"]
        self._last_sim_step = sims[-1] if sims else None
        self._step_times = [t for t, kind, _ in self.schedule if kind == "step"]
        self.event_time = next((t for t, kind, idx in self.schedule
                                if kind == "cycle" and self.record[idx]["cycle"] >= event_cycle), 0.0)

    def step_time(self, n_steps: int) -> float:
        """Seconds into the replay at which the n-th step appeared (the event's time before the first step)."""
        return self._step_times[n_steps - 1] if 0 < n_steps <= len(self._step_times) else self.event_time

    @property
    def recorded_on(self) -> str:
        return str(self.trace.get("started_at") or "")[:10]

    def elapsed(self, now: float | None = None) -> float:
        return (time.time() if now is None else now) - self.start

    def finished(self, now: float | None = None) -> bool:
        return self.elapsed(now) >= self.duration_s

    def counts(self, now: float | None = None) -> tuple[int, int]:
        """(plant cycles shown, agent steps shown) at this moment."""
        e = self.elapsed(now)
        cycles = steps = 0
        for t, kind, idx in self.schedule:
            if t > e:
                break
            if kind == "cycle":
                cycles = idx + 1
            else:
                steps = idx + 1
        return max(cycles, 1), steps

    def partial(self, now: float | None = None) -> dict[str, Any]:
        """The trace as far as it has been played; the result and the truth appear at the end."""
        _, n_steps = self.counts(now)
        done = self.finished(now)
        out = {k: self.trace.get(k) for k in ("scenario", "started_at", "event", "triage", "nameplate", "seed")}
        out["steps"] = self.trace.get("steps", [])[:n_steps]
        for key in ("final", "work_order", "actions", "sources", "ranking", "active_test", "totals",
                    "validation_notes", "hidden_truth", "stop_reason"):
            out[key] = self.trace.get(key) if done else None
        # The overlays come from the last twin ranking: show them once that step has been played.
        shown_sim = self._last_sim_step is not None and self._last_sim_step < n_steps
        out["overlays"] = self.trace.get("overlays") if done or shown_sim else None
        out["finished"] = done
        return out
