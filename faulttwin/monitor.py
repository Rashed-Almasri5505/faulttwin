"""Edge monitor and the textbook baseline diagnoser.

The monitor is deterministic and runs once per fundamental cycle on the measured
currents. It raises an Event on the rising edge of each threshold check, then keeps
the check latched until the feature has been back below clear_ratio x threshold for
clear_cycles cycles, so a condition that sits near a threshold does not chatter.
The baseline diagnoser is deliberately naive and is used only in the evaluation.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from faulttwin.features import Features, compute_features, park_modulus_A
from faulttwin.plant import LEGS, PlantChunk

EventType = Literal["mean_current", "kcl_dc", "kcl_w1", "amplitude_ratio", "phase_loss", "no_current", "transient"]
Severity = Literal["low", "medium", "high"]
CHECKS: tuple[EventType, ...] = (
    "mean_current", "kcl_dc", "kcl_w1", "amplitude_ratio", "phase_loss", "no_current", "transient",
)
SEVERITY: dict[EventType, Severity] = {
    "mean_current": "high",
    "phase_loss": "high",
    "no_current": "high",
    "kcl_dc": "medium",
    "kcl_w1": "medium",
    "amplitude_ratio": "medium",
    "transient": "low",
}


class MonitorThresholds(BaseModel):
    """All values come from config.yaml; a missing or misspelt key is an error."""

    model_config = ConfigDict(extra="forbid")

    mu_abs_max: float
    mu_consecutive_cycles: int
    r_kcl_dc_max: float
    r_kcl_w1_max: float
    amp_ratio_band: tuple[float, float]
    low_rms_frac: float
    i_s_step_frac: float
    warmup_cycles: int
    min_i_s_A: float
    clear_ratio: float
    clear_cycles: int
    window_cycles: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> MonitorThresholds:
        return cls(**cfg["monitor"], window_cycles=cfg["features"]["window_cycles"])


class Event(BaseModel):
    type: EventType
    severity: Severity
    cycle_index: int
    message: str
    features: Features | None


class MonitorStatus(BaseModel):
    cycle_index: int
    flags: dict[EventType, bool]      # threshold exceeded in this cycle
    active: dict[EventType, bool]     # latched state (cleared with hysteresis)
    features: Features | None
    new_events: list[Event]
    data_ok: bool = True


class EdgeMonitor:
    """Per-cycle threshold monitor with a small buffer of recent cycles."""

    def __init__(self, thresholds: MonitorThresholds, w1_rad_s: float) -> None:
        self.th = thresholds
        self.w1_rad_s = w1_rad_s
        self.reset()

    def reset(self) -> None:
        self._t: deque[np.ndarray] = deque(maxlen=self.th.window_cycles)
        self._i: deque[np.ndarray] = deque(maxlen=self.th.window_cycles)
        self._seen = 0
        self._mu_streak = 0
        self._prev_i_s_A: float | None = None
        self._hold = 0
        self._active: dict[EventType, bool] = dict.fromkeys(CHECKS, False)
        self._clear_streak: dict[EventType, int] = dict.fromkeys(CHECKS, 0)

    def expect_operating_point_change(self, cycles: int | None = None) -> None:
        """Hold new events while the windows still span a commanded change of m.

        Latched checks stay as they are, so a fault that is already reported does not
        fire again; only the windows after the change are judged.
        """
        self._hold = max(self._hold, cycles if cycles is not None else self.th.window_cycles + 1)

    def process_chunk(self, chunk: PlantChunk) -> list[MonitorStatus]:
        """Feed a plant chunk to the monitor one cycle at a time."""
        n = len(chunk.t_s) // chunk.n_cycles
        return [
            self.update_cycle(chunk.t_s[c * n:(c + 1) * n], chunk.i_meas_A[c * n:(c + 1) * n],
                              chunk.first_cycle + c, commanded_m=chunk.m)
            for c in range(chunk.n_cycles)
        ]

    def update_cycle(self, t_s: np.ndarray, i_meas_A: np.ndarray, cycle_index: int,
                     commanded_m: float | None = None) -> MonitorStatus:
        """Add one fundamental cycle of measured currents and run the checks.

        commanded_m is the modulation index the inverter was given; 0 means stopped.
        Without it, the plant counts as stopped when |i_s| is below min_i_s_A.
        """
        flags: dict[EventType, bool] = dict.fromkeys(CHECKS, False)
        if not (np.isfinite(i_meas_A).all() and np.isfinite(t_s).all()):
            # Bad samples: skip the cycle and keep the latched states. The window buffer is
            # emptied, because a window must be contiguous in time.
            self._t.clear()
            self._i.clear()
            return MonitorStatus(cycle_index=cycle_index, flags=flags, active=dict(self._active),
                                 features=None, new_events=[], data_ok=False)

        self._t.append(t_s)
        self._i.append(i_meas_A)
        self._seen += 1
        th = self.th
        cycle_i_s_A = park_modulus_A(i_meas_A)
        low_current = cycle_i_s_A < th.min_i_s_A
        stopped = commanded_m == 0.0 if commanded_m is not None else low_current
        ready = self._seen > th.warmup_cycles and len(self._t) == th.window_cycles
        features = compute_features(np.concatenate(self._t), np.concatenate(self._i), self.w1_rad_s) if ready else None
        clear: dict[EventType, bool] = dict.fromkeys(CHECKS, True)
        holding = self._hold > 0

        if stopped or features is None:
            self._mu_streak = 0
        elif low_current:
            # Commanded on, but no current flows in any phase.
            flags["no_current"] = True
            clear["no_current"] = False
            self._mu_streak = 0
        else:
            if not holding:
                # "|mu_k| > limit for N consecutive cycles", judged on each cycle on its own.
                cycle_mu = np.abs(i_meas_A.mean(axis=0)) / cycle_i_s_A
                self._mu_streak = self._mu_streak + 1 if cycle_mu.max() > th.mu_abs_max else 0
            mu_max = max(abs(x) for x in features.mu)
            flags["mean_current"] = self._mu_streak >= th.mu_consecutive_cycles
            clear["mean_current"] = mu_max < th.clear_ratio * th.mu_abs_max
            flags["kcl_dc"] = features.r_kcl_dc > th.r_kcl_dc_max
            clear["kcl_dc"] = features.r_kcl_dc < th.clear_ratio * th.r_kcl_dc_max
            flags["kcl_w1"] = features.r_kcl_w1 > th.r_kcl_w1_max
            clear["kcl_w1"] = features.r_kcl_w1 < th.clear_ratio * th.r_kcl_w1_max
            lo, hi = th.amp_ratio_band
            flags["amplitude_ratio"] = any(not lo <= r <= hi for r in features.amp_ratio)
            lo_c, hi_c = 1.0 - th.clear_ratio * (1.0 - lo), 1.0 + th.clear_ratio * (hi - 1.0)
            clear["amplitude_ratio"] = all(lo_c <= r <= hi_c for r in features.amp_ratio)
            flags["phase_loss"] = _phase_loss(features.rms_A, th.low_rms_frac, th.min_i_s_A)
            clear["phase_loss"] = not _phase_loss(features.rms_A, th.low_rms_frac / th.clear_ratio, th.min_i_s_A)
            if self._prev_i_s_A and not holding:
                change = abs(cycle_i_s_A - self._prev_i_s_A) / self._prev_i_s_A
                flags["transient"] = change > th.i_s_step_frac
                clear["transient"] = change < th.clear_ratio * th.i_s_step_frac

        self._prev_i_s_A = None if stopped or low_current else cycle_i_s_A
        new_events: list[Event] = []
        if features is not None and not holding and not stopped:
            for check in CHECKS:
                if flags[check] and not self._active[check]:
                    new_events.append(Event(type=check, severity=SEVERITY[check], cycle_index=cycle_index,
                                            message=_describe(check, features, th, commanded_m), features=features))
                    self._active[check] = True
                    self._clear_streak[check] = 0
                elif self._active[check] and not flags[check] and clear[check]:
                    self._clear_streak[check] += 1
                    if self._clear_streak[check] >= th.clear_cycles:
                        self._active[check] = False
                elif self._active[check]:
                    self._clear_streak[check] = 0
        self._hold = max(0, self._hold - 1)
        return MonitorStatus(cycle_index=cycle_index, flags=flags, active=dict(self._active),
                             features=features, new_events=new_events)


def _phase_loss(rms_A: list[float], frac: float, min_ref_A: float) -> bool:
    for k in range(3):
        others = [rms_A[j] for j in range(3) if j != k]
        ref_A = sum(others) / 2.0
        if ref_A > min_ref_A and rms_A[k] < frac * ref_A:
            return True
    return False


def _describe(check: EventType, f: Features | None, th: MonitorThresholds, commanded_m: float | None) -> str:
    if check == "no_current" or f is None:
        m_text = f" (m = {commanded_m:.2f})" if commanded_m is not None else ""
        return f"no phase current although the inverter is commanded on{m_text}"
    if check == "mean_current":
        k = int(np.argmax(np.abs(f.mu)))
        return f"phase {LEGS[k]}: normalized mean current mu = {f.mu[k]:+.3f} (limit +/-{th.mu_abs_max:.2f})"
    if check == "kcl_dc":
        return f"measured currents do not sum to zero: r_KCL,dc = {f.r_kcl_dc:.3f} (limit {th.r_kcl_dc_max:.2f})"
    if check == "kcl_w1":
        return f"KCL residual at w1: r_KCL,w1 = {f.r_kcl_w1:.3f} (limit {th.r_kcl_w1_max:.2f})"
    if check == "amplitude_ratio":
        ratios = ", ".join(f"{leg} {r:.2f}" for leg, r in zip(LEGS, f.amp_ratio))
        return f"w1 amplitude ratios outside [{th.amp_ratio_band[0]:.2f}, {th.amp_ratio_band[1]:.2f}]: {ratios}"
    if check == "phase_loss":
        k = int(np.argmin(f.rms_A))
        return f"phase {LEGS[k]} RMS {f.rms_A[k]:.2f} A is below {th.low_rms_frac:.0%} of the other phases"
    return f"Park's vector modulus changed by more than {th.i_s_step_frac:.0%} in one cycle"


def baseline_diagnose(features: Features, mu_abs_max: float) -> str:
    """Textbook baseline: the leg with the largest |mu| is faulty; mu < 0 means its upper switch is open."""
    k = int(np.argmax(np.abs(features.mu)))
    mu = features.mu[k]
    if abs(mu) <= mu_abs_max:
        return "healthy"
    return f"{LEGS[k]}{'U' if mu < 0 else 'L'}_open"
