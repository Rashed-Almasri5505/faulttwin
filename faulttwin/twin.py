"""Digital twin: simulate candidate causes and score them against the measurement.

The twin runs the plant equations with nominal parameters. It knows nothing the plant
draws per session (R and L mismatch, noise) and none of the plant's optional
non-idealities; device drops reach it only from a datasheet lookup, via with_params().
It has no noise and no sensor errors, except when the cause being tested is itself a
sensor error or a load change. It shares the plant's time base, so predicted and measured
waveforms are aligned by the commanded-voltage phase.

Match error (lower is better), in percent of the measured |i_s|:
    wave error     RMS of (measured - predicted) currents over the window / |i_s|
    feature error  RMS difference of the normalized features (mu, r_KCL,dc, r_KCL,w1, amplitude ratios - 1)
    tolerance      a penalty for a look-alike whose fitted size is so small that it only soaks up the
                   plant's normal parameter tolerance. It is full below the family's minimum effect and
                   falls linearly to zero at (1 + tolerance_ramp_frac) x minimum effect. Such results are
                   flagged within_tolerance; one can still rank first if it beats "healthy" by more than
                   the penalty, and the ranking report then says so.

The twin only judges steady state: a window that still holds a fault onset or the transient
after a change of m, or a stopped plant, raises WindowNotDiagnosable.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from faulttwin.features import Features, compute_features, park_modulus_A
from faulttwin.hypotheses import SPECS, Hypothesis, HypothesisSpec
from faulttwin.plant import Condition, PlantChunk, PlantParams, measure, simulate

_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_MAX_GOLDEN_STEPS = 100
_ORDER = {h: i for i, h in enumerate(Hypothesis)}


class WindowNotDiagnosable(ValueError):
    """The window cannot be compared with the twin (stopped plant, or not in steady state)."""


class MinEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_offset: float = Field(gt=0)   # A
    sensor_gain: float = Field(gt=0)
    load_unbalance: float = Field(gt=0)
    load_step: float = Field(gt=0)


class TwinSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settle_cycles: int = Field(ge=2)
    window_cycles: int = Field(ge=1)
    fit_grid_points: int = Field(ge=3)
    fit_tolerance: float = Field(gt=0)   # stop the golden-section search when the bracket is this narrow
    min_effect: MinEffect
    within_tolerance_penalty_pct: float = Field(ge=0)
    tolerance_ramp_frac: float = Field(ge=0)
    max_cycle_i_s_change: float = Field(gt=0)
    max_cycle_mu_change: float = Field(gt=0)
    min_i_s_A: float = Field(gt=0)
    max_cache_entries: int = Field(ge=1)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TwinSettings:
        return cls(**cfg["twin"], window_cycles=cfg["features"]["window_cycles"])


@dataclass
class MeasuredWindow:
    """A window of measured currents spanning whole cycles, with its place on the plant time base."""

    t_s: np.ndarray
    i_meas_A: np.ndarray
    step0: int        # plant step index of t_s[0]
    m: float          # modulation index during the window
    steps_per_cycle: int

    @classmethod
    def from_chunk(cls, chunk: PlantChunk, params: PlantParams, cycles: int) -> MeasuredWindow:
        """The last `cycles` cycles of a plant chunk."""
        n = cycles * params.steps_per_cycle
        if len(chunk.t_s) < n:
            raise ValueError(f"chunk has fewer than {cycles} cycles")
        t = chunk.t_s[-n:]
        return cls(t_s=t, i_meas_A=chunk.i_meas_A[-n:], step0=round(t[0] / params.dt_s), m=chunk.m,
                   steps_per_cycle=params.steps_per_cycle)

    @property
    def n_cycles(self) -> int:
        return len(self.t_s) // self.steps_per_cycle

    def cycle_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cycle |i_s| (A) and per-cycle mu (cycles x 3)."""
        n = self.steps_per_cycle
        cycles = [self.i_meas_A[c * n:(c + 1) * n] for c in range(self.n_cycles)]
        i_s = np.array([park_modulus_A(c) for c in cycles])
        mu = np.array([c.mean(axis=0) / max(s, 1e-9) for c, s in zip(cycles, i_s)])
        return i_s, mu


class FittedParam(BaseModel):
    name: str
    value: float
    unit: str
    fitted: bool = True   # False when fitting was switched off and a typical value was assumed


@dataclass
class HypothesisResult:
    hypothesis: Hypothesis
    spec: HypothesisSpec
    match_error_pct: float
    wave_error_pct: float
    feature_error_pct: float
    tolerance_penalty_pct: float
    fitted_param: FittedParam | None
    within_tolerance: bool    # fitted look-alike too small to tell apart from a healthy plant
    predicted_features: Features
    t_s: np.ndarray
    i_pred_A: np.ndarray      # predicted measured currents over the window

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-ready result for the agent and the UI."""
        f = self.predicted_features
        out: dict[str, Any] = {
            "hypothesis": self.hypothesis.value,
            "converter_fault": self.spec.converter_fault,
            "match_error_pct": round(self.match_error_pct, 2),
            "wave_error_pct": round(self.wave_error_pct, 2),
            "feature_error_pct": round(self.feature_error_pct, 2),
            "fitted_param": self.fitted_param.model_dump() if self.fitted_param else None,
            "predicted_key_features": {
                "mu": [round(x, 3) for x in f.mu],
                "r_kcl_dc": round(f.r_kcl_dc, 3),
                "r_kcl_w1": round(f.r_kcl_w1, 3),
                "amp_ratio": [round(x, 3) for x in f.amp_ratio],
            },
        }
        if self.within_tolerance:
            out["note"] = "fitted size is within the plant's normal tolerance; consistent with a healthy plant"
        return out


def feature_vector(f: Features) -> np.ndarray:
    """Normalized features compared between measurement and twin (all dimensionless)."""
    return np.array([*f.mu, f.r_kcl_dc, f.r_kcl_w1, *(r - 1.0 for r in f.amp_ratio)])


def ranking_report(results: Sequence[HypothesisResult], top_n: int = 5) -> dict[str, Any]:
    """The top of a ranking as JSON for the agent, with notes that must not be missed."""
    notes = []
    if results and results[0].within_tolerance:
        notes.append(f"The best match ({results[0].hypothesis.value}) has a fitted size within the plant's normal "
                     "tolerance; treat it as consistent with a healthy plant, not as a confirmed look-alike.")
    return {"ranked": [r.summary() for r in results[:top_n]], "n_tested": len(results), "notes": notes}


def nominal_params(cfg: dict[str, Any]) -> PlantParams:
    """Nominal plant parameters for the twin: no dead time and no device drops (plant truth)."""
    return replace(PlantParams.from_config(cfg["plant"]), dead_time_s=0.0, v_ce_sat_V=0.0, v_f_V=0.0)


class DigitalTwin:
    def __init__(self, params: PlantParams, settings: TwinSettings) -> None:
        self.params = params
        self.settings = settings
        self._cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DigitalTwin:
        return cls(nominal_params(cfg), TwinSettings.from_config(cfg))

    def with_params(self, params: PlantParams) -> DigitalTwin:
        """A twin with updated parameters, e.g. device drops from a datasheet."""
        return DigitalTwin(params, self.settings)

    def window_from_chunk(self, chunk: PlantChunk) -> MeasuredWindow:
        return MeasuredWindow.from_chunk(chunk, self.params, self.settings.window_cycles)

    # --- checks -----------------------------------------------------------------------------

    def check_diagnosable(self, window: MeasuredWindow) -> None:
        """Raise WindowNotDiagnosable unless the window is a running plant in steady state."""
        s = self.settings
        i_s, mu = window.cycle_stats()
        if window.m == 0.0 or i_s.min() < s.min_i_s_A:
            raise WindowNotDiagnosable("the plant is stopped or carries almost no current; nothing to compare")
        i_s_change = float((i_s.max() - i_s.min()) / i_s.max())
        mu_change = float(np.max(mu.max(axis=0) - mu.min(axis=0)))
        if i_s_change > s.max_cycle_i_s_change or mu_change > s.max_cycle_mu_change:
            raise WindowNotDiagnosable(
                f"the window is not in steady state (per-cycle |i_s| changed by {i_s_change:.1%}, mu by "
                f"{mu_change:.3f}); it still holds a fault onset or an operating-point change. Run at least "
                f"{s.window_cycles} more cycles and try again.")

    # --- simulation -------------------------------------------------------------------------

    def true_currents(self, condition: Condition, window: MeasuredWindow) -> np.ndarray:
        """Twin's true currents over the window, after settling from zero current."""
        key = (condition.open_switches, condition.r_factor, window.m, window.step0, len(window.t_s))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        n_window = len(window.t_s)
        n_settle = self.settings.settle_cycles * self.params.steps_per_cycle
        r = tuple(x * f for x, f in zip(self.params.r_ohm, condition.r_factor))
        ts, i, _ = simulate(0.0, n_settle + n_window, (0.0, 0.0, 0.0), self.params, m=window.m,
                            open_switches=condition.open_switches, r_ohm=r, step0=window.step0 - n_settle)
        if abs(ts[n_settle] - window.t_s[0]) > 0.5 * self.params.dt_s:
            raise AssertionError("twin and plant time bases are misaligned")
        self._cache[key] = i[n_settle:].copy()
        while len(self._cache) > self.settings.max_cache_entries:
            self._cache.popitem(last=False)
        return self._cache[key]

    def predict(self, spec: HypothesisSpec, value: float | None, window: MeasuredWindow) -> np.ndarray:
        """Predicted measured currents for a hypothesis (value = its fitted parameter)."""
        condition = spec.condition(value)
        return measure(self.true_currents(condition, window), condition)   # noise-free

    # --- scoring ----------------------------------------------------------------------------

    def score(self, window: MeasuredWindow, measured: Features,
              i_pred_A: np.ndarray) -> tuple[float, float, float, Features]:
        """Raw match error (wave + feature), its two parts, and the predicted features."""
        predicted = compute_features(window.t_s, i_pred_A, self.params.w1_rad_s)
        i_s = max(measured.i_s_A, 1e-9)
        wave = 100.0 * float(np.sqrt(np.mean((window.i_meas_A - i_pred_A) ** 2))) / i_s
        feat = 100.0 * float(np.sqrt(np.mean((feature_vector(measured) - feature_vector(predicted)) ** 2)))
        return wave + feat, wave, feat, predicted

    def tolerance_penalty(self, spec: HypothesisSpec, value: float) -> float:
        """Penalty for a fitted look-alike of this size (0 for sizes well above the minimum effect)."""
        assert spec.fit is not None
        band = getattr(self.settings.min_effect, spec.family)
        size = abs(value - spec.fit.neutral)
        ramp = self.settings.tolerance_ramp_frac * band
        if size < band:
            weight = 1.0
        elif ramp > 0.0:
            weight = max(0.0, 1.0 - (size - band) / ramp)
        else:
            weight = 0.0
        return self.settings.within_tolerance_penalty_pct * weight

    def evaluate(self, hypothesis: Hypothesis, window: MeasuredWindow, measured: Features,
                 fit_params: bool = True) -> HypothesisResult:
        spec = SPECS[hypothesis]
        if spec.fit is None:
            i_pred = self.predict(spec, None, window)
            total, wave, feat, predicted = self.score(window, measured, i_pred)
            return HypothesisResult(hypothesis, spec, total, wave, feat, 0.0, None, False, predicted,
                                    window.t_s, i_pred)
        if fit_params:
            value = self._fit(spec, window, measured)
        else:
            # Without a fit, try the typical value on each side of neutral and keep the better one.
            value = min(spec.fit.defaults, key=lambda x: self.score(window, measured, self.predict(spec, x, window))[0])
        i_pred = self.predict(spec, value, window)
        raw, wave, feat, predicted = self.score(window, measured, i_pred)
        penalty = self.tolerance_penalty(spec, value)
        fitted = FittedParam(name=spec.fit.name, value=value, unit=spec.fit.unit, fitted=fit_params)
        return HypothesisResult(hypothesis, spec, raw + penalty, wave, feat, penalty, fitted, penalty > 0.0,
                                predicted, window.t_s, i_pred)

    def simulate_hypotheses(self, hypotheses: Iterable[Hypothesis | str] | Hypothesis | str, window: MeasuredWindow,
                            fit_params: bool = True) -> list[HypothesisResult]:
        """Simulate each cause on the twin and rank them by match error (best first).

        Ties are broken in favour of causes that are not converter faults, then by the
        order of the Hypothesis enum, never by the order of the input list.
        """
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        unique = list(dict.fromkeys(Hypothesis(h) for h in hypotheses))
        if not unique:
            raise ValueError("no hypotheses to test")
        self.check_diagnosable(window)
        measured = compute_features(window.t_s, window.i_meas_A, self.params.w1_rad_s)
        results = [self.evaluate(h, window, measured, fit_params) for h in unique]
        return sorted(results, key=lambda r: (round(r.match_error_pct, 9), r.spec.converter_fault, _ORDER[r.hypothesis]))

    # --- fitting ----------------------------------------------------------------------------

    def _fit(self, spec: HypothesisSpec, window: MeasuredWindow, measured: Features) -> float:
        """1-D fit of the look-alike parameter: coarse grid, then golden-section refinement."""
        assert spec.fit is not None

        def cost(x: float) -> float:
            return self.score(window, measured, self.predict(spec, x, window))[0]

        grid = np.linspace(spec.fit.lower, spec.fit.upper, self.settings.fit_grid_points)
        costs = [cost(float(x)) for x in grid]
        best = int(np.argmin(costs))
        lo = float(grid[max(best - 1, 0)])
        hi = float(grid[min(best + 1, len(grid) - 1)])
        a, b = lo + (1 - _GOLDEN) * (hi - lo), lo + _GOLDEN * (hi - lo)
        fa, fb = cost(a), cost(b)
        for _ in range(_MAX_GOLDEN_STEPS):
            if hi - lo <= self.settings.fit_tolerance:
                break
            if fa < fb:
                hi, b, fb = b, a, fa
                a = lo + (1 - _GOLDEN) * (hi - lo)
                fa = cost(a)
            else:
                lo, a, fa = a, b, fb
                b = lo + _GOLDEN * (hi - lo)
                fb = cost(b)
        x_best = 0.5 * (lo + hi)
        return x_best if cost(x_best) <= costs[best] else float(grid[best])


def rank_of(results: Sequence[HypothesisResult], hypothesis: Hypothesis | str) -> int:
    """1-based rank of a hypothesis in a ranked result list."""
    for i, r in enumerate(results, start=1):
        if r.hypothesis == hypothesis:
            return i
    raise ValueError(f"{hypothesis} is not in the results")
