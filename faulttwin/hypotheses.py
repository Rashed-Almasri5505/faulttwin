"""Possible causes of a monitor event, and how each one maps onto the plant model.

A hypothesis is either a converter fault (open switches) or a look-alike (sensor
error, load change). Look-alikes have one parameter that the digital twin fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import Literal

from faulttwin.plant import LEGS, Condition

Family = Literal[
    "healthy", "switch_open", "leg_open", "two_switch_open",
    "sensor_offset", "sensor_gain", "load_unbalance", "load_step",
]

class Hypothesis(StrEnum):
    healthy = "healthy"
    # one open switch (textbook names: AU=S1, AL=S4, BU=S3, BL=S6, CU=S5, CL=S2)
    AU_open = "AU_open"
    AL_open = "AL_open"
    BU_open = "BU_open"
    BL_open = "BL_open"
    CU_open = "CU_open"
    CL_open = "CL_open"
    # both switches of one leg open
    leg_A_open = "leg_A_open"
    leg_B_open = "leg_B_open"
    leg_C_open = "leg_C_open"
    # two switches open in different legs
    AU_BU_open = "AU_BU_open"
    AU_BL_open = "AU_BL_open"
    AL_BU_open = "AL_BU_open"
    AL_BL_open = "AL_BL_open"
    AU_CU_open = "AU_CU_open"
    AU_CL_open = "AU_CL_open"
    AL_CU_open = "AL_CU_open"
    AL_CL_open = "AL_CL_open"
    BU_CU_open = "BU_CU_open"
    BU_CL_open = "BU_CL_open"
    BL_CU_open = "BL_CU_open"
    BL_CL_open = "BL_CL_open"
    # look-alikes: the converter is healthy
    sensor_offset_A = "sensor_offset_A"
    sensor_offset_B = "sensor_offset_B"
    sensor_offset_C = "sensor_offset_C"
    sensor_gain_A = "sensor_gain_A"
    sensor_gain_B = "sensor_gain_B"
    sensor_gain_C = "sensor_gain_C"
    load_unbalance_A = "load_unbalance_A"
    load_unbalance_B = "load_unbalance_B"
    load_unbalance_C = "load_unbalance_C"
    load_step = "load_step"


_PAIRS = [(a, b) for i, a in enumerate(LEGS) for b in LEGS[i + 1:]]
_TWO_SWITCH = [f"{a}{sa}_{b}{sb}_open" for a, b in _PAIRS for sa, sb in product("UL", "UL")]


@dataclass(frozen=True)
class FitSpec:
    """One fitted parameter of a look-alike hypothesis."""

    name: str
    unit: str
    lower: float
    upper: float
    defaults: tuple[float, float]   # typical values on both sides of neutral, tried when fitting is off
    neutral: float                  # value at which the look-alike has no effect

    @property
    def default(self) -> float:
        return self.defaults[0]


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis: Hypothesis
    family: Family
    converter_fault: bool
    description: str
    leg: int | None = None
    switches: frozenset[str] = frozenset()
    fit: FitSpec | None = None

    def condition(self, value: float | None = None) -> Condition:
        """Plant condition this hypothesis describes; value is the fitted parameter."""
        if self.switches:
            return Condition(open_switches=self.switches)
        if self.fit is None:
            return Condition()
        x = self.fit.default if value is None else value
        if self.family == "load_step":
            return Condition(r_factor=(x, x, x))
        per_phase = [self.fit.neutral] * 3
        per_phase[self.leg] = x  # type: ignore[index]
        triple = (per_phase[0], per_phase[1], per_phase[2])
        if self.family == "sensor_offset":
            return Condition(sensor_offset_A=triple)
        if self.family == "sensor_gain":
            return Condition(sensor_gain=triple)
        return Condition(r_factor=triple)


OFFSET_FIT = FitSpec("offset", "A", -6.0, 6.0, defaults=(2.0, -2.0), neutral=0.0)
GAIN_FIT = FitSpec("gain", "1", 0.5, 1.5, defaults=(0.85, 1.15), neutral=1.0)
R_FACTOR_FIT = FitSpec("r_factor", "1", 0.5, 1.6, defaults=(1.3, 0.7), neutral=1.0)


def _build() -> dict[Hypothesis, HypothesisSpec]:
    specs: dict[Hypothesis, HypothesisSpec] = {
        Hypothesis.healthy: HypothesisSpec(Hypothesis.healthy, "healthy", False, "no fault; converter and sensors healthy"),
    }
    for k, leg in enumerate(LEGS):
        for side, where in (("U", "upper"), ("L", "lower")):
            sid = f"{leg}{side}"
            h = Hypothesis(f"{sid}_open")
            specs[h] = HypothesisSpec(h, "switch_open", True, f"{where} switch of leg {leg} open", k, frozenset({sid}))
    for k, leg in enumerate(LEGS):
        h = Hypothesis(f"leg_{leg}_open")
        specs[h] = HypothesisSpec(h, "leg_open", True, f"both switches of leg {leg} open", k,
                                  frozenset({f"{leg}U", f"{leg}L"}))
    for name in _TWO_SWITCH:
        a, b, _ = name.split("_")
        h = Hypothesis(name)
        specs[h] = HypothesisSpec(h, "two_switch_open", True, f"switches {a} and {b} open", None, frozenset({a, b}))
    for k, leg in enumerate(LEGS):
        for family, fit, text in (
            ("sensor_offset", OFFSET_FIT, f"current sensor of phase {leg} has a DC offset"),
            ("sensor_gain", GAIN_FIT, f"current sensor of phase {leg} has a gain error"),
            ("load_unbalance", R_FACTOR_FIT, f"load resistance of phase {leg} differs from the others"),
        ):
            h = Hypothesis(f"{family}_{leg}")
            specs[h] = HypothesisSpec(h, family, False, text, k, fit=fit)  # type: ignore[arg-type]
    specs[Hypothesis.load_step] = HypothesisSpec(
        Hypothesis.load_step, "load_step", False, "load resistance of all phases changed together", fit=R_FACTOR_FIT)
    return specs


SPECS: dict[Hypothesis, HypothesisSpec] = _build()

# Two-switch combinations are tested only when explicitly requested (CLAUDE.md 5.4).
TWO_SWITCH_HYPOTHESES: tuple[Hypothesis, ...] = tuple(h for h, s in SPECS.items() if s.family == "two_switch_open")
DEFAULT_HYPOTHESES: tuple[Hypothesis, ...] = tuple(h for h in SPECS if h not in TWO_SWITCH_HYPOTHESES)


def hypothesis_for_condition(condition: Condition) -> Hypothesis | None:
    """The hypothesis that describes a single-cause condition (None for combined causes)."""
    if condition.is_healthy:
        return Hypothesis.healthy
    for h, spec in SPECS.items():
        if spec.switches and spec.switches == condition.open_switches and condition.r_factor == (1.0, 1.0, 1.0) \
                and condition.sensor_offset_A == (0.0, 0.0, 0.0) and condition.sensor_gain == (1.0, 1.0, 1.0):
            return h
    if condition.open_switches:
        return None
    changed = [
        ("sensor_offset", [k for k, x in enumerate(condition.sensor_offset_A) if x != 0.0]),
        ("sensor_gain", [k for k, x in enumerate(condition.sensor_gain) if x != 1.0]),
        ("load_unbalance", [k for k, x in enumerate(condition.r_factor) if x != 1.0]),
    ]
    active = [(fam, ks) for fam, ks in changed if ks]
    if len(active) != 1:
        return None
    family, ks = active[0]
    if family == "load_unbalance" and len(ks) == 3 and len(set(condition.r_factor)) == 1:
        return Hypothesis.load_step
    return Hypothesis(f"{family}_{LEGS[ks[0]]}") if len(ks) == 1 else None
