"""Named plant scenarios for the CLI demo, the "Mystery fault" button and the evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from faulttwin.hypotheses import Hypothesis
from faulttwin.plant import Condition


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    condition: Condition
    truth: Hypothesis
    kind: Literal["fault", "look-alike", "healthy"]
    m: float = 0.8


def _switch(sid: str, where: str) -> Scenario:
    names = {"AU": "S1", "AL": "S4", "BU": "S3", "BL": "S6", "CU": "S5", "CL": "S2"}
    return Scenario(f"{sid}_open", f"{where} switch {sid} ({names[sid]}) open", Condition(open_switches={sid}),
                    Hypothesis(f"{sid}_open"), "fault")


SCENARIOS: dict[str, Scenario] = {s.name: s for s in [
    Scenario("healthy", "Healthy converter", Condition(), Hypothesis.healthy, "healthy"),
    Scenario("low_load", "Healthy converter at low load (m = 0.3)", Condition(), Hypothesis.healthy, "healthy", m=0.3),
    _switch("AU", "Upper"), _switch("AL", "Lower"), _switch("BU", "Upper"),
    _switch("BL", "Lower"), _switch("CU", "Upper"), _switch("CL", "Lower"),
    *[Scenario(f"leg_{leg}_open", f"Both switches of leg {leg} open", Condition(open_switches={f"{leg}U", f"{leg}L"}),
               Hypothesis(f"leg_{leg}_open"), "fault") for leg in "ABC"],
    Scenario("AU_BU_open", "Switches AU and BU open", Condition(open_switches={"AU", "BU"}), Hypothesis.AU_BU_open, "fault"),
    Scenario("sensor_offset_A", "Current sensor A offset +2 A", Condition(sensor_offset_A=(2.0, 0.0, 0.0)),
             Hypothesis.sensor_offset_A, "look-alike"),
    Scenario("sensor_offset_C", "Current sensor C offset -2.5 A", Condition(sensor_offset_A=(0.0, 0.0, -2.5)),
             Hypothesis.sensor_offset_C, "look-alike"),
    # A 3 A offset lifts |mu| above the textbook 0.10 threshold: the baseline calls it an open switch (BU_open).
    Scenario("sensor_offset_B_3A", "Current sensor B offset -3 A", Condition(sensor_offset_A=(0.0, -3.0, 0.0)),
             Hypothesis.sensor_offset_B, "look-alike"),
    Scenario("sensor_gain_B", "Current sensor B gain 0.85", Condition(sensor_gain=(1.0, 0.85, 1.0)),
             Hypothesis.sensor_gain_B, "look-alike"),
    Scenario("sensor_gain_A", "Current sensor A gain 1.15", Condition(sensor_gain=(1.15, 1.0, 1.0)),
             Hypothesis.sensor_gain_A, "look-alike"),
    # The edge monitor does not raise an event for this one (its amplitude ratios stay inside the band):
    # a +30 % load unbalance is not an alarm. Kept for the evaluation's no-false-alarm cases.
    Scenario("load_unbalance_A", "Phase A load resistance x1.3", Condition(r_factor=(1.3, 1.0, 1.0)),
             Hypothesis.load_unbalance_A, "look-alike"),
    Scenario("load_step", "Load resistance of all phases x0.7", Condition(r_factor=(0.7, 0.7, 0.7)),
             Hypothesis.load_step, "look-alike"),
]}

# The "Mystery fault" button draws from these: real faults and the look-alikes that fool thresholds.
MYSTERY_POOL = ["AU_open", "BL_open", "CU_open", "leg_B_open", "sensor_offset_A", "sensor_offset_C",
                "sensor_offset_B_3A", "sensor_gain_B", "sensor_gain_A"]
