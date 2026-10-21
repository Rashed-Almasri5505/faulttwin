"""Edge monitor behaviour and threshold calibration.

Calibration uses config.calibration_seeds, which the evaluation must not reuse.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from faulttwin.config import load_config
from faulttwin.features import compute_features
from faulttwin.monitor import EdgeMonitor, Event, MonitorThresholds, baseline_diagnose
from faulttwin.plant import Condition, InverterPlant, PlantParams, rated_peak_current_A

CFG = load_config()
SEEDS = CFG["calibration_seeds"]
TH = MonitorThresholds.from_config(CFG)
NOISE_A = CFG["plant"]["noise_sigma_frac"] * rated_peak_current_A(PlantParams())


def realistic_plant(seed: int, m: float = 0.8, noise_A: float = NOISE_A) -> InverterPlant:
    return InverterPlant(PlantParams(m=m), seed=seed, noise_sigma_A=noise_A,
                         mismatch_frac=CFG["plant"]["param_mismatch_frac"])


def new_monitor(plant: InverterPlant) -> EdgeMonitor:
    return EdgeMonitor(TH, plant.params.w1_rad_s)


def events_after(condition: Condition, seed: int, m: float = 0.8, before: int = 15, after: int = 10) -> list[Event]:
    """Run healthy, inject the condition, and return the events raised after injection."""
    plant = realistic_plant(seed, m)
    monitor = new_monitor(plant)
    monitor.process_chunk(plant.run_cycles(before))
    plant.set_condition(condition)
    statuses = monitor.process_chunk(plant.run_cycles(after))
    return [e for s in statuses for e in s.new_events]


def test_thresholds_come_from_config_and_typos_fail() -> None:
    assert TH.mu_abs_max == CFG["monitor"]["mu_abs_max"]
    bad = dict(CFG["monitor"], mu_abs_maxx=0.2)
    with pytest.raises(ValidationError):
        MonitorThresholds(**bad, window_cycles=2)
    missing = {k: v for k, v in CFG["monitor"].items() if k != "r_kcl_dc_max"}
    with pytest.raises(ValidationError):
        MonitorThresholds(**missing, window_cycles=2)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("m", [0.3, 0.8])
def test_no_false_alarms_on_healthy_plant(seed: int, m: float) -> None:
    plant = realistic_plant(seed, m)
    statuses = new_monitor(plant).process_chunk(plant.run_cycles(40))
    assert [e.message for s in statuses for e in s.new_events] == []


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_switch_fault_raises_mean_current_event_at_the_second_faulty_cycle(seed: int) -> None:
    events = events_after(Condition(open_switches={"AU"}), seed)
    mean_events = [e for e in events if e.type == "mean_current"]
    assert len(mean_events) == 1 and mean_events[0].severity == "high"
    # Fault from cycle 15 on; "2 consecutive cycles" means the event comes at cycle 16.
    assert mean_events[0].cycle_index == 16
    assert "phase A" in mean_events[0].message


def test_one_cycle_glitch_does_not_satisfy_two_consecutive_cycles() -> None:
    plant = realistic_plant(SEEDS[0])
    monitor = new_monitor(plant)
    events = []
    for c in range(30):
        chunk = plant.run_cycles(1)
        i_meas = chunk.i_meas_A.copy()
        if c == 20:
            i_meas[:, 0] += 6.0   # one cycle with |mu_a| of about 0.22
        events += monitor.update_cycle(chunk.t_s, i_meas, c, commanded_m=chunk.m).new_events
    assert "mean_current" not in {e.type for e in events}


@pytest.mark.parametrize("seed", SEEDS[:3])
def test_leg_open_is_caught_by_rms_and_amplitude_ratio(seed: int) -> None:
    types = {e.type for e in events_after(Condition(open_switches={"BU", "BL"}), seed)}
    assert {"phase_loss", "amplitude_ratio"} <= types
    assert "mean_current" not in types


def test_sensor_offset_raises_kcl_dc() -> None:
    types = {e.type for e in events_after(Condition(sensor_offset_A=(1.5, 0, 0)), SEEDS[0])}
    assert "kcl_dc" in types


def test_sensor_gain_raises_kcl_w1() -> None:
    types = {e.type for e in events_after(Condition(sensor_gain=(1.0, 1.0, 0.85)), SEEDS[0])}
    assert "kcl_w1" in types


def test_load_step_down_is_a_low_severity_transient() -> None:
    events = events_after(Condition(r_factor=(0.7, 0.7, 0.7)), SEEDS[0])
    assert {e.type for e in events} == {"transient"}
    assert all(e.severity == "low" for e in events)


def test_all_currents_lost_while_commanded_on_is_a_high_severity_event() -> None:
    events = events_after(Condition(open_switches={"AU", "AL", "BU", "BL"}), SEEDS[0])
    no_current = [e for e in events if e.type == "no_current"]
    assert no_current and no_current[0].severity == "high"


@pytest.mark.parametrize(
    ("condition", "m", "seed"),
    [
        (Condition(sensor_offset_A=(1.02, 0.0, 0.0)), 0.3, 9008),   # mu_a sits near 0.10
        (Condition(sensor_gain=(1.0, 1.17, 1.0)), 0.8, 9001),       # amplitude ratio B sits near 1.10
    ],
)
def test_near_threshold_condition_raises_each_event_once(condition: Condition, m: float, seed: int) -> None:
    events = events_after(condition, seed, m=m, after=100)
    types = [e.type for e in events]
    assert types, "expected at least one event"
    assert len(types) == len(set(types)), f"chatter: {types}"


@pytest.mark.parametrize(("m_from", "m_to", "seed"),
                         [(0.8, 0.4, SEEDS[0]), (0.4, 0.8, SEEDS[1]), (0.3, 0.9, SEEDS[2]), (0.9, 0.3, 126)])
def test_commanded_modulation_change_raises_no_events(m_from: float, m_to: float, seed: int) -> None:
    plant = realistic_plant(seed, m_from)
    monitor = new_monitor(plant)
    monitor.process_chunk(plant.run_cycles(15))
    plant.set_modulation_index(m_to)
    monitor.expect_operating_point_change()
    statuses = monitor.process_chunk(plant.run_cycles(10))
    assert [e.message for s in statuses for e in s.new_events] == []


def test_fault_already_reported_does_not_fire_again_after_an_m_step() -> None:
    plant = realistic_plant(SEEDS[0])
    monitor = new_monitor(plant)
    monitor.process_chunk(plant.run_cycles(15))
    plant.set_condition(Condition(open_switches={"AU"}))
    first = [e.type for s in monitor.process_chunk(plant.run_cycles(5)) for e in s.new_events]
    plant.set_modulation_index(0.4)
    monitor.expect_operating_point_change()
    later = [e.type for s in monitor.process_chunk(plant.run_cycles(10)) for e in s.new_events]
    assert "mean_current" in first
    assert "mean_current" not in later


def test_events_fire_once_per_rising_edge() -> None:
    events = events_after(Condition(open_switches={"AU"}), SEEDS[0], after=20)
    assert len([e for e in events if e.type == "mean_current"]) == 1


def test_cleared_and_reinjected_fault_fires_again() -> None:
    plant = realistic_plant(SEEDS[2])
    monitor = new_monitor(plant)
    monitor.process_chunk(plant.run_cycles(15))
    events = []
    for condition in (Condition(open_switches={"AU"}), Condition(), Condition(open_switches={"AU"})):
        plant.set_condition(condition)
        events += [e.type for s in monitor.process_chunk(plant.run_cycles(10)) for e in s.new_events]
    assert events.count("mean_current") == 2


@pytest.mark.parametrize("offset_A", [0.0, 2.5])
def test_stopped_plant_raises_no_events(offset_A: float) -> None:
    plant = realistic_plant(SEEDS[1])
    plant.set_condition(Condition(sensor_offset_A=(offset_A, 0.0, 0.0)))
    monitor = new_monitor(plant)
    monitor.process_chunk(plant.run_cycles(15))
    plant.stop()
    monitor.expect_operating_point_change()
    statuses = monitor.process_chunk(plant.run_cycles(10))
    assert [e.type for s in statuses for e in s.new_events] == []


def test_non_finite_samples_are_skipped_without_losing_state() -> None:
    plant = realistic_plant(SEEDS[0])
    monitor = new_monitor(plant)
    plant.set_condition(Condition(open_switches={"AU"}))
    monitor.process_chunk(plant.run_cycles(15))
    chunk = plant.run_cycles(1)
    bad = chunk.i_meas_A.copy()
    bad[10, 1] = np.nan
    status = monitor.update_cycle(chunk.t_s, bad, 15, commanded_m=chunk.m)
    assert not status.data_ok and status.new_events == [] and status.active["mean_current"]
    later = [e.type for s in monitor.process_chunk(plant.run_cycles(5)) for e in s.new_events]
    assert "mean_current" not in later


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        (Condition(), "healthy"),
        (Condition(open_switches={"AU"}), "AU_open"),
        (Condition(open_switches={"CL"}), "CL_open"),
        # Naive by design: these are wrong, which the evaluation will show.
        (Condition(open_switches={"AU", "BU"}), "CL_open"),
        (Condition(sensor_offset_A=(0.0, 3.0, 0.0)), "BL_open"),
    ],
)
def test_baseline_diagnoser(condition: Condition, expected: str) -> None:
    plant = InverterPlant(PlantParams(m=0.3 if condition.sensor_offset_A[1] else 0.8))
    plant.run_cycles(15)
    plant.set_condition(condition)
    chunk = plant.run_cycles(5)
    f = compute_features(chunk.t_s[-2000:], chunk.i_meas_A[-2000:], plant.params.w1_rad_s)
    assert baseline_diagnose(f, TH.mu_abs_max) == expected
