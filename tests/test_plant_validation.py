"""Validation targets from CLAUDE.md section 5.1.

Noise-free plant with default parameters: 15 healthy cycles, then the condition is
applied, then 15 more cycles. Features use the last 2 cycles of the measured currents.
"""

import time

import numpy as np
import pytest

from faulttwin import plant as plant_module
from faulttwin.features import Features, compute_features
from faulttwin.plant import Condition, InverterPlant, PlantParams, simulate

W1 = PlantParams().w1_rad_s
TWO_CYCLES = 2 * PlantParams().steps_per_cycle


def run_case(condition: Condition, m_after: float | None = None) -> tuple[Features, np.ndarray]:
    plant = InverterPlant(PlantParams())
    first = plant.run_cycles(15)
    plant.set_condition(condition)
    if m_after is not None:
        plant.set_modulation_index(m_after)
    second = plant.run_cycles(15)
    features = compute_features(second.t_s[-TWO_CYCLES:], second.i_meas_A[-TWO_CYCLES:], W1)
    i_true = np.vstack([first.i_true_A, second.i_true_A])
    return features, i_true


def amp_a(f: Features, harmonic: int) -> float:
    return f.amp_A[0][harmonic]


def assert_kcl(i_true: np.ndarray) -> None:
    assert np.max(np.abs(i_true.sum(axis=1))) < 1e-6


def test_healthy() -> None:
    f, i_true = run_case(Condition())
    assert max(abs(x) for x in f.mu) < 0.01
    assert f.r_kcl_dc < 0.005
    assert amp_a(f, 1) == pytest.approx(27.1, abs=0.3)
    assert_kcl(i_true)


def test_au_open_s1() -> None:
    f, i_true = run_case(Condition(open_switches={"AU"}))
    assert f.mu == pytest.approx([-0.42, 0.21, 0.21], abs=0.03)
    assert f.r_kcl_dc < 0.005
    assert amp_a(f, 0) == pytest.approx(9.4, abs=0.5)
    assert amp_a(f, 1) == pytest.approx(14.1, abs=0.7)
    assert amp_a(f, 2) == pytest.approx(5.0, abs=0.5)
    assert_kcl(i_true)


def test_al_open_s4() -> None:
    f, i_true = run_case(Condition(open_switches={"AL"}))
    assert f.mu == pytest.approx([0.42, -0.21, -0.21], abs=0.03)
    assert_kcl(i_true)


def test_leg_a_open() -> None:
    f, i_true = run_case(Condition(open_switches={"AU", "AL"}))
    assert f.rms_A[0] < 0.2
    assert f.r_kcl_dc < 0.005
    # Normalized means alone miss this fault ...
    assert max(abs(x) for x in f.mu) < 0.01
    # ... but the amplitude ratios do not.
    assert f.amp_ratio[0] < 0.1
    assert_kcl(i_true)


def test_au_and_bu_open() -> None:
    f, i_true = run_case(Condition(open_switches={"AU", "BU"}))
    assert f.mu == pytest.approx([-0.43, -0.46, 0.90], abs=0.05)
    assert_kcl(i_true)


def test_sensor_offset_a() -> None:
    f, i_true = run_case(Condition(sensor_offset_A=(2.0, 0.0, 0.0)))
    assert f.mu[0] == pytest.approx(0.074, abs=0.005)
    assert f.r_kcl_dc == pytest.approx(0.074, abs=0.005)
    # The true currents still sum to zero; only the measured ones do not.
    assert_kcl(i_true)


def test_sensor_gain_b() -> None:
    f, i_true = run_case(Condition(sensor_gain=(1.0, 0.85, 1.0)))
    assert max(abs(x) for x in f.mu) < 0.01
    assert f.r_kcl_w1 == pytest.approx(0.158, abs=0.01)
    assert_kcl(i_true)


def test_load_unbalance_ra() -> None:
    f, i_true = run_case(Condition(r_factor=(1.3, 1.0, 1.0)))
    assert max(abs(x) for x in f.mu) < 0.01
    assert f.r_kcl_dc < 0.005 and f.r_kcl_w1 < 0.005
    assert amp_a(f, 1) == pytest.approx(23.65, abs=0.3)
    assert_kcl(i_true)


def test_modulation_index_step_separates_switch_fault_from_sensor_offset() -> None:
    au_08, _ = run_case(Condition(open_switches={"AU"}))
    au_04, _ = run_case(Condition(open_switches={"AU"}), m_after=0.4)
    assert au_08.mu[0] == pytest.approx(-0.42, abs=0.03)
    assert au_04.mu[0] == pytest.approx(-0.42, abs=0.03)

    off_08, _ = run_case(Condition(sensor_offset_A=(2.0, 0.0, 0.0)))
    off_04, _ = run_case(Condition(sensor_offset_A=(2.0, 0.0, 0.0)), m_after=0.4)
    assert off_08.mu[0] == pytest.approx(0.074, abs=0.005)
    assert off_04.mu[0] == pytest.approx(0.147, abs=0.005)


@pytest.mark.parametrize("switches", [{"BU"}, {"BL"}, {"CU"}, {"CL"}, {"BU", "BL"}, {"CU", "CL"}, {"AU", "CL"}])
def test_kcl_holds_for_other_faults(switches: set[str]) -> None:
    _, i_true = run_case(Condition(open_switches=switches, r_factor=(1.2, 0.9, 1.0)))
    assert_kcl(i_true)


def test_kcl_holds_with_mismatch_noise_and_non_idealities() -> None:
    params = PlantParams(dead_time_s=1e-6, v_ce_sat_V=1.6, v_f_V=1.5)
    plant = InverterPlant(params, seed=9001, noise_sigma_A=0.135, mismatch_frac=0.05,
                          condition=Condition(open_switches={"AU"}))
    chunk = plant.run_cycles(10)
    assert np.max(np.abs(chunk.i_true_A.sum(axis=1))) < 1e-6
    assert len({round(r, 6) for r in plant.params.l_H}) == 3   # unequal inductances


def steady_amp_a(params: PlantParams) -> float:
    plant = InverterPlant(params)
    chunk = plant.run_cycles(15)
    f = compute_features(chunk.t_s[-TWO_CYCLES:], chunk.i_true_A[-TWO_CYCLES:], W1)
    return amp_a(f, 1)


def test_dead_time_lowers_the_fundamental_by_the_expected_amount() -> None:
    # Square-wave error of height Vdc*td*fsw opposing the current: its fundamental (4/pi)*4 V,
    # projected on the current through |Z| and the load angle phi.
    z = complex(5.0, W1 * 10e-3)
    expected_drop_A = (4 / np.pi) * 400 * 1e-6 * 10e3 / abs(z) * np.cos(np.angle(z))
    drop = steady_amp_a(PlantParams()) - steady_amp_a(PlantParams(dead_time_s=1e-6))
    assert drop == pytest.approx(expected_drop_A, rel=0.15)


def test_in_phase_back_emf_lowers_the_fundamental_by_e_over_z() -> None:
    z = abs(complex(5.0, W1 * 10e-3))
    drop = steady_amp_a(PlantParams()) - steady_amp_a(PlantParams(emf_V=20.0))
    assert drop == pytest.approx(20.0 / z, rel=0.02)


def test_device_drops_lower_the_fundamental() -> None:
    assert steady_amp_a(PlantParams(v_ce_sat_V=1.6, v_f_V=1.5)) < steady_amp_a(PlantParams()) - 0.1


def test_measurement_noise_has_the_requested_sigma() -> None:
    plant = InverterPlant(PlantParams(), seed=9003, noise_sigma_A=0.135)
    chunk = plant.run_cycles(10)
    assert np.std(chunk.i_meas_A - chunk.i_true_A) == pytest.approx(0.135, rel=0.05)


def test_model_limits_are_enforced() -> None:
    with pytest.raises(ValueError):
        PlantParams(m=1.2)
    with pytest.raises(ValueError):
        InverterPlant(PlantParams()).set_modulation_index(-0.1)
    with pytest.raises(ValueError):
        simulate(0.0, 100, (0.0, 0.0, 0.0), PlantParams(emf_V=10.0), open_switches=frozenset({"AU", "AL"}))
    with pytest.raises(ValueError):
        PlantParams(w1_rad_s=2 * np.pi * 60)   # 833.3 samples per period at dt = 20 us


def test_device_drops_are_read_from_config_only_when_enabled() -> None:
    from faulttwin.config import load_config

    plant_cfg = dict(load_config()["plant"])
    assert PlantParams.from_config(plant_cfg).v_ce_sat_V == 0.0
    plant_cfg["device_drops_enabled"] = True
    with pytest.raises(ValueError):
        PlantParams.from_config(plant_cfg)
    plant_cfg.update(v_ce_sat_V=1.6, v_f_V=1.5)
    params = PlantParams.from_config(plant_cfg)
    assert (params.v_ce_sat_V, params.v_f_V) == (1.6, 1.5)


def test_stop_and_resume() -> None:
    plant = InverterPlant(PlantParams())
    plant.run_cycles(10)
    plant.stop()
    stopped = plant.run_cycles(5)
    assert stopped.m == 0.0
    assert np.max(np.abs(stopped.i_true_A[-1000:])) < 1e-3
    plant.resume()
    assert plant.run_cycles(5).m == pytest.approx(0.8)


@pytest.mark.skipif(plant_module._integrate_numba is None, reason="numba not installed")
@pytest.mark.parametrize("switches", [set(), {"AU"}, {"BU", "BL"}, {"AU", "CL"}])
def test_numba_and_python_backends_agree(switches: set[str]) -> None:
    params = PlantParams(dead_time_s=1e-6, v_ce_sat_V=1.6, v_f_V=1.5, l_H=(10e-3, 10.4e-3, 9.7e-3),
                         emf_V=20.0, emf_lag_rad=0.2)
    kwargs = {"open_switches": frozenset(switches), "r_ohm": (5.0, 5.5, 4.8), "step0": 12345}
    if {"BU", "BL"} <= switches:
        params = PlantParams(dead_time_s=1e-6, v_ce_sat_V=1.6, v_f_V=1.5, l_H=params.l_H)   # no EMF with an open leg
    _, a, a_final = simulate(0.0, 3000, (1.0, -0.5, -0.5), params, backend="python", **kwargs)
    _, b, b_final = simulate(0.0, 3000, (1.0, -0.5, -0.5), params, backend="numba", **kwargs)
    assert np.max(np.abs(a - b)) < 1e-9
    assert np.max(np.abs(a.sum(axis=1))) < 1e-6 and np.max(np.abs(b.sum(axis=1))) < 1e-6
    assert np.allclose(a_final, b_final, atol=1e-9)


def test_faster_than_real_time() -> None:
    plant = InverterPlant(PlantParams(), condition=Condition(open_switches={"AU", "BL"}))
    plant.run_cycles(1)   # warm up (and JIT-compile with numba)
    best = float("inf")
    for _ in range(5):    # best of 5, so a busy machine does not fail the test
        t0 = time.perf_counter()
        for _ in range(25):
            plant.run_cycles(1)   # 0.5 s of simulated time, streamed one cycle at a time
        best = min(best, time.perf_counter() - t0)
    assert 0.5 / best >= 5.0, f"only {0.5 / best:.1f}x real time"
