"""Digital twin: ranking, fits, scoring and alignment (Phase 2 acceptance).

Acceptance: for each validation case the true cause ranks #1 with noise on, for 3 seeds.
The plant here is realistic (noise and +/-5 % parameter mismatch). The seeds are neither the
calibration seeds (9001-9010) nor review-sweep seeds (5000-5999) nor evaluation seeds.
"""

import copy
import json

import numpy as np
import pytest
from pydantic import ValidationError

from faulttwin import plant as plant_module
from faulttwin.config import load_config
from faulttwin.hypotheses import DEFAULT_HYPOTHESES, SPECS, TWO_SWITCH_HYPOTHESES, Hypothesis, hypothesis_for_condition
from faulttwin.plant import Condition, InverterPlant, PlantChunk
from faulttwin.twin import DigitalTwin, MeasuredWindow, TwinSettings, WindowNotDiagnosable, rank_of, ranking_report

CFG = load_config()
# Three seeds as the acceptance criterion asks; one without numba, where the twin is ~10x slower.
TEST_SEEDS = [7001, 7002, 7003] if plant_module._integrate_numba is not None else [7001]
TWIN = DigitalTwin.from_config(CFG)
NOMINAL = TWIN.params

VALIDATION_CASES = [
    ("healthy", Condition(), Hypothesis.healthy, 0.8),
    ("AU open (S1)", Condition(open_switches={"AU"}), Hypothesis.AU_open, 0.8),
    ("AL open (S4)", Condition(open_switches={"AL"}), Hypothesis.AL_open, 0.8),
    ("leg A open", Condition(open_switches={"AU", "AL"}), Hypothesis.leg_A_open, 0.8),
    ("AU+BU open", Condition(open_switches={"AU", "BU"}), Hypothesis.AU_BU_open, 0.8),
    ("sensor A offset +2 A", Condition(sensor_offset_A=(2.0, 0.0, 0.0)), Hypothesis.sensor_offset_A, 0.8),
    ("sensor B gain 0.85", Condition(sensor_gain=(1.0, 0.85, 1.0)), Hypothesis.sensor_gain_B, 0.8),
    ("R_A x1.3", Condition(r_factor=(1.3, 1.0, 1.0)), Hypothesis.load_unbalance_A, 0.8),
    # Modulation-index row of the validation table
    ("AU open at m=0.4", Condition(open_switches={"AU"}), Hypothesis.AU_open, 0.4),
    ("sensor A offset +2 A at m=0.4", Condition(sensor_offset_A=(2.0, 0.0, 0.0)), Hypothesis.sensor_offset_A, 0.4),
]


def plant_after(condition: Condition, seed: int | None, m: float = 0.8, realistic: bool = True) -> tuple[InverterPlant, PlantChunk]:
    plant = InverterPlant.from_config(CFG, seed=seed, realistic=realistic)
    plant.set_modulation_index(m)
    plant.run_cycles(15)
    plant.set_condition(condition)
    return plant, plant.run_cycles(15)


def measured_window(condition: Condition, seed: int | None, m: float = 0.8, realistic: bool = True) -> MeasuredWindow:
    return TWIN.window_from_chunk(plant_after(condition, seed, m, realistic)[1])


def hypotheses_for(truth: Hypothesis) -> list[Hypothesis]:
    extra = list(TWO_SWITCH_HYPOTHESES) if SPECS[truth].family == "two_switch_open" else []   # only on request
    return list(DEFAULT_HYPOTHESES) + extra


def ranked(condition: Condition, seed: int | None, truth: Hypothesis, m: float = 0.8, realistic: bool = True):
    return TWIN.simulate_hypotheses(hypotheses_for(truth), measured_window(condition, seed, m, realistic))


def last_cycle(chunk: PlantChunk) -> PlantChunk:
    n = NOMINAL.steps_per_cycle
    return PlantChunk(chunk.t_s[-n:], chunk.i_true_A[-n:], chunk.i_meas_A[-n:], chunk.m,
                      chunk.first_cycle + chunk.n_cycles - 1, 1)


def joined_window(chunks: list[PlantChunk], m: float) -> MeasuredWindow:
    t = np.concatenate([c.t_s for c in chunks])
    i = np.concatenate([c.i_meas_A for c in chunks])
    return MeasuredWindow(t, i, round(t[0] / NOMINAL.dt_s), m, NOMINAL.steps_per_cycle)


# --- acceptance ------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize(("name", "condition", "truth", "m"), VALIDATION_CASES, ids=[c[0] for c in VALIDATION_CASES])
def test_true_cause_ranks_first_with_noise_and_mismatch(name: str, condition: Condition, truth: Hypothesis,
                                                        m: float, seed: int) -> None:
    results = ranked(condition, seed, truth, m)
    top = [(r.hypothesis.value, round(r.match_error_pct, 2)) for r in results[:3]]
    assert results[0].hypothesis == truth, f"{name}, seed {seed}: top 3 = {top}"
    assert not results[0].within_tolerance


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize(
    ("condition", "truth", "m"),
    [
        (Condition(open_switches={"BU"}), Hypothesis.BU_open, 0.8),
        (Condition(open_switches={"BL"}), Hypothesis.BL_open, 0.8),
        (Condition(open_switches={"CU"}), Hypothesis.CU_open, 0.8),
        (Condition(open_switches={"CL"}), Hypothesis.CL_open, 0.35),
        (Condition(open_switches={"BU", "BL"}), Hypothesis.leg_B_open, 0.8),
        (Condition(open_switches={"CU", "CL"}), Hypothesis.leg_C_open, 0.8),
        (Condition(open_switches={"AL", "CU"}), Hypothesis.AL_CU_open, 0.8),
        (Condition(), Hypothesis.healthy, 0.3),
        (Condition(sensor_offset_A=(0.0, -1.5, 0.0)), Hypothesis.sensor_offset_B, 0.8),
        (Condition(sensor_offset_A=(0.0, 0.0, 3.0)), Hypothesis.sensor_offset_C, 0.3),
        (Condition(sensor_gain=(1.15, 1.0, 1.0)), Hypothesis.sensor_gain_A, 0.8),
        (Condition(r_factor=(1.0, 1.0, 0.7)), Hypothesis.load_unbalance_C, 0.8),
        (Condition(r_factor=(1.3, 1.3, 1.3)), Hypothesis.load_step, 0.8),
        (Condition(r_factor=(0.7, 0.7, 0.7)), Hypothesis.load_step, 0.8),
    ],
)
def test_other_faults_and_look_alikes_rank_first(condition: Condition, truth: Hypothesis, m: float, seed: int) -> None:
    results = ranked(condition, seed, truth, m)
    top = [(r.hypothesis.value, round(r.match_error_pct, 2)) for r in results[:3]]
    assert results[0].hypothesis == truth, f"seed {seed}, m {m}: top 3 = {top}"


@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize(("condition", "m"), [(Condition(r_factor=(1.0, 1.15, 1.0)), 0.8),
                                              (Condition(r_factor=(0.85, 1.0, 1.0)), 0.3)])
def test_small_unbalance_is_never_called_a_converter_fault(condition: Condition, m: float, seed: int) -> None:
    # Known limitation: a ~15 % unbalance is at the edge of the +/-5 % R/L tolerance, so the exact
    # label may be "healthy". It must never be a converter fault.
    results = ranked(condition, seed, Hypothesis.healthy, m)
    assert not results[0].spec.converter_fault


# --- fits and scoring --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", TEST_SEEDS)
@pytest.mark.parametrize(
    ("condition", "truth", "expected", "tol"),
    [   # values between the coarse grid points, so the golden-section refinement is exercised
        (Condition(sensor_offset_A=(2.0, 0.0, 0.0)), Hypothesis.sensor_offset_A, 2.0, 0.1),
        (Condition(sensor_gain=(1.0, 0.87, 1.0)), Hypothesis.sensor_gain_B, 0.87, 0.012),
        (Condition(r_factor=(1.27, 1.0, 1.0)), Hypothesis.load_unbalance_A, 1.27, 0.06),
        (Condition(r_factor=(0.73, 0.73, 0.73)), Hypothesis.load_step, 0.73, 0.02),
    ],
)
def test_fitted_parameters_recover_the_injected_values(condition: Condition, truth: Hypothesis,
                                                      expected: float, tol: float, seed: int) -> None:
    result = next(r for r in ranked(condition, seed, truth) if r.hypothesis == truth)
    assert result.fitted_param is not None and result.fitted_param.fitted
    assert result.fitted_param.value == pytest.approx(expected, abs=tol)
    assert not result.within_tolerance


def test_match_error_is_wave_plus_feature_plus_tolerance_penalty() -> None:
    for r in ranked(Condition(), TEST_SEEDS[0], Hypothesis.healthy):
        assert r.match_error_pct == pytest.approx(r.wave_error_pct + r.feature_error_pct + r.tolerance_penalty_pct)
        assert r.within_tolerance == (r.tolerance_penalty_pct > 0)


def test_predicted_features_are_the_hypothesis_physics() -> None:
    results = {r.hypothesis: r for r in ranked(Condition(open_switches={"AU"}), None, Hypothesis.AU_open, realistic=False)}
    assert results[Hypothesis.AU_open].predicted_features.mu == pytest.approx([-0.42, 0.21, 0.21], abs=0.03)
    assert results[Hypothesis.healthy].predicted_features.mu == pytest.approx([0.0, 0.0, 0.0], abs=0.01)


def test_feature_distance_separates_look_alikes_from_a_switch_fault() -> None:
    window = measured_window(Condition(open_switches={"AU"}), TEST_SEEDS[0])
    measured_mu = np.array(TWIN.simulate_hypotheses([Hypothesis.AU_open], window)[0].predicted_features.mu)
    for h in (Hypothesis.sensor_offset_A, Hypothesis.load_unbalance_A):
        # Neither look-alike can produce mu_b = mu_c = +0.21 of an open upper switch in leg A.
        r = TWIN.simulate_hypotheses([h], window)[0]
        assert np.max(np.abs(np.array(r.predicted_features.mu) - measured_mu)) > 0.15
        assert r.feature_error_pct > 1.0


def test_twin_matches_an_ideal_plant_exactly() -> None:
    """Same equations, nominal parameters, no noise: the twin is aligned with the plant time base."""
    cases = [(Condition(), Hypothesis.healthy), (Condition(open_switches={"AU"}), Hypothesis.AU_open),
             (Condition(open_switches={"AU", "AL"}), Hypothesis.leg_A_open),
             (Condition(open_switches={"AU", "BU"}), Hypothesis.AU_BU_open)]
    for condition, truth in cases:
        window = measured_window(condition, None, realistic=False)
        best = TWIN.simulate_hypotheses(hypotheses_for(truth), window)[0]
        assert best.hypothesis == truth
        assert best.match_error_pct < 1e-6
        assert np.max(np.abs(best.i_pred_A - window.i_meas_A)) < 1e-9
        shifted = np.roll(best.i_pred_A, 25, axis=0)   # a 0.5 ms misalignment must show up clearly
        assert 100 * np.sqrt(np.mean((window.i_meas_A - shifted) ** 2)) / best.predicted_features.i_s_A > 3.0


def test_twin_does_not_know_the_plant_mismatch() -> None:
    plant, chunk = plant_after(Condition(), TEST_SEEDS[0])
    assert TWIN.params.r_ohm != plant.params.r_ohm and TWIN.params.l_H != plant.params.l_H
    window = TWIN.window_from_chunk(chunk)
    nominal = TWIN.simulate_hypotheses([Hypothesis.healthy], window)[0]
    oracle = DigitalTwin(plant.params, TWIN.settings).simulate_hypotheses([Hypothesis.healthy], window)[0]
    # With the plant's true R and L only noise is left; the nominal twin is clearly worse.
    assert nominal.wave_error_pct > oracle.wave_error_pct + 0.5


def test_twin_from_config_ignores_plant_only_non_idealities() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["plant"].update(dead_time_enabled=True, device_drops_enabled=True, v_ce_sat_V=1.6, v_f_V=1.5)
    twin = DigitalTwin.from_config(cfg)
    assert (twin.params.dead_time_s, twin.params.v_ce_sat_V, twin.params.v_f_V) == (0.0, 0.0, 0.0)


def test_small_look_alike_fits_on_a_healthy_plant_are_flagged_within_tolerance() -> None:
    results = ranked(Condition(), TEST_SEEDS[1], Hypothesis.healthy)
    fitted = [r for r in results if r.fitted_param is not None]
    assert fitted and all(r.within_tolerance for r in fitted)
    assert "note" in fitted[0].summary()


def test_tolerance_penalty_ramps_down_without_a_cliff() -> None:
    spec = SPECS[Hypothesis.load_unbalance_A]
    p = [TWIN.tolerance_penalty(spec, 1.0 + d) for d in (0.05, 0.099, 0.101, 0.1125, 0.13)]
    full = TWIN.settings.within_tolerance_penalty_pct
    assert p[0] == p[1] == full
    assert full > p[2] > p[3] > p[4] == 0.0


def test_ranking_report_warns_when_the_best_match_is_within_tolerance() -> None:
    results = ranked(Condition(), TEST_SEEDS[0], Hypothesis.healthy)
    assert ranking_report(results)["notes"] == []
    flagged = sorted(results, key=lambda r: not r.within_tolerance)
    report = ranking_report(flagged, top_n=3)
    assert len(report["ranked"]) == 3 and report["notes"]
    json.dumps(report)


# --- operating limits ---------------------------------------------------------------------------

def test_window_spanning_an_m_change_is_not_diagnosable() -> None:
    plant = InverterPlant.from_config(CFG, seed=TEST_SEEDS[0])
    before = plant.run_cycles(15)
    plant.set_modulation_index(0.4)
    after = plant.run_cycles(1)
    with pytest.raises(WindowNotDiagnosable):
        TWIN.simulate_hypotheses(DEFAULT_HYPOTHESES, joined_window([last_cycle(before), after], 0.4))


def test_window_at_the_fault_onset_is_not_diagnosable_but_two_cycles_later_it_is() -> None:
    plant = InverterPlant.from_config(CFG, seed=TEST_SEEDS[1])
    healthy = plant.run_cycles(15)
    plant.set_condition(Condition(open_switches={"AU"}))
    first = plant.run_cycles(1)
    onset = joined_window([last_cycle(healthy), first], 0.8)
    with pytest.raises(WindowNotDiagnosable):
        TWIN.simulate_hypotheses(DEFAULT_HYPOTHESES, onset)
    later = TWIN.window_from_chunk(plant.run_cycles(2))
    assert TWIN.simulate_hypotheses(DEFAULT_HYPOTHESES, later)[0].hypothesis == Hypothesis.AU_open


def test_stopped_plant_is_not_diagnosable() -> None:
    plant = InverterPlant.from_config(CFG, seed=TEST_SEEDS[0])
    plant.run_cycles(15)
    plant.stop()
    with pytest.raises(WindowNotDiagnosable):
        TWIN.simulate_hypotheses(DEFAULT_HYPOTHESES, TWIN.window_from_chunk(plant.run_cycles(5)))


def test_ranking_does_not_depend_on_input_order() -> None:
    window = measured_window(Condition(open_switches={"CU"}), TEST_SEEDS[0])
    forward = [r.hypothesis for r in TWIN.simulate_hypotheses(DEFAULT_HYPOTHESES, window)]
    backward = [r.hypothesis for r in TWIN.simulate_hypotheses(list(reversed(DEFAULT_HYPOTHESES)), window)]
    assert forward == backward


def test_hypothesis_input_forms() -> None:
    window = measured_window(Condition(), TEST_SEEDS[0])
    assert len(TWIN.simulate_hypotheses("healthy", window)) == 1
    assert len(TWIN.simulate_hypotheses(Hypothesis.AU_open, window)) == 1
    assert len(TWIN.simulate_hypotheses(["healthy", Hypothesis.healthy, "AU_open"], window)) == 2
    with pytest.raises(ValueError):
        TWIN.simulate_hypotheses([], window)
    with pytest.raises(ValueError):
        TWIN.simulate_hypotheses(["not_a_cause"], window)
    with pytest.raises(ValueError):
        rank_of(TWIN.simulate_hypotheses(["healthy"], window), Hypothesis.AU_open)


@pytest.mark.parametrize(("condition", "truth"), [(Condition(sensor_offset_A=(0.0, -2.5, 0.0)), Hypothesis.sensor_offset_B),
                                                  (Condition(sensor_gain=(1.0, 1.0, 1.15)), Hypothesis.sensor_gain_C),
                                                  (Condition(r_factor=(0.7, 1.0, 1.0)), Hypothesis.load_unbalance_A)])
def test_fit_params_false_tries_both_signs(condition: Condition, truth: Hypothesis) -> None:
    window = measured_window(condition, TEST_SEEDS[0])
    results = TWIN.simulate_hypotheses(DEFAULT_HYPOTHESES, window, fit_params=False)
    assert results[0].hypothesis == truth
    assert results[0].fitted_param is not None and not results[0].fitted_param.fitted


# --- configuration and enum ----------------------------------------------------------------------

def test_twin_settings_reject_typos_and_bad_values() -> None:
    bad_key = copy.deepcopy(CFG)
    bad_key["twin"]["min_effect"]["sensor_gian"] = bad_key["twin"]["min_effect"].pop("sensor_gain")
    with pytest.raises(ValidationError):
        TwinSettings.from_config(bad_key)
    bad_value = copy.deepcopy(CFG)
    bad_value["twin"]["fit_tolerance"] = 0.0
    with pytest.raises(ValidationError):
        TwinSettings.from_config(bad_value)


def test_two_switch_hypotheses_only_when_requested() -> None:
    assert not set(TWO_SWITCH_HYPOTHESES) & set(DEFAULT_HYPOTHESES)
    assert len(DEFAULT_HYPOTHESES) == 20 and len(TWO_SWITCH_HYPOTHESES) == 12


@pytest.mark.parametrize("hypothesis", list(Hypothesis))
def test_every_hypothesis_round_trips_through_its_condition(hypothesis: Hypothesis) -> None:
    assert hypothesis_for_condition(SPECS[hypothesis].condition()) == hypothesis


def test_combined_causes_have_no_single_hypothesis() -> None:
    assert hypothesis_for_condition(Condition(open_switches={"AU"}, sensor_offset_A=(1.0, 0, 0))) is None
    assert hypothesis_for_condition(Condition(r_factor=(1.2, 0.8, 1.0))) is None
    assert hypothesis_for_condition(Condition(sensor_offset_A=(1.0, 0, 0), sensor_gain=(1, 0.9, 1))) is None


def test_results_are_json_ready() -> None:
    results = ranked(Condition(open_switches={"AU"}), TEST_SEEDS[0], Hypothesis.AU_open)
    json.dumps([r.summary() for r in results])
    assert results[0].i_pred_A.shape == (TWIN.settings.window_cycles * NOMINAL.steps_per_cycle, 3)
