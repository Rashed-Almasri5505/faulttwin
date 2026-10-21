"""Feature extraction on synthetic signals with known answers."""

import math

import numpy as np
import pytest

from faulttwin.features import amplitude_A, clarke, compute_features, window_cycles

W1 = 2 * math.pi * 50
DT = 20e-6
T = np.arange(2000) * DT + 0.0123   # 2 periods, arbitrary start time


def balanced(amp_A: float, phase_rad: float = 0.3) -> np.ndarray:
    return np.column_stack([amp_A * np.sin(W1 * T + phase_rad + ph) for ph in (0.0, -2 * math.pi / 3, 2 * math.pi / 3)])


def test_balanced_set_gives_circle_of_radius_equal_to_amplitude() -> None:
    i = balanced(10.0)
    a, b = clarke(i)
    assert np.hypot(a, b) == pytest.approx(10.0, rel=1e-9)
    f = compute_features(T, i, W1)
    assert f.i_s_A == pytest.approx(10.0, rel=1e-9)
    assert f.mu == pytest.approx([0, 0, 0], abs=1e-9)
    assert f.r_kcl_dc == pytest.approx(0, abs=1e-9)
    assert f.r_kcl_w1 == pytest.approx(0, abs=1e-9)
    assert f.amp_ratio == pytest.approx([1, 1, 1], rel=1e-9)
    assert f.rms_A == pytest.approx([10 / math.sqrt(2)] * 3, rel=1e-9)


def test_single_bin_dft_recovers_each_component() -> None:
    x = 1.5 + 4.0 * np.sin(W1 * T + 0.7) + 2.0 * np.cos(2 * W1 * T) + 0.5 * np.sin(3 * W1 * T - 1.0)
    assert amplitude_A(x, T, 0.0) == pytest.approx(1.5, abs=1e-9)
    assert amplitude_A(x, T, W1) == pytest.approx(4.0, abs=1e-9)
    assert amplitude_A(x, T, 2 * W1) == pytest.approx(2.0, abs=1e-9)
    assert amplitude_A(x, T, 3 * W1) == pytest.approx(0.5, abs=1e-9)


def test_spectrum_bins_are_in_rad_per_s() -> None:
    f = compute_features(T, balanced(5.0), W1)
    assert f.omega_rad_s == pytest.approx([0.0, W1, 2 * W1, 3 * W1])


def test_sensor_offset_shows_in_mu_and_kcl_dc() -> None:
    i = balanced(10.0)
    i[:, 1] += 1.0
    f = compute_features(T, i, W1)
    # The offset also deforms the Park's vector circle slightly, so |i_s| is a little above 10 A.
    assert f.mu[1] == pytest.approx(1.0 / f.i_s_A, rel=1e-9)
    assert f.mu[1] == pytest.approx(0.1, abs=1e-3)
    assert f.r_kcl_dc == pytest.approx(f.mu[1], rel=1e-9)


def test_window_must_span_whole_periods() -> None:
    assert window_cycles(T, W1) == 2
    with pytest.raises(ValueError):
        compute_features(T[:1500], balanced(1.0)[:1500], W1)


def test_non_finite_samples_are_rejected() -> None:
    i = balanced(10.0)
    i[5, 2] = np.inf
    with pytest.raises(ValueError):
        compute_features(T, i, W1)


def test_whole_period_check_survives_large_absolute_times() -> None:
    t_late = (np.arange(2000) + 48 * 3600 * 50_000) * DT   # 48 h after start
    assert window_cycles(t_late, W1) == 2


def test_stopped_plant_gives_finite_features() -> None:
    f = compute_features(T, np.zeros((2000, 3)), W1)
    values = [f.i_s_A, f.r_kcl_dc, f.r_kcl_w1, *f.mu, *f.amp_ratio, *f.hd_ratio]
    assert all(math.isfinite(v) for v in values)
    assert f.mu == [0.0, 0.0, 0.0]
