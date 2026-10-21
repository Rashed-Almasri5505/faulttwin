"""Diagnostic features computed over whole fundamental periods.

Clarke transform and Park's vector modulus, normalized mean phase currents, KCL
residuals and single-bin DFT amplitudes at 0, w1, 2*w1 and 3*w1. All frequencies
are angular frequencies in rad/s.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, Field

HARMONICS = (0, 1, 2, 3)
_TINY_A = 1e-9


class Features(BaseModel):
    """Features of one window of three phase currents (A, B, C order)."""

    window_cycles: int
    w1_rad_s: float
    omega_rad_s: list[float] = Field(description="DFT bins: 0, w1, 2*w1, 3*w1 (rad/s)")
    i_s_A: float = Field(description="Park's vector modulus |i_s|, mean over the window")
    mean_A: list[float]
    mu: list[float] = Field(description="normalized mean phase currents mean(i_k) / |i_s|")
    r_kcl_dc: float = Field(description="|mean(i_a + i_b + i_c)| / |i_s|")
    r_kcl_w1: float = Field(description="2 |DFT_w1(i_a + i_b + i_c)| / |i_s|")
    amp_A: list[list[float]] = Field(description="per phase: amplitude at each omega_rad_s bin")
    amp_ratio: list[float] = Field(description="A_k(w1) / mean_k A_k(w1)")
    hd_ratio: list[float] = Field(description="sqrt(A_k(2w1)^2 + A_k(3w1)^2) / A_k(w1)")
    rms_A: list[float]

    def compact(self) -> dict[str, object]:
        """Rounded key values, small enough to hand to a language model."""
        return {
            "i_s_A": round(self.i_s_A, 3),
            "mu": [round(x, 4) for x in self.mu],
            "r_kcl_dc": round(self.r_kcl_dc, 4),
            "r_kcl_w1": round(self.r_kcl_w1, 4),
            "amp_ratio": [round(x, 4) for x in self.amp_ratio],
            "hd_ratio": [round(x, 4) for x in self.hd_ratio],
            "rms_A": [round(x, 3) for x in self.rms_A],
            "spectrum_A": {f"{w:.1f} rad/s": [round(a[h], 3) for a in self.amp_A]
                           for h, w in enumerate(self.omega_rad_s)},
        }


def clarke(i_A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude-invariant Clarke transform of an (n, 3) array of phase currents."""
    ia, ib, ic = i_A[:, 0], i_A[:, 1], i_A[:, 2]
    i_alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
    i_beta = (ib - ic) / math.sqrt(3.0)
    return i_alpha, i_beta


def park_modulus_A(i_A: np.ndarray) -> float:
    i_alpha, i_beta = clarke(i_A)
    return float(np.mean(np.hypot(i_alpha, i_beta)))


def dft_bin(x: np.ndarray, t_s: np.ndarray, omega_rad_s: float) -> complex:
    """Single-bin DFT, scaled so that a sinusoid of amplitude A gives |X| = A / 2."""
    if omega_rad_s == 0.0:
        return complex(np.mean(x))
    return complex(np.mean(x * np.exp(-1j * omega_rad_s * t_s)))


def amplitude_A(x: np.ndarray, t_s: np.ndarray, omega_rad_s: float) -> float:
    """Amplitude of the component at omega (the absolute mean value for omega = 0)."""
    x_bin = dft_bin(x, t_s, omega_rad_s)
    return abs(x_bin) if omega_rad_s == 0.0 else 2.0 * abs(x_bin)


def window_cycles(t_s: np.ndarray, w1_rad_s: float) -> int:
    """Number of whole fundamental periods spanned by uniformly sampled times t_s."""
    n = len(t_s)
    dt_s = float(t_s[-1] - t_s[0]) / (n - 1)   # robust to rounding at large absolute times
    cycles = n * dt_s * w1_rad_s / (2.0 * math.pi)
    whole = round(cycles)
    if whole < 1 or abs(cycles - whole) > 1e-6 * cycles:
        raise ValueError(f"window must span a whole number of periods, got {cycles:.4f}")
    return whole


def compute_features(t_s: np.ndarray, i_A: np.ndarray, w1_rad_s: float) -> Features:
    """Features of a window of currents i_A[n, 3] sampled at t_s[n]."""
    if not (np.isfinite(i_A).all() and np.isfinite(t_s).all()):
        raise ValueError("non-finite samples in the window")
    n_cycles = window_cycles(t_s, w1_rad_s)
    omegas = [h * w1_rad_s for h in HARMONICS]
    i_s = park_modulus_A(i_A)
    scale = i_s if i_s > _TINY_A else float("inf")   # stopped plant: normalized features are 0

    mean = i_A.mean(axis=0)
    total = i_A.sum(axis=1)
    amp = [[amplitude_A(i_A[:, k], t_s, w) for w in omegas] for k in range(3)]
    a1 = [amp[k][1] for k in range(3)]
    a1_mean = sum(a1) / 3.0
    amp_ratio = [a / a1_mean if a1_mean > _TINY_A else 1.0 for a in a1]
    hd = [math.hypot(amp[k][2], amp[k][3]) / amp[k][1] if amp[k][1] > _TINY_A else 0.0 for k in range(3)]

    return Features(
        window_cycles=n_cycles,
        w1_rad_s=w1_rad_s,
        omega_rad_s=omegas,
        i_s_A=i_s,
        mean_A=mean.tolist(),
        mu=(mean / scale).tolist(),
        r_kcl_dc=abs(float(total.mean())) / scale,
        r_kcl_w1=amplitude_A(total, t_s, w1_rad_s) / scale,
        amp_A=amp,
        amp_ratio=amp_ratio,
        hd_ratio=hd,
        rms_A=np.sqrt(np.mean(i_A**2, axis=0)).tolist(),
    )
