"""Virtual inverter plant.

A two-level three-phase voltage-source inverter with sinusoidal PWM, modelled as a
switching-cycle average, driving a star-connected RL load with an isolated neutral.
Open-circuit switch faults follow the diode-conduction rules of the averaged model;
sensor errors affect only the measured copy of the currents.

Units are SI throughout; angles in rad, angular frequency in rad/s. Phase currents are
positive when they flow out of the inverter leg into the load.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

LEGS = ("A", "B", "C")
PHASE_SHIFTS_RAD = (0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0)
SWITCH_IDS = ("AU", "AL", "BU", "BL", "CU", "CL")
# Textbook numbering of a two-level bridge.
TEXTBOOK_NAMES = {"AU": "S1", "AL": "S4", "BU": "S3", "BL": "S6", "CU": "S5", "CL": "S2"}

Triple = tuple[float, float, float]


@dataclass(frozen=True)
class PlantParams:
    """Electrical parameters. Defaults match config.yaml.

    Limits of the model: the modulation index must be in [0, 1] (no overmodulation),
    and a back-EMF cannot be combined with a fully open leg, because diode re-conduction
    of a floating phase driven by the back-EMF is not modelled.
    """

    v_dc_V: float = 400.0
    r_ohm: Triple = (5.0, 5.0, 5.0)
    l_H: Triple = (10e-3, 10e-3, 10e-3)
    w1_rad_s: float = 2.0 * math.pi * 50.0
    m: float = 0.8
    dt_s: float = 20e-6
    eps_open_leg_A: float = 0.05
    f_sw_Hz: float = 10e3
    dead_time_s: float = 0.0        # 0 disables the dead-time voltage error
    v_ce_sat_V: float = 0.0         # 0 disables device voltage drops
    v_f_V: float = 0.0
    emf_V: float = 0.0              # optional sinusoidal back-EMF amplitude
    emf_lag_rad: float = 0.0        # back-EMF phase lag behind the commanded voltage

    def __post_init__(self) -> None:
        samples = 2.0 * math.pi / (self.w1_rad_s * self.dt_s)
        if abs(samples - round(samples)) > 1e-6 * samples:
            raise ValueError(f"dt_s must divide the fundamental period exactly; got {samples:.4f} samples per period")
        if not 0.0 <= self.m <= 1.0:
            raise ValueError(f"modulation index must be in [0, 1], got {self.m}")

    @property
    def steps_per_cycle(self) -> int:
        return round(2.0 * math.pi / (self.w1_rad_s * self.dt_s))

    @classmethod
    def from_config(cls, plant_cfg: dict[str, Any]) -> PlantParams:
        r, l_ = float(plant_cfg["r_ohm"]), float(plant_cfg["l_H"])
        v_ce_sat_V = v_f_V = 0.0
        if plant_cfg.get("device_drops_enabled"):
            if plant_cfg.get("v_ce_sat_V") is None or plant_cfg.get("v_f_V") is None:
                raise ValueError("device_drops_enabled needs plant.v_ce_sat_V and plant.v_f_V in config.yaml")
            v_ce_sat_V, v_f_V = float(plant_cfg["v_ce_sat_V"]), float(plant_cfg["v_f_V"])
        return cls(
            v_dc_V=float(plant_cfg["v_dc_V"]),
            r_ohm=(r, r, r),
            l_H=(l_, l_, l_),
            w1_rad_s=float(plant_cfg["w1_rad_s"]),
            m=float(plant_cfg["m"]),
            dt_s=float(plant_cfg["dt_s"]),
            eps_open_leg_A=float(plant_cfg["eps_open_leg_A"]),
            f_sw_Hz=float(plant_cfg["f_sw_Hz"]),
            dead_time_s=float(plant_cfg["dead_time_s"]) if plant_cfg.get("dead_time_enabled") else 0.0,
            v_ce_sat_V=v_ce_sat_V,
            v_f_V=v_f_V,
        )


def rated_peak_current_A(params: PlantParams) -> float:
    """Peak phase current of the healthy, balanced plant at the nominal operating point."""
    z_ohm = math.hypot(params.r_ohm[0], params.w1_rad_s * params.l_H[0])
    return params.m * params.v_dc_V / 2.0 / z_ohm


@dataclass(frozen=True)
class Condition:
    """What is wrong with the plant right now. Healthy by default.

    open_switches  converter faults, e.g. {"AU"} or {"AU", "AL"} (whole leg A open)
    r_factor       load change: per-phase multiplier on R (unbalance or load step)
    sensor_offset_A, sensor_gain   current-sensor errors; they affect only the measured copy
    """

    open_switches: frozenset[str] = frozenset()
    r_factor: Triple = (1.0, 1.0, 1.0)
    sensor_offset_A: Triple = (0.0, 0.0, 0.0)
    sensor_gain: Triple = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        unknown = set(self.open_switches) - set(SWITCH_IDS)
        if unknown:
            raise ValueError(f"unknown switch IDs: {sorted(unknown)}")
        object.__setattr__(self, "open_switches", frozenset(self.open_switches))

    @property
    def is_healthy(self) -> bool:
        return self == Condition()


def simulate(
    t0_s: float,
    n_steps: int,
    i0_A: Triple,
    params: PlantParams,
    *,
    m: float | None = None,
    open_switches: frozenset[str] = frozenset(),
    r_ohm: Triple | None = None,
    step0: int | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, Triple]:
    """Integrate the averaged plant with forward Euler.

    Returns (t_s[n], i_A[n, 3], i_final_A). The currents are the true (not measured)
    currents after each step; t_s[s] is the time at the start of step s, as in the
    validated reference implementation. Pass step0 to compute times as (step0 + s) * dt
    without floating-point drift.

    backend: "numba" (compiled), "python" (pure Python), or "auto" (numba when installed).
    Both backends run the same equations and agree to rounding error.
    """
    m = params.m if m is None else m
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"modulation index must be in [0, 1], got {m}")
    if params.emf_V and any(f"{leg}U" in open_switches and f"{leg}L" in open_switches for leg in LEGS):
        raise ValueError("a back-EMF with a fully open leg is not modelled (no diode re-conduction)")
    r = params.r_ohm if r_ohm is None else r_ohm
    steps = np.arange(n_steps, dtype=float)
    ts = (step0 + steps) * params.dt_s if step0 is not None else t0_s + steps * params.dt_s
    angle_rad = params.w1_rad_s * ts
    half_V = params.v_dc_V / 2.0
    v_cmd_V = np.column_stack([m * half_V * np.sin(angle_rad + ph) for ph in PHASE_SHIFTS_RAD])
    if params.emf_V:
        emf_V = np.column_stack([params.emf_V * np.sin(angle_rad + ph - params.emf_lag_rad)
                                 for ph in PHASE_SHIFTS_RAD])
    else:
        emf_V = np.zeros((n_steps, 3))
    up_open = np.array([f"{leg}U" in open_switches for leg in LEGS])
    lo_open = np.array([f"{leg}L" in open_switches for leg in LEGS])
    args = (
        v_cmd_V, emf_V, np.asarray(i0_A, dtype=float), np.asarray(r, dtype=float),
        1.0 / np.asarray(params.l_H, dtype=float), up_open, lo_open,
        params.v_dc_V, params.eps_open_leg_A, params.dt_s,
        params.v_dc_V * params.dead_time_s * params.f_sw_Hz, params.v_ce_sat_V, params.v_f_V,
    )
    use_numba = backend == "numba" or (backend == "auto" and _integrate_numba is not None)
    if use_numba:
        if _integrate_numba is None:
            raise RuntimeError("numba is not installed")
        out = np.empty((n_steps, 3))
        i_final = _integrate_numba(*args, out)
        return ts, out, (float(i_final[0]), float(i_final[1]), float(i_final[2]))
    return (ts, *_integrate_python(*args))


def _integrate_python(
    v_cmd_V: np.ndarray, emf_V: np.ndarray, i0_A: np.ndarray, r_ohm: np.ndarray, inv_l_1_H: np.ndarray,
    up_open: np.ndarray, lo_open: np.ndarray, v_dc_V: float, eps_A: float, dt_s: float,
    dead_V: float, v_ce_V: float, v_f_V: float,
) -> tuple[np.ndarray, Triple]:
    """Pure-Python integration loop (scalar math on lists, the fastest form without numba)."""
    n_steps = len(v_cmd_V)
    half = v_dc_V / 2.0
    v_cmd = v_cmd_V.T.tolist()
    emf = emf_V.T.tolist()
    r = r_ohm.tolist()
    inv_l = inv_l_1_H.tolist()
    up = up_open.tolist()
    lo = lo_open.tolist()
    non_ideal = bool(dead_V or v_ce_V or v_f_V)

    out = np.empty((n_steps, 3))
    i = i0_A.tolist()
    v = [0.0, 0.0, 0.0]
    e = [0.0, 0.0, 0.0]
    conducting = [True, True, True]
    for s in range(n_steps):
        for k in range(3):
            ik = i[k]
            vk = v_cmd[k][s]
            e[k] = emf[k][s]
            conducting[k] = True
            if up[k] and lo[k]:                           # both open: diodes only
                if ik > eps_A:
                    vk = -half - v_f_V
                elif ik < -eps_A:
                    vk = half + v_f_V
                else:
                    conducting[k] = False
            elif up[k] and ik > 0.0:                      # + current via the lower diode
                vk = -half - v_f_V
            elif lo[k] and ik < 0.0:                      # - current via the upper diode
                vk = half + v_f_V
            elif non_ideal:                                # normal PWM operation
                d = 0.5 + vk / v_dc_V
                if ik > 0.0:
                    vk -= dead_V + d * v_ce_V + (1.0 - d) * v_f_V
                elif ik < 0.0:
                    vk += dead_V + (1.0 - d) * v_ce_V + d * v_f_V
            v[k] = vk

        cond = [k for k in range(3) if conducting[k]]
        if len(cond) < 2:
            i = [0.0, 0.0, 0.0]
        else:
            # Neutral voltage from KCL over the conducting phases (reduces to the plain
            # mean of (v - R i - e) when the inductances are equal).
            num = 0.0
            den = 0.0
            for k in cond:
                num += (v[k] - r[k] * i[k] - e[k]) * inv_l[k]
                den += inv_l[k]
            vn = num / den
            new = [0.0, 0.0, 0.0]
            for k in cond:
                new[k] = i[k] + dt_s * (v[k] - vn - r[k] * i[k] - e[k]) * inv_l[k]
            if len(cond) == 2:                             # one phase floating
                x, y = cond
                cm = 0.5 * (new[x] + new[y])
                new[x] -= cm
                new[y] -= cm
            i = new
        out[s, 0] = i[0]
        out[s, 1] = i[1]
        out[s, 2] = i[2]
    return out, (i[0], i[1], i[2])


def _integrate_arrays(
    v_cmd_V: np.ndarray, emf_V: np.ndarray, i0_A: np.ndarray, r_ohm: np.ndarray, inv_l_1_H: np.ndarray,
    up_open: np.ndarray, lo_open: np.ndarray, v_dc_V: float, eps_A: float, dt_s: float,
    dead_V: float, v_ce_V: float, v_f_V: float, out: np.ndarray,
) -> np.ndarray:
    """The same loop as _integrate_python, written on arrays so numba can compile it."""
    half = v_dc_V / 2.0
    non_ideal = dead_V != 0.0 or v_ce_V != 0.0 or v_f_V != 0.0
    i = i0_A.copy()
    v = np.zeros(3)
    new = np.zeros(3)
    conducting = np.ones(3, dtype=np.bool_)
    for s in range(v_cmd_V.shape[0]):
        for k in range(3):
            ik = i[k]
            vk = v_cmd_V[s, k]
            conducting[k] = True
            if up_open[k] and lo_open[k]:
                if ik > eps_A:
                    vk = -half - v_f_V
                elif ik < -eps_A:
                    vk = half + v_f_V
                else:
                    conducting[k] = False
            elif up_open[k] and ik > 0.0:
                vk = -half - v_f_V
            elif lo_open[k] and ik < 0.0:
                vk = half + v_f_V
            elif non_ideal:
                d = 0.5 + vk / v_dc_V
                if ik > 0.0:
                    vk -= dead_V + d * v_ce_V + (1.0 - d) * v_f_V
                elif ik < 0.0:
                    vk += dead_V + (1.0 - d) * v_ce_V + d * v_f_V
            v[k] = vk
        n_cond = 0
        num = 0.0
        den = 0.0
        for k in range(3):
            if conducting[k]:
                n_cond += 1
                num += (v[k] - r_ohm[k] * i[k] - emf_V[s, k]) * inv_l_1_H[k]
                den += inv_l_1_H[k]
        if n_cond < 2:
            for k in range(3):
                i[k] = 0.0
        else:
            vn = num / den
            for k in range(3):
                if conducting[k]:
                    new[k] = i[k] + dt_s * (v[k] - vn - r_ohm[k] * i[k] - emf_V[s, k]) * inv_l_1_H[k]
                else:
                    new[k] = 0.0
            if n_cond == 2:
                cm = 0.5 * (new[0] + new[1] + new[2])   # the floating phase contributes 0
                for k in range(3):
                    if conducting[k]:
                        new[k] -= cm
            for k in range(3):
                i[k] = new[k]
        for k in range(3):
            out[s, k] = i[k]
    return i


try:
    if os.environ.get("FAULTTWIN_PURE_PYTHON"):
        raise ImportError("pure Python requested")
    from numba import njit

    _integrate_numba = njit(cache=True)(_integrate_arrays)
except ImportError:
    _integrate_numba = None


def measure(i_true_A: np.ndarray, condition: Condition, noise_sigma_A: float = 0.0,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply sensor gain, offset and Gaussian noise to the true currents."""
    i_meas = i_true_A * np.asarray(condition.sensor_gain) + np.asarray(condition.sensor_offset_A)
    if noise_sigma_A > 0.0:
        rng = rng or np.random.default_rng()
        i_meas = i_meas + rng.normal(0.0, noise_sigma_A, size=i_meas.shape)
    return i_meas


@dataclass
class PlantChunk:
    """Samples produced by one call to InverterPlant.run_cycles."""

    t_s: np.ndarray
    i_true_A: np.ndarray
    i_meas_A: np.ndarray
    m: float
    first_cycle: int
    n_cycles: int


@dataclass
class InverterPlant:
    """Stateful plant, streamed in whole fundamental cycles.

    The plant carries effects the digital twin does not know in advance: measurement
    noise and per-session parameter mismatch (R and L drawn within +/- mismatch_frac).
    """

    params: PlantParams = field(default_factory=PlantParams)
    seed: int | None = None
    noise_sigma_A: float = 0.0
    mismatch_frac: float = 0.0
    condition: Condition = field(default_factory=Condition)

    def __post_init__(self) -> None:
        self.nominal = self.params
        self._rng = np.random.default_rng(self.seed)
        if self.mismatch_frac > 0.0:
            f = self.mismatch_frac
            r = tuple(float(x * self._rng.uniform(1 - f, 1 + f)) for x in self.nominal.r_ohm)
            l_ = tuple(float(x * self._rng.uniform(1 - f, 1 + f)) for x in self.nominal.l_H)
            self.params = replace(self.nominal, r_ohm=r, l_H=l_)
        self.m = self.params.m
        self.step = 0
        self.cycle = 0
        self.i_A: Triple = (0.0, 0.0, 0.0)
        self.stopped = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, seed: int | None = None, realistic: bool = True) -> InverterPlant:
        """Plant from config.yaml; realistic=True adds the configured noise and parameter mismatch."""
        params = PlantParams.from_config(cfg["plant"])
        if not realistic:
            return cls(params, seed=seed)
        return cls(params, seed=seed,
                   noise_sigma_A=cfg["plant"]["noise_sigma_frac"] * rated_peak_current_A(params),
                   mismatch_frac=cfg["plant"]["param_mismatch_frac"])

    @property
    def t_s(self) -> float:
        return self.step * self.params.dt_s

    def set_condition(self, condition: Condition) -> None:
        self.condition = condition

    def set_modulation_index(self, m: float) -> None:
        if not 0.0 <= m <= 1.0:
            raise ValueError(f"modulation index must be in [0, 1], got {m}")
        self.m = float(m)

    def stop(self) -> None:
        """Stop by applying the zero voltage vector (m = 0).

        The load current then decays through the bridge. This de-energises the default
        RL load; with a back-EMF it would keep a short-circuit current flowing.
        """
        self.stopped = True

    def resume(self) -> None:
        self.stopped = False

    def run_cycles(self, n_cycles: int = 1) -> PlantChunk:
        n_steps = n_cycles * self.params.steps_per_cycle
        r = tuple(x * f for x, f in zip(self.params.r_ohm, self.condition.r_factor))
        m = 0.0 if self.stopped else self.m
        ts, i_true, self.i_A = simulate(
            0.0, n_steps, self.i_A, self.params, m=m, open_switches=self.condition.open_switches,
            r_ohm=r, step0=self.step,
        )
        i_meas = measure(i_true, self.condition, self.noise_sigma_A, self._rng)
        chunk = PlantChunk(ts, i_true, i_meas, m, self.cycle, n_cycles)
        self.step += n_steps
        self.cycle += n_cycles
        return chunk
