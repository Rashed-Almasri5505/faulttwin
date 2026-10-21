"""Calibrate and check the digital twin on the calibration seeds.

For each case the plant runs with noise and parameter mismatch (seeded) and the twin ranks
every hypothesis with the settings in config.yaml. The script reports, per case and seed,
the rank of the true cause and its margin over the best wrong cause (positive = ranked
first), and, for healthy plants, the largest fitted look-alike sizes per family, which
must stay below twin.min_effect.

Usage:  python scripts/calibrate_twin.py [--seeds 9001 9002 ...]
Only calibration seeds (config.yaml calibration_seeds) may be used here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faulttwin.config import load_config  # noqa: E402
from faulttwin.hypotheses import DEFAULT_HYPOTHESES, SPECS, TWO_SWITCH_HYPOTHESES, Hypothesis  # noqa: E402
from faulttwin.plant import Condition, InverterPlant  # noqa: E402
from faulttwin.twin import DigitalTwin, rank_of  # noqa: E402

CASES: list[tuple[str, Condition, float, Hypothesis]] = [
    ("healthy m=0.8", Condition(), 0.8, Hypothesis.healthy),
    ("healthy m=0.3", Condition(), 0.3, Hypothesis.healthy),
    *[(f"{s} open", Condition(open_switches={s}), 0.8, Hypothesis(f"{s}_open"))
      for s in ("AU", "AL", "BU", "BL", "CU", "CL")],
    ("AU open m=0.35", Condition(open_switches={"AU"}), 0.35, Hypothesis.AU_open),
    ("CL open m=0.35", Condition(open_switches={"CL"}), 0.35, Hypothesis.CL_open),
    *[(f"leg {leg} open", Condition(open_switches={f"{leg}U", f"{leg}L"}), 0.8, Hypothesis(f"leg_{leg}_open"))
      for leg in "ABC"],
    ("AU+BU open", Condition(open_switches={"AU", "BU"}), 0.8, Hypothesis.AU_BU_open),
    ("AL+CU open", Condition(open_switches={"AL", "CU"}), 0.8, Hypothesis.AL_CU_open),
    ("offset A +2 A", Condition(sensor_offset_A=(2.0, 0, 0)), 0.8, Hypothesis.sensor_offset_A),
    ("offset B -1.5 A", Condition(sensor_offset_A=(0, -1.5, 0)), 0.8, Hypothesis.sensor_offset_B),
    ("offset C +3 A m=0.3", Condition(sensor_offset_A=(0, 0, 3.0)), 0.3, Hypothesis.sensor_offset_C),
    ("gain B 0.85", Condition(sensor_gain=(1, 0.85, 1)), 0.8, Hypothesis.sensor_gain_B),
    ("gain A 1.15", Condition(sensor_gain=(1.15, 1, 1)), 0.8, Hypothesis.sensor_gain_A),
    ("gain C 0.9 m=0.3", Condition(sensor_gain=(1, 1, 0.9)), 0.3, Hypothesis.sensor_gain_C),
    ("R_A x1.3", Condition(r_factor=(1.3, 1, 1)), 0.8, Hypothesis.load_unbalance_A),
    ("R_C x0.7", Condition(r_factor=(1, 1, 0.7)), 0.8, Hypothesis.load_unbalance_C),
    ("load step x1.3", Condition(r_factor=(1.3, 1.3, 1.3)), 0.8, Hypothesis.load_step),
    ("load step x0.7", Condition(r_factor=(0.7, 0.7, 0.7)), 0.8, Hypothesis.load_step),
    # Small look-alikes at the edge of the CLAUDE.md ranges.
    ("offset A +1 A", Condition(sensor_offset_A=(1.0, 0, 0)), 0.8, Hypothesis.sensor_offset_A),
    ("offset C -1 A m=0.3", Condition(sensor_offset_A=(0, 0, -1.0)), 0.3, Hypothesis.sensor_offset_C),
    ("gain A 0.9", Condition(sensor_gain=(0.9, 1, 1)), 0.8, Hypothesis.sensor_gain_A),
    ("gain B 1.1", Condition(sensor_gain=(1, 1.1, 1)), 0.8, Hypothesis.sensor_gain_B),
    ("R_B x1.15", Condition(r_factor=(1, 1.15, 1)), 0.8, Hypothesis.load_unbalance_B),
    ("R_A x0.85 m=0.3", Condition(r_factor=(0.85, 1, 1)), 0.3, Hypothesis.load_unbalance_A),
    ("R_A x0.86 m=0.85", Condition(r_factor=(0.86, 1, 1)), 0.85, Hypothesis.load_unbalance_A),
    ("load step x1.15", Condition(r_factor=(1.15, 1.15, 1.15)), 0.8, Hypothesis.load_step),
]
# Single-phase unbalances of about 15 % are at the edge of the plant's +/-5 % R/L tolerance and may be
# called "healthy" (a known limitation, see PROGRESS.md). They must never be called a converter fault.
KNOWN_LIMITATION = {"R_B x1.15", "R_A x0.85 m=0.3", "R_A x0.86 m=0.85"}


def run_case(cfg: dict, condition: Condition, m: float, seed: int, twin: DigitalTwin, two_switch: bool):
    plant = InverterPlant.from_config(cfg, seed=seed)
    plant.set_modulation_index(m)
    plant.run_cycles(15)
    plant.set_condition(condition)
    window = twin.window_from_chunk(plant.run_cycles(15))
    hyps = list(DEFAULT_HYPOTHESES) + (list(TWO_SWITCH_HYPOTHESES) if two_switch else [])
    return twin.simulate_hypotheses(hyps, window)


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg["calibration_seeds"])
    args = parser.parse_args()
    if not set(args.seeds) <= set(cfg["calibration_seeds"]):
        print("Use calibration seeds only (config.yaml calibration_seeds).", file=sys.stderr)
        return 2

    twin = DigitalTwin.from_config(cfg)
    settings = twin.settings
    fault_vs_not_errors = 0
    print(f"{'case':<22} {'seed':>5} {'rank':>4} {'true err %':>10} {'best wrong':<24} {'err %':>7} {'margin':>7} "
          f"{'fit':>12} {'s':>5}")
    worst: dict[str, tuple[int, float]] = {}
    healthy_size: dict[str, float] = {}
    for name, condition, m, truth in CASES:
        for seed in args.seeds:
            t0 = time.perf_counter()
            results = run_case(cfg, condition, m, seed, twin, two_switch=SPECS[truth].family == "two_switch_open")
            rank = rank_of(results, truth)
            true = results[rank - 1]
            wrong = next(r for r in results if r.hypothesis != truth)
            margin = wrong.match_error_pct - true.match_error_pct
            fit = f"{true.fitted_param.value:+.3f} {true.fitted_param.unit}" if true.fitted_param else ""
            wrong_name = f"{wrong.hypothesis}{' (tol)' if wrong.within_tolerance else ''}"
            prev_rank, prev_margin = worst.get(name, (1, float("inf")))
            worst[name] = (max(prev_rank, rank), min(prev_margin, margin))
            fault_vs_not_errors += results[0].spec.converter_fault != SPECS[truth].converter_fault
            if truth == Hypothesis.healthy:
                for r in results:
                    if r.fitted_param:
                        size = abs(r.fitted_param.value - r.spec.fit.neutral)   # type: ignore[union-attr]
                        healthy_size[r.spec.family] = max(healthy_size.get(r.spec.family, 0.0), size)
            print(f"{name:<22} {seed:>5} {rank:>4} {true.match_error_pct:>10.2f} {wrong_name:<24} "
                  f"{wrong.match_error_pct:>7.2f} {margin:>7.2f} {fit:>12} {time.perf_counter() - t0:>5.2f}", flush=True)
    print()
    print("Per case: worst rank of the true cause, and its smallest margin (> 0 means ranked first):")
    unexpected = []
    for name, (rank, margin) in worst.items():
        note = " (known limitation)" if name in KNOWN_LIMITATION and rank > 1 else ""
        if rank > 1 and name not in KNOWN_LIMITATION:
            unexpected.append(name)
        print(f"  {name:<22} worst rank {rank}  margin {margin:+.2f}{note}")
    print("Healthy plants: largest fitted look-alike size vs twin.min_effect:")
    band_violations = []
    for family, size in healthy_size.items():
        band = getattr(settings.min_effect, family)
        if size >= band:
            band_violations.append(family)
        print(f"  {family:<16} {size:.3f}  (min_effect {band})")
    print(f"Converter-fault-or-not errors: {fault_vs_not_errors}")
    print(f"Unexpected misrankings: {unexpected or 'none'}; band violations: {band_violations or 'none'}")
    return 1 if unexpected or band_violations or fault_vs_not_errors else 0


if __name__ == "__main__":
    sys.exit(main())
