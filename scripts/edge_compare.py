"""Compare the edge node's features and alarms with the Python monitor's, cycle by cycle, on a connected board.

    python scripts/edge_compare.py                  # finds the board; writes eval/results/edge_compare.md
    python scripts/edge_compare.py --port COM5 --scenarios AU_open sensor_offset_A

Each scenario runs on a fresh plant session: healthy cycles, the fault, a modulation-index step and back
(as in the agent's active test), an operating-point notice, then a monitor reset with the fault still in the
plant (so the alarms fire again). The board receives every
cycle over USB serial and reports its features; the laptop computes the same cycle with the Python monitor
(full rate, float64) and with edge.BoardModel (the firmware's arithmetic in Python). Pass: |delta mu| < 0.01
on every cycle and the same alarms on the same cycles. No model or web calls are made.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FAULTTWIN_OFFLINE", "1")          # no model or Tavily client with real keys

from faulttwin import edge  # noqa: E402
from faulttwin.config import REPO_ROOT, load_config  # noqa: E402
from faulttwin.runtime import build_session  # noqa: E402
from faulttwin.scenarios import SCENARIOS  # noqa: E402

DEFAULT_SCENARIOS = ["healthy", "AU_open", "BL_open", "leg_C_open", "AU_BU_open", "sensor_offset_A", "sensor_offset_C",
                     "sensor_gain_A", "sensor_gain_B", "load_unbalance_A", "load_step", "low_load"]
SEED = 6301          # not a calibration or evaluation seed
MU_LIMIT = 0.01


def drive(s: Any, scenario: str) -> None:
    sc = SCENARIOS[scenario]
    with s.lock:
        s.plant.set_modulation_index(sc.m)
        s.advance(10)
        s.plant.set_condition(sc.condition)
        s.advance(15)
        s.plant.set_modulation_index(max(0.3, sc.m - 0.3))
        s.monitor.expect_operating_point_change()
        s.advance(6)
        s.plant.set_modulation_index(sc.m)
        s.monitor.expect_operating_point_change()
        s.advance(6)
        s.monitor.expect_operating_point_change()
        s.advance(4)
        s.monitor.reset()
        s.advance(8)


def compare(link: edge.EdgeLink, scenario: str, cfg: dict[str, Any], seed: int = SEED,
            timeout_s: float = 15.0) -> dict[str, Any]:
    s = build_session(cfg, seed=seed)
    python: dict[int, Any] = {}
    model = edge.BoardModel(s.monitor.th)
    seen: list[tuple[int, int]] = []
    oracle: dict[int, edge.BoardFeatures] = {}
    oracle_events: list[edge.BoardEvent] = []

    def on_cycle(chunk: Any, status: Any) -> None:
        now = (s.monitor.resets, s.monitor.operating_point_notices)
        flags = (edge.FLAG_RESET if not seen or now[0] != seen[-1][0] else 0) | \
                (edge.FLAG_OP_CHANGE if seen and now[1] != seen[-1][1] else 0)
        seen.append(now)
        f, ev = model.process(chunk.first_cycle, chunk.m, flags,
                              edge.quantize(edge.decimate(chunk.i_meas_A)) * edge.SAMPLE_LSB_A)
        oracle[chunk.first_cycle] = f
        oracle_events.extend(ev)
        python[chunk.first_cycle] = status.features

    _, first_event = link.events_since(0)
    errors_before = len(link.rx.errors)
    link.attach(s)                         # its listener runs first, so both see the same monitor counters
    s.cycle_listeners.append(on_cycle)
    t0 = time.perf_counter()
    drive(s, scenario)
    last = s.cycle - 1
    deadline = time.time() + timeout_s
    while link.features_for(last) is None:
        if time.time() > deadline or link.dead:
            raise RuntimeError(f"{scenario}: the board did not report cycle {last}")
        time.sleep(0.02)
    time.sleep(0.2)                        # any event lines printed after the last feature line
    wall_s = time.perf_counter() - t0
    link.detach()
    board_events, _ = link.events_since(first_event)

    d_mu = d_mu_model = d_dc = d_w1 = d_ar = d_is = 0.0
    compared = missing = ready_mismatch = 0
    for cycle, p in python.items():
        b = link.features_for(cycle)
        if b is None:
            missing += 1
            continue
        if b.ready != (p is not None):
            ready_mismatch += 1
        if p is None or not b.ready:
            continue
        compared += 1
        d_mu = max(d_mu, float(np.max(np.abs(np.subtract(b.mu, p.mu)))))
        d_mu_model = max(d_mu_model, float(np.max(np.abs(np.subtract(b.mu, oracle[cycle].mu)))))
        d_dc = max(d_dc, abs(b.r_kcl_dc - p.r_kcl_dc))
        d_w1 = max(d_w1, abs(b.r_kcl_w1 - p.r_kcl_w1))
        d_ar = max(d_ar, float(np.max(np.abs(np.subtract(b.amp_ratio, p.amp_ratio)))))
        d_is = max(d_is, abs(b.i_s_A - p.i_s_A) / p.i_s_A)
    alarms_python = [(e.cycle_index, e.type, e.severity) for e in s.events]
    alarms_board = [(e.cycle, e.type, e.severity) for e in board_events]
    matrix_board = list(link.rx.shown)                     # cleared when this session was attached
    matrix_model = [e.shown for e in oracle_events if e.shown]
    ok = (missing == 0 and ready_mismatch == 0 and compared > 0 and d_mu < MU_LIMIT
          and alarms_board == alarms_python and matrix_board == matrix_model and len(link.rx.errors) == errors_before)
    return {"scenario": scenario, "cycles": len(python), "compared": compared, "missing": missing,
            "ready_mismatch": ready_mismatch, "max_abs_d_mu": d_mu, "max_abs_d_mu_vs_model": d_mu_model,
            "max_abs_d_r_kcl_dc": d_dc, "max_abs_d_r_kcl_w1": d_w1, "max_abs_d_amp_ratio": d_ar,
            "max_rel_d_i_s": d_is, "alarms_python": alarms_python, "alarms_board": alarms_board,
            "alarms_match": alarms_board == alarms_python, "matrix_board": matrix_board, "matrix_model": matrix_model,
            "matrix_match": matrix_board == matrix_model, "board_errors": list(link.rx.errors)[errors_before:],
            "wall_s": round(wall_s, 2), "pass": ok}


def _markdown(rows: list[dict[str, Any]], link: edge.EdgeLink, stamp: str) -> str:
    lines = [
        "# Edge node vs Python monitor",
        "",
        f"Measured {stamp} on {link.board} ({link.port}), firmware `{link.rx.heartbeat.split()[2]}`, "
        f"by `python scripts/edge_compare.py`.",
        "",
        "The board receives each simulated cycle as 100 block-averaged int16 samples per phase and computes",
        "the monitor's features in float32; the Python monitor uses all 1000 samples per cycle in float64.",
        f"Pass: |Δμ| < {MU_LIMIT} on every compared cycle, the same alarms on the same cycles, and the LED matrix",
        "showing the phase and arrow that the firmware's rule, run on the laptop (edge.matrix_fault), gives.",
        "",
        "| Scenario | Cycles compared | max \\|Δμ\\| | max \\|Δr_KCL,dc\\| | max \\|Δr_KCL,ω₁\\| | max \\|Δ amplitude ratio\\| "
        "| Alarms (board = Python) | Matrix (board = laptop) | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        alarms = ", ".join(sorted({a[1] for a in r["alarms_python"]})) or "none"
        matrix = " ".join(r["matrix_board"]) or "check mark only"
        lines.append(f"| {r['scenario']} | {r['compared']} | {r['max_abs_d_mu']:.4f} | {r['max_abs_d_r_kcl_dc']:.4f} "
                     f"| {r['max_abs_d_r_kcl_w1']:.4f} | {r['max_abs_d_amp_ratio']:.4f} "
                     f"| {'same' if r['alarms_match'] else 'DIFFERENT'}: {alarms} "
                     f"| {'same' if r['matrix_match'] else 'DIFFERENT'}: {matrix} | {'pass' if r['pass'] else 'FAIL'} |")
    lines += ["", f"{sum(r['pass'] for r in rows)} of {len(rows)} scenarios pass. "
              f"Largest |Δμ| against the firmware's arithmetic run on the laptop (edge.BoardModel): "
              f"{max(r['max_abs_d_mu_vs_model'] for r in rows):.1e}.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port (default: find the board)")
    ap.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS, choices=sorted(SCENARIOS))
    ap.add_argument("--no-write", action="store_true", help="print only; do not write eval/results/edge_compare.*")
    args = ap.parse_args()
    link = edge.EdgeLink.open(args.port, "edge node") if args.port else edge.find_board()
    if link is None:
        print("No FaultTwin edge node found. Flash firmware/faulttwin_edge and connect the board.")
        return 2
    cfg = load_config()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    try:
        print(f"{link.board} on {link.port}: {link.rx.heartbeat}")
        for name in args.scenarios:
            r = compare(link, name, cfg)
            rows.append(r)
            print(f"{name:20s} cycles {r['compared']:3d}  max|dmu| {r['max_abs_d_mu']:.4f}  "
                  f"vs model {r['max_abs_d_mu_vs_model']:.1e}  alarms {'same' if r['alarms_match'] else 'DIFFERENT'}"
                  f"  {'pass' if r['pass'] else 'FAIL'}")
            if not r["alarms_match"]:
                print(f"    python {r['alarms_python']}\n    board  {r['alarms_board']}")
            if r["board_errors"]:
                print(f"    board errors: {r['board_errors']}")
    finally:
        link.close()
    if not args.no_write:
        out = REPO_ROOT / "eval" / "results"
        (out / "edge_compare.md").write_text(_markdown(rows, link, stamp), encoding="utf-8")
        (out / "edge_compare.json").write_text(json.dumps({"measured": stamp, "board": link.board,
                                                           "heartbeat": link.rx.heartbeat, "rows": rows}, indent=1),
                                               encoding="utf-8")
        print(f"wrote {out / 'edge_compare.md'}")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
