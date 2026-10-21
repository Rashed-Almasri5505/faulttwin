"""Triage accuracy and latency per model on the steady-state features saved in data/traces/ (live, about $0.003).

Replays the triage input of every saved trace against Lightning and Nano (reasoning off) with the current
prompt, twice each, and prints the share of correct families (switch or leg fault / sensor error /
load change) and the latencies. Used for the Phase 4 triage numbers in the feedback data.

Usage:  python scripts/triage_study.py
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faulttwin.agent import TRIAGE_PROMPT, TRIAGE_SCHEMA, triage_checks  # noqa: E402
from faulttwin.config import load_config  # noqa: E402
from faulttwin.hypotheses import SPECS, Hypothesis  # noqa: E402
from faulttwin.runtime import build_session  # noqa: E402

WANT = {"switch_open": "switch_or_leg_fault", "leg_open": "switch_or_leg_fault", "two_switch_open": "switch_or_leg_fault",
        "sensor_offset": "sensor_error", "sensor_gain": "sensor_error", "load_unbalance": "load_change",
        "load_step": "load_change"}
MODES = [("triage", "off"), ("triage", "off"), ("triage_fallback", "off"), ("triage_fallback", "off")]


def main() -> int:
    cfg = load_config()
    s = build_session(cfg, seed=1)
    if not s.llm.available:
        print("NEBIUS_API_KEY is not set.", file=sys.stderr)
        return 2
    labels = {h.value for h in Hypothesis}
    cases = []
    for p in sorted((ROOT / "data" / "traces").glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        truth = (t.get("hidden_truth") or {}).get("truth") or t.get("scenario")
        if truth in labels and t.get("first_message"):
            cases.append((truth, t["event"], t["first_message"]["latest_features"]))
    print(f"{len(cases)} saved triage inputs")
    for role, thinking in MODES:
        ok, lat, errors = collections.Counter(), [], 0
        for truth, ev, f in cases:
            content = json.dumps({"event": ev["type"], "severity": ev["severity"],
                                  "rules": triage_checks(f, s.monitor.th)})
            msgs = [{"role": "system", "content": TRIAGE_PROMPT}, {"role": "user", "content": content}]
            want = WANT[SPECS[Hypothesis(truth)].family]
            try:
                res = s.llm.chat(role, msgs, label="triage-study", thinking=thinking, max_tokens=1500, timeout_s=30.0,
                                 response_format={"type": "json_schema", "json_schema": TRIAGE_SCHEMA}, max_retries=1)
                family = json.loads(res.content)["family"]
                lat.append(res.call.latency_s)
            except Exception as exc:  # noqa: BLE001 - a measurement: count it and go on
                errors += 1
                family = f"error {type(exc).__name__}"
            ok[family == want] += 1
            if family != want:
                print(f"  {role}/{thinking}: {truth} -> {family}")
        med = f"{statistics.median(lat):.2f} s" if lat else "n/a"
        print(f"{role}/{thinking}: correct {ok[True]}/{len(cases)}, errors {errors}, median latency {med}")
    print(f"cost ${s.llm.spent_usd():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
