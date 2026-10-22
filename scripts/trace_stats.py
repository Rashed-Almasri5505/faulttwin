"""Per-model numbers from saved agent traces (for the README and the feedback data).

Reads data/traces/*.json (local development traces, gitignored), eval/results/traces/*.json (the
evaluation) and eval/results/dev/traces/*.json (development-seed runs; both committed), or the files
and wildcard patterns given, and prints, per model:
calls, median and 95th-percentile latency, median tokens, cost; and for the agent: tool calls,
invalid tool arguments, finalize rejections (and how many broke the answer schema), JSON-fallback
use, and the diagnosis outcome per trace.

Usage:  python scripts/trace_stats.py [trace.json | "dir/*.json" ...]
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faulttwin.config import REPO_ROOT  # noqa: E402


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def trace_paths(args: list[str]) -> list[Path]:
    """The files named, with wildcards expanded here (PowerShell passes them through unexpanded)."""
    explicit = bool(args)
    if not args:
        args = [str(REPO_ROOT / "data" / "traces" / "*.json"), str(REPO_ROOT / "eval" / "results" / "traces" / "*.json"),
                str(REPO_ROOT / "eval" / "results" / "dev" / "traces" / "*.json")]
    paths: list[Path] = []
    for a in args:
        if not any(c in a for c in "*?["):
            paths.append(Path(a))
            continue
        found = sorted(Path(p) for p in glob.glob(a))
        if not found and explicit:
            print(f"warning: no file matches {a}", file=sys.stderr)
        paths += found
    return paths


def main() -> int:
    paths = trace_paths(sys.argv[1:])
    if not paths:
        print("no traces found")
        return 1
    by_model: dict[str, list[dict]] = defaultdict(list)
    tool_steps = by_code = invalid = rejected = bad_schema = fallback = 0
    rows = []
    for path in paths:
        t = json.loads(path.read_text(encoding="utf-8"))
        for s in t["steps"]:
            if s["kind"] in ("llm", "triage") and "prompt_tokens" in s:
                by_model[s["model"]].append(s)
            if s["kind"] == "tool":
                if s["result_summary"].startswith("background evidence"):   # issued by code, not by the model
                    by_code += 1
                    continue
                tool_steps += 1
                invalid += "error" in (s.get("result") or {}) and "invalid arguments" in s["result"]["error"]
                if s.get("tool") == "finalize_diagnosis" and s["result_summary"].startswith("rejected"):
                    rejected += 1
                    bad_schema += "final answer is not valid" in json.dumps(s.get("result"))
                fallback += bool(s.get("via_json_fallback"))
        for call in (t.get("work_order") or {}).get("calls", []):
            by_model[call["model"]].append(call)
        truth = (t.get("hidden_truth") or {}).get("truth")
        label = (t.get("final") or {}).get("label")
        rows.append((path.name, truth, label, t["totals"].get("tool_calls"), t["totals"].get("llm_cost_usd_incl_report",
                     t["totals"].get("llm_cost_usd")), t["totals"].get("wall_s_incl_report", t["totals"].get("wall_s"))))
    print(f"{len(paths)} traces\n")
    print("| Model | Calls | Median latency (s) | p95 latency (s) | Median prompt tok | Median completion tok | Cost (USD) |")
    print("|---|---|---|---|---|---|---|")
    for model, calls in sorted(by_model.items()):
        lat = [c["latency_s"] for c in calls]
        print(f"| `{model}` | {len(calls)} | {statistics.median(lat):.2f} | {pct(lat, 0.95):.2f} | "
              f"{statistics.median(c['prompt_tokens'] for c in calls):.0f} | "
              f"{statistics.median(c['completion_tokens'] for c in calls):.0f} | "
              f"{sum(c['cost_usd'] for c in calls):.4f} |")
    print(f"\nAgent tool calls: {tool_steps} (plus {by_code} background steps issued by code); invalid arguments: "
          f"{invalid}; finalize rejections: {rejected} ({bad_schema} broke the answer schema); calls outside the "
          f"declared schema: {invalid + bad_schema}; JSON fallback used: {fallback}")
    print("\n| Trace | Truth | Diagnosis | Tool calls | Cost (USD) | Time (s) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
