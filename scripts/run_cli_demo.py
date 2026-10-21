"""Run one FaultTwin diagnosis from the command line.

The plant runs healthy, the scenario is injected, the edge monitor raises an event, Lightning
triages it, Super investigates with tools (twin, active test, Tavily, interlock), and Ultra
writes the work order. Prints each step, the diagnosis, the work order and the cost, and
saves the trace to data/traces/.

Usage:  python scripts/run_cli_demo.py --scenario AU_open
        python scripts/run_cli_demo.py --scenario sensor_offset_A --seed 4242
        python scripts/run_cli_demo.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faulttwin.agent import DiagnosisAgent, Trace  # noqa: E402
from faulttwin.config import REPO_ROOT, load_config  # noqa: E402
from faulttwin.llm import BudgetExceeded  # noqa: E402
from faulttwin.report import write_work_order  # noqa: E402
from faulttwin.runtime import build_session, run_to_event  # noqa: E402
from faulttwin.scenarios import SCENARIOS  # noqa: E402

TRACE_DIR = REPO_ROOT / "data" / "traces"


def print_step(step: dict) -> None:
    kind = step["kind"]
    if kind == "triage":
        r = step["result"]
        print(f"  [{step.get('badge', '?'):>9}] triage: {r['family']}, urgency {r['urgency']}, "
              f"escalate={r['escalate']}  ({step.get('latency_s', 0)} s)")
    elif kind == "llm":
        calls = ", ".join(c["name"] for c in step.get("tool_calls", [])) or "(text)"
        print(f"  [{step['badge']:>9}] {calls}   {step['prompt_tokens']}+{step['completion_tokens']} tok, "
              f"{step['latency_s']} s, ${step['cost_usd']:.4f}")
    elif kind == "tool":
        print(f"  [{'tool':>9}] {step['tool']}: {step['result_summary']}")
    elif kind == "note":
        print(f"  [{'note':>9}] {step['text']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="AU_open")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--no-report", action="store_true", help="skip the work order")
    parser.add_argument("--wait-for-ultra", action="store_true", help="wait for Ultra's full work order")
    parser.add_argument("--list", action="store_true", help="list the scenarios")
    args = parser.parse_args()
    if args.list:
        for name, s in SCENARIOS.items():
            print(f"{name:<18} {s.title}")
        return 0
    if args.scenario not in SCENARIOS:
        print(f"Unknown scenario {args.scenario!r}; use --list.", file=sys.stderr)
        return 2

    cfg = load_config()
    scenario = SCENARIOS[args.scenario]
    session = build_session(cfg, seed=args.seed)
    if not session.llm.available:
        print("NEBIUS_API_KEY is not set: the agent cannot run live.", file=sys.stderr)
        return 2
    try:
        missing = session.llm.missing_models()
    except Exception as exc:  # noqa: BLE001 - the listing is a check, not a requirement
        missing = []
        print(f"Warning: could not list Token Factory models ({type(exc).__name__}).")
    if missing:
        print(f"Warning: Token Factory does not list these configured models: {missing}")
    if not session.nameplate.igbt_module_part_number:
        print("Warning: no verified nameplate in config.yaml; datasheet lookups will be refused.")
    print(f"Scenario: {scenario.title}  (seed {args.seed}, m = {scenario.m})")
    t0 = time.perf_counter()
    event = run_to_event(session, scenario)
    if event is None:
        print("The edge monitor raised no medium or high severity event: nothing to diagnose.")
        return 0
    print(f"Monitor event at cycle {event.cycle_index}: [{event.severity}] {event.type}: {event.message}")

    trace = Trace(scenario=scenario.name)
    trace.event = {"type": event.type, "severity": event.severity, "cycle": event.cycle_index, "message": event.message}
    trace.hidden_truth = {"scenario": scenario.name, "truth": scenario.truth.value, "title": scenario.title}
    agent = DiagnosisAgent(session, session.llm, cfg, on_step=print_step)
    try:
        final = agent.run(event, trace)
    except BudgetExceeded as exc:
        print(f"Budget cap reached: {exc}")
        return 3

    print()
    print(f"Diagnosis: {final.label}  (confidence {final.confidence:.2f}, twin error {final.twin_match_error_pct} %)")
    print(f"Truth:     {scenario.truth.value}  -> {'CORRECT' if final.label == scenario.truth.value else 'WRONG'}")
    print(f"Summary:   {final.summary}")
    for v in session.verdicts:
        print(f"Interlock: {v.request}: {'approved' if v.approved else 'rejected'} - {v.reason}")
    print(f"Citations: {len(final.citations)}")
    for c in final.citations:
        src = session.sources.get(c["url"])
        print(f"  - {c['title'][:80]}  {c['url']}  [{src.label if src else '?'}]")
    if trace.validation_notes:
        print("Checks:    " + "; ".join(trace.validation_notes))

    if not args.no_report:
        asset = session.nameplate.inverter_rating or "two-level inverter"
        work_order = write_work_order(session.llm, cfg, trace.final or {}, session, asset,
                                      wait_for_ultra=args.wait_for_ultra)
        trace.work_order = work_order.as_dict()
        print()
        print(f"--- Work order ({work_order.model or 'template'}, {work_order.latency_s} s, Ultra {work_order.ultra_status}) ---")
        print(work_order.markdown)
        for note in work_order.notes:
            print(f"(work order note: {note})")

    totals = trace.totals
    totals["llm_cost_usd_incl_report"] = round(session.llm.spent_usd(), 5)
    totals["wall_s_incl_report"] = round(time.perf_counter() - t0, 2)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TRACE_DIR / f"{stamp}_{scenario.name}.json"
    path.write_text(trace.to_json(), encoding="utf-8")
    print()
    print(f"Cost: ${totals['llm_cost_usd_incl_report']:.4f} (agent ${totals['llm_cost_usd']:.4f}), "
          f"tokens {totals['prompt_tokens']}+{totals['completion_tokens']}, tool calls {totals['tool_calls']}, "
          f"Tavily {totals['tavily_live_calls']} live calls / {totals['tavily_credits']:g} credits, "
          f"time {totals['wall_s_incl_report']} s")
    print(f"Trace: {path.relative_to(REPO_ROOT)}")
    return 0 if final.label == scenario.truth.value else 1


if __name__ == "__main__":
    sys.exit(main())
