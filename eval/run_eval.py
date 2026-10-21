"""FaultTwin evaluation (CLAUDE.md section 8): methods A, B and C on 24 scenarios, and C-T on the switch faults.

A    Textbook baseline: if the largest |mu| is above 0.10, its leg is faulty (mu < 0: upper switch open,
     mu > 0: lower switch open); otherwise "healthy". Deliberately naive.
B    Nemotron 3 Super given only the edge monitor's features (no tools, one call).
C    The full FaultTwin pipeline as the app runs it: the edge monitor must raise a medium or high alarm;
     then Lightning triage, the Super agent with the twin, the active test, Tavily and the interlock, and the
     work order. No alarm means no diagnosis: C then reports "healthy" (and spends nothing).
C-T  The full agent without any web tools (Tavily ablation), on the 8 single-switch faults.
Twin The twin's match error for the true cause on a plant with device voltage drops, with and without the
     drops from the datasheet (no model calls).

Every method sees the same plant data: one session per scenario, seeded from eval/scenarios.yaml. Results
are saved after each scenario (eval/results/results.json), so an interrupted run resumes where it stopped.
Live calls go to Nebius Token Factory and Tavily and are booked in the spend ledger.

Usage:  python eval/run_eval.py --estimate          print the cost estimate and stop
        python eval/run_eval.py                     run everything not yet in results.json, then report
        python eval/run_eval.py --only AU_open      run some scenarios again
        python eval/run_eval.py --dev --only ...    development seeds (6700 + n), results in eval/results/dev/
        python eval/run_eval.py --report            rebuild results.md and the confusion matrix only
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from faulttwin.agent import SIGNATURES, Trace  # noqa: E402
from faulttwin.config import load_config  # noqa: E402
from faulttwin.hypotheses import SPECS, Hypothesis  # noqa: E402
from faulttwin.llm import LLMError, SpendLedger, TokenFactory  # noqa: E402
from faulttwin.monitor import Event, baseline_from_mu  # noqa: E402
from faulttwin.plant import Condition  # noqa: E402
from faulttwin.runtime import build_session, run_diagnosis  # noqa: E402
from faulttwin.scenarios import Scenario  # noqa: E402
from faulttwin.twin import WindowNotDiagnosable  # noqa: E402
from faulttwin.web import TavilyWeb  # noqa: E402

HERE = Path(__file__).resolve().parent
SCENARIOS_PATH = HERE / "scenarios.yaml"
HEALTHY_CYCLES = 10       # healthy plant before the condition is applied
WATCH_CYCLES = 25         # cycles the monitor gets to raise a medium or high alarm
DEV_SEED_BASE = 6700      # development seeds (a Phase 3-4 range), never the evaluation's 8101-8124
EVAL_DAILY_CAP_USD = 3.0  # the evaluation's own cap on the shared ledger (CLAUDE.md: "well under $3")
LABELS = [h.value for h in Hypothesis] + ["inconclusive"]
# Typical cost per run, measured in Phases 3-4 (USD): used for the estimate only.
COST_C = {"fault": 0.021, "look-alike": 0.012, "healthy": 0.0}
COST_B, COST_CT = 0.0025, 0.016

B_SCHEMA = {
    "name": "diagnosis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": LABELS}, "confidence": {"type": "number"},
                       "reason": {"type": "string"}},
        "required": ["label", "confidence", "reason"],
        "additionalProperties": False,
    },
}
B_SYSTEM = (
    "You diagnose a two-level three-phase voltage-source inverter (sinusoidal PWM, star-connected RL load, isolated "
    "neutral) from the features its edge monitor computed over the last two fundamental cycles. Spectra are in rad/s.\n"
    f"{SIGNATURES}\nGive the single most likely cause as one label from the list, a confidence between 0 and 1 and a "
    "one-sentence reason. You have no tools and no other data. Use 'healthy' if nothing is wrong and 'inconclusive' "
    "if the features do not decide."
)


# --- scenarios ----------------------------------------------------------------------------------------

def load_scenarios(dev: bool) -> list[dict[str, Any]]:
    items = yaml.safe_load(SCENARIOS_PATH.read_text(encoding="utf-8"))["scenarios"]
    if dev:
        items = [dict(s, seed=DEV_SEED_BASE + n) for n, s in enumerate(items, start=1)]
    return items


def to_scenario(item: dict[str, Any]) -> Scenario:
    c = item.get("condition") or {}
    cond = Condition(open_switches=frozenset(c.get("open", [])), sensor_offset_A=tuple(c.get("offset_A", (0, 0, 0))),
                     sensor_gain=tuple(c.get("gain", (1, 1, 1))), r_factor=tuple(c.get("r_factor", (1, 1, 1))))
    return Scenario(item["id"], item["id"], cond, Hypothesis(item["truth"]), item["kind"], m=float(item["m"]))


def family(label: str | None) -> str:
    if label in (None, "inconclusive"):
        return "inconclusive"
    return SPECS[Hypothesis(label)].family


def is_fault(label: str | None) -> bool:
    return label not in (None, "inconclusive") and SPECS[Hypothesis(label)].converter_fault


# --- one scenario -----------------------------------------------------------------------------------

def new_session(cfg: dict[str, Any], seed: int, ledger: SpendLedger, plant_cfg: dict[str, Any] | None = None,
                realistic: bool = True) -> Any:
    run_cfg = cfg if plant_cfg is None else dict(cfg, plant=plant_cfg)
    llm = TokenFactory(run_cfg, ledger=ledger)
    return build_session(run_cfg, seed=seed, llm=llm, web=TavilyWeb(run_cfg, ledger=ledger), realistic=realistic)


def observe(session: Any, scenario: Scenario) -> tuple[Event | None, int]:
    """Run healthy, apply the condition, and watch for the first medium or high alarm."""
    with session.lock:
        session.plant.set_modulation_index(scenario.m)
        session.advance(HEALTHY_CYCLES)
        session.events.clear()
        session.plant.set_condition(scenario.condition)
        inject = session.cycle
        for _ in range(WATCH_CYCLES):
            for status in session.advance(1):
                for event in status.new_events:
                    if event.severity in ("medium", "high"):
                        return event, inject
    return None, inject


def steady_features(session: Any) -> dict[str, Any]:
    with session.lock:
        try:
            return session.features(session.steady_window()).compact()
        except WindowNotDiagnosable:
            return session.features().compact()


def method_b(llm: TokenFactory, features: dict[str, Any], event: Event | None, m: float) -> dict[str, Any]:
    n0, c0 = len(llm.calls), llm.spent_usd()
    alarm = {"type": event.type, "severity": event.severity, "message": event.message} if event else "none"
    content = json.dumps({"operating_point": {"m": m, "w1_rad_s": 314.16}, "monitor_alarm": alarm,
                          "features": features, "labels": LABELS})
    try:
        res = llm.chat("agent", [{"role": "system", "content": B_SYSTEM}, {"role": "user", "content": content}],
                       label="eval:B", thinking="low", max_tokens=2000,
                       response_format={"type": "json_schema", "json_schema": B_SCHEMA}, timeout_s=60.0)
        out = json.loads(res.content)
        label = out["label"] if out.get("label") in LABELS else "inconclusive"
        result = {"label": label, "confidence": out.get("confidence"), "reason": out.get("reason", "")[:400]}
    except (LLMError, json.JSONDecodeError, KeyError) as exc:
        result = {"label": "inconclusive", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    calls = llm.calls[n0:]
    result.update(cost_usd=round(llm.spent_usd() - c0, 6), latency_s=round(sum(c.latency_s for c in calls), 2),
                  prompt_tokens=sum(c.prompt_tokens for c in calls),
                  completion_tokens=sum(c.completion_tokens for c in calls))
    return result


def summarize_c(trace: Any, work_order: Any, inject: int, event: Event) -> dict[str, Any]:
    final = trace.final or {}
    totals = trace.totals or {}
    derates = [v for v in trace.actions if v["approved"] and v["request"].startswith(("derate", "stop"))]
    derate = derates[-1] if derates else None
    tests = [v for v in trace.actions if v["request"].startswith("active test")]
    return {
        "label": final.get("label"), "confidence": final.get("confidence"),
        "twin_match_error_pct": final.get("twin_match_error_pct"), "proposed_label": final.get("proposed_label"),
        "triage_family": (trace.triage or {}).get("family"), "triage_model": (trace.triage or {}).get("model"),
        "tool_calls": totals.get("tool_calls"), "cost_usd": totals.get("llm_cost_usd_incl_report"),
        "agent_cost_usd": totals.get("llm_cost_usd"), "wall_s": totals.get("wall_s_incl_report"),
        "prompt_tokens": totals.get("prompt_tokens"), "completion_tokens": totals.get("completion_tokens"),
        "tavily_live_calls": totals.get("tavily_live_calls"), "tavily_credits": totals.get("tavily_credits"),
        "cycles_to_alarm": event.cycle_index - inject,
        "cycles_to_diagnosis": (totals.get("plant_cycle_at_end") or inject) - inject,
        "active_test": [{"request": v["request"], "approved": v["approved"]} for v in tests],
        "test_result": ((trace.active_test or {}).get("analysis") or {}).get("consistent_with"),
        "derate": ({"request": derate["request"], "m": derate["applied"].get("m"),
                    "from_datasheet": bool(derate["applied"].get("limit_from_datasheet")),
                    "needs_datasheet": bool(derate.get("needs_datasheet"))} if derate else None),
        "citations": len(final.get("citations", [])), "citation_urls": [c["url"] for c in final.get("citations", [])],
        "stop_reason": trace.stop_reason, "validation_notes": trace.validation_notes,
        "service_errors": trace.service_errors,
        "work_order_model": work_order.model if work_order else None,
        "ultra_status": work_order.ultra_status if work_order else None,
        "summary": final.get("summary", "")[:600],
    }


def save_trace(trace: Any, path: Path) -> None:
    data = json.loads(trace.to_json())
    data.pop("plant_record", None)            # the plant record is for replays; it would make each file ~0.5 MB
    data.pop("overlays", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def run_scenario(item: dict[str, Any], cfg: dict[str, Any], ledger: SpendLedger, out_dir: Path) -> dict[str, Any]:
    sc = to_scenario(item)
    t0 = time.perf_counter()
    session = new_session(cfg, item["seed"], ledger)
    event, inject = observe(session, sc)
    features = steady_features(session)
    th = cfg["monitor"]
    rec: dict[str, Any] = {
        "id": item["id"], "truth": item["truth"], "kind": item["kind"], "family": family(item["truth"]),
        "m": item["m"], "seed": item["seed"],
        "alarm": ({"type": event.type, "severity": event.severity, "cycle_after_inject": event.cycle_index - inject,
                   "message": event.message} if event else None),
        "features": features,
        "A": {"label": baseline_from_mu(features["mu"], th["mu_abs_max"])},
        "B": method_b(session.llm, features, event, sc.m),
    }
    if event is None:
        rec["C"] = {"label": "healthy", "no_alarm": True, "cost_usd": 0.0, "wall_s": 0.0, "tool_calls": 0}
    else:
        trace, work_order = run_diagnosis(session, event, cfg, trace=Trace(scenario=sc.name), scenario=sc, save=False)
        rec["C"] = summarize_c(trace, work_order, inject, event)
        save_trace(trace, out_dir / "traces" / f"{item['id']}__C.json")
    if sc.kind == "fault" and len(sc.condition.open_switches) == 1:
        rec["C_T"] = run_without_web(item, sc, cfg, ledger, out_dir)
        rec["twin_device_drops"] = twin_with_device_drops(item, sc, cfg, ledger)
    rec["wall_s_total"] = round(time.perf_counter() - t0, 1)
    return rec


def run_without_web(item: dict[str, Any], sc: Scenario, cfg: dict[str, Any], ledger: SpendLedger,
                    out_dir: Path) -> dict[str, Any]:
    """C-T: the same plant data, the same agent, no web tools."""
    session = new_session(cfg, item["seed"], ledger)
    event, inject = observe(session, sc)
    if event is None:
        return {"label": "healthy", "no_alarm": True}
    trace, work_order = run_diagnosis(session, event, cfg, trace=Trace(scenario=sc.name), scenario=sc, save=False,
                                      web_tools=False)
    save_trace(trace, out_dir / "traces" / f"{item['id']}__C-T.json")
    return summarize_c(trace, work_order, inject, event)


SETTLE_CYCLES = 10   # the device-drop experiment's window starts this many cycles after the alarm (no onset)


def twin_with_device_drops(item: dict[str, Any], sc: Scenario, cfg: dict[str, Any], ledger: SpendLedger,
                           realistic: bool = True) -> dict:
    """The true cause's twin match error on a plant with device drops, without and with the datasheet's drops.

    Assumption, stated in the results: the simulated devices have the datasheet's typical V_CE(sat) and V_F.
    The window is taken SETTLE_CYCLES after the alarm, so it holds no fault-onset transient. With
    realistic=False (no noise, no R/L mismatch) it is a check that the drops are applied as modelled.
    """
    facts_path = ROOT / "data" / "cache" / "datasheets" / f"{cfg['nameplate']['igbt_module_part_number']}.json"
    from faulttwin.web import DatasheetFacts, _fix_typ_columns
    try:
        facts = DatasheetFacts.model_validate_json(facts_path.read_text(encoding="utf-8"))
        _fix_typ_columns(facts)
    except (FileNotFoundError, ValueError):
        return {"skipped": "no verified datasheet in the cache"}
    if not (facts.V_CE_sat_typ.verified and facts.V_F_typ.verified):
        return {"skipped": "the datasheet's V_CE(sat) or V_F is not verified"}
    v_ce, v_f = float(facts.V_CE_sat_typ.value), float(facts.V_F_typ.value)
    plant_cfg = dict(copy.deepcopy(cfg["plant"]), device_drops_enabled=True, v_ce_sat_V=v_ce, v_f_V=v_f)
    session = new_session(cfg, item["seed"], ledger, plant_cfg=plant_cfg, realistic=realistic)
    observe(session, sc)
    with session.lock:
        session.advance(SETTLE_CYCLES)
        window = session.steady_window()
    out = {"V_CE_sat_V": v_ce, "V_F_V": v_f, "source": facts.source_url, "settle_cycles": SETTLE_CYCLES,
           "realistic_plant": realistic}
    for name, twin in (("without_datasheet", session.twin),
                       ("with_datasheet", session.twin.with_params(replace(session.twin.params, v_ce_sat_V=v_ce,
                                                                           v_f_V=v_f)))):
        ranking = twin.simulate_hypotheses(list(Hypothesis), window)
        truth = next(r for r in ranking if r.hypothesis == sc.truth)
        out[name] = {"truth_match_error_pct": round(truth.match_error_pct, 3), "rank": ranking.index(truth) + 1,
                     "best": ranking[0].hypothesis.value}
    return out


# --- estimate and report ------------------------------------------------------------------------------

def estimate(items: list[dict[str, Any]]) -> tuple[float, int]:
    single = sum(1 for s in items if s["kind"] == "fault" and len(s["condition"].get("open", [])) == 1)
    usd = sum(COST_C[s["kind"]] for s in items) + COST_B * len(items) + COST_CT * single
    credits = sum(2 for s in items if s["kind"] != "healthy")
    return usd, credits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="scenario ids to (re)run")
    ap.add_argument("--dev", action="store_true", help="development seeds, separate results folder")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild the report from results.json")
    ap.add_argument("--twin-only", action="store_true",
                    help="redo only the device-drop experiment (no model calls) and keep the first version")
    args = ap.parse_args()
    cfg = load_config()
    items = load_scenarios(args.dev)
    out_dir = HERE / "results" / ("dev" if args.dev else "")
    results_path = out_dir / "results.json"
    todo = [s for s in items if not args.only or s["id"] in args.only]
    usd, credits = estimate(todo)
    print(f"{len(todo)} scenarios; estimated cost ${usd:.2f} (Token Factory) and about {credits} Tavily credits")
    if args.estimate:
        return 0
    from eval.report import write_report       # matplotlib is only needed for the report
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"scenarios": {}}
    if args.twin_only:
        ledger = SpendLedger(ROOT / "data" / "cache" / "twin_only_unused.json", 0.0, 0.0)   # no model is called
        for item in todo:
            rec = results["scenarios"].get(item["id"])
            if rec is None or "twin_device_drops" not in rec:
                continue
            sc = to_scenario(item)
            rec.setdefault("twin_device_drops_first_window", rec["twin_device_drops"])
            rec["twin_device_drops"] = twin_with_device_drops(item, sc, cfg, ledger)
            rec["twin_device_drops_ideal_plant"] = twin_with_device_drops(item, sc, cfg, ledger, realistic=False)
            print(item["id"], rec["twin_device_drops"]["without_datasheet"]["truth_match_error_pct"], "->",
                  rec["twin_device_drops"]["with_datasheet"]["truth_match_error_pct"], "| ideal:",
                  rec["twin_device_drops_ideal_plant"]["with_datasheet"]["truth_match_error_pct"], flush=True)
        results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    elif not args.report:
        ledger = SpendLedger.from_config(cfg)
        ledger = SpendLedger(ledger.path, daily_cap_usd=EVAL_DAILY_CAP_USD, total_cap_usd=ledger.total_cap_usd)
        spent0 = ledger.spent()[1]
        for item in todo:
            if item["id"] in results["scenarios"] and not args.only:
                continue
            print(f"--- {item['id']} (seed {item['seed']})", flush=True)
            rec = run_scenario(item, cfg, ledger, out_dir)
            c = rec["C"]
            print(f"    alarm: {rec['alarm']['type'] if rec['alarm'] else 'none'} | A {rec['A']['label']} | "
                  f"B {rec['B']['label']} | C {c['label']} (${c.get('cost_usd') or 0:.4f}, {c.get('wall_s')} s)"
                  + (f" | C-T {rec['C_T']['label']}" if "C_T" in rec else ""), flush=True)
            results["scenarios"][item["id"]] = rec
            results["seeds"] = "development" if args.dev else "evaluation (8101-8124)"
            out_dir.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"spent in this run: ${ledger.spent()[1] - spent0:.4f}")
    write_report(results, out_dir, [s["id"] for s in items])
    print(f"report: {out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
