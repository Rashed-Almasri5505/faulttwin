"""Tables and the confusion matrix for the evaluation (eval/results/results.md, confusion_matrix.png)."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faulttwin.config import load_config
from faulttwin.hypotheses import SPECS, Hypothesis

METHODS = (("A", "Textbook baseline"), ("B", "Super, one call, no tools"), ("C", "FaultTwin (full agent)"))
FAMILIES = ["healthy", "switch_open", "leg_open", "two_switch_open", "sensor_offset", "sensor_gain",
            "load_unbalance", "load_step", "inconclusive"]
FAMILY_TEXT = {"healthy": "healthy", "switch_open": "open switch", "leg_open": "open leg",
               "two_switch_open": "two switches", "sensor_offset": "sensor offset", "sensor_gain": "sensor gain",
               "load_unbalance": "load unbalance", "load_step": "load step", "inconclusive": "inconclusive"}
GROUPS = [("Single switch, nominal load", lambda r: r["family"] == "switch_open" and r["m"] >= 0.5),
          ("Single switch, low load (m = 0.35)", lambda r: r["family"] == "switch_open" and r["m"] < 0.5),
          ("Whole leg", lambda r: r["family"] == "leg_open"),
          ("Two switches, different legs", lambda r: r["family"] == "two_switch_open"),
          ("Sensor offset", lambda r: r["family"] == "sensor_offset"),
          ("Sensor gain", lambda r: r["family"] == "sensor_gain"),
          ("Load unbalance", lambda r: r["family"] == "load_unbalance"),
          ("Load step (healthy converter)", lambda r: r["family"] == "load_step"),
          ("Healthy, low load (m = 0.3)", lambda r: r["family"] == "healthy")]


def family(label: str | None) -> str:
    if label in (None, "inconclusive"):
        return "inconclusive"
    return SPECS[Hypothesis(label)].family


def is_fault(label: str | None) -> bool:
    return label not in (None, "inconclusive") and SPECS[Hypothesis(label)].converter_fault


def metrics(rows: list[dict[str, Any]], m: str) -> dict[str, Any]:
    faults = [r for r in rows if is_fault(r["truth"])]
    others = [r for r in rows if not is_fault(r["truth"])]
    return {
        "n": len(rows),
        "exact": sum(r[m]["label"] == r["truth"] for r in rows),
        "fault_or_not": sum(is_fault(r[m]["label"]) == is_fault(r["truth"]) for r in rows),
        "false_alarms": sum(is_fault(r[m]["label"]) for r in others), "n_not_fault": len(others),
        "missed_faults": sum(not is_fault(r[m]["label"]) for r in faults), "n_fault": len(faults),
        "inconclusive": sum(r[m]["label"] == "inconclusive" for r in rows),
    }


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f} %)" if n else "-"


def confusion_png(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    truth_fams = [f for f in FAMILIES if any(r["family"] == f for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    for ax, (m, title) in zip(axes, METHODS):
        mat = np.zeros((len(truth_fams), len(FAMILIES)), dtype=int)
        for r in rows:
            mat[truth_fams.index(r["family"]), FAMILIES.index(family(r[m]["label"]))] += 1
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, mat.max()))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j]:
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            color="white" if mat[i, j] > mat.max() / 2 else "#1F2933", fontsize=11)
        ax.set_xticks(range(len(FAMILIES)), [FAMILY_TEXT[f] for f in FAMILIES], rotation=45, ha="right")
        ax.set_yticks(range(len(truth_fams)), [FAMILY_TEXT[f] for f in truth_fams])
        exact = sum(r[m]["label"] == r["truth"] for r in rows)
        ax.set_title(f"{m}: {title}\nexact label {exact}/{len(rows)}", fontsize=11)
        ax.set_xlabel("diagnosed family")
        ax.set_ylabel("true family")
    fig.suptitle("FaultTwin evaluation: confusion by cause family (24 scenarios, fixed seeds)", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _run_window(out_dir: Path) -> str:
    """When the run happened, from its traces (the report may be rebuilt later)."""
    import json
    times = sorted(json.loads(p.read_text(encoding="utf-8")).get("started_at", "")
                   for p in (out_dir / "traces").glob("*__C.json"))
    times = [t for t in times if t]
    if not times:
        return ""
    return f" on {times[0][:10]}, {times[0][11:16]}-{times[-1][11:16]} UTC"


def write_report(results: dict[str, Any], out_dir: Path, order: list[str]) -> None:
    cfg = load_config()
    rows = [results["scenarios"][i] for i in order if i in results["scenarios"]]
    if not rows:
        return
    summary = {m: metrics(rows, m) for m, _ in METHODS}
    c_runs = [r["C"] for r in rows if not r["C"].get("no_alarm")]
    b_runs = [r["B"] for r in rows]
    cost = {"B_total_usd": round(sum(b.get("cost_usd") or 0 for b in b_runs), 4),
            "C_total_usd": round(sum(c.get("cost_usd") or 0 for c in c_runs), 4),
            "C_T_total_usd": round(sum(r["C_T"].get("cost_usd") or 0 for r in rows if "C_T" in r), 4),
            "C_per_diagnosis_median_usd": _median([c.get("cost_usd") for c in c_runs]),
            "C_per_diagnosis_max_usd": max((c.get("cost_usd") or 0 for c in c_runs), default=None),
            "C_wall_median_s": _median([c.get("wall_s") for c in c_runs]),
            "C_wall_max_s": max((c.get("wall_s") or 0 for c in c_runs), default=None),
            "C_cycles_to_alarm_median": _median([c.get("cycles_to_alarm") for c in c_runs]),
            "C_cycles_to_diagnosis_median": _median([c.get("cycles_to_diagnosis") for c in c_runs]),
            "C_tokens_median": _median([(c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0)
                                        for c in c_runs]),
            "B_latency_median_s": _median([b.get("latency_s") for b in b_runs]),
            "B_cost_median_usd": _median([b.get("cost_usd") for b in b_runs])}
    results["summary"] = {"metrics": summary, "cost_and_time": cost,
                          "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    confusion_png(rows, out_dir / "confusion_matrix.png")
    window = _run_window(out_dir)
    b_gated = sum((r["B"]["label"] if r["alarm"] else "healthy") == r["truth"] for r in rows)
    seeds = "evaluation seeds 8101-8124" if "8101" in str(results.get("seeds", "")) else str(results.get("seeds", ""))

    md = ["# FaultTwin evaluation results", "",
          f"{len(rows)} scenarios with fixed {seeds}, one full run{window}. Models: triage "
          f"`{cfg['llm']['models']['triage']}`, agent `{cfg['llm']['models']['agent']}`, work order "
          f"`{cfg['llm']['models']['report']}` (hedged by Super, as in the app), all on Nebius Token Factory; web "
          "evidence through Tavily. Every run is reported, failures included.", "",
          "**Methods.** A: textbook threshold detector (largest |μ| above 0.10 picks the leg, its sign the switch). "
          "B: Nemotron 3 Super in one call, given the edge monitor's features, its alarm message, the operating point "
          "and the same fault-signature notes as the agent; no tools. B runs on every scenario; if it were only "
          f"asked after an alarm, like C, its exact label would be right in {b_gated}/{len(rows)}. C: the app's "
          "diagnosis pipeline (the edge monitor must raise a medium or high alarm, then Lightning triage, the Super "
          "agent with the digital twin, the active test, Tavily and the safety interlock, and the work order); in "
          "the evaluation the plant is paused while the models run. When the monitor raises no alarm, C reports "
          "\"healthy\" and spends nothing (a correct outcome for a healthy converter, a wrong exact label for a "
          "look-alike).", "",
          "## Summary", "",
          "| Method | Exact label | Converter fault or not | False alarms on non-faults | Missed faults | Inconclusive |",
          "|---|---|---|---|---|---|"]
    for m, title in METHODS:
        s = summary[m]
        md.append(f"| {m}. {title} | {_pct(s['exact'], s['n'])} | {_pct(s['fault_or_not'], s['n'])} | "
                  f"{s['false_alarms']}/{s['n_not_fault']} | {s['missed_faults']}/{s['n_fault']} | {s['inconclusive']} |")
    alarmed = [r for r in rows if r["alarm"]]
    md += ["", "Missed faults include inconclusive answers on a fault. On the "
           f"{len(alarmed)} scenarios where the edge monitor raised an alarm, the full agent's exact label was right in "
           f"{sum(r['C']['label'] == r['truth'] for r in alarmed)}.", "",
           "## Cost and time", "",
           f"- C (full agent, {len(c_runs)} diagnoses after an alarm): median ${cost['C_per_diagnosis_median_usd']} of "
           f"Token Factory cost per diagnosis, work order included (max ${cost['C_per_diagnosis_max_usd']}), plus one "
           f"live Tavily credit; median {round(cost['C_tokens_median'] or 0)} triage and agent tokens (the work order "
           "adds about 2,000).",
           f"- Time to diagnosis: median {cost['C_wall_median_s']} s wall-clock from the alarm to the checked answer "
           f"and its work order (max {cost['C_wall_max_s']} s: in leg_B_open a last model turn started 0.5 s before "
           "the 90 s budget ran out and got the 5 s minimum timeout; a turn now needs 8 s of budget left).",
           f"- Simulated time: the alarm came in the first cycle after the condition was applied (median). The "
           f"agent's tools then advanced the plant by a median {round(cost['C_cycles_to_diagnosis_median'] or 0)} "
           "cycles (steady windows, the active test); the plant is paused while the models think, so this is not a "
           "time to diagnosis.",
           f"- B: median ${cost['B_cost_median_usd']} and {cost['B_latency_median_s']} s per call.",
           f"- Total: B ${cost['B_total_usd']}, C ${cost['C_total_usd']}, C-T ${cost['C_T_total_usd']}.", "",
           "## By scenario group", "", "| Group | n | A exact | B exact | C exact | C fault-or-not |", "|---|---|---|---|---|---|"]
    for name, pick in GROUPS:
        g = [r for r in rows if pick(r)]
        if g:
            md.append(f"| {name} | {len(g)} | " + " | ".join(str(sum(r[m]['label'] == r['truth'] for r in g))
                                                            for m in ("A", "B", "C"))
                      + f" | {sum(is_fault(r['C']['label']) == is_fault(r['truth']) for r in g)} |")
    md += ["", "## Every scenario", "",
           "| Scenario | Truth | Monitor alarm | A | B | C | C cost | C time | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["C"]
        alarm = f"{r['alarm']['type']} ({r['alarm']['severity']})" if r["alarm"] else "none"
        notes = []
        if c.get("no_alarm"):
            notes.append("no alarm, no diagnosis")
        if c.get("stop_reason"):
            notes.append(f"stopped: {c['stop_reason'][:80]}")
        if c.get("proposed_label"):
            notes.append(f"proposed {c['proposed_label']}")
        if c.get("test_result"):
            notes.append(f"m-test: {c['test_result']}")
        elif any(t["approved"] for t in c.get("active_test", [])):
            notes.append("m-test: not informative (no significant mean current)")
        elif c.get("active_test"):
            notes.append("m-test refused by the interlock")

        def mark(label: str) -> str:
            return f"{label} ✓" if label == r["truth"] else f"{label} ✗"

        md.append(f"| {r['id']} | {r['truth']} | {alarm} | {mark(r['A']['label'])} | {mark(r['B']['label'])} | "
                  f"{mark(c['label'])} | ${c.get('cost_usd') or 0:.4f} | {c.get('wall_s') or 0} s | "
                  f"{'; '.join(notes)} |")
    ct = [r for r in rows if "C_T" in r]
    if ct:
        md += ["", "## What Tavily adds (C vs C-T, the 8 single-switch faults)", "",
               "C-T is the same agent on the same plant data with the web tools removed.", "",
               "| Scenario | C label | C-T label | C derate | C-T derate | C sources | C-T sources |",
               "|---|---|---|---|---|---|---|"]

        def not_needed(x: dict[str, Any]) -> bool:
            return bool(x.get("derate")) and x["derate"]["request"].endswith("(not needed)")

        def der(x: dict[str, Any]) -> str:
            d = x.get("derate")
            if not d:
                return "none"
            if not_needed(x):
                return f"not needed (m stays {d['m']})"
            src = "limit from datasheet I_C,nom" if d["from_datasheet"] else "conservative default, no datasheet"
            return f"m = {d['m']} ({src})" if d.get("m") is not None else d["request"]

        for r in ct:
            md.append(f"| {r['id']} | {r['C']['label']} | {r['C_T']['label']} | {der(r['C'])} | {der(r['C_T'])} | "
                      f"{r['C'].get('citations', 0)} | {r['C_T'].get('citations', 0)} |")
        changed = [r for r in ct if r["C"].get("derate") and r["C_T"].get("derate")
                   and r["C"]["derate"].get("m") != r["C_T"]["derate"].get("m")]
        both_idle = [r for r in ct if not_needed(r["C"]) and not_needed(r["C_T"])]
        cost_c = sum(r["C"].get("cost_usd") or 0 for r in ct)
        cost_ct = sum(r["C_T"].get("cost_usd") or 0 for r in ct)
        md += ["", f"- The verified datasheet rating changed the interlock's derate in {len(changed)}/{len(ct)} runs "
               "(at nominal load the derated drive kept m = 0.66 instead of the conservative default 0.40, i.e. more "
               f"output while still under 50 % of I_C,nom); in {len(both_idle)}/{len(ct)} (low load) no derate was "
               "needed either way. The datasheet came from a Tavily extract cached on 2026-10-21; each diagnosis made one "
               "live Tavily search (failure evidence).",
               f"- Diagnoses with at least one source URL: C {sum(1 for r in ct if r['C'].get('citations'))}/{len(ct)}, "
               f"C-T {sum(1 for r in ct if r['C_T'].get('citations'))}/{len(ct)} (C-T has no web tools, so 0 is "
               "expected by design).",
               f"- Exact label: C {sum(r['C']['label'] == r['truth'] for r in ct)}/{len(ct)}, C-T "
               f"{sum(r['C_T']['label'] == r['truth'] for r in ct)}/{len(ct)}: Tavily did not change the labels.",
               f"- Token Factory cost on these runs: ${cost_c:.4f} with the web tools, ${cost_ct:.4f} without."]
        tw = [r for r in ct if "with_datasheet" in (r.get("twin_device_drops") or {})]
        if tw:
            d0 = tw[0]["twin_device_drops"]
            settle = d0.get("settle_cycles")
            md += ["", "### The twin with the datasheet's device drops", "",
                   f"A separate run per scenario on a plant whose IGBTs and diodes drop V_CE(sat) = {d0['V_CE_sat_V']} V "
                   f"and V_F = {d0['V_F_V']} V (the datasheet's typical values, read through Tavily from "
                   f"{d0['source']}; assumption: the simulated devices match their datasheet). The twin without the "
                   "datasheet ignores the drops; with it, it models them. No model calls."
                   + (f" The window starts {settle} cycles after the alarm, so it holds no fault-onset transient (a "
                      "first version took the window right after the alarm; it also gave 4 down and 4 up)."
                      if settle else ""), "",
                   "| Scenario | Truth's match error without | with datasheet | Rank without | Rank with |",
                   "|---|---|---|---|---|"]
            deltas = []
            for r in tw:
                d = r["twin_device_drops"]
                deltas.append(d["with_datasheet"]["truth_match_error_pct"] - d["without_datasheet"]["truth_match_error_pct"])
                md.append(f"| {r['id']} | {d['without_datasheet']['truth_match_error_pct']} % | "
                          f"{d['with_datasheet']['truth_match_error_pct']} % | {d['without_datasheet']['rank']} | "
                          f"{d['with_datasheet']['rank']} |")
            md += ["", f"With the datasheet's drops the true cause's match error went down in "
                   f"{sum(x < 0 for x in deltas)} of {len(deltas)} scenarios and up in {sum(x > 0 for x in deltas)} "
                   f"(mean change {statistics.mean(deltas):+.2f} points); the true cause ranked first either way."]
            ideal = [r["twin_device_drops_ideal_plant"] for r in tw if "with_datasheet" in
                     (r.get("twin_device_drops_ideal_plant") or {})]
            if ideal:
                wo = statistics.mean(d["without_datasheet"]["truth_match_error_pct"] for d in ideal)
                wi = max(d["with_datasheet"]["truth_match_error_pct"] for d in ideal)
                md += ["", f"Check on an ideal plant (no noise, no R/L tolerance, the same drops): the truth's match "
                       f"error is {wo:.2f} % on average without the datasheet and at most {wi:.3f} % with it, so the "
                       "drops are modelled correctly; on the realistic plant their effect (about 2 V on a 400 V DC "
                       "link) is smaller than that of the ±5 % R/L tolerance and the noise."]
    findings = Path(__file__).resolve().parent / "findings.md"
    if findings.exists() and "development" not in str(results.get("seeds", "")):
        md += ["", findings.read_text(encoding="utf-8").strip()]
    md += ["", "## Confusion matrix", "", "![Confusion by cause family](confusion_matrix.png)", "",
           "## Limitations", "",
           "- 24 scenarios, one seed each: the numbers show behaviour, not statistically tight rates.",
           "- The plant is a switching-cycle-average simulation with ±5 % R/L mismatch and 0.5 % noise; no hardware.",
           "- The edge monitor raised no medium or high alarm for these load unbalances (±25 % on one phase) or load "
           "steps, so the full pipeline did not diagnose them (and raised no false alarm). Its amplitude-ratio band "
           "[0.90, 1.10] is crossed only beyond about ±35-50 % on one phase; larger unbalances do raise an alarm.",
           "- The look-alike sizes were chosen within CLAUDE.md's ranges, except the second load step (-25 % instead "
           "of -30 %). The baseline's false alarms depend on the offset size: at m = 0.8 its 0.10 threshold on μ is "
           "about 2.7 A, so only the 3 A offsets fooled it.",
           "- The two-switch causes are tested only when the agent asks for them (CLAUDE.md 5.4).",
           "- Recorded replays in the app are selected correct runs; this page reports every evaluation run.", ""]
    (out_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    import json
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
