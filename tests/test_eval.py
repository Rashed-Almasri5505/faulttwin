"""Phase 5: the evaluation's scenarios, metrics and report (offline; the live run is eval/run_eval.py)."""

from pathlib import Path

import shutil

import eval.run_eval as run_eval
from eval.report import metrics, write_report
from eval.run_eval import estimate, load_scenarios, stored_datasheet, to_scenario
from faulttwin.config import REPO_ROOT, load_config
from faulttwin.hypotheses import SPECS, Hypothesis

RESERVED = set(range(9001, 9011)) | set(range(7001, 7004)) | set(range(5000, 6000)) | set(range(6000, 7000)) | \
    {4242, 4343, 1} | set(range(10_000, 100_000))


def test_the_device_drop_experiment_finds_the_shipped_datasheet_on_a_fresh_clone(tmp_path: Path, monkeypatch) -> None:
    cfg = load_config()
    seed = cfg["tavily"]["seed_datasheets_dir"]
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)                     # a clone: no data/cache at all
    assert stored_datasheet(cfg) is None
    shutil.copytree(REPO_ROOT / seed, tmp_path / seed)
    facts = stored_datasheet(cfg)
    assert facts is not None and facts.V_CE_sat_typ.verified and facts.V_CE_sat_typ.value == 1.95
    assert facts.V_F_typ.verified and facts.V_F_typ.value == 1.85


def test_the_24_scenarios_match_the_spec_and_use_their_own_seeds() -> None:
    items = load_scenarios(dev=False)
    assert len(items) == 24 and len({s["id"] for s in items}) == 24
    fam = [SPECS[Hypothesis(s["truth"])].family for s in items]
    single = [s for s in items if SPECS[Hypothesis(s["truth"])].family == "switch_open"]
    assert sum(1 for s in single if s["m"] == 0.8) == 6 and sum(1 for s in single if s["m"] < 0.5) == 2
    assert (fam.count("leg_open"), fam.count("two_switch_open"), fam.count("sensor_offset"), fam.count("sensor_gain"),
            fam.count("load_unbalance"), fam.count("load_step"), fam.count("healthy")) == (3, 2, 3, 2, 2, 2, 2)
    assert all(s["m"] == 0.3 for s in items if s["truth"] == "healthy")
    seeds = [s["seed"] for s in items]
    assert len(set(seeds)) == 24 and not set(seeds) & RESERVED
    for s in items:                                   # every scenario builds a valid plant condition
        sc = to_scenario(s)
        assert sc.truth.value == s["truth"]
    dev = load_scenarios(dev=True)
    assert not {s["seed"] for s in dev} & set(seeds)
    usd, _ = estimate(items)
    assert usd < 3.0                                  # a full run must stay well under $3


def _row(truth: str, a: str, b: str, c: str, **extra) -> dict:
    return {"id": truth + str(len(extra)), "truth": truth, "family": SPECS[Hypothesis(truth)].family, "m": 0.8,
            "alarm": None, "A": {"label": a}, "B": {"label": b}, "C": {"label": c, **extra}}


def test_metrics_count_false_alarms_missed_faults_and_inconclusive() -> None:
    rows = [_row("AU_open", "AU_open", "AL_open", "AU_open"),
            _row("sensor_offset_B", "BU_open", "sensor_offset_B", "sensor_offset_B"),
            _row("leg_A_open", "healthy", "leg_A_open", "inconclusive"),
            _row("healthy", "healthy", "load_unbalance_A", "healthy")]
    a, b, c = (metrics(rows, m) for m in "ABC")
    assert (a["exact"], a["false_alarms"], a["missed_faults"]) == (2, 1, 1)
    assert (b["exact"], b["fault_or_not"], b["false_alarms"]) == (2, 4, 0)
    assert (c["exact"], c["missed_faults"], c["inconclusive"]) == (3, 1, 1)


def test_the_report_and_its_figure_are_written(tmp_path: Path) -> None:
    rows = {r["id"]: r for r in [_row("AU_open", "AU_open", "AU_open", "AU_open", cost_usd=0.02, wall_s=30),
                                 _row("healthy", "healthy", "healthy", "healthy", no_alarm=True, cost_usd=0.0)]}
    results = {"scenarios": rows, "seeds": "test"}
    write_report(results, tmp_path, list(rows))
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "| C. FaultTwin (full agent) | 2/2 (100 %)" in md and "no alarm, no diagnosis" in md
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 10_000
    assert results["summary"]["metrics"]["A"]["exact"] == 2
