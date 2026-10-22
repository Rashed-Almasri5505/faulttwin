"""scripts/trace_stats.py: runs on the committed evaluation traces, expands wildcards, counts schema breaks."""

import importlib.util
import json
import sys
from pathlib import Path

from faulttwin.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("trace_stats", REPO_ROOT / "scripts" / "trace_stats.py")
trace_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace_stats)  # type: ignore[union-attr]

EVAL_TRACES = REPO_ROOT / "eval" / "results" / "traces"


def _trace(steps: list[dict]) -> dict:
    return {"steps": steps, "totals": {"tool_calls": len(steps), "llm_cost_usd": 0.0, "wall_s": 1.0}}


def test_default_paths_include_the_committed_evaluation_traces() -> None:
    paths = trace_stats.trace_paths([])
    committed = sorted(EVAL_TRACES.glob("*.json"))
    assert committed, "the evaluation traces are committed and must be found"
    assert set(committed) <= set(paths)


def test_default_paths_work_without_local_traces(tmp_path: Path, monkeypatch) -> None:
    """A fresh clone has no data/traces/: the evaluation traces alone are read."""
    (tmp_path / "eval" / "results" / "traces").mkdir(parents=True)
    only = tmp_path / "eval" / "results" / "traces" / "x.json"
    only.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(trace_stats, "REPO_ROOT", tmp_path)
    assert trace_stats.trace_paths([]) == [only]


def test_wildcards_are_expanded_by_the_script(tmp_path: Path, capsys) -> None:
    for name in ("a.json", "b.json", "c.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert trace_stats.trace_paths([str(tmp_path / "*.json")]) == [tmp_path / "a.json", tmp_path / "b.json"]
    assert trace_stats.trace_paths([str(tmp_path / "c.txt")]) == [tmp_path / "c.txt"]
    assert trace_stats.trace_paths([str(tmp_path / "*" / "none.json")]) == []
    assert "no file matches" in capsys.readouterr().err


def test_counts_schema_breaks_and_skips_code_issued_steps(tmp_path: Path, monkeypatch, capsys) -> None:
    rejected = {"kind": "tool", "tool": "finalize_diagnosis", "result_summary": "rejected: 1 problem",
                "result": {"accepted": False, "problems": ["final answer is not valid: alternatives_rejected"]}}
    invalid = {"kind": "tool", "tool": "apply_action", "result_summary": "error",
               "result": {"error": "invalid arguments for apply_action: target_m"}}
    background = {"kind": "tool", "tool": "web_failure_evidence", "result_summary": "background evidence: done",
                  "result": {}}                                     # an older trace: no by_code flag yet
    fail_safe = {"kind": "tool", "tool": "apply_action", "by_code": True, "result_summary": "fail-safe (code): ok",
                 "result": {}}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(_trace([rejected, invalid, background, fail_safe])), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["trace_stats.py", str(path)])
    assert trace_stats.main() == 0
    out = capsys.readouterr().out
    assert "Agent tool calls: 2 (plus 2 steps issued by code)" in out
    assert "invalid arguments: 1" in out and "(1 broke the answer schema)" in out
    assert "calls outside the declared schema: 2" in out


def test_runs_on_the_evaluation_traces(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["trace_stats.py", str(EVAL_TRACES / "*.json")])
    assert trace_stats.main() == 0
    assert f"{len(list(EVAL_TRACES.glob('*.json')))} traces" in capsys.readouterr().out
