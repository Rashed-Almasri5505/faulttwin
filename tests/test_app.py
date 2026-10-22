"""Phase 4: replay player, budget guard, and the app itself (headless AppTest), offline and with fakes."""

import json
import shutil
from pathlib import Path

import pytest

import faulttwin.replay as replay_mod
from faulttwin.config import REPO_ROOT, load_config
from faulttwin.llm import SpendLedger, TokenFactory
from faulttwin.replay import ReplayPlayer, check_replay, list_replays, load_replay, save_replay
from faulttwin.ui.runner import Guard
from faulttwin.ui.views import view_from_record
from tests.fakes import FakeOpenAI, chat_response

CFG = load_config()
APP = str(REPO_ROOT / "app.py")


def synthetic_trace(n_cycles: int = 20, n_steps: int = 6) -> dict:
    record = [{"cycle": c, "t_wall": 100.0 + 0.001 * c, "m": 0.8, "stopped": False, "t0_s": c * 0.02,
               "i_meas_A": [[1.0, -0.5, -0.5]] * 100, "features": None, "active_checks": [],
               "events": [{"type": "mean_current", "severity": "high", "message": "x"}] if c == 12 else []}
              for c in range(n_cycles)]
    steps = [{"index": j + 1, "kind": "note", "text": f"s{j}", "t_wall": 100.02 + 2.0 * j} for j in range(n_steps)]
    return {"scenario": "AU_open", "event": {"cycle": 12}, "plant_record": record, "steps": steps,
            "final": {"label": "AU_open", "confidence": 0.9, "summary": "s"},
            "work_order": {"markdown": "# Work order: x", "latency_s": 1.0},
            "hidden_truth": {"scenario": "AU_open", "truth": "AU_open", "title": "Upper switch AU (S1) open"},
            "started_at": "2026-10-21T00:00:00+00:00"}


# --- replay player ------------------------------------------------------------------------------------

def test_replay_plays_cycles_and_steps_in_order_and_reveals_at_the_end() -> None:
    p = ReplayPlayer(synthetic_trace(), start=0.0)
    assert p.record[0]["cycle"] == 4                      # lead-in of 8 cycles before the event at cycle 12
    assert p.counts(now=0.0)[1] == 0
    early = p.partial(now=0.5)
    assert early["final"] is None and early["hidden_truth"] is None and not early["finished"]
    cycles_mid, steps_mid = p.counts(now=p.duration_s / 2)
    assert 0 < cycles_mid <= len(p.record) and 0 < steps_mid < 6
    end = p.partial(now=p.duration_s + 0.1)
    assert end["finished"] and end["final"]["label"] == "AU_open" and end["hidden_truth"]["truth"] == "AU_open"
    assert len(end["steps"]) == 6


def test_replay_gives_every_item_some_time_on_screen_and_the_healthy_lead_in_about_3_s() -> None:
    p = ReplayPlayer(synthetic_trace(), start=0.0)
    times = [t for t, _, _ in p.schedule]
    assert all(b - a >= 0.15 - 1e-9 for a, b in zip(times, times[1:]))
    first_event = next(t for t, kind, idx in p.schedule if kind == "cycle" and p.record[idx]["cycle"] >= 12)
    assert 2.5 <= first_event <= 3.5
    assert p.duration_s < 60


def test_replay_marks_the_derate_from_the_recorded_interlock_verdict() -> None:
    trace = synthetic_trace()
    trace["steps"].append({"kind": "tool", "tool": "apply_action", "t_wall": 100.0105, "result": {
        "interlock": {"approved": True, "applied": {"m": 0.66, "fault": "AU_open"}}}})
    p = ReplayPlayer(trace, start=0.0)
    assert [r["derated_m"] for r in p.record if r["cycle"] in (9, 11)] == [None, 0.66]
    v = view_from_record(p.record, len(p.record), CFG["plant"]["dt_s"])
    assert v is not None and v.derated_m == 0.66


def test_replay_overlays_appear_once_the_last_ranking_has_played() -> None:
    trace = synthetic_trace()
    trace["steps"][3] = {"kind": "tool", "tool": "simulate_hypotheses", "t_wall": 106.02, "result": {}}
    trace["overlays"] = {"t_s": [0.0], "measured_A": [[0, 0, 0]], "predicted_A": {"AU_open": [[0, 0, 0]]}}
    p = ReplayPlayer(trace, start=0.0)
    t_sim = next(t for t, kind, idx in p.schedule if kind == "step" and idx == 3)
    assert p.partial(now=t_sim - 0.01)["overlays"] is None
    assert p.partial(now=t_sim + 0.01)["overlays"] is not None


def test_view_from_record_shows_the_last_three_cycles() -> None:
    p = ReplayPlayer(synthetic_trace(), start=0.0)
    v = view_from_record(p.record, 10, CFG["plant"]["dt_s"])
    assert v is not None and v.i_A.shape == (300, 3) and v.cycle == p.record[9]["cycle"] + 1


def test_saved_replays_need_everything_the_ui_shows(tmp_path: Path) -> None:
    trace = synthetic_trace()
    path = save_replay(json.dumps(trace), "AU_open", directory=tmp_path)
    assert load_replay(path) is not None
    with pytest.raises(ValueError):
        save_replay(json.dumps({**trace, "plant_record": []}), "bad", directory=tmp_path)
    for broken in ({**trace, "final": {"label": "AU_open"}}, {**trace, "work_order": None},
                   {**trace, "plant_record": [{"cycle": 1}]}, {**trace, "hidden_truth": None}, [trace]):
        with pytest.raises(ValueError):
            check_replay(broken)


def test_unreadable_replay_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "not_utf8.json").write_bytes(b"\xff\xfe\x00garbage")
    (tmp_path / "a_list.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "truncated.json").write_text('{"plant_record": [', encoding="utf-8")
    save_replay(json.dumps(synthetic_trace()), "good", directory=tmp_path)
    found = {name: load_replay(path) for name, path in list_replays(tmp_path).items()}
    assert {n for n, d in found.items() if d is not None} == {"good"}


def test_committed_replays_load_are_correct_and_render() -> None:
    found = list_replays()
    if not found:
        pytest.skip("no recorded replays yet")
    for name, path in found.items():
        data = load_replay(path)
        assert data is not None, name
        assert data["final"]["label"] == data["hidden_truth"]["truth"], name   # only correct runs are replays
        assert data["work_order"]["markdown"].startswith("# Work order"), name
        p = ReplayPlayer(data, start=0.0)
        assert view_from_record(p.record, len(p.record), CFG["plant"]["dt_s"]) is not None, name
        assert p.partial(now=p.duration_s + 1)["finished"], name


def test_plant_figures_updated_in_place_match_fresh_ones() -> None:
    from faulttwin.runtime import build_session
    from faulttwin.scenarios import SCENARIOS
    from faulttwin.ui.plots import PlantFigures, currents_figure, park_figure, spectrum_figure
    from faulttwin.ui.views import view_from_session

    s = build_session(CFG, seed=6160)
    s.advance(6)
    figs = PlantFigures()
    figs.update(view_from_session(s))
    s.plant.set_condition(SCENARIOS["AU_open"].condition)
    s.advance(4)
    v = view_from_session(s)
    for updated, fresh in zip(figs.update(v), (currents_figure(v), park_figure(v), spectrum_figure(v))):
        a, b = updated.to_dict(), fresh.to_dict()
        assert len(a["data"]) == len(b["data"])
        for ta, tb in zip(a["data"], b["data"]):
            assert ta.get("x") is None or list(ta["x"]) == list(tb["x"])
            assert list(ta["y"]) == list(tb["y"])
        assert a["layout"].get("xaxis", {}).get("range") == b["layout"].get("xaxis", {}).get("range")


# --- guard --------------------------------------------------------------------------------------------

def test_guard_switches_to_replay_for_missing_keys_runs_and_budget(tmp_path: Path, monkeypatch) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", daily_cap_usd=1.0, total_cap_usd=15.0)
    guard = Guard(ledger, runs_per_session=3)
    monkeypatch.setenv("NEBIUS_API_KEY", "x" * 20)
    monkeypatch.setenv("TAVILY_API_KEY", "y" * 20)
    monkeypatch.delenv("FAULTTWIN_OFFLINE", raising=False)
    assert guard.decide(0, False)[2] == "ok"
    assert guard.decide(3, False)[2] == "session"
    assert guard.decide(0, True)[2] == "api"
    ledger.add(0.98)
    ok, why, kind = guard.decide(0, False)
    assert not ok and kind == "budget" and "budget" in why and "$" not in why   # daily cap (needs room for one run)
    monkeypatch.setenv("FAULTTWIN_OFFLINE", "1")
    ok, why = Guard(SpendLedger(tmp_path / "s2.json", 1.0, 15.0), 3).live_allowed(0, False)
    assert not ok and "not configured" in why


# --- the app, offline ---------------------------------------------------------------------------------

def _app(monkeypatch, replay_dir: Path | None = None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    if replay_dir is not None:
        monkeypatch.setattr(replay_mod, "REPLAY_DIR", replay_dir)
    st.cache_resource.clear()                   # the app caches the replays per process
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def _button(at, label: str):
    return next(b for b in at.button if b.label == label)


def _finish_replay(at) -> None:
    at.session_state.player.start -= 10_000
    at.run()


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("FAULTTWIN_OFFLINE", "1")
    return monkeypatch


def test_app_starts_without_keys_in_replay_mode(offline) -> None:
    at = _app(offline)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "FaultTwin" in text and "Recorded runs" in text
    assert any("not configured" in m.value for m in at.sidebar.markdown)


def test_two_mysteries_in_a_row_without_keys_each_finish_once(offline) -> None:
    if len(list_replays()) < 2:
        pytest.skip("needs recorded replays")
    at = _app(offline)
    _button(at, "Mystery fault").click().run()
    first = at.session_state.player
    assert first is not None and at.session_state.hidden and at.session_state.runs_used == 0
    _finish_replay(at)
    assert at.session_state.score["recorded"] == 1 and not at.exception
    _button(at, "Another mystery").click().run()
    assert at.session_state.player is not first
    assert [b for b in at.sidebar.button if b.label == "Mystery fault" and b.disabled]   # redrawn: playing again
    _finish_replay(at)
    assert at.session_state.score == {"correct": 0, "total": 0, "recorded": 2}
    assert not [b for b in at.sidebar.button if b.label in ("Inject", "Mystery fault") and b.disabled]
    assert not at.exception


def test_a_corrupt_or_missing_replay_folder_never_breaks_the_page(offline, tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "AU_open.json").write_bytes(b"\xff\xfe broken")
    (bad / "BL_open.json").write_text("[]", encoding="utf-8")
    at = _app(offline, bad)
    _button(at, "Mystery fault").click().run()
    assert not at.exception and at.session_state.player is None
    assert "No recordings are available" in at.session_state.notice
    empty = tmp_path / "empty"
    empty.mkdir()
    at = _app(offline, empty)
    at.sidebar.selectbox[0].select("AU_open").run()
    _button(at, "Inject").click().run()
    notice = at.session_state.notice
    assert not at.exception and "There are no recorded diagnoses" in notice and "so recorded" not in notice


def test_a_look_alike_without_an_alarm_is_explained(offline) -> None:
    import random
    offline.setattr(random.SystemRandom, "randrange", lambda self, a, b: 6201)   # a seed checked to stay quiet
    at = _app(offline)
    at.sidebar.selectbox[0].select("load_unbalance_A").run()
    _button(at, "Inject").click().run()
    assert not at.session_state.notice
    for _ in range(8):                      # 2 plant cycles per run
        at.run()
    assert any("no false alarm" in s.value for s in at.success), [s.value for s in at.success]
    assert not at.exception


# --- the app, live with fake models and a fake Tavily -------------------------------------------------

def _live_app(monkeypatch, tmp_path: Path, responder, pool=("AU_open",)):
    import faulttwin.runtime as runtime
    import faulttwin.scenarios as scenarios
    import faulttwin.ui.runner as runner
    from tests.test_agent_offline import fake_web

    monkeypatch.setenv("NEBIUS_API_KEY", "x" * 20)
    monkeypatch.setenv("TAVILY_API_KEY", "y" * 20)
    monkeypatch.delenv("FAULTTWIN_OFFLINE", raising=False)
    real_build = runtime.build_session

    def build(cfg, *, seed, **_):
        ledger = SpendLedger(tmp_path / "spend.json", daily_cap_usd=1.0, total_cap_usd=15.0)
        llm = TokenFactory(cfg, ledger=ledger, client=FakeOpenAI(responder))
        return real_build(cfg, seed=6150, llm=llm, web=fake_web(tmp_path))

    monkeypatch.setattr(runtime, "build_session", build)
    monkeypatch.setattr(scenarios, "MYSTERY_POOL", list(pool))
    monkeypatch.setattr(runner.Guard, "from_config", classmethod(
        lambda cls, cfg: cls(SpendLedger(tmp_path / "spend.json", 1.0, 15.0), 3)))
    monkeypatch.setattr(runtime, "TRACE_DIR", tmp_path / "traces")
    return _app(monkeypatch)


def _wait_for_run(at, max_runs: int = 40) -> None:
    attempt = at.session_state.attempt
    for _ in range(max_runs):                # each run advances the plant 2 cycles, until the event starts a run
        ss = at.session_state
        if ss.run is not None or ss.player is not None or (ss.finished_key or (None, -1))[1] == attempt:
            break
        at.run()
    run = at.session_state.run
    if run is not None:
        run._thread.join(timeout=120)
    at.run()
    assert at.session_state.finished_key is not None and at.session_state.finished_key[1] == attempt


def _au_open_agent():
    from tests.test_agent_offline import DEFAULT, MODULE, final_args, scripted_agent

    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    turns = [
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
        chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT}),
                                  ("web_datasheet_lookup", {"part_number": MODULE})]),
        chat_response(tool_calls=[("apply_action", {"action": "derate", "reason": "AU open"}),
                                  ("web_failure_evidence", {"hypothesis": "AU_open", "part_number": MODULE})]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args("AU_open", citations=review))]),
    ]
    state = {"script": None}

    def responder(kw):
        msgs = kw.get("messages", [])
        if not kw.get("stream") and not kw.get("response_format") and len(msgs) == 2:
            state["script"] = scripted_agent(list(turns))      # a new investigation starts its script again
        return (state["script"] or scripted_agent(list(turns)))(kw)
    return responder


def test_live_mysteries_of_the_same_scenario_are_each_scored(monkeypatch, tmp_path: Path) -> None:
    at = _live_app(monkeypatch, tmp_path, _au_open_agent())
    assert any("Live AI ready" in m.value for m in at.markdown)
    for n in (1, 2):
        _button(at, "Mystery fault" if n == 1 else "Another mystery").click().run()
        _wait_for_run(at)
        assert at.session_state.run.status == "done", at.session_state.run.trace.stop_reason
        assert at.session_state.score == {"correct": n, "total": n, "recorded": 0}
    assert at.session_state.runs_used == 2 and not at.exception
    assert any("2 of 2 correct" in c.value for c in at.sidebar.caption)


def test_an_outage_switches_to_the_recording_and_does_not_count(monkeypatch, tmp_path: Path) -> None:
    if "AU_open" not in list_replays():
        pytest.skip("needs the AU_open recording")

    def down(kw):
        raise ConnectionError("simulated Token Factory outage")

    at = _live_app(monkeypatch, tmp_path, down)
    _button(at, "Mystery fault").click().run()
    _wait_for_run(at)
    ss = at.session_state
    assert ss.api_failed and ss.runs_used == 0 and ss.player is not None and ss.condition == "AU_open"
    assert ss.score["total"] == 0 and "did not answer" in ss.player_note
    assert any("did not answer" in w.value for w in at.warning)          # the banner under the header
    assert not at.exception


def test_replay_shows_what_the_agent_is_waiting_for_and_ends_paused(offline) -> None:
    if "AU_open" not in list_replays():
        pytest.skip("needs the AU_open recording")
    at = _app(offline)
    at.sidebar.selectbox[0].select("AU_open").run()
    _button(at, "Inject").click().run()
    p = at.session_state.player
    t_event = p.event_time
    p.start -= t_event + 0.5                       # just after the alarm, before the first agent step
    at.run()
    assert any("⏳" in m.value for m in at.markdown), "no pending line after the alarm"
    assert all(s.disabled for s in at.sidebar.slider)        # a recording's plant cannot be changed
    _finish_replay(at)
    assert any("end of the recording" in m.value for m in at.markdown)
    assert not any("⏳" in m.value for m in at.markdown) and not at.exception


def test_replay_files_with_broken_inner_fields_are_rejected() -> None:
    import copy
    good = synthetic_trace()
    good["plant_record"][3]["features"] = {"i_s_A": 27.0, "r_kcl_dc": 0.0, "r_kcl_w1": 0.0, "mu": [0, 0, 0],
                                           "amp_ratio": [1, 1, 1], "rms_A": [19, 19, 19],
                                           "spectrum_A": {"0.0 rad/s": [0, 0, 0]}}
    check_replay(good)
    broken = []
    b = copy.deepcopy(good); b["plant_record"][3]["features"]["mu"] = [0, 0]; broken.append(b)
    b = copy.deepcopy(good); b["plant_record"][12]["events"][0]["severity"] = "huge"; broken.append(b)
    b = copy.deepcopy(good); b["steps"].append({"kind": "tool", "tool": "x"}); broken.append(b)
    b = copy.deepcopy(good); b["final"]["confidence"] = "high"; broken.append(b)
    for x in broken:
        with pytest.raises(ValueError):
            check_replay(x)


def test_long_recordings_are_shortened_after_the_alarm_only() -> None:
    trace = synthetic_trace(n_steps=12)
    for j, s in enumerate(trace["steps"]):
        s["t_wall"] = 100.02 + 9.0 * j                      # slow model calls: 9 s between steps
    record = trace["plant_record"]
    for c in range(20, 160):                                 # the plant keeps running while the models think
        record.append(dict(record[-1], cycle=c, t_wall=100.02 + 0.4 * (c - 20), events=[]))
    p = ReplayPlayer(trace, start=0.0)
    times = [t for t, _, _ in p.schedule]
    assert p.speedup > 1.0 and p.duration_s <= 41.0 + 1e-9 and times == sorted(times)
    lead = [t for t, kind, idx in p.schedule if kind == "cycle" and p.record[idx]["cycle"] < 12]
    assert 2.5 <= max(lead) <= 3.5                           # the healthy lead-in keeps its pace


def test_guard_switches_to_replay_before_the_tavily_credits_run_out(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "x" * 20)
    monkeypatch.setenv("TAVILY_API_KEY", "y" * 20)
    monkeypatch.delenv("FAULTTWIN_OFFLINE", raising=False)
    ledger = SpendLedger(tmp_path / "spend.json", 1.0, 15.0, daily_cap_credits=25, total_cap_credits=400)
    guard = Guard(ledger, runs_per_session=3)
    assert guard.decide(0, False)[2] == "ok"
    ledger.add_credits(23)                            # 2 left, a diagnosis may need 3
    ok, why, kind = guard.decide(0, False)
    assert not ok and kind == "budget" and "web search" in why


def test_results_tab_shows_the_evaluation(offline) -> None:
    if not (REPO_ROOT / "eval" / "results" / "results.md").exists():
        pytest.skip("no evaluation results yet")
    at = _app(offline)
    text = " ".join(m.value for m in at.markdown)
    assert "FaultTwin (full agent)" in text and "![" not in text and not at.exception


def test_a_link_opens_a_recorded_case_at_a_moment(offline) -> None:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    if "AU_open" not in list_replays():
        pytest.skip("needs the AU_open recording")
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["case"] = "AU_open"
    at.query_params["at"] = "20"
    at.run()
    p = at.session_state.player
    assert p is not None and at.session_state.condition == "AU_open" and not at.session_state.hidden
    assert 19.5 <= p.elapsed() <= 25 and not at.exception
    at2 = AppTest.from_file(APP, default_timeout=120)
    at2.query_params["view"] = "results"
    at2.run()
    assert at2.tabs[0].label == "Results" and not at2.exception
