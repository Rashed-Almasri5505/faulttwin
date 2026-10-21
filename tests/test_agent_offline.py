"""Phase 3 logic without network: interlock, tools, agent loop checks, Tavily layer, report hedge, budget."""

import copy
import json
import threading
from pathlib import Path

import pytest

from faulttwin.agent import DiagnosisAgent, Trace
from faulttwin.config import load_config
from faulttwin.hypotheses import Hypothesis
from faulttwin.llm import BudgetExceeded, SpendLedger, TokenFactory
from faulttwin.report import write_work_order
from faulttwin.runtime import build_session, run_to_event
from faulttwin.scenarios import SCENARIOS
from faulttwin.tools import Nameplate, run_tool, tool_specs
from faulttwin.web import TavilyWeb
from tests.fakes import FakeOpenAI, FakeTavily, chat_response, stream_chunks

CFG = load_config()
SEED = 6101   # not a calibration, test-sweep or review seed
MODULE, DRIVER = "A1P50S65M2", "STDRIVE601"
PDF = "https://www.st.com/resource/en/datasheet/a1p50s65m2.pdf"
DATASHEET_TEXT = ("A1P50S65M2 ACEPACK 1 sixpack topology, 650 V, 50 A trench gate field-stop IGBT. "
                  "Table 1. Absolute maximum ratings of the IGBT VCES Collector-emitter voltage (VGE = 0 V) 650 V "
                  "IC Continuous collector current (TC = 100 C) 50 A. VCE(sat) Collector-emitter saturation voltage "
                  "VGE = 15 V, IC = 50 A 1.65 V. VF Diode forward voltage IF = 50 A 1.55 V. " + "x" * 6000)
SEARCH_RESULTS = [
    {"title": "A1P50S65M2 datasheet", "url": PDF},
    {"title": "IGBT open-circuit fault review", "url": "https://example.org/igbt-open-circuit"},
    {"title": "A1P50S65M2 at a distributor", "url": "https://www.mouser.com/ProductDetail/A1P50S65M2"},
]


def fake_web(tmp_path: Path, **kw) -> TavilyWeb:
    return TavilyWeb(CFG, client=FakeTavily(pages={PDF: DATASHEET_TEXT}, results=SEARCH_RESULTS, **kw),
                     cache_dir=tmp_path / "tavily")


def facts_reply(quote_ic: str = "IC Continuous collector current (TC = 100 C) 50 A") -> str:
    return json.dumps({
        "V_CES_V": {"value": 650, "condition": None, "quote": "VCES Collector-emitter voltage (VGE = 0 V) 650 V"},
        "I_C_nom_A": {"value": 50, "condition": "T_C = 100 C", "quote": quote_ic},
        "V_CE_sat_typ_V": {"value": 1.65, "condition": "IC = 50 A", "quote": "IC = 50 A 1.65 V"},
        "V_F_typ_V": {"value": 1.55, "condition": "IF = 50 A", "quote": "IF = 50 A 1.55 V"},
    })


def make_llm(responder, tmp_path: Path, cfg=CFG) -> TokenFactory:
    ledger = SpendLedger(tmp_path / "spend.json", daily_cap_usd=1.0, total_cap_usd=15.0)
    return TokenFactory(cfg, ledger=ledger, client=FakeOpenAI(responder))


def session_for(scenario: str, tmp_path: Path, responder=None):
    llm = make_llm(responder or (lambda kw: chat_response(facts_reply())), tmp_path)
    s = build_session(CFG, seed=SEED, llm=llm, web=fake_web(tmp_path))
    s.nameplate = Nameplate(igbt_module_part_number=MODULE, gate_driver_part_number=DRIVER, inverter_rating="test")
    event = run_to_event(s, SCENARIOS[scenario])
    return s, event


def call(s, name: str, **args):
    return run_tool(s, {t.name: t for t in tool_specs(s.nameplate)}, name, json.dumps(args))


DEFAULT = ["healthy", "AU_open", "AL_open", "BU_open", "BL_open", "CU_open", "CL_open", "leg_A_open", "leg_B_open",
           "leg_C_open", "sensor_offset_A", "sensor_offset_B", "sensor_offset_C", "sensor_gain_A", "sensor_gain_B",
           "sensor_gain_C", "load_unbalance_A", "load_unbalance_B", "load_unbalance_C", "load_step"]


# --- interlock and tools ------------------------------------------------------------------------

def test_derate_is_decided_by_the_interlocks_own_ranking_and_the_datasheet(tmp_path: Path) -> None:
    s, event = session_for("AU_open", tmp_path)
    assert event is not None and event.type in ("mean_current", "amplitude_ratio")
    # The interlock ranks all causes itself; what the model tested does not matter.
    call(s, "simulate_hypotheses", hypotheses=["healthy", "sensor_gain_A", "load_step"])
    no_ds = s.interlock.check_action(s, "derate", None)
    assert no_ds.approved and no_ds.applied["m"] == CFG["interlock"]["default_derate_m"] and no_ds.needs_datasheet
    ds = call(s, "web_datasheet_lookup", part_number=MODULE)
    assert ds["datasheet"]["I_C_nom"].startswith("50 A")
    approved = call(s, "apply_action", action="derate", reason="AU open confirmed")["interlock"]
    assert approved["approved"] and not approved["needs_datasheet"]
    assert approved["applied"]["predicted_peak_A"] <= 25.0 + 1e-6      # 50 % of I_C,nom = 50 A, with model margin
    assert s.plant.m == approved["applied"]["m"] < 0.8 and "0.80 ->" in approved["reason"]


@pytest.mark.parametrize("scenario", ["AU_open", "BL_open", "CU_open", "leg_A_open"])
@pytest.mark.parametrize("seed", [6004, 6008])   # seeds where the plant's impedance is below nominal
def test_real_plant_stays_within_the_derate_limit(scenario: str, seed: int, tmp_path: Path) -> None:
    llm = make_llm(lambda kw: chat_response(facts_reply()), tmp_path)
    s = build_session(CFG, seed=seed, llm=llm, web=fake_web(tmp_path))
    s.nameplate = Nameplate(igbt_module_part_number=MODULE, gate_driver_part_number=DRIVER)
    run_to_event(s, SCENARIOS[scenario])
    call(s, "web_datasheet_lookup", part_number=MODULE)
    assert call(s, "apply_action", action="derate", reason="x")["interlock"]["approved"]
    s.advance(4)                                   # let the pre-derate current die out
    peak = max(abs(c.i_true_A).max() for c in [s.plant.run_cycles(1) for _ in range(6)])
    assert peak <= 25.0, f"true peak {peak:.2f} A after the derate"


def test_upward_test_is_refused_whenever_a_fault_is_plausible_or_the_plant_is_derated(tmp_path: Path) -> None:
    s, _ = session_for("AU_open", tmp_path)
    call(s, "simulate_hypotheses", hypotheses=["healthy"])               # a model hiding the fault
    s.plant.set_modulation_index(0.6)
    s.advance(4)
    v = call(s, "run_active_test", m_new=0.9)["interlock"]
    assert not v["approved"] and "converter fault is plausible" in v["reason"]
    s2, _ = session_for("sensor_offset_A", tmp_path)
    s2.derated_m = 0.6
    assert not call(s2, "run_active_test", m_new=0.9)["interlock"]["approved"]


def test_resume_after_a_stop_resumes_derated_for_the_current_fault(tmp_path: Path) -> None:
    s, _ = session_for("AU_open", tmp_path)
    call(s, "web_datasheet_lookup", part_number=MODULE)
    s.derated_m, s.derated_for = 0.8, Hypothesis.leg_B_open            # stale derate from another fault
    assert call(s, "apply_action", action="stop", reason="x")["interlock"]["approved"]
    resume = call(s, "apply_action", action="resume", reason="x")["interlock"]
    assert resume["approved"] and resume["applied"]["m"] < 0.8 and resume["applied"]["fault"] == "AU_open"
    assert not s.plant.stopped and s.plant.m == resume["applied"]["m"]


def test_derate_not_needed_is_reported_as_such(tmp_path: Path) -> None:
    s, _ = session_for("leg_B_open", tmp_path)
    call(s, "web_datasheet_lookup", part_number=MODULE)
    v = call(s, "apply_action", action="derate", reason="x")["interlock"]
    assert v["approved"] and v["request"].endswith("(not needed)") and v["applied"]["m"] == 0.8


def test_session_evidence_is_reset_for_each_diagnosis(tmp_path: Path) -> None:
    s, _ = session_for("AU_open", tmp_path)
    call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    call(s, "run_active_test", m_new=0.5)
    call(s, "web_failure_evidence", hypothesis="AU_open", part_number=MODULE)
    s.begin_diagnosis()
    assert s.last_ranking is None and s.last_test is None and not s.diag_sources and not s.diag_verdicts()
    assert s.sources   # the session-wide list for the UI is kept


def test_derate_is_rejected_for_a_look_alike(tmp_path: Path) -> None:
    s, _ = session_for("sensor_offset_A", tmp_path)
    call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    assert s.last_ranking[0].hypothesis == Hypothesis.sensor_offset_A
    assert not call(s, "apply_action", action="derate", reason="x")["interlock"]["approved"]
    assert s.plant.m == 0.8


def test_stop_and_resume_rules(tmp_path: Path) -> None:
    s, _ = session_for("AU_open", tmp_path)
    call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    assert call(s, "apply_action", action="stop", reason="x")["interlock"]["approved"] and s.plant.stopped
    assert "error" in call(s, "simulate_hypotheses", hypotheses=DEFAULT)    # a stopped plant is not diagnosable
    assert not call(s, "apply_action", action="derate", reason="x")["interlock"]["approved"]
    resume = call(s, "apply_action", action="resume", reason="x")["interlock"]
    assert resume["approved"] and resume["applied"]["m"] == CFG["interlock"]["default_derate_m"]   # no datasheet


@pytest.mark.parametrize(("scenario", "expected"), [("AU_open", "switch_fault"), ("sensor_offset_A", "sensor_offset")])
def test_modulation_step_test_separates_offset_from_switch_fault(scenario: str, expected: str, tmp_path: Path) -> None:
    s, _ = session_for(scenario, tmp_path)
    call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    result = call(s, "run_active_test", test="modulation_step", m_new=0.5, n_cycles=6)
    assert result["interlock"]["approved"]
    assert result["analysis"]["consistent_with"] == expected
    assert s.plant.m == 0.8                                                  # returned to the operating point


def test_active_test_limits(tmp_path: Path) -> None:
    s, _ = session_for("AU_open", tmp_path)
    call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    assert not call(s, "run_active_test", m_new=0.9)["interlock"]["approved"]     # raises m on a faulty converter
    assert not call(s, "run_active_test", m_new=0.75)["interlock"]["approved"]    # step too small
    assert not call(s, "run_active_test", m_new=1.5)["interlock"]["approved"]     # outside [0.3, 0.9]
    assert call(s, "run_active_test", m_new=0.5, n_cycles=99)["interlock"]["applied"]["n_cycles"] == 10


def test_tool_argument_errors_come_back_to_the_model(tmp_path: Path) -> None:
    s, _ = session_for("AU_open", tmp_path)
    specs = {t.name: t for t in tool_specs(s.nameplate)}
    assert "error" in call(s, "simulate_hypotheses", hypotheses=["not_a_cause"])
    assert "error" in call(s, "web_datasheet_lookup", part_number="SOMETHING-ELSE")
    assert "error" in run_tool(s, {}, "no_such_tool", "{}")
    for bad in ("[1, 2]", "42", "\"text\"", "{not json", "null"):
        result = run_tool(s, specs, "run_active_test", bad)
        assert "error" in result or not result.get("interlock", {}).get("approved", True), bad


def test_every_mystery_scenario_raises_an_event(tmp_path: Path) -> None:
    from faulttwin.scenarios import MYSTERY_POOL
    for name in MYSTERY_POOL:
        for seed in (6300, 6301):
            llm = make_llm(lambda kw: chat_response("{}"), tmp_path)
            s = build_session(CFG, seed=seed, llm=llm, web=fake_web(tmp_path))
            assert run_to_event(s, SCENARIOS[name]) is not None, f"{name} seed {seed}"


# --- agent loop ---------------------------------------------------------------------------------

def scripted_agent(turns: list):
    """Responder that plays scripted agent turns, answers triage and datasheet calls."""
    queue = list(turns)

    def responder(kw):
        if kw.get("stream"):
            return stream_chunks("# Work order: upper switch of leg A open")
        fmt = (kw.get("response_format") or {}).get("json_schema", {}).get("name")
        if fmt == "triage":
            return chat_response(json.dumps({"family": "switch_or_leg_fault", "urgency": "high", "escalate": True,
                                             "reason": "mu_a is -0.42"}))
        if fmt == "datasheet_facts":
            return chat_response(facts_reply())
        return queue.pop(0) if queue else chat_response("I am done.")
    return responder


def final_args(label: str, citations=None, parts=None) -> dict:
    return {"label": label, "confidence": 0.9, "summary": "s", "evidence": ["e"], "alternatives_rejected": [],
            "tests_run": [], "actions": [], "citations": citations or [], "recommended_maintenance": ["replace"],
            "replacement_parts": parts or []}


def test_agent_completes_an_au_open_investigation(tmp_path: Path) -> None:
    turns = [
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})], reasoning="think"),
        # top 3 holds a switch fault and a sensor error, so the modulation-step test is required
        chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
        chat_response(tool_calls=[("web_datasheet_lookup", {"part_number": MODULE})]),
        chat_response(tool_calls=[("apply_action", {"action": "derate", "reason": "AU open"})]),
        chat_response(tool_calls=[("web_failure_evidence", {"hypothesis": "AU_open", "part_number": MODULE}),
                                  ("web_replacement_parts", {"part_number": MODULE})]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args(
            "AU_open", citations=[{"title": "review", "url": "https://example.org/igbt-open-circuit"},
                                  {"title": "made up", "url": "https://invented.example/x"}]))]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args(
            "AU_open", citations=[{"title": "review", "url": "https://example.org/igbt-open-circuit"}]))]),
    ]
    s, event = session_for("AU_open", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="AU_open")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    assert final.label == "AU_open"
    assert final.twin_match_error_pct is not None
    # The invented URL is dropped; the datasheet that set the derate limit is cited by code.
    assert [c["url"] for c in final.citations] == ["https://example.org/igbt-open-circuit", PDF]
    finals = [x for x in trace.steps if x.get("tool") == "finalize_diagnosis"]
    assert finals[0]["result_summary"].startswith("rejected") and "citations" in finals[0]["result_summary"]
    assert trace.totals["tool_calls"] == 9 and trace.overlays is not None
    assert trace.active_test["analysis"]["consistent_with"] == "switch_fault"
    assert any(v["request"].startswith("derate") and v["approved"] for v in trace.actions)
    assert trace.steps[0]["kind"] == "triage"
    json.loads(trace.to_json())
    work_order = write_work_order(s.llm, CFG, trace.final, s, "test inverter")
    assert work_order.markdown.startswith("# Work order")


def test_agent_label_must_agree_with_the_twin_and_converter_faults_need_action(tmp_path: Path) -> None:
    turns = [
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
        chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args("BL_open"))]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args("AU_open"))]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args("AU_open"))]),
    ]
    s, event = session_for("AU_open", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="AU_open")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    summaries = [x["result_summary"] for x in trace.steps if x.get("tool") == "finalize_diagnosis"]
    assert "the twin's best match is AU_open" in summaries[0]
    assert "apply_action" in summaries[1]
    # Still failing a hard check after two rejections: the answer becomes inconclusive, not AU_open.
    assert final.label == "inconclusive" and final.proposed_label == "AU_open"
    assert any("apply_action" in n for n in trace.validation_notes)
    assert final.tests_run and any("modulation step" in t for t in final.tests_run)   # gathered evidence is kept


def test_json_fallback_and_tool_call_limit(tmp_path: Path) -> None:
    turns = [chat_response(json.dumps({"action": "get_latest_features", "args": {}}))] * 12
    s, event = session_for("sensor_offset_A", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="sensor_offset_A")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    tools = [x for x in trace.steps if x["kind"] == "tool" and not x["result_summary"].startswith("background")]
    assert len(tools) == 10 and all(x["via_json_fallback"] for x in tools)
    assert final.label == "inconclusive" and "tool-call limit" in trace.stop_reason


def test_m_test_is_required_when_a_sensor_error_and_a_switch_fault_compete(tmp_path: Path) -> None:
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    turns = [
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": ["sensor_offset_A", "AU_open", "healthy"]})]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args("sensor_offset_A"))]),
        chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
        chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": ["sensor_offset_A", "AU_open", "healthy"]}),
                                  ("web_failure_evidence", {"hypothesis": "sensor_offset_A", "part_number": MODULE})]),
        chat_response(tool_calls=[("finalize_diagnosis", final_args("sensor_offset_A", citations=review))]),
    ]
    s, event = session_for("sensor_offset_A", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="sensor_offset_A")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    summaries = [x["result_summary"] for x in trace.steps if x.get("tool") == "finalize_diagnosis"]
    assert "run_active_test" in summaries[0] and "web_failure_evidence" in summaries[0]
    assert summaries[-1] == "accepted as sensor_offset_A" and final.label == "sensor_offset_A"
    assert trace.active_test["analysis"]["consistent_with"] == "sensor_offset"


def test_claims_about_a_test_that_never_ran_are_sent_back(tmp_path: Path) -> None:
    claim = dict(final_args("sensor_offset_A"), summary="The offset scales inversely with |i_s| in the m-step test.")
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("finalize_diagnosis", claim)])]
    s, event = session_for("sensor_offset_A", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="sensor_offset_A")
    DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    first = next(x for x in trace.steps if x.get("tool") == "finalize_diagnosis")
    assert "mentions a modulation or active test, but none was run" in first["result_summary"]


def test_invented_numbers_are_sent_back_and_finally_removed(tmp_path: Path) -> None:
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    invented = dict(final_args("sensor_offset_A", citations=review),
                    summary="The offset is 2.0 A. The sensor drifted 7.77 A last week.",
                    evidence=["mu_a measured", "temperature was 93.4 C"])
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
             chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT}),
                                       ("web_failure_evidence", {"hypothesis": "sensor_offset_A", "part_number": MODULE})]),
             *[chat_response(tool_calls=[("finalize_diagnosis", invented)])] * 3]
    s, event = session_for("sensor_offset_A", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="sensor_offset_A")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    first = next(x for x in trace.steps if x.get("tool") == "finalize_diagnosis")
    assert "7.77" in first["result_summary"] and "93.4" in first["result_summary"]
    assert final.label == "sensor_offset_A"
    assert "7.77" not in final.summary and all("93.4" not in e for e in final.evidence)


def test_a_final_answer_at_the_limit_is_settled_not_thrown_away(tmp_path: Path) -> None:
    # leg_B_open: the modulation test cannot separate the causes, and the answer still draws a conclusion from it.
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    answer = dict(final_args("leg_B_open", citations=review),
                  summary="The twin's best match is an open leg B. The modulation step test was consistent with a "
                          "switch fault.",
                  evidence=["phase B carries no current", "active test: mu unchanged, consistent with a switch fault"])
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
             chat_response(tool_calls=[("apply_action", {"action": "derate", "reason": "leg B open"}),
                                       ("web_failure_evidence", {"hypothesis": "leg_B_open", "part_number": MODULE})]),
             # the 11th call of the investigation arrives in the turn that crosses the limit of 10
             chat_response(tool_calls=[("get_latest_features", {})] * 6 + [("finalize_diagnosis", answer)])]
    s, event = session_for("leg_B_open", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="leg_B_open")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    assert not trace.active_test["analysis"]["informative"]
    assert final.label == "leg_B_open"
    assert "consistent with a switch fault" not in final.summary and "open leg B" in final.summary
    assert final.evidence == ["phase B carries no current"]
    assert any("not informative" in n for n in trace.validation_notes)
    assert trace.totals["tool_calls"] == 10                  # the settled answer is not counted as an 11th call


def test_a_switch_described_with_the_wrong_side_is_sent_back_and_finally_removed(tmp_path: Path) -> None:
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    wrong = dict(final_args("AU_open", citations=review),
                 summary="The lower switch of phase A is open. The twin confirms it.",
                 evidence=["upper switch of leg A open", "phase A lower switch failed"],
                 alternatives_rejected=[{"label": "AL_open", "why": "the lower switch would give the opposite sign"}])
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
             chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT}),
                                       ("web_datasheet_lookup", {"part_number": MODULE})]),
             chat_response(tool_calls=[("apply_action", {"action": "derate", "reason": "AU open"}),
                                       ("web_failure_evidence", {"hypothesis": "AU_open", "part_number": MODULE})]),
             *[chat_response(tool_calls=[("finalize_diagnosis", wrong)])] * 3]
    s, event = session_for("AU_open", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="AU_open")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    first = next(x for x in trace.steps if x.get("tool") == "finalize_diagnosis")
    assert "upper switch of leg A" in first["result_summary"] and "do not call it the lower switch" in first["result_summary"]
    assert final.label == "AU_open"
    assert "lower switch" not in final.summary and final.evidence == ["upper switch of leg A open"]
    assert final.alternatives_rejected                     # naming the rejected AL_open's side is fine


@pytest.mark.parametrize("scenario, family", [
    ("AU_open", "switch_or_leg_fault"), ("BL_open", "switch_or_leg_fault"), ("leg_B_open", "switch_or_leg_fault"),
    ("sensor_offset_A", "sensor_error"), ("sensor_offset_C", "sensor_error"), ("sensor_gain_A", "sensor_error"),
    ("sensor_gain_B", "sensor_error"),
])
def test_the_triage_rule_table_points_to_the_right_family(scenario: str, family: str, tmp_path: Path) -> None:
    from faulttwin.agent import triage_checks
    s, event = session_for(scenario, tmp_path)
    rules = triage_checks(s.features(s.steady_window()).compact(), s.monitor.th)
    first = next(r for r in rules if r["holds"])
    assert first["family_if_holds"] == family, rules


def test_tool_call_limit_holds_inside_one_turn(tmp_path: Path) -> None:
    many = [("get_latest_features", {})] * 14
    s, event = session_for("sensor_offset_A", tmp_path, scripted_agent([chat_response(tool_calls=many)]))
    trace = Trace(scenario="sensor_offset_A")
    DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    ran = [x for x in trace.steps if x.get("tool") == "get_latest_features" and "error" not in x["result"]]
    assert len(ran) == 10 and trace.totals["tool_calls"] == 10


def test_diagnosis_accepts_list_fields_sent_as_json_strings() -> None:
    from faulttwin.agent import Diagnosis
    d = Diagnosis.model_validate({"label": "healthy", "confidence": 0.5, "summary": "s",
                                  "alternatives_rejected": json.dumps([{"label": "AU_open", "why": "w"}])})
    assert d.alternatives_rejected == [{"label": "AU_open", "why": "w"}]


def test_work_order_parts_and_sources_come_from_code_and_urls_are_removed(tmp_path: Path) -> None:
    def responder(kw):
        return stream_chunks("# Work order: x\n## Diagnosis\nSee https://evil.example/x for details.\n"
                             "## Sources\n- [made up](https://invented.example/y)")
    s, _ = session_for("AU_open", tmp_path, responder)
    call(s, "web_failure_evidence", hypothesis="AU_open", part_number=MODULE)
    final = final_args("AU_open", citations=[{"title": "IGBT open-circuit fault review",
                                              "url": "https://example.org/igbt-open-circuit"}])
    wo = write_work_order(s.llm, CFG, final, s, "inv")
    assert "evil.example" not in wo.markdown and "invented.example" not in wo.markdown
    assert "## Sources\n- [IGBT open-circuit fault review](https://example.org/igbt-open-circuit) · via Tavily" in wo.markdown


def test_work_order_cut_off_at_max_tokens_is_not_used(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["llm"]["report"]["max_tokens"] = 1

    def responder(kw):
        chunks = list(stream_chunks(f"# Work order by {kw['model']}"))
        chunks[-2].choices[0].finish_reason = "length" if "Ultra" in kw["model"] else "stop"
        return iter(chunks)
    s, _ = session_for("AU_open", tmp_path, responder)
    wo = write_work_order(s.llm, cfg, final_args("AU_open"), s, "inv")
    assert wo.ultra_status == "cut off" and "super" in (wo.model or "").lower()
    assert any("cut off" in n for n in wo.notes)


def test_corrupt_spend_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    path.write_text('{"total_usd": 0.1, "days": {"20', encoding="utf-8")
    ledger = SpendLedger(path, daily_cap_usd=1.0, total_cap_usd=15.0)
    with pytest.raises(BudgetExceeded):
        ledger.check()


def test_spend_ledger_waits_for_a_file_held_open_elsewhere(tmp_path: Path, monkeypatch) -> None:
    # On Windows a reader in another process (the app) makes the rename fail with a sharing violation.
    import faulttwin.llm as llm_mod
    real, fails = llm_mod.os.replace, {"n": 3}

    def busy(src, dst):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise PermissionError(5, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(llm_mod.os, "replace", busy)
    ledger = SpendLedger(tmp_path / "spend.json", daily_cap_usd=1.0, total_cap_usd=15.0)
    ledger.add(0.01)
    assert ledger.spent()[1] == pytest.approx(0.01)
    fails["n"] = 10 ** 6                                     # held open for good: refuse, never crash or leak files
    with pytest.raises(BudgetExceeded):
        ledger.add(0.01)
    assert not list(tmp_path.glob("*.tmp"))


def test_manufacturer_domains_are_checked_on_the_host() -> None:
    from faulttwin.web import _is_manufacturer
    assert _is_manufacturer("https://www.st.com/resource/en/datasheet/a1p50s65m2.pdf")
    assert not _is_manufacturer("https://evil.example/st.com/a1p50s65m2.pdf")
    assert not _is_manufacturer("https://st.com.evil.example/x.pdf")


def test_datasheet_quote_must_name_the_parameter(tmp_path: Path) -> None:
    wrong = json.loads(facts_reply())
    wrong["I_C_nom_A"] = {"value": 650, "condition": None, "quote": "VCES Collector-emitter voltage (VGE = 0 V) 650 V"}
    llm = make_llm(lambda kw: chat_response(json.dumps(wrong)), tmp_path)
    facts = fake_web(tmp_path).datasheet_facts(MODULE, llm, cache=False)
    assert not facts.I_C_nom.verified and facts.I_C_nom.value is None
    assert any("I_C_nom" in n for n in facts.notes)


def test_every_live_diagnosis_makes_a_tavily_call_even_when_stopped_early(tmp_path: Path) -> None:
    s, event = session_for("AU_open", tmp_path, scripted_agent([]))   # the model never calls a tool
    trace = Trace(scenario="AU_open")
    DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    assert s.web.live_calls >= 1
    assert any(x.get("result_summary", "").startswith("background evidence") for x in trace.steps)


# --- Tavily layer -------------------------------------------------------------------------------

def test_first_search_of_a_diagnosis_is_live_then_cached(tmp_path: Path) -> None:
    web = fake_web(tmp_path)
    assert web.search("q")[0].via == "live"
    assert web.search("q")[0].via == "cached"
    web.begin_diagnosis()
    assert web.search("q")[0].via == "live"
    assert web.live_calls == 2 and web.credits_used == 2


def test_blocked_pages_and_pdf_retry(tmp_path: Path) -> None:
    blocked = "Access Denied. You are using automation tools to browse the website."
    client = FakeTavily(pages={PDF: DATASHEET_TEXT, "https://www.mouser.com/x.pdf": blocked}, fail_basic_pdf=True)
    web = TavilyWeb(CFG, client=client, cache_dir=tmp_path / "t")
    pages = {p.url: p for p in web.extract([PDF, "https://www.mouser.com/x.pdf"])}
    assert pages[PDF].depth == "advanced" and not pages[PDF].blocked
    assert "https://www.mouser.com/x.pdf" not in web.seen_urls


def test_datasheet_values_need_a_quote_that_is_really_in_the_page(tmp_path: Path) -> None:
    llm = make_llm(lambda kw: chat_response(facts_reply(quote_ic="IC nominal current 50 A")), tmp_path)
    facts = fake_web(tmp_path).datasheet_facts(MODULE, llm)
    assert facts.V_CES.verified and facts.V_CES.value == 650
    assert not facts.I_C_nom.verified and any("I_C_nom" in n for n in facts.notes)
    assert facts.source_url == PDF


# --- report and budget --------------------------------------------------------------------------

def _report_session(tmp_path: Path, responder):
    s, _ = session_for("AU_open", tmp_path, responder)
    return s, final_args("AU_open")


def test_work_order_uses_ultra_when_it_is_fast(tmp_path: Path) -> None:
    def responder(kw):
        return stream_chunks(f"# Work order by {kw['model']}")
    s, final = _report_session(tmp_path, responder)
    wo = write_work_order(s.llm, CFG, final, s, "inv")
    assert wo.ultra_status == "used" and "Ultra" in wo.model


def test_work_order_falls_back_to_super_when_ultra_is_late(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["llm"]["report"].update(first_token_deadline_s=0.3, total_deadline_s=0.6)

    def responder(kw):
        if "Ultra" in kw["model"]:
            return stream_chunks("# Ultra review", first_delay_s=1.0)
        return stream_chunks("# Work order by Super")
    s, final = _report_session(tmp_path, responder)
    wo = write_work_order(s.llm, cfg, final, s, "inv")
    assert wo.ultra_status == "late" and "super" in wo.model.lower()
    assert wo.senior_review is not None and wo.senior_review.result(timeout=5).content.startswith("# Ultra")


def test_work_order_without_any_model_uses_the_template(tmp_path: Path) -> None:
    s, final = _report_session(tmp_path, None)
    s.llm._client = None
    wo = write_work_order(s.llm, CFG, final, s, "inv")
    assert wo.ultra_status == "not available" and wo.markdown.startswith("# Work order")


def test_spend_ledger_enforces_the_caps(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", daily_cap_usd=0.01, total_cap_usd=1.0)
    ledger.add(0.009)
    ledger.check()
    ledger.add(0.002)
    with pytest.raises(BudgetExceeded):
        ledger.check()
    llm = TokenFactory(CFG, ledger=ledger, client=FakeOpenAI(lambda kw: chat_response("x")))
    with pytest.raises(BudgetExceeded):
        llm.chat("triage", [{"role": "user", "content": "hi"}], label="t")


def test_reasoning_is_split_from_the_answer(tmp_path: Path) -> None:
    llm = make_llm(lambda kw: chat_response("<think>private</think>Answer", reasoning="more"), tmp_path)
    res = llm.chat("agent", [{"role": "user", "content": "q"}], label="t", thinking="low")
    assert res.content == "Answer" and "private" in res.reasoning and "more" in res.reasoning
    sent = llm._client.requests[-1]
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
    assert res.call.cost_usd > 0


# --- review round 2: evidence relevance, text checks ------------------------------------------------

OFF_TOPIC = [{"title": "CAUSE | English meaning - Cambridge Dictionary", "url": "https://dictionary.example/cause",
              "content": "the reason why something happens", "score": 0.03},
             {"title": "What Causes ME/CFS", "url": "https://health.example/mecfs", "content": "doctors do not know",
              "score": 0.02}]
ON_TOPIC = {"title": "Hall-effect current sensor drift", "url": "https://ti.example/hall-drift",
            "content": "## Gain drift\n- [Sign in](x)\nCurrent sensor gain drifts with **temperature**.", "score": 0.6}


def test_off_topic_failure_evidence_is_dropped_and_retried_on_technical_sites(tmp_path: Path) -> None:
    client = FakeTavily(results=OFF_TOPIC + [ON_TOPIC])
    web = TavilyWeb(CFG, client=client, cache_dir=tmp_path / "t1")
    found = web.failure_evidence("sensor_gain", "inverter phase current sensor")
    assert [s.url for s in found] == [ON_TOPIC["url"]]
    assert found[0].snippet == "Gain drift Current sensor gain drifts with temperature."     # markdown removed
    assert "inverter phase current sensor inverter" not in client.calls[0][1]["query"]        # no repeated words
    only_off = FakeTavily(results=OFF_TOPIC)
    web = TavilyWeb(CFG, client=only_off, cache_dir=tmp_path / "t2")
    assert web.failure_evidence("sensor_gain", "inverter phase current sensor") == []
    assert [c[1].get("include_domains") is not None for c in only_off.calls] == [False, True]   # one retry


def test_a_diagnosis_without_relevant_sources_is_not_forced_to_cite(tmp_path: Path) -> None:
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("web_failure_evidence", {"hypothesis": "sensor_gain_B", "part_number": MODULE})]),
             chat_response(tool_calls=[("finalize_diagnosis", final_args("sensor_gain_B"))])]
    llm = make_llm(scripted_agent(turns), tmp_path)
    web = TavilyWeb(CFG, client=FakeTavily(results=OFF_TOPIC), cache_dir=tmp_path / "tavily")
    s = build_session(CFG, seed=SEED, llm=llm, web=web)
    s.nameplate = Nameplate(igbt_module_part_number=MODULE, gate_driver_part_number=DRIVER, inverter_rating="test")
    event = run_to_event(s, SCENARIOS["sensor_gain_B"])
    trace = Trace(scenario="sensor_gain_B")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    evidence = next(x for x in trace.steps if x.get("tool") == "web_failure_evidence")
    assert evidence["result"]["sources"] == [] and "No relevant source" in evidence["result"]["note"]
    assert final.label == "sensor_gain_B" and final.citations == []
    assert final.recommended_maintenance[0].startswith("Check the gain of the phase B current sensor")


def test_number_checks_read_unicode_minus_signs() -> None:
    from faulttwin.report import _check_prose, _known_numbers
    facts = {"diagnosis": {"label": "sensor_offset_C", "summary": "an offset of about \u20132.5 A"}}
    known = _known_numbers(facts)
    assert known == {-2.5}                               # not 20132.5 from the escaped "–" + "2.5"
    assert _check_prose("The phase C sensor reads \u22122.5 A too low.", known) == []


def test_work_order_text_must_name_the_diagnosed_cause_and_side() -> None:
    from faulttwin.report import _check_prose
    facts = {"diagnosis": {"label": "leg_B_open"}, "evidence": ["leg_B_open match error 1.3 %",
                                                                "Rejected: sensor_gain_B 88 %"]}
    assert _check_prose("Replace the lower IGBT (BL_open) of leg B.", set(), facts)
    assert not _check_prose("Both the upper and lower switch of leg B are open; sensor_gain_B was rejected.",
                            set(), facts)
    au = {"diagnosis": {"label": "AU_open"}}
    assert _check_prose("The lower switch of leg A failed.", set(), au)
    assert not _check_prose("The upper switch of leg A failed.", set(), au)


def test_part_links_are_not_cited_as_evidence(tmp_path: Path) -> None:
    sensor = {"title": "Current sensor gain drift", "url": "https://ti.example/gain",
              "content": "current sensor gain drift in motor drives", "score": 0.6}
    shop = {"title": "Buy A1P50S65M2", "url": "https://shop.example/A1P50S65M2",
            "content": "A1P50S65M2 IGBT module in stock", "score": 0.5}
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT}),
                                       ("web_failure_evidence", {"hypothesis": "sensor_gain_B", "part_number": MODULE}),
                                       ("web_replacement_parts", {"part_number": MODULE})]),
             chat_response(tool_calls=[("finalize_diagnosis", final_args(
                 "sensor_gain_B", citations=[{"title": "t", "url": sensor["url"]}, {"title": "t", "url": shop["url"]}]))])]
    llm = make_llm(scripted_agent(turns), tmp_path)
    web = TavilyWeb(CFG, client=FakeTavily(results=[sensor, shop]), cache_dir=tmp_path / "tavily")
    s = build_session(CFG, seed=SEED, llm=llm, web=web)
    s.nameplate = Nameplate(igbt_module_part_number=MODULE, gate_driver_part_number=DRIVER, inverter_rating="test")
    event = run_to_event(s, SCENARIOS["sensor_gain_B"])
    final = DiagnosisAgent(s, s.llm, CFG).run(event, Trace(scenario="sensor_gain_B"))
    assert final.label == "sensor_gain_B" and [c["url"] for c in final.citations] == [sensor["url"]]


def test_negated_test_mentions_are_not_claims() -> None:
    from faulttwin.agent import claims_test
    assert not claims_test("This explains the residual without requiring any active test.")
    assert not claims_test("No modulation step test was needed.")
    assert claims_test("The modulation step test showed that mu stays the same.")


def test_a_claim_about_a_test_that_never_ran_is_removed_at_the_end(tmp_path: Path) -> None:
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    claim = dict(final_args("sensor_gain_B", citations=review),
                 summary="Sensor B has a gain error. The modulation step test confirmed it.",
                 alternatives_rejected=[{"label": "BL_open", "why": "the active test showed no mean current"}])
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT}),
                                       ("web_failure_evidence", {"hypothesis": "sensor_gain_B", "part_number": MODULE})]),
             *[chat_response(tool_calls=[("finalize_diagnosis", claim)])] * 3]
    s, event = session_for("sensor_gain_B", tmp_path, scripted_agent(turns))
    trace = Trace(scenario="sensor_gain_B")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    assert final.label == "sensor_gain_B"
    assert final.summary == "Sensor B has a gain error." and final.alternatives_rejected == []


def test_first_maintenance_step_names_the_switch_and_the_six_pack(tmp_path: Path) -> None:
    from faulttwin.agent import first_maintenance_step
    from faulttwin.hypotheses import SPECS, Hypothesis
    plate = Nameplate(igbt_module_part_number=MODULE, gate_driver_part_number=DRIVER)
    step = first_maintenance_step(SPECS[Hypothesis.AU_open], plate)
    assert "AU (S1)" in step and DRIVER in step and "six-pack" in step and MODULE in step
    assert "leg B" in first_maintenance_step(SPECS[Hypothesis.leg_B_open], plate)


def test_ultras_late_review_gets_the_same_checks_as_the_work_order() -> None:
    from concurrent.futures import Future
    from faulttwin.llm import ChatResult, LLMCall
    from faulttwin.report import WorkOrder
    late: Future = Future()
    late.set_result(ChatResult(content="The offset is 9.99 A.", reasoning=None, tool_calls=[],
                               call=LLMCall(label="report", model="nvidia/Nemotron-3-Ultra-550b-a55b",
                                            finish_reason="stop")))
    wo = WorkOrder("# Work order: x", "nvidia/nemotron-3-super-120b-a12b", 3.0, "late", late,
                   known={2.0}, facts={"diagnosis": {"label": "sensor_offset_A"}})
    assert wo.as_dict()["senior_review"] is None and any("not shown" in n for n in wo.notes)


def test_alarms_raised_by_the_agents_own_test_are_labelled(tmp_path: Path) -> None:
    s, _ = session_for("sensor_offset_A", tmp_path)
    call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    n = len(s.events)
    call(s, "run_active_test", test="modulation_step", m_new=0.4, n_cycles=8)
    during = [e for e in s.events[n:] if "modulation-step test" in e.message]
    assert during, [e.message for e in s.events[n:]]


# --- Phase 5 preparation: low load, the Tavily ablation, datasheet table columns ---------------------

def test_typ_column_is_read_from_the_quoted_row() -> None:
    from faulttwin.web import DatasheetFacts, FactValue, _fix_typ_columns, typ_from_row
    assert typ_from_row("VCE(sat) (terminal) Collector-emitter saturation voltage VGE = 15 V, I C= 50 A 1.95 2.3") == 1.95
    assert typ_from_row("VF (terminal) Forward voltage IF = 50 A - 1.85 2.65 V") == 1.85
    assert typ_from_row("IF = 50 A 1.55 V") is None                      # one value: nothing to correct
    facts = DatasheetFacts(part_number=MODULE, V_CES=FactValue(value=650, unit="V", verified=True),
                           I_C_nom=FactValue(value=50, unit="A", verified=True),
                           V_CE_sat_typ=FactValue(value=2.3, unit="V", verified=True,
                                                  quote="VCE(sat) VGE = 15 V, I C= 50 A 1.95 2.3"),
                           V_F_typ=FactValue(value=1.85, unit="V", verified=True, quote="VF IF = 50 A - 1.85 2.65 V"),
                           source_url=PDF, source_title="x", via="cached")
    _fix_typ_columns(facts)
    assert facts.V_CE_sat_typ.value == 1.95 and facts.V_F_typ.value == 1.85
    assert any("not the Typ column" in n for n in facts.notes)


def test_at_low_load_an_impossible_test_is_not_demanded(tmp_path: Path) -> None:
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),          # upward: refused
             chat_response(tool_calls=[("web_datasheet_lookup", {"part_number": MODULE}),
                                       ("apply_action", {"action": "derate", "reason": "AL open"}),
                                       ("web_failure_evidence", {"hypothesis": "AL_open", "part_number": MODULE})]),
             chat_response(tool_calls=[("finalize_diagnosis", final_args("AL_open", citations=review))])]
    llm = make_llm(scripted_agent(turns), tmp_path)
    s = build_session(CFG, seed=SEED, llm=llm, web=fake_web(tmp_path))
    s.nameplate = Nameplate(igbt_module_part_number=MODULE, gate_driver_part_number=DRIVER, inverter_rating="test")
    from dataclasses import replace as dc_replace
    event = run_to_event(s, dc_replace(SCENARIOS["AL_open"], m=0.35))
    trace = Trace(scenario="AL_open")
    agent = DiagnosisAgent(s, s.llm, CFG)
    assert "probably not possible" in agent.test_hint()
    final = agent.run(event, trace)
    assert not any(v["approved"] for v in trace.actions if v["request"].startswith("active test"))
    assert final.label == "AL_open"
    assert any("not possible" in t for t in final.tests_run)


def test_the_tavily_ablation_has_no_web_tools_and_still_concludes(tmp_path: Path) -> None:
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
             chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT}),
                                       ("apply_action", {"action": "derate", "reason": "AU open"})]),
             chat_response(tool_calls=[("finalize_diagnosis", final_args("AU_open"))])]
    s, event = session_for("AU_open", tmp_path, scripted_agent(turns))
    agent = DiagnosisAgent(s, s.llm, CFG, web_tools=False)
    assert not any(name.startswith("web_") for name in agent.specs)
    trace = Trace(scenario="AU_open")
    final = agent.run(event, trace)
    derate = next(v for v in trace.actions if v["request"].startswith("derate"))
    assert final.label == "AU_open" and final.citations == []
    assert derate["approved"] and not derate["applied"].get("limit_from_datasheet") and derate["needs_datasheet"]


def test_an_answer_the_full_ranking_contradicts_is_sent_back(tmp_path: Path) -> None:
    # Found by the evaluation: AL_CL_open was called BU_open at a 107 % match error (two-switch causes untested).
    review = [{"title": "review", "url": "https://example.org/igbt-open-circuit"}]
    two = DEFAULT + ["AU_BU_open"]
    turns = [chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": DEFAULT})]),
             chat_response(tool_calls=[("run_active_test", {"m_new": 0.5})]),
             chat_response(tool_calls=[("web_datasheet_lookup", {"part_number": MODULE}),
                                       ("apply_action", {"action": "derate", "reason": "fault"}),
                                       ("web_failure_evidence", {"hypothesis": "BU_open", "part_number": MODULE})]),
             chat_response(tool_calls=[("finalize_diagnosis", final_args("WRONG", citations=review))]),
             chat_response(tool_calls=[("simulate_hypotheses", {"hypotheses": two})]),
             chat_response(tool_calls=[("finalize_diagnosis", final_args("AU_BU_open", citations=review))])]
    s, event = session_for("AU_BU_open", tmp_path, scripted_agent(turns))
    first = call(s, "simulate_hypotheses", hypotheses=DEFAULT)
    wrong = first["ranked"][0]["hypothesis"]
    assert "AU_BU_open" in first["untested_better_cause"]
    turns[3] = chat_response(tool_calls=[("finalize_diagnosis", final_args(wrong, citations=review))])
    s, event = session_for("AU_BU_open", tmp_path / "2", scripted_agent(turns))
    trace = Trace(scenario="AU_BU_open")
    final = DiagnosisAgent(s, s.llm, CFG).run(event, trace)
    summaries = [x["result_summary"] for x in trace.steps if x.get("tool") == "finalize_diagnosis"]
    assert "ranking of all causes puts AU_BU_open first" in summaries[0]
    assert final.label == "AU_BU_open"


def test_tavily_credits_are_booked_and_capped(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "spend.json", 1.0, 15.0, daily_cap_credits=3, total_cap_credits=400)
    web = TavilyWeb(CFG, client=FakeTavily(results=SEARCH_RESULTS), cache_dir=tmp_path / "tavily", ledger=ledger)
    web.search("first query")
    web.search("second query")
    assert ledger.credits()[1] == 2 and ledger.remaining_credits() == 1
    web.search("third query")
    with pytest.raises(BudgetExceeded):               # the daily cap of 3 is reached: no fourth live call
        web.search("fourth query")
    assert len(web._client.calls) == 3
    web.search("first query")                         # cached results still work
    assert ledger.spent()[1] == 0.0                   # dollars and credits are kept apart


def test_test_conclusions_are_judged_sentence_by_sentence() -> None:
    from faulttwin.agent import _is_test_conclusion, _sentences
    text = ("The modulation step test could not separate the causes. The amplitude ratio indicates no current in "
            "phase B.")
    assert not any(_is_test_conclusion(x) for x in _sentences(text))       # outcome word in another sentence
    assert _is_test_conclusion("The modulation\u2011step test showed mu unchanged, consistent with a switch fault.")


def test_no_model_turn_starts_without_enough_budget(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["agent"]["max_wall_s"] = 7.0                                      # below the 8 s a turn needs
    s, event = session_for("AU_open", tmp_path, scripted_agent([]))
    trace = Trace(scenario="AU_open")
    final = DiagnosisAgent(s, s.llm, cfg).run(event, trace)
    assert final.label == "inconclusive" and trace.stop_reason.startswith("time limit")
    assert not any(c.label.startswith("agent:turn") for c in s.llm.calls)
