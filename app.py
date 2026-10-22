"""FaultTwin: Streamlit app.

A live simulated inverter runs in the browser session. Inject a fault or a look-alike (or press
"Mystery fault"); the edge monitor raises an event; Nemotron 3.5 Lightning triages it; Nemotron 3
Super investigates with the digital twin, a safe active test and Tavily evidence; the safety
interlock approves or rejects every action; Nemotron 3 Ultra writes the work order. When live AI
calls are not possible (no keys, budget caps, API errors), recorded diagnoses are replayed in the
same interface.

Two fragments keep the page cheap: the plant section updates on a timer (slower when nobody has
touched the page for a while), and the agent panel updates on a timer only while a diagnosis or a
replay is playing. Each finished run or replay is handled exactly once (score, failures, refresh).

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import streamlit as st

from faulttwin.config import REPO_ROOT, load_config
from faulttwin.hypotheses import SPECS, Hypothesis
from faulttwin.llm import badge_of
from faulttwin.monitor import baseline_from_mu
from faulttwin.plant import Condition, rated_peak_current_A
from faulttwin.replay import ReplayPlayer, list_replays, load_replay
from faulttwin.runtime import PLANT_CYCLES_PER_TICK, PLANT_TICK_S, build_session
from faulttwin.scenarios import MYSTERY_POOL, SCENARIOS
from faulttwin.ui.plots import PlantFigures, overlay_figure, ranking_figure
from faulttwin.ui.runner import ESTIMATED_RUN_USD, Guard, LiveRun, friendly_budget_reason
from faulttwin.ui.views import view_from_record, view_from_session
from faulttwin.web import clean_snippet

st.set_page_config(page_title="FaultTwin", page_icon="⚡", layout="wide")

IDLE_TICK_S = 2.5            # plant update interval when nobody has used the page for a while
IDLE_AFTER_S = 90.0
BUSY_TICK_S = 1.0            # one timer for plant and agent while a diagnosis or replay plays
VIEWER_TIMEOUT_S = 30.0      # a visitor counts as active this long after their last update
MAX_TICK_SCALE = 3.0         # timers slow down by at most this factor when many visitors are active
QUIET_CYCLES = 12            # cycles after an inject with no alarm before the page says so
PLOT_CONFIG = {"displayModeBar": False}
BADGE = {"Lightning": ":green-background[Lightning]", "Nano": ":green-background[Nano]",
         "Super": ":blue-background[Super]", "Ultra": ":violet-background[Ultra]", "Rules": ":gray-background[Rules]"}
FAMILY = {"switch_or_leg_fault": "switch or leg fault", "sensor_error": "sensor error", "load_change": "load change",
          "transient": "transient", "unclear": "unclear"}
INJECTABLE = [n for n in SCENARIOS if n not in ("healthy", "low_load")]
# Measured on five seeds: these look-alikes stay inside the alarm limits (a load step logs a low-severity event).
NO_ALARM = {"load_unbalance_A", "load_step"}


# --- state ------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def config() -> dict[str, Any]:
    return load_config()


@st.cache_resource(show_spinner=False)
def replays() -> dict[str, dict[str, Any]]:
    """Recorded diagnoses that pass the replay checks; a bad file is skipped, never shown."""
    out = {}
    for name, path in list_replays().items():
        data = load_replay(path)
        if data is not None:
            out[name] = data
    return out


def state() -> Any:
    ss = st.session_state
    if "session" not in ss:
        cfg = config()
        ss.seed = random.SystemRandom().randrange(10_000, 99_999)   # each visitor gets their own plant tolerances
        ss.session = build_session(cfg, seed=ss.seed)
        ss.session.advance(6)
        ss.runs_used = 0
        ss.api_failed = False
        ss.run = None
        ss.player = None
        ss.player_note = ""
        ss.condition = "healthy"
        ss.hidden = False
        ss.handled_events = 0
        ss.attempt = 0                       # one per inject or mystery
        ss.finished_key = None               # the run or replay whose end has been handled
        ss.injected_cycle = 0
        ss.injected_events = 0
        ss.score = {"correct": 0, "total": 0, "recorded": 0}
        ss.notice = ""
        ss.rng = random.Random(ss.seed)
        ss.last_activity = time.time()
        ss.plant_slow = False
        ss.figs = {}
        ss.plant_figs = PlantFigures()
        ss.watched = set()                   # mystery cases seen in this visit
        ss.last_mystery = None
    # The demo limits depend on the date (higher in the judging period), so they are set again on each new UTC day:
    # a tab left open across 30 Nov gets the judging limits too. The session's models and Tavily check the same
    # ledger as the guard, so a run the guard allows is not stopped halfway by an older, lower cap.
    today = datetime.now(timezone.utc).date()
    if ss.get("limits_day") != today:
        ss.limits_day = today
        ss.guard = Guard.from_config(config())
        ss.session.llm.ledger = ss.session.web.ledger = ss.guard.ledger
    return ss


def decide(ss: Any) -> tuple[bool, str, str]:
    return ss.guard.decide(ss.runs_used, ss.api_failed)


def live_allowed(ss: Any) -> tuple[bool, str]:
    ok, why, _ = decide(ss)
    return ok, why


def busy(ss: Any) -> bool:
    return (ss.run is not None and ss.run.running) or (ss.player is not None and not ss.player.finished())


def idle(ss: Any) -> bool:
    return not busy(ss) and time.time() - ss.last_activity > IDLE_AFTER_S


def review_pending(ss: Any) -> bool:
    """Ultra's late work order (the senior review) has not arrived yet."""
    wo = ss.run.work_order if ss.run is not None else None
    return wo is not None and wo.senior_review is not None and not wo.senior_review.done()


def mark_activity() -> None:
    """A widget inside a fragment was used: count it as activity (the plant timer speeds up on the next page run)."""
    st.session_state.last_activity = time.time()


def refresh_app(ss: Any) -> None:
    """Rerun the whole page (sidebar, header, timers) because the state changed, not the visitor."""
    ss.forced_rerun = True
    st.rerun()


# --- text helpers -------------------------------------------------------------------------------------

KEY_NAMES = {"mu_ratio_after_over_before": "μ ratio after/before", "expected_ratio_if_sensor_offset":
             "ratio expected for a sensor offset", "match_error_pct": "match error (%)", "amp_ratio": "amplitude ratios",
             "r_kcl_dc": "r_KCL,dc",
             "r_kcl_w1": "r_KCL,ω₁", "i_s_A": "|i_s| (A)", "switch_fault": "a switch fault",
             "sensor_offset": "a sensor offset"}


def pretty(text: str) -> str:
    """Display notation: μ, ω₁, ±, arrows, and words for raw keys the models sometimes copy."""
    text = re.sub(r"\b(" + "|".join(map(re.escape, KEY_NAMES)) + r")\b(?!_)", lambda m: KEY_NAMES[m.group(1)], text)
    text = re.sub(r"\bmu(_[abcABC])?\b", lambda m: "μ" + (m.group(1) or ""), text)
    text = re.sub(r"\b2\*w1\b", "2ω₁", text)
    text = re.sub(r"\bw1\b", "ω₁", text)
    return text.replace("+/-", "±").replace("->", "→").replace("<=", "≤").replace(">=", "≥")


def num(x: float, fmt: str = "+.3f") -> str:
    out = format(0.0 if abs(x) < 5e-4 else x, fmt)
    return out[1:] if out.startswith("+") and float(out) == 0 else out


def cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def short_cause(label: str) -> str:
    """Compact cause name for buttons: 'AU open', 'leg B open', 'sensor gain A'."""
    return label.replace("_open", " open").replace("_", " ")


def near_limit(x: float, limits: list[float], digits: int, sign: str = "") -> str:
    """A value rounded to the limit would look like it equals it: show one more digit there."""
    close = any(abs(abs(x) - abs(lim)) < 0.5 * 10 ** -digits for lim in limits)
    return num(x, f"{sign}.{digits + 1 if close else digits}f")


def cause_text(label: str) -> str:
    if label in SPECS:
        d = SPECS[Hypothesis(label)].description
        return f"{d[0].upper()}{d[1:]} ({label})"
    return "Inconclusive" if label == "inconclusive" else label


def recording_text(ss: Any, text: str) -> str:
    """Source labels of a recording: 'live' meant live when it was recorded."""
    if ss.player is None:
        return text
    rec = f"via Tavily (recorded {ss.player.recorded_on})"
    return text.replace("(via Tavily (live))", f"· {rec}").replace("via Tavily (live)", rec)


def no_recording_notice(why: str, name: str) -> str:
    cause = why.split(", so ")[0].rstrip(".") + "."       # the cause only: no recording will be shown here
    if not replays():
        return f"{cause} There are no recorded diagnoses on this deployment; the plant and the edge monitor run live."
    return (f"{cause} There is no recorded diagnosis of “{SCENARIOS[name].title}”: the plant and the edge monitor run "
            "live, but no agent runs. Pick a condition marked **· recorded** in the selector, or press **Mystery "
            "fault** for a random hidden one.")


# --- actions --------------------------------------------------------------------------------------

def back_to_live_plant(ss: Any) -> None:
    if ss.run is not None and ss.run.running:
        return                                   # never leave an agent thread acting on the plant
    s = ss.session
    with s.lock:
        s.plant.set_condition(Condition())
        s.plant.resume()
        s.derated_m, s.derated_for = None, None
        s.plant.set_modulation_index(ss.get("m_slider", 0.8))
        s.monitor.reset()
        s.events.clear()
        s.history.clear()
        s.statuses.clear()
        s.advance(6)
    ss.handled_events = 0
    ss.run, ss.player, ss.player_note = None, None, ""
    ss.condition, ss.hidden, ss.notice = "healthy", False, ""
    ss.figs = {}


def start_replay(ss: Any, name: str, note: str, hidden: bool) -> bool:
    try:
        player = ReplayPlayer(replays()[name], attempt=ss.attempt)
    except Exception:  # noqa: BLE001 - a recording that cannot be played: stay on the live plant
        ss.notice = f"The recording of {SCENARIOS[name].title} could not be played; the plant runs live."
        return False
    ss.player, ss.player_note = player, note
    ss.condition, ss.hidden = name, hidden
    ss.run = None
    ss.figs = {}                                 # never reuse a live run's figures for a recording
    ss.watched.add(name)
    return True


def inject(ss: Any, name: str, hidden: bool = False) -> None:
    ok, why = live_allowed(ss)
    back_to_live_plant(ss)
    ss.attempt += 1
    ss.figs = {}
    if not ok:
        if name in replays() and start_replay(ss, name, why, hidden):
            return
        if name not in NO_ALARM:                 # those never start a diagnosis: the empty state explains them
            ss.notice = no_recording_notice(why, name)
    s = ss.session
    with s.lock:
        s.plant.set_condition(SCENARIOS[name].condition)
        ss.injected_cycle, ss.injected_events = s.cycle, len(s.events)
    ss.condition, ss.hidden = name, hidden


def mystery(ss: Any) -> None:
    ok, why = live_allowed(ss)
    pool = MYSTERY_POOL if ok else [n for n in MYSTERY_POOL if n in replays()] or list(replays())
    if not pool:
        ss.notice = why + " No recordings are available."
        return
    # Prefer cases not seen yet in this visit, and never the same one twice in a row.
    fresh = [n for n in pool if n not in ss.watched and n != ss.last_mystery]
    choice = ss.rng.choice(fresh or [n for n in pool if n != ss.last_mystery] or pool)
    ss.last_mystery = choice
    inject(ss, choice, hidden=True)
    if ss.player is None:
        ss.watched.add(choice)


def tick(ss: Any) -> bool:
    """Advance the live plant and start a diagnosis when the monitor raises a new event.

    Returns True when the whole page must refresh (a diagnosis or a replay started, or a notice changed).
    """
    if ss.player is not None:
        return False
    s = ss.session
    s.advance(PLANT_CYCLES_PER_TICK)
    events = s.events
    if len(events) <= ss.handled_events:
        return False
    new = [e for e in events[ss.handled_events:] if e.severity in ("medium", "high")]
    ss.handled_events = len(events)
    # Diagnose once per injected condition (Clear faults starts over); a rare noise event on a healthy
    # plant does not spend one of the session's live diagnoses.
    if not new or ss.run is not None or ss.condition == "healthy" or ss.notice:
        return False
    ok, why = live_allowed(ss)
    if ok:
        ss.runs_used += 1
        ss.run = LiveRun(s, config(), new[0], SCENARIOS.get(ss.condition), attempt=ss.attempt).start()
        return True
    if ss.condition in replays() and start_replay(ss, ss.condition, why, ss.hidden):
        return True
    ss.notice = no_recording_notice(why, ss.condition)
    return True


def fail_safe_verdict(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The approved interlock verdict on a derate that code itself (the fail-safe), not the model, requested."""
    for step in steps:
        verdict = (step.get("result") or {}).get("interlock") or {}
        if step.get("by_code") and step.get("tool") == "apply_action" and verdict.get("approved"):
            return verdict
    return None


def on_finished(ss: Any) -> bool:
    """Handle the end of a live run or a replay once: score, outages, caps. True: refresh the page."""
    obj = ss.run if ss.run is not None else ss.player
    key = (type(obj).__name__, obj.attempt)
    if ss.finished_key == key:
        return False
    ss.finished_key = key
    if ss.player is not None:
        if ss.hidden:
            ss.score["recorded"] += 1
        return True
    run = ss.run
    title = SCENARIOS[ss.condition].title if ss.condition in SCENARIOS else ss.condition
    if run.status in ("failed", "budget"):
        # An outage or a spend cap is not the agent's answer: no score, and it does not use up a live run.
        ss.runs_used = max(0, ss.runs_used - 1)
        if run.status == "failed":
            ss.api_failed = True
            web_only = run.trace.service_errors and not (run.trace.stop_reason or "").startswith("model call failed")
            cause = ("The live web search (Tavily) did not answer." if web_only else
                     "The live AI service did not answer.")
        else:
            cause = friendly_budget_reason(run.error or run.trace.stop_reason or "").split(", so ")[0] + "."
        if ss.condition in replays() and start_replay(ss, ss.condition, f"{cause} A recorded diagnosis is shown.",
                                                      ss.hidden):
            return True
        verdict = fail_safe_verdict(run.trace.steps)
        if verdict is not None:
            # No recording to show, but the fail-safe derated (or capped) the live plant: keep this run on
            # screen, so the derate on the plant line comes with the interlock's verdict.
            m = float((verdict.get("applied") or {}).get("m", ss.session.plant.m))
            what = (f"code asked the interlock for a derate; m stays at {m:.2f} and is capped there"
                    if verdict.get("request", "").endswith("(not needed)") else f"code derated the drive to m = {m:.2f}")
            ss.notice = (f"{cause} The safety interlock's own twin ranking confirmed a converter fault, so {what} "
                         "(the fail-safe step below).")
            ss.hidden = False
            return True
        ss.run = None
        ss.notice = (f"{cause} This diagnosis stopped, and there is no recording of **{title}** (the injected "
                     "condition). Press **Clear faults**, then **Mystery fault** to watch a recorded diagnosis.")
        ss.hidden = False
        return True
    final = run.trace.final or {}
    if ss.hidden and final:
        ss.score["total"] += 1
        ss.score["correct"] += int(final.get("label") == SCENARIOS[ss.condition].truth.value)
    return True


# --- rendering: header and sidebar ------------------------------------------------------------------

def header(ss: Any) -> None:
    left, right = st.columns([6, 2], vertical_alignment="center")
    with left:
        st.markdown("## FaultTwin")
        st.caption("An edge AI agent that catches open-circuit switch faults in inverters and checks its diagnosis on "
                   "a physics digital twin before it acts. Built for the Nebius x NVIDIA Global AI Hackathon, Physical "
                   "AI track: NVIDIA Nemotron models via Nebius Token Factory; web evidence via Tavily.")
    ok, why, kind = decide(ss)
    with right:
        if ss.run is not None and ss.run.running:
            badge = ":green-background[Live AI · diagnosing]"
        elif ss.player is not None:
            badge = ":orange-background[Recorded run]"
        else:
            badge = ":green-background[Live AI ready]" if ok else ":orange-background[Recorded runs]"
        st.markdown(f"#### {badge}")
    if kind in ("budget", "api"):
        st.warning(why)


def sidebar(ss: Any) -> None:
    cfg = config()
    s = ss.session
    sb = st.sidebar
    sb.header("Plant")
    disabled = busy(ss)
    recorded = ss.player is not None                 # a recording's plant cannot be changed
    m = sb.slider("Modulation index m", 0.3, 0.9, 0.8, 0.05, key="m_slider", disabled=disabled or recorded,
                  help="Operating point of the inverter. ω₁ = 314.16 rad/s.")
    sb.caption("ω₁ = 314.16 rad/s · V_dc = 400 V · RL load 5 Ω, 10 mH")
    noise = sb.slider("Measurement noise (% of rated peak current)", 0.0, 2.0,
                      100 * cfg["plant"]["noise_sigma_frac"], 0.1, key="noise_pct", disabled=disabled or recorded)
    if recorded:
        sb.caption("A recording is shown. Press **Clear faults** to return to the live plant.")
    with s.lock:
        limit = s.derated_m if s.derated_m is not None else 1.0
        target = min(m, limit)
        if ss.player is None and not disabled:
            if not s.plant.stopped and abs(s.plant.m - target) > 1e-9:
                s.plant.set_modulation_index(target)
                s.monitor.expect_operating_point_change()
            sigma = noise / 100 * rated_peak_current_A(s.plant.nominal)
            if abs(s.plant.noise_sigma_A - sigma) > 1e-12:
                s.plant.noise_sigma_A = sigma
        derated = s.derated_m
    if derated is not None and m > derated and not recorded:
        sb.warning(f"The safety interlock limits m to {derated:.2f} after the derate.")

    sb.header("Faults and look-alikes")
    ok, why, kind = decide(ss)

    def label(n: str) -> str:
        text = SCENARIOS[n].title
        if n in NO_ALARM:
            text += " (no alarm expected)"
        if not ok and n in replays():
            text += " · recorded"
        return text

    choice = sb.selectbox("Condition to inject", INJECTABLE, format_func=label, disabled=disabled)
    c1, c2 = sb.columns(2)
    if c1.button("Inject", type="primary", disabled=disabled, width="stretch"):
        inject(ss, choice)
        st.rerun()
    if c2.button("Clear faults", disabled=disabled and ss.player is None, width="stretch"):
        back_to_live_plant(ss)
        st.rerun()
    if sb.button("Mystery fault", icon="❓", disabled=disabled, width="stretch",
                 help="Injects a random hidden fault or look-alike. The answer is shown after the agent finishes."):
        mystery(ss)
        st.rerun()
    sc = ss.score
    sb.caption(f"Agent score on live mystery diagnoses: {sc['correct']} of {sc['total']} correct" if sc["total"]
               else "Agent score on live mystery diagnoses: none yet")
    if sc["recorded"]:
        sb.caption(f"Recorded mysteries watched: {sc['recorded']} (recordings are selected correct runs)")

    sb.header("Live AI")
    if kind == "no_keys":
        sb.markdown("Live AI is **not configured** on this deployment.")
        sb.caption("Recorded diagnoses play in the same interface.")
        return
    sb.markdown(f"Next diagnosis: **{'live' if ok else 'recorded'}**")
    if not ok:
        sb.caption(why)
    if kind != "api":
        sb.caption(f"Live diagnoses left in this visit: {max(0, ss.guard.runs_per_session - ss.runs_used)}")
        remaining = ss.guard.remaining_usd()
        sb.caption("Demo budget left today: " + (f"\\${remaining:.2f}" if remaining is not None and
                                                 remaining >= ESTIMATED_RUN_USD else "paused"))
        try:
            credits = ss.guard.ledger.remaining_credits()
        except Exception:  # noqa: BLE001 - an unreadable ledger already pauses live mode above
            credits = None
        if credits is not None:
            sb.caption(f"Web search (Tavily) credits left today: {credits:.0f}")


# --- rendering: plant ---------------------------------------------------------------------------------

def plant_rows(view: Any, ss: Any) -> None:
    if view is None:
        st.info("Starting the plant simulation…")
        return
    replaying = ss.player is not None
    obj = ss.run if ss.run is not None else ss.player
    # The answer is revealed only once the end has been handled (an outage replays the mystery, still hidden).
    revealed = obj is not None and ss.finished_key == (type(obj).__name__, obj.attempt) and \
        (ss.run is None or ss.run.status == "done")
    if ss.condition not in SCENARIOS:
        cond = "Healthy"
    elif ss.hidden and not revealed:
        cond = "Mystery fault (hidden)"
    else:
        cond = SCENARIOS[ss.condition].title + (" (the mystery)" if ss.hidden else "")
    status = "stopped by the interlock" if view.stopped else f"running at m = {view.m:.2f}"
    if view.derated_m is not None and not view.stopped:
        status += f", derated by the interlock to m ≤ {view.derated_m:.2f}"
    if replaying and ss.player.finished():
        status = f"end of the recording ({status.replace('running', 'was running')})"
    # Few elements per update: each one is a separate message to the browser.
    head = (f"**Plant:** {status} · cycle {view.cycle} · condition: **{cond}**"
            + (" · :orange[recorded plant data]" if replaying else ""))
    strip = status_strip(ss)
    st.markdown(head + (f"  \n{strip}" if strip else ""))
    a, b, c = st.columns(3)
    currents, park, spectrum = ss.plant_figs.update(view)
    a.plotly_chart(currents, width="stretch", config=PLOT_CONFIG, key="fig_currents")
    b.plotly_chart(park, width="stretch", config=PLOT_CONFIG, key="fig_park")
    c.plotly_chart(spectrum, width="stretch", config=PLOT_CONFIG, key="fig_spectrum")

    th = config()["monitor"]
    f = view.features
    with st.container():
        if f is None:
            st.markdown("**Edge monitor** (2-cycle window, checked every cycle)  \nWarming up…")
        else:
            lo, hi = th["amp_ratio_band"]

            def cell(text: str, bad: bool) -> str:
                return f":red[**{text} ▲**]" if bad else text

            mu_lim, dc_lim, w1_lim = th["mu_abs_max"], th["r_kcl_dc_max"], th["r_kcl_w1_max"]
            values = [cell(near_limit(f["mu"][k], [mu_lim], 3, "+"), abs(f["mu"][k]) > mu_lim) for k in range(3)]
            values += [cell(num(f["r_kcl_dc"], ".3f"), f["r_kcl_dc"] > dc_lim),
                       cell(num(f["r_kcl_w1"], ".3f"), f["r_kcl_w1"] > w1_lim)]
            values += [cell(near_limit(r, [lo, hi], 2), not lo <= r <= hi) for r in f["amp_ratio"]]
            limits = [f"±{mu_lim:.2f}"] * 3 + [f"{dc_lim:.2f}", f"{w1_lim:.2f}"] + [f"{lo:.2f}–{hi:.2f}"] * 3
            heads = ["μ_a", "μ_b", "μ_c", "r_KCL,dc", "r_KCL,ω₁", "A_a/mean", "A_b/mean", "A_c/mean"]
            st.markdown("**Edge monitor** (2-cycle window, checked every cycle)\n\n"
                        "| | " + " | ".join(heads) + " |\n|---|" + "---|" * len(heads) + "\n"
                        "| value | " + " | ".join(values) + " |\n| limit | " + " | ".join(limits) + " |")
            base = baseline_from_mu(f["mu"], mu_lim)
            st.caption(f"|i_s| = {f['i_s_A']:.2f} A · RMS = " + ", ".join(f"{x:.2f}" for x in f["rms_A"]) + " A · "
                       "red ▲ = outside its limit · A_k/mean: ω₁ amplitude of phase k over the mean of the three · "
                       "textbook threshold detector (for comparison): "
                       f"**{'healthy' if base == 'healthy' else cause_text(base)}**")
    icons = {"high": "🔴", "medium": "🟠", "low": "⚪"}
    lines = [f"{icons[e['severity']]} **{e['severity'].upper()}** · cycle {e['cycle']}: {pretty(e['message'])}"
             for e in reversed(view.events[-4:])]
    st.markdown("**Event log**  \n" + ("  \n".join(lines) if lines else "No events. The monitor checks every cycle."))


def status_strip(ss: Any) -> str:
    """One line next to the plots, so the plant and the agent's state share one screen."""
    if ss.run is not None:
        t = ss.run.trace
        if ss.run.running:
            tools = [x["tool"] for x in t.steps if x["kind"] == "tool"]
            last = f" · last tool: {tools[-1]}" if tools else ""
            return f":blue-background[**Agent:** investigating · {len(t.steps)} steps{last}]"
        if t.final and ss.run.status == "done":
            return (f":blue-background[**Diagnosis:** {cause_text(t.final['label'])} · confidence "
                    f"{t.final['confidence']:.2f}]")
    elif ss.player is not None:
        p = ss.player.partial()
        if p["final"]:
            return (f":orange-background[**Recorded diagnosis:** {cause_text(p['final']['label'])} · confidence "
                    f"{p['final']['confidence']:.2f}]")
        return (f":orange-background[**Agent (recorded run):** {len(p['steps'])} of "
                f"{len(ss.player.trace.get('steps', []))} steps]")
    return ""


def plant_body(ss: Any) -> bool:
    """The plant section. True: the whole page must refresh."""
    refresh = False
    try:
        refresh = tick(ss)
        if ss.player is not None:
            n, _ = ss.player.counts()
            view = view_from_record(ss.player.record, n, ss.session.plant.params.dt_s)
        else:
            view = view_from_session(ss.session)
        plant_rows(view, ss)
        if not busy(ss) and idle(ss) != ss.plant_slow:     # slow the timer down (or back up)
            refresh = True
    except Exception as exc:  # noqa: BLE001 - never show a stack trace to a visitor
        st.warning(f"The plant panel hit a problem and will retry on the next update ({type(exc).__name__}).")
    return refresh


def plant_section() -> None:
    ss = state()
    tick_scale(ss)
    if plant_body(ss):
        refresh_app(ss)


# --- rendering: agent ---------------------------------------------------------------------------------

def _args_text(args: Any) -> str:
    if not isinstance(args, dict):
        return cut(str(args), 60) if args else ""
    parts = []
    for k, v in args.items():
        if isinstance(v, list):
            v = f"{len(v)} items" if len(v) > 3 else ", ".join(map(str, v))
        parts.append(f"{k}={cut(str(v), 40)}")
    return ", ".join(parts)


def money(usd: float) -> str:
    """A dollar amount for markdown: '$' is escaped, or two amounts on a line would render as LaTeX math."""
    return "<\\$0.0001" if 0 < usd < 0.0001 else f"\\${usd:.4f}"


def _meter(step: dict[str, Any]) -> str:
    return (f"{step.get('prompt_tokens', 0)}+{step.get('completion_tokens', 0)} tok · {step.get('latency_s', 0)} s · "
            f"{money(step.get('cost_usd', 0))}")


def _step_line(step: dict[str, Any]) -> tuple[str, str | None, Any]:
    """(line, reasoning text, result) for one trace step."""
    kind = step["kind"]
    if kind == "triage":
        r = step["result"]
        badge = step.get("badge", "Rules")
        meter = f" · {_meter(step)}" if badge != "Rules" else ""
        rule = f" ({pretty(r['rule'])})" if r.get("rule") else ""
        return (f"{BADGE.get(badge, badge)} edge triage: **{FAMILY.get(r['family'], r['family'])}**, urgency "
                f"{r['urgency']}{rule}{meter}", step.get("reasoning"), {"model's reason": r.get("reason", "")})
    if kind == "llm":
        calls = ", ".join(c["name"] for c in step.get("tool_calls", [])) or ("inside " + step["inside_tool"]
                                                                            if step.get("inside_tool") else "text")
        return f"{BADGE.get(step.get('badge'), step.get('badge'))} → {calls} · {_meter(step)}", step.get("reasoning"), None
    if kind == "tool":
        args = _args_text(step.get("arguments"))
        if step["tool"] == "finalize_diagnosis":    # the full answer is in the card; name the label only
            raw = step.get("arguments")
            label = re.search(r'"label"\s*:\s*"([^"]+)"', raw if isinstance(raw, str) else json.dumps(raw))
            args = f"label={label.group(1)}" if label else ""
        tag = ":gray-background[code]" if step.get("by_code") else ":orange-background[tool]"   # not the model's call
        return (f"{tag} **{step['tool']}**({args}) · {step.get('latency_s', 0)} s → "
                f"{pretty(cut(step['result_summary'], 220))}", None, step.get("result"))
    return f":gray-background[note] {pretty(step.get('text', ''))}", None, None


def steps_timeline(trace: dict[str, Any], ss: Any) -> None:
    """All steps as one markdown block (cheap to send every update); details for one step once it is over."""
    st.markdown("**Steps** (model, tool and arguments, tokens, latency, cost)")
    steps = trace.get("steps", [])
    lines = [f"{i}. {_step_line(step)[0]}" for i, step in enumerate(steps, start=1)]
    wo = trace.get("work_order") or {}
    for c in wo.get("calls", []):
        badge = badge_of(c.get("model", ""))
        used = "" if c.get("used", True) else " (standby, not used)"
        lines.append(f"- {BADGE.get(badge, badge)} → work order{used} · {_meter(c)}")
    pending = pending_line(trace, ss)
    if pending:
        lines.append(f"- {pending}")
    with st.container(height=440, autoscroll=True):
        st.markdown("\n".join(lines) or "Waiting for the first step…")
    if trace.get("finished") and steps:
        with st.expander("Details of one step (tool result or model reasoning)"):
            pick = st.selectbox("Step", list(range(1, len(steps) + 1)), index=len(steps) - 1,
                                key=f"detail_{ss.attempt}", format_func=lambda i: f"{i}. {_step_title(steps[i - 1])}")
            line, reasoning, result = _step_line(steps[pick - 1])
            if result is not None:
                st.json(result, expanded=False)
            if reasoning:
                st.caption("Model reasoning (kept in the trace, not used as the answer):")
                st.text(reasoning[:3000])
            if result is None and not reasoning:
                st.caption("No further details for this step.")


def _step_title(step: dict[str, Any]) -> str:
    if step["kind"] == "tool":
        return f"tool {step['tool']}"
    if step["kind"] == "llm":
        return f"{step.get('badge')} model call"
    return step["kind"]


def pending_line(trace: dict[str, Any], ss: Any) -> str | None:
    """What the agent is waiting for, so a long model call does not look like a frozen page."""
    if trace.get("finished"):
        return None
    steps = trace.get("steps", [])
    if ss.player is not None:
        if ss.player.elapsed() < ss.player.event_time:
            return None                          # still the healthy lead-in before the event
        # the recorded wait, not the shortened one on screen
        waited = (ss.player.elapsed() - ss.player.step_time(len(steps))) * ss.player.speedup
    elif ss.run is not None:
        last = steps[-1].get("t_wall") if steps else ss.run.started
        waited = time.time() - (last or ss.run.started)
    else:
        return None
    if trace.get("final") and not trace.get("work_order"):
        who = "Ultra is writing the work order"
    elif not steps:
        who = "edge triage is classifying the event"
    elif steps[-1]["kind"] == "llm" and steps[-1].get("tool_calls"):
        who = f"running {steps[-1]['tool_calls'][0]['name']}"
    else:
        who = "Super is choosing the next step"
    return f"⏳ {who}… {max(0.0, waited):.0f} s"


def agent_panel(trace: dict[str, Any] | None, ss: Any) -> None:
    st.subheader("Diagnosis agent")
    if ss.notice:
        st.info(ss.notice)
    if ss.player is not None and ss.player_note:
        faster = (f" Waits between steps play {ss.player.speedup:.1f}× faster than recorded; each step shows its "
                  "recorded latency." if ss.player.speedup > 1.05 else "")
        st.info(f"**Recorded run** (made on {ss.player.recorded_on}). {ss.player_note}{faster}")
    if trace is None:
        empty_state(ss)
        return
    finished = trace.get("finished")
    if not finished:
        st.caption("Investigating…" if ss.player is None else "Playing the recorded investigation…")
    left, right = st.columns([3, 2])
    with left:
        steps_timeline(trace, ss)
    with right:
        final_card(trace, ss)
        ranked = next((s["result"]["ranked"] for s in reversed(trace.get("steps", []))
                       if s.get("tool") == "simulate_hypotheses" and "ranked" in (s.get("result") or {})), None)
        if ranked:
            key = ("rank", ss.attempt, id(ranked))
            fig = ss.figs.get(key) or ss.figs.setdefault(key, ranking_figure(ranked))
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key="fig_rank")
            st.caption("Red bars are converter faults; grey bars are look-alikes and healthy.")
        verdicts = [s["result"]["interlock"] for s in trace.get("steps", [])
                    if isinstance(s.get("result"), dict) and "interlock" in s["result"]]
        if verdicts:
            st.markdown("**Safety interlock** (code decides; the model only asks)")
            for v in verdicts:
                st.markdown(f"{'✅ approved' if v['approved'] else '⛔ rejected'} · {v['request']}: "
                            f"{pretty(v['reason'])}")
    overlays_section(trace, ss)
    sources_section(trace, ss)
    work_order_section(trace, ss)
    totals = trace.get("totals") or {}
    if finished and totals:
        prefix = "Recorded run: " if ss.player is not None else ""
        st.caption(f"{prefix}cost {money(totals.get('llm_cost_usd_incl_report', totals.get('llm_cost_usd', 0)))} · "
                   f"tokens {totals.get('prompt_tokens', 0)}+{totals.get('completion_tokens', 0)} · tool calls "
                   f"{totals.get('tool_calls', 0)} · Tavily live calls {totals.get('tavily_live_calls', 0)} · "
                   f"{totals.get('wall_s_incl_report', totals.get('wall_s', 0))} s")


def empty_state(ss: Any) -> None:
    s = ss.session
    if ss.condition == "healthy":
        st.caption("No diagnosis yet. Inject a fault or press **Mystery fault**: the edge monitor raises an event, "
                   "edge triage classifies it, and the agent starts.")
        return
    if ss.notice:
        return
    with s.lock:
        quiet_for = s.cycle - ss.injected_cycle
        since = s.events[ss.injected_events:]
    alarms = [e for e in since if e.severity in ("medium", "high")]
    if alarms or quiet_for <= QUIET_CYCLES:
        st.caption("Waiting for the edge monitor…")
        return
    spec = SPECS[SCENARIOS[ss.condition].truth]
    low = " A low-severity event was logged, which does not start a diagnosis." if since else ""
    if spec.converter_fault:
        st.warning(f"The edge monitor has not raised an alarm for this condition after {quiet_for} cycles.{low}")
    else:
        st.success("The edge monitor raised no alarm: this look-alike is not a converter fault, and its features stay "
                   "inside the alarm limits by design, so no diagnosis is started and no live AI call is spent (no "
                   f"false alarm).{low} Press **Clear faults** to start over.")


def final_card(trace: dict[str, Any], ss: Any) -> None:
    final = trace.get("final")
    if not final:
        return
    label = final["label"]
    fault = label in SPECS and SPECS[Hypothesis(label)].converter_fault
    err = final.get("twin_match_error_pct")
    with st.container(border=True):
        st.markdown(f"#### {'⚠️' if fault else '✅' if label != 'inconclusive' else '❔'} {cause_text(label)}")
        st.caption(f"confidence {final['confidence']:.2f} · twin match error "
                   + (f"{err:.2f} % of |i_s|" if isinstance(err, (int, float)) else "—"))
        st.write(pretty(final["summary"]))
        for t in final.get("tests_run", []):
            st.caption(f"• {pretty(t)}")
        truth = trace.get("hidden_truth")
        if truth and trace.get("finished") and ss.hidden:
            correct = truth["truth"] == label
            st.markdown(f"**The mystery was:** {truth['title']} — {'✅ correct' if correct else '❌ not correct'}")
        # Why the investigation stopped is in the summary; the expander is only for problems in an actual answer.
        checks = [n for n in trace.get("validation_notes") or [] if not n.startswith("stopped early:")]
        if checks:
            with st.expander("Code checks on the agent's final answer"):
                st.caption("These problems were still in the agent's last answer after the retries. Code removed the "
                           "sentences concerned (or, for a missing piece of evidence, made the answer inconclusive):")
                for n in checks:
                    st.caption(f"• {pretty(n)}")
    if trace.get("finished"):
        c1, c2 = st.columns(2)
        if c1.button("Clear faults", key=f"clear_{ss.attempt}", width="stretch",
                     help="Remove the fault and return to the healthy plant"):
            back_to_live_plant(ss)
            st.rerun()
        if c2.button("Another mystery", key=f"again_{ss.attempt}", icon="❓", width="stretch"):
            mystery(ss)
            st.rerun()


def overlays_section(trace: dict[str, Any], ss: Any) -> None:
    overlays = trace.get("overlays")
    version: Any = "final"
    s = ss.session
    if overlays is None and ss.player is None:
        with s.lock:
            ranking, window = s.last_ranking, s.last_window
        if ranking and window is not None:
            step = max(1, len(window.t_s) // 400)
            overlays = {"t_s": window.t_s[::step], "measured_A": window.i_meas_A[::step],
                        "predicted_A": {r.hypothesis.value: r.i_pred_A[::step] for r in ranking[:3]}}
            version = id(window)                 # a new twin ranking brings a new window
    if not overlays:
        return
    names = list(overlays["predicted_A"])
    st.markdown("**Measured vs digital twin** for the top causes")
    pick = st.segmented_control("Cause", names, default=names[0], key=f"ov_{ss.attempt}", format_func=short_cause,
                                label_visibility="collapsed", on_change=mark_activity) or names[0]
    if pick not in names:
        pick = names[0]
    key = ("ov", type(ss.player or ss.run).__name__, ss.attempt, pick, version)
    fig = ss.figs.get(key) or ss.figs.setdefault(key, overlay_figure(overlays, pick))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key="fig_overlay")


def sources_section(trace: dict[str, Any], ss: Any) -> None:
    seen: dict[str, dict[str, Any]] = {}
    for s in trace.get("steps", []):
        r = s.get("result") or {}
        for item in r.get("sources", []) + r.get("links", []):
            seen.setdefault(item["url"], item)
        if r.get("source"):
            ds = r.get("datasheet") or {}
            when = (ds.get("fetched_at") or (trace.get("started_at") if ds.get("via") == "cached" else "") or "")[:10]
            label = "via Tavily (live)" if ds.get("via") == "live" else f"via Tavily (cached {when})".replace(" ()", "")
            seen.setdefault(r["source"], {"title": f"{ds.get('part_number', 'IGBT module')} datasheet",
                                          "url": r["source"], "source": label})
    if not seen:
        return
    with st.expander(f"Sources (via Tavily): {len(seen)}", expanded=True):
        for item in seen.values():
            title = _escape(item.get("title") or item.get("name") or item["url"])
            host = urlparse(item["url"]).netloc
            st.markdown(f"- [{title}]({item['url']}) · `{host}` · "
                        f"*{recording_text(ss, item.get('source', 'via Tavily'))}*")
            if item.get("snippet") and item["snippet"] != "datasheet":
                st.caption(_escape(cut(clean_snippet(item["snippet"]), 240)))


def _escape(text: str) -> str:
    """Web text shown as plain text, not as markdown."""
    return re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", text)


def work_order_section(trace: dict[str, Any], ss: Any) -> None:
    wo = trace.get("work_order")
    if not wo:
        if trace.get("final"):
            st.caption("Writing the work order…")
        return
    who = BADGE.get(wo.get("badge") or "", "code (template)")
    md = re.sub(r"(\*\*Asset:\*\*[^\n]*?)[ \t]*\n(\*\*Time)", r"\1  \n\2", recording_text(ss, wo["markdown"]))
    shown = re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", pretty(md), flags=re.MULTILINE)
    with st.expander(f"Technician work order · written by {who} · {wo.get('latency_s', '?')} s", expanded=True):
        st.markdown(shown)
        if wo.get("senior_review"):
            with st.expander("Senior review by Nemotron 3 Ultra (arrived after the deadline; same checks passed)"):
                review = pretty(recording_text(ss, wo["senior_review"]["content"]))
                st.markdown(re.sub(r"^(#{1,4}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", review,
                                   flags=re.MULTILINE))
        st.download_button("Download work order (.md)", md, file_name="faulttwin_work_order.md",
                           mime="text/markdown", key=f"dl_{ss.attempt}", on_click="ignore")


def agent_body(ss: Any) -> bool:
    """The agent panel. True: the whole page must refresh."""
    refresh = False
    try:
        obj = ss.run if ss.run is not None else ss.player
        if obj is not None and ((ss.run is not None and not ss.run.running) or
                                (ss.player is not None and ss.player.finished())):
            refresh = on_finished(ss)
        if ss.run is not None and not ss.run.running and not review_pending(ss) and \
                ss.get("review_seen") != ss.attempt and ss.run.work_order is not None and \
                ss.run.work_order.senior_review is not None:
            ss.review_seen, refresh = ss.attempt, True        # the late review arrived: stop the timer
        if ss.player is not None:
            trace = ss.player.partial()
        else:
            trace = ss.run.snapshot() if ss.run is not None else None
        agent_panel(trace, ss)
    except Exception as exc:  # noqa: BLE001 - never show a stack trace to a visitor
        st.warning(f"The agent panel hit a problem and will retry on the next update ({type(exc).__name__}).")
    return refresh


def agent_section() -> None:
    ss = state()
    if agent_body(ss):
        refresh_app(ss)


def live_section() -> None:
    """Plant and agent together on one timer while a diagnosis or a replay plays (one timer per tab)."""
    ss = state()
    tick_scale(ss)
    refresh = plant_body(ss)
    st.divider()
    refresh = agent_body(ss) or refresh
    if refresh:
        refresh_app(ss)


@st.cache_resource(show_spinner=False)
def viewers() -> dict[int, float]:
    """Last update time per visitor session, shared by the whole server process."""
    return {}


def tick_scale(ss: Any) -> float:
    """Slow every timer down when many visitors use the page at once, so one small server keeps up."""
    reg = viewers()
    now = time.time()
    reg[ss.seed] = now
    for key, seen in list(reg.items()):
        if now - seen > VIEWER_TIMEOUT_S:
            reg.pop(key, None)
    return min(MAX_TICK_SCALE, max(1.0, len(reg) / 2))


# --- other tabs -----------------------------------------------------------------------------------

def how_it_works_tab() -> None:
    st.subheader("How FaultTwin works")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#F4F6F8", fontname="Helvetica", fontsize=11];
  P [label="Virtual inverter plant\\n2-level VSI, faults, sensor errors"];
  M [label="Edge monitor\\nfeatures in ω + thresholds"];
  T [label="Edge triage\\nNemotron 3.5 Lightning"];
  A [label="Diagnosis agent\\nNemotron 3 Super + tools"];
  D [label="Digital twin"]; S [label="Safety interlock\\n(code)"]; W [label="Tavily"];
  R [label="Work order\\nNemotron 3 Ultra"];
  P -> M [label="i_a, i_b, i_c"]; M -> T [label="event"]; T -> A [label="escalate"];
  A -> D [dir=both, label="simulate causes"]; A -> S [dir=both, label="test / action"]; S -> P;
  A -> W [dir=both, label="datasheets, evidence, parts"]; A -> R [label="checked diagnosis"];
}""", width="stretch")
    st.markdown("**Models** (all NVIDIA Nemotron, all served by Nebius Token Factory)")
    m = config()["llm"]["models"]
    st.table([
        {"Role": "Edge triage", "Model": m["triage"],
         "Why": "small and fast; reasoning off; reads the edge monitor's rule checks; JSON schema"},
        {"Role": "Triage fallback", "Model": m["triage_fallback"],
         "Why": "same price as Lightning, used if the first fails"},
        {"Role": "Diagnosis agent", "Model": m["agent"], "Why": "tool calling with brief reasoning"},
        {"Role": "Work order", "Model": m["report"],
         "Why": "largest Nemotron model, for the final text; Super writes in parallel and is shown if Ultra has no "
                "first token within 10 s or is not done within 30 s"},
    ])
    st.markdown("**The physics in six lines**")
    st.markdown("- Two-level inverter, switching-cycle average model; an open switch forces its leg to the opposite rail "
                "whenever current would need it.")
    st.latex(r"\frac{di_k}{dt} = \frac{v_k - v_n - R_k i_k}{L},\qquad v_n = \frac{1}{|C|}\sum_{k \in C}(v_k - R_k i_k)")
    st.markdown("- Park's vector modulus and normalized mean currents:")
    st.latex(r"|i_s| = \overline{\sqrt{i_\alpha^2 + i_\beta^2}},\qquad \mu_k = \overline{i_k} / |i_s|")
    st.markdown("- KCL residuals expose sensor errors (the true currents always sum to zero):")
    st.latex(r"r_{KCL,dc} = \frac{|\overline{i_a + i_b + i_c}|}{|i_s|},\qquad r_{KCL,\omega_1} = \frac{2\,|X_{\omega_1}(i_a+i_b+i_c)|}{|i_s|}")
    st.markdown("- An open upper switch gives μ ≈ [−0.42, +0.21, +0.21]; a sensor offset gives μ ≈ offset / |i_s|.\n"
                "- The modulation-step test: an offset's μ scales with 1/|i_s|, a switch fault's μ does not.\n"
                "- The digital twin runs the same equations with nominal parameters and fits each look-alike's size; "
                "the plant has noise and ±5 % parameter mismatch the twin does not know.")
    evaluated = (REPO_ROOT / "eval" / "results" / "results.md").exists()
    st.markdown("**Honest limitations**")
    st.markdown("- The plant is simulated (the Physical AI track allows this); hardware-in-the-loop is the next step.\n"
                "- By design the edge monitor does not alarm on a single-phase load unbalance up to about ±35 % (it "
                "is not a converter fault). When the twin is asked, an unbalance below about 15 % cannot be told apart "
                "from normal tolerance and may be called healthy, never a converter fault.\n"
                "- Triage is designed for edge deployment but runs on Token Factory here; the next step is to run it "
                "locally on a Jetson Orin Nano.\n"
                "- Recorded diagnoses (replay mode) are selected correct runs; the evaluation in the Results tab "
                "reports every run, failures included" + ("." if evaluated else ", once it has been run."))


def results_tab() -> None:
    st.subheader("Evaluation")
    results = REPO_ROOT / "eval" / "results"
    md, png = results / "results.md", results / "confusion_matrix.png"
    if not md.exists():
        st.info("The evaluation has not been run yet. It will compare a textbook threshold detector, Nemotron Super "
                "without tools, and the full FaultTwin agent on 24 fixed scenarios, and report every run.")
        return
    text = md.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("!["))   # the figure is shown below
    text = re.sub(r"^# .*\n", "", text, count=1)                                       # the tab has its own title
    text = text.replace("$", "\\$")                    # "$0.0177 ... $0.0262" would otherwise render as LaTeX
    # The wide per-scenario table is shown as a scrollable table from results.json instead of markdown.
    head, _, rest = text.partition("## Every scenario")
    tail = rest[rest.find("\n## "):] if "\n## " in rest else ""
    demote = lambda t: re.sub(r"^(#{2,3}) ", lambda m: "#" * (len(m.group(1)) + 1) + " ", t, flags=re.MULTILINE)
    st.markdown(demote(head))
    rj = results / "results.json"
    data = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    if rest and data.get("scenarios"):
        st.markdown("#### Every scenario")
        st.dataframe(scenario_rows(data["scenarios"]), hide_index=True, width="stretch")
    st.markdown(demote(tail))
    if png.exists():
        st.image(str(png), caption="Confusion by cause family for the three methods (24 scenarios, fixed seeds)")
    if data:
        with st.expander("Raw results"):
            st.json(data, expanded=False)


def scenario_rows(scenarios: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per evaluation scenario for the Results tab."""
    rows = []
    for r in scenarios.values():
        c = r["C"]

        def mark(label: str) -> str:
            return f"{label} {'✓' if label == r['truth'] else '✗'}"

        rows.append({"Scenario": r["id"], "Truth": r["truth"],
                     "Monitor alarm": f"{r['alarm']['type']} ({r['alarm']['severity']})" if r["alarm"] else "none",
                     "A: baseline": mark(r["A"]["label"]), "B: Super only": mark(r["B"]["label"]),
                     "C: FaultTwin": mark(c["label"]), "C cost (USD)": round(c.get("cost_usd") or 0.0, 4),
                     "C time (s)": c.get("wall_s") or 0.0,
                     "Note": "no alarm, no diagnosis" if c.get("no_alarm") else (c.get("stop_reason") or "")})
    return rows


def open_link(ss: Any) -> str:
    """A link can open a recorded case (?case=AU_open), at a moment (&at=35, seconds), and pick the first tab
    (&view=how or &view=results). Read once per visit; used for the README, the video and screenshots."""
    q = st.query_params
    if ss.get("link_done"):
        return ss.get("link_view", "live")
    ss.link_done = True
    ss.link_view = q.get("view", "live") if q.get("view") in ("how", "results") else "live"
    case = q.get("case")
    if case in replays() and start_replay(ss, case, "Opened from a link to this recorded case.", hidden=False):
        ss.attempt += 1
        ss.player.attempt = ss.attempt
        try:
            ss.player.start -= max(0.0, min(float(q.get("at", 0)), ss.player.duration_s))
        except ValueError:
            pass
    return ss.link_view


def main() -> None:
    ss = state()
    view = open_link(ss)
    if not ss.pop("forced_rerun", False):
        ss.last_activity = time.time()          # a visitor's click or slider, not a timer
    ss.plant_slow = idle(ss)
    if ss.figs and len(ss.figs) > 24:
        ss.figs = {}
    scale = tick_scale(ss)
    header(ss)
    sidebar(ss)
    names = {"live": ["Live", "How it works", "Results"], "how": ["How it works", "Live", "Results"],
             "results": ["Results", "Live", "How it works"]}[view]
    tabs = dict(zip(names, st.tabs(names)))
    live, how, res = tabs["Live"], tabs["How it works"], tabs["Results"]
    with live:
        if busy(ss):
            st.fragment(live_section, run_every=BUSY_TICK_S * scale, key="live")()
        else:
            # The plant keeps running on its own timer; a finished replay's recorded plant does not change.
            paused = ss.player is not None and ss.player.finished()
            every = None if paused else (IDLE_TICK_S if ss.plant_slow else PLANT_TICK_S) * scale
            st.fragment(plant_section, run_every=every, key="plant")()
            st.divider()
            st.fragment(agent_section, run_every=IDLE_TICK_S if review_pending(ss) else None, key="agent")()
    with how:
        how_it_works_tab()
    with res:
        results_tab()


main()
