"""Technician work order, written by Nemotron 3 Ultra with Nemotron 3 Super as a hedge.

Both models start at the same time. Ultra's work order is used when its first token comes
within first_token_deadline_s and it finishes within total_deadline_s. Otherwise Super's
work order is shown at once, and Ultra's text, if it arrives later, is shown as a
"senior review". The UI never waits for Ultra past the deadlines. Replays and the
evaluation can wait for Ultra's full output (wait_for_ultra=True).

The model writes only the prose sections. The Parts and Sources sections are built by code
from the checked diagnosis (Tavily results only), and any URL in the model's text is removed.
A model text that was cut off at max_tokens, or that contains numbers that are not in the
facts, is not used.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from faulttwin.hypotheses import SPECS, Hypothesis
from faulttwin.llm import ChatResult, LLMError, TokenFactory, badge_of, utc_now

REPORT_SYSTEM = (
    "You write technician work orders for inverter faults, in markdown. Write exactly these sections: a title line "
    "'# Work order: <diagnosis in plain words>', then '**Asset:**', '**Time (UTC):**', '## Diagnosis' (with the "
    "confidence), '## Evidence' (the evidence and tests given; you may point the reader to the plots by name - phase "
    "currents, Park's vector, spectrum in rad/s, measured-vs-twin overlay - but do not describe what a plot shows "
    "beyond the facts given), '## Actions taken' (with the safety interlock's verdicts) and '## Recommended steps'. "
    "Do not write Parts or Sources sections and do not write any URL: the system adds them. Use only the facts given "
    "and only their numbers; report only the tests in tests_run. Never invent part numbers, ratings or prices; if "
    "something is not given, say it is not verified. At most 300 words."
)
URL_RE = re.compile(r"\(?https?://\S+\)?")
_NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+\.\d+")


@dataclass
class WorkOrder:
    markdown: str
    model: str | None
    latency_s: float
    ultra_status: str                       # used / late / failed / cut off / not available
    senior_review: Future | None = None     # Ultra's text when it was late
    calls: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        review = None
        if self.senior_review is not None and self.senior_review.done() and self.senior_review.exception() is None:
            review = self.senior_review.result()
        return {"markdown": self.markdown, "model": self.model, "badge": badge_of(self.model) if self.model else None,
                "latency_s": self.latency_s, "ultra_status": self.ultra_status, "senior_review": review,
                "calls": self.calls, "notes": self.notes}


def report_facts(final: dict[str, Any], session: Any, asset: str) -> dict[str, Any]:
    label = final["label"]
    plain = SPECS[Hypothesis(label)].description if label in {h.value for h in Hypothesis} else "inconclusive"
    nameplate = {k: v for k, v in session.nameplate.model_dump(exclude_none=True).items() if not k.endswith("_url")}
    return {
        "asset": asset,
        "time_utc": utc_now(),
        "diagnosis": {"label": label, "in_plain_words": plain, "confidence": final.get("confidence"),
                      "summary": final.get("summary"), "twin_match_error_pct": final.get("twin_match_error_pct")},
        "evidence": final.get("evidence", []),
        "alternatives_rejected": final.get("alternatives_rejected", []),
        "tests_run": final.get("tests_run", []),
        "interlock_verdicts": [{"request": v.request, "approved": v.approved, "reason": v.reason,
                                "applied": v.applied} for v in session.diag_verdicts()],
        "datasheet": session.datasheet.compact() if session.datasheet else "not looked up",
        "nameplate": nameplate,
        "recommended_maintenance": final.get("recommended_maintenance", []),
        "replacement_part_links": len(final.get("replacement_parts", [])),
        "sources_listed_by_system": len(final.get("citations", [])),
    }


def code_sections(final: dict[str, Any], session: Any) -> str:
    """Parts and Sources, built from the checked diagnosis: only Tavily results, with their labels."""
    def label(url: str) -> str:
        src = session.diag_sources.get(url)
        return f" ({src.label})" if src else ""

    parts = [f"- [{p['name']}]({p['url']}){label(p['url'])}" for p in final.get("replacement_parts", [])]
    sources = [f"- [{c['title']}]({c['url']}){label(c['url'])}" for c in final.get("citations", [])]
    return "\n".join(["## Parts", *(parts or ["- none listed"]), "", "## Sources", *(sources or ["- none"])])


def _known_numbers(facts: dict[str, Any]) -> set[float]:
    return {float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(facts, default=str))}


def _check_prose(text: str, known: set[float]) -> list[str]:
    bad = [float(x) for x in _NUM_RE.findall(text)
           if not any(abs(abs(float(x)) - abs(y)) <= max(0.006, 0.02 * abs(y)) or
                      abs(abs(float(x)) - 100 * abs(y)) <= max(0.06, 2 * abs(y)) for y in known)]
    return [f"numbers not in the facts: {bad[:5]}"] if bad else []


def _clean(text: str) -> str:
    """Remove URLs and any Parts/Sources sections the model wrote anyway."""
    text = re.split(r"^##\s*(Parts|Sources)\b", text, flags=re.MULTILINE)[0]
    return URL_RE.sub("", text).rstrip()


def write_work_order(llm: TokenFactory, cfg: dict[str, Any], final: dict[str, Any], session: Any,
                     asset: str, *, wait_for_ultra: bool = False) -> WorkOrder:
    rc = cfg["llm"]["report"]
    facts = report_facts(final, session, asset)
    known = _known_numbers(facts)
    messages = [{"role": "system", "content": REPORT_SYSTEM}, {"role": "user", "content": json.dumps(facts)}]
    thinking = "default" if rc["enable_thinking"] else "off"
    t0 = time.perf_counter()
    first_token = threading.Event()
    cancel_hedge = threading.Event()
    notes: list[str] = []
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="report")

    def stream(role: str, flag: threading.Event | None, cancel: threading.Event | None = None) -> ChatResult:
        return llm.stream_text(role, messages, label=f"report:{role}", thinking=thinking,
                               max_tokens=int(rc["max_tokens"]), timeout_s=120.0, first_token=flag, cancel=cancel)

    ultra = pool.submit(stream, "report", first_token)
    hedge = (pool.submit(stream, "report_fallback", None, cancel_hedge)
             if rc["hedge_with_fallback"] and not wait_for_ultra else None)
    pool.shutdown(wait=False)

    def usable(res: ChatResult, who: str) -> bool:
        if res.call.finish_reason == "length":
            notes.append(f"{who} text was cut off at max_tokens; not used")
            return False
        problems = _check_prose(_clean(res.content), known)
        if problems:
            notes.append(f"{who} text not used: {problems[0]}")
            return False
        return True

    def done(res: ChatResult, status: str, review: Future | None = None) -> WorkOrder:
        cancel_hedge.set()
        if hedge is not None and hedge is not review:
            try:
                hedge.result(timeout=3)   # stops quickly once cancelled; its cost is then booked
            except Exception:  # noqa: BLE001
                pass
        calls = [f.result().call.as_dict() for f in (ultra, hedge) if f is not None and f.done()
                 and f.exception() is None]
        markdown = _clean(res.content) + "\n\n" + code_sections(final, session)
        return WorkOrder(markdown, res.call.model, round(time.perf_counter() - t0, 2), status, review, calls, notes)

    status = "failed"
    if wait_for_ultra:
        for attempt in (1, 2):
            try:
                res = ultra.result(timeout=150) if attempt == 1 else stream("report", None)
                if usable(res, "Ultra"):
                    return done(res, "used")
                status = "cut off" if res.call.finish_reason == "length" else "failed"
                break
            except Exception:  # noqa: BLE001 - one retry, then the fallback model writes it
                continue
        try:
            res = stream("report_fallback", None)
            if usable(res, "Super"):
                return done(res, status)
        except LLMError:
            pass
    else:
        deadline_first = t0 + float(rc["first_token_deadline_s"])
        while not first_token.is_set() and not ultra.done() and time.perf_counter() < deadline_first:
            first_token.wait(0.05)
        if first_token.is_set():
            remaining = float(rc["total_deadline_s"]) - (time.perf_counter() - t0)
            try:
                res = ultra.result(timeout=max(0.0, remaining))
                if usable(res, "Ultra"):
                    return done(res, "used")
                status = "cut off" if res.call.finish_reason == "length" else "failed"
            except LLMError:
                status = "failed"
            except Exception:  # noqa: BLE001 - concurrent.futures.TimeoutError
                status = "late"
        else:
            status = "failed" if ultra.done() and ultra.exception() is not None else "late"
        if hedge is not None:
            try:
                res = hedge.result(timeout=120)
                if usable(res, "Super"):
                    return done(res, status, ultra if status == "late" else None)
            except Exception:  # noqa: BLE001
                pass
    reason = "no language model was available" if not llm.available else "the model texts did not pass the checks"
    return WorkOrder(_fallback_markdown(final, session, asset, reason), None, round(time.perf_counter() - t0, 2),
                     "not available" if not llm.available else status, None, [], notes)


def _fallback_markdown(final: dict[str, Any], session: Any, asset: str, reason: str) -> str:
    """A plain work order built from the checked diagnosis when no model text can be used."""
    f = report_facts(final, session, asset)
    d = f["diagnosis"]

    def bullets(items: list[str]) -> list[str]:
        return [f"- {x}" for x in items] or ["- none"]

    lines = [f"# Work order: {d['in_plain_words']}", f"**Asset:** {asset}", f"**Time (UTC):** {f['time_utc']}",
             "## Diagnosis", f"{d['label']} (confidence {d['confidence']}). {d['summary']}",
             "## Evidence", *bullets(f["evidence"] + f["tests_run"]),
             "## Actions taken", *bullets([f"{v['request']}: {'approved' if v['approved'] else 'rejected'} "
                                          f"({v['reason']})" for v in f["interlock_verdicts"]]),
             "## Recommended steps", *bullets(f["recommended_maintenance"]),
             "", code_sections(final, session), "", f"_Written from the checked diagnosis by code ({reason})._"]
    return "\n".join(lines)
