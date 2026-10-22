"""Phase 0 smoke test for Nebius Token Factory.

Checks, in order:
  1. which base URL answers, and which Nemotron model IDs the account can see
  2. one short chat with each Nemotron role (triage, agent, report)
  3. OpenAI-style tool calling, including the tool-result round trip
  4. JSON-schema structured output
  5. how reasoning is returned, and whether it can be switched off

Prints a summary and saves the full report to data/cache/smoke_nebius.json.
The API key is read from the environment / .env and is never printed.

Usage:  python scripts/smoke_test_nebius.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from faulttwin.config import REPO_ROOT, get_secret, load_config  # noqa: E402

REPORT_PATH = REPO_ROOT / "data" / "cache" / "smoke_nebius.json"
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
ULTRA_TIMEOUT_S = 240
NO_THINKING = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

# Role -> (ID known before the smoke test, substrings that identify the family). Matching is case-insensitive.
ROLE_HINTS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "lightning": (None, ("lightning",)),
    "nano": ("nvidia/nvidia-nemotron-3-nano-30b-a3b", ("nemotron-3-nano",)),
    "super": ("nvidia/nemotron-3-super-120b-a12b", ("nemotron-3-super",)),
    "ultra": ("nvidia/nemotron-3-ultra-550b-a55b", ("nemotron-3-ultra",)),
}

FEATURES_TOOL = {
    "type": "function",
    "function": {
        "name": "get_latest_features",
        "description": "Return the latest inverter diagnostic features computed over a whole number of fundamental cycles.",
        "parameters": {
            "type": "object",
            "properties": {
                "window_cycles": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Number of fundamental cycles."}
            },
            "required": ["window_cycles"],
        },
    },
}

TRIAGE_SCHEMA = {
    "name": "triage",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "family": {"type": "string", "enum": ["switch_fault", "sensor_error", "load_change", "transient", "healthy"]},
            "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
            "reason": {"type": "string"},
        },
        "required": ["family", "urgency", "reason"],
        "additionalProperties": False,
    },
}


def tier_of(model_id: str) -> str | None:
    low = model_id.lower()
    for tier in ("ultra", "super", "lightning", "nano"):
        if tier in low:
            return tier
    return None


class Meter:
    """Counts tokens and dollars across all calls in this run."""

    def __init__(self, prices: dict[str, list[float]]) -> None:
        self.prices = prices
        self.cost_usd = 0.0
        self.calls: list[dict[str, Any]] = []

    def add(self, model: str, usage: Any, latency_s: float, label: str) -> dict[str, Any]:
        p_tok = getattr(usage, "prompt_tokens", 0) or 0
        c_tok = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        r_tok = getattr(details, "reasoning_tokens", None) if details else None
        price_in, price_out = self.prices.get(tier_of(model) or "", [0.0, 0.0])
        cost = (p_tok * price_in + c_tok * price_out) / 1e6
        self.cost_usd += cost
        row = {
            "label": label,
            "model": model,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "reasoning_tokens": r_tok,
            "latency_s": round(latency_s, 2),
            "cost_usd": round(cost, 6),
        }
        self.calls.append(row)
        return row


def inspect_message(msg: Any) -> dict[str, Any]:
    """Describe where (if anywhere) reasoning text appears in a chat message."""
    content = msg.content or ""
    extra = getattr(msg, "model_extra", None) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    return {
        "content_preview": THINK_RE.sub("", content).strip()[:200],
        "has_think_tags": "<think>" in content,
        "reasoning_field": "reasoning_content" if extra.get("reasoning_content") else ("reasoning" if extra.get("reasoning") else None),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "think_chars": sum(len(m) for m in THINK_RE.findall(content)),
    }


def pick_base_url(api_key: str, cfg: dict[str, Any]) -> tuple[OpenAI, str, list[str]]:
    candidates = [get_secret("NEBIUS_BASE_URL"), cfg["llm"]["base_url"], *cfg["llm"]["base_url_alternatives"]]
    errors = []
    for url in dict.fromkeys(c for c in candidates if c):
        client = OpenAI(base_url=url, api_key=api_key, timeout=cfg["llm"]["timeout_s"], max_retries=cfg["llm"]["max_retries"])
        try:
            ids = sorted(m.id for m in client.models.list())
            return client, url, ids
        except Exception as exc:  # report and try the next URL
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:200]}")
    raise SystemExit("No base URL answered:\n  " + "\n  ".join(errors))


def resolve_roles(ids: list[str]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for role, (known, needles) in ROLE_HINTS.items():
        exact = [i for i in ids if known and i.lower() == known.lower()]
        matches = [i for i in ids if "nemotron" in i.lower() and any(n in i.lower() for n in needles)]
        # Prefer the known ID; otherwise the shortest match (avoids "-fast" or quantized variants).
        resolved[role] = exact[0] if exact else (min(matches, key=len) if matches else None)
    return resolved


def chat(client: OpenAI, meter: Meter, model: str, label: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model, **kwargs)
    row = meter.add(model, resp.usage, time.perf_counter() - t0, label)
    return resp, row


def run_step(results: dict[str, Any], key: str, fn: Any) -> None:
    try:
        results[key] = fn()
    except Exception as exc:
        results[key] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def main() -> int:
    cfg = load_config()
    api_key = get_secret("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2

    meter = Meter(cfg["llm"]["prices_usd_per_1m_tokens"])
    client, base_url, ids = pick_base_url(api_key, cfg)
    nemotron_ids = [i for i in ids if "nemotron" in i.lower()]
    roles = resolve_roles(ids)
    report: dict[str, Any] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "model_count": len(ids),
        "nemotron_models": nemotron_ids,
        "resolved_roles": roles,
    }

    # 2. One short chat per role.
    chats: dict[str, Any] = {}
    for role, model in roles.items():
        if not model:
            chats[role] = {"ok": False, "error": "model not found in the list"}
            continue

        # Ultra can be slow to start: one long attempt instead of several 60 s retries.
        role_client = client.with_options(timeout=ULTRA_TIMEOUT_S, max_retries=0) if role == "ultra" else client

        def one(model: str = model, role: str = role, role_client: OpenAI = role_client) -> dict[str, Any]:
            resp, row = chat(
                role_client, meter, model, f"chat:{role}",
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=1024, temperature=0.6,
            )
            return {"ok": True, **row, "finish_reason": resp.choices[0].finish_reason, **inspect_message(resp.choices[0].message)}

        run_step(chats, role, one)
    report["chat"] = chats

    # 3. Tool calling on the agent model, then the small model.
    tools: dict[str, Any] = {}
    for role in ("super", "nano", "lightning"):
        model = roles.get(role)
        if not model:
            continue

        def tool_round_trip(model: str = model, role: str = role) -> dict[str, Any]:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": "You diagnose inverter faults. Use the tools to get data; never invent numbers."},
                {"role": "user", "content": "Get the latest features over a 2-cycle window, then tell me the value of mu_a."},
            ]
            resp, row1 = chat(client, meter, model, f"tools:{role}:call", messages=messages, tools=[FEATURES_TOOL],
                              tool_choice="auto", max_tokens=2048, temperature=0.6)
            msg = resp.choices[0].message
            calls = msg.tool_calls or []
            out: dict[str, Any] = {"ok": False, "first_call": row1, "finish_reason": resp.choices[0].finish_reason,
                                   "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls]}
            if not calls:
                out["content_preview"] = (msg.content or "")[:300]
                return out
            args = json.loads(calls[0].function.arguments or "{}")
            out["args_valid"] = calls[0].function.name == "get_latest_features" and args.get("window_cycles") == 2
            messages.append({"role": "assistant", "content": msg.content,
                             "tool_calls": [c.model_dump(exclude_none=True) for c in calls]})
            messages.append({"role": "tool", "tool_call_id": calls[0].id,
                             "content": json.dumps({"mu_a": -0.42, "mu_b": 0.21, "mu_c": 0.21, "r_kcl_dc": 0.001})})
            resp2, row2 = chat(client, meter, model, f"tools:{role}:answer", messages=messages, tools=[FEATURES_TOOL],
                               max_tokens=2048, temperature=0.6)
            answer = THINK_RE.sub("", resp2.choices[0].message.content or "")
            out.update(ok=bool(out["args_valid"]) and "-0.42" in answer, second_call=row2, answer_preview=answer.strip()[:200])
            return out

        run_step(tools, role, tool_round_trip)
    report["tool_calling"] = tools

    # 4. Structured output (JSON schema) on the triage candidates, with thinking on and off.
    structured: dict[str, Any] = {}
    for role in ("lightning", "nano", "super"):
        model = roles.get(role)
        if not model:
            continue
        for variant, extra in {"thinking_on": {}, "thinking_off": NO_THINKING}.items():

            def structured_call(model: str = model, role: str = role, variant: str = variant,
                                extra: dict = extra) -> dict[str, Any]:
                resp, row = chat(
                    client, meter, model, f"json:{role}:{variant}",
                    messages=[{"role": "user", "content": "Inverter event: mu_a=-0.42, mu_b=0.21, mu_c=0.21, "
                                                          "r_kcl_dc=0.001. Classify the event family and urgency."}],
                    response_format={"type": "json_schema", "json_schema": TRIAGE_SCHEMA},
                    max_tokens=2048, temperature=0.6, **extra,
                )
                text = THINK_RE.sub("", resp.choices[0].message.content or "").strip()
                try:
                    parsed = json.loads(text)
                    valid = isinstance(parsed, dict) and set(parsed) == {"family", "urgency", "reason"}
                except json.JSONDecodeError:
                    parsed, valid = None, False
                return {"ok": valid, **row, "finish_reason": resp.choices[0].finish_reason,
                        "parsed": parsed, "raw_preview": text[:200]}

            run_step(structured, f"{role}:{variant}", structured_call)
    report["structured_output"] = structured

    # 5. Reasoning probe: same prompt with and without thinking disabled.
    probe: dict[str, Any] = {}
    for role in ("lightning", "nano", "super"):
        model = roles.get(role)
        if not model:
            continue
        prompt = [{"role": "user", "content": "Is 17 a prime number? Answer yes or no."}]
        variants: dict[str, dict[str, Any]] = {
            "default": {},
            "enable_thinking_false": NO_THINKING,
            "system_no_think": {"messages": [{"role": "system", "content": "/no_think"}, *prompt]},
            "reasoning_effort_low": {"reasoning_effort": "low"},
        }
        if role == "super":
            # Model-card option for Super: think, but with fewer reasoning tokens.
            variants["low_effort"] = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}}
        for name, extra in variants.items():
            def probe_call(model: str = model, role: str = role, name: str = name, extra: dict = extra) -> dict[str, Any]:
                kwargs = {"messages": prompt, "max_tokens": 1024, "temperature": 0.6, **extra}
                resp, row = chat(client, meter, model, f"probe:{role}:{name}", **kwargs)
                return {"ok": True, **row, **inspect_message(resp.choices[0].message)}

            run_step(probe, f"{role}:{name}", probe_call)
    report["reasoning_probe"] = probe

    report["calls"] = meter.calls
    report["total_cost_usd"] = round(meter.cost_usd, 6)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report)
    return 0


def print_summary(report: dict[str, Any]) -> None:
    print(f"Base URL: {report['base_url']}  ({report['model_count']} models listed)")
    print("Nemotron models listed:")
    for i in report["nemotron_models"]:
        print(f"  {i}")
    print("Resolved roles:")
    for role, model in report["resolved_roles"].items():
        print(f"  {role:<10} {model}")

    def status(section: str) -> None:
        print(f"{section}:")
        for key, r in report[section].items():
            if not r.get("ok", False) and "error" in r:
                print(f"  {key:<32} FAIL  {r['error']}")
                continue
            bits = [("ok" if r.get("ok") else "not ok")]
            for field in ("latency_s", "prompt_tokens", "completion_tokens", "reasoning_tokens"):
                if r.get(field) is not None:
                    bits.append(f"{field}={r[field]}")
            if "reasoning_field" in r:
                bits.append(f"reasoning_field={r['reasoning_field']} reasoning_chars={r['reasoning_chars']} think_tags={r['has_think_tags']}")
            if "tool_calls" in r:
                bits.append(f"tool_calls={r['tool_calls']}")
            print(f"  {key:<32} " + "  ".join(bits))

    for section in ("chat", "tool_calling", "structured_output", "reasoning_probe"):
        status(section)
    print(f"Total cost of this smoke test: ${report['total_cost_usd']:.4f}")
    print(f"Full report: {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
