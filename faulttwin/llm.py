"""Nebius Token Factory client: model routing, reasoning handling, cost accounting, spend caps.

Token Factory is OpenAI-compatible, so this wraps the OpenAI SDK. Reasoning is switched
with chat_template_kwargs (verified in Phase 0): enable_thinking false turns it off, and
low_effort true makes Super reason briefly. Reasoning text is never shown to users as an
answer; it is kept in the trace. Every call is metered from `usage` and booked in a small
JSON ledger that enforces the daily and total spend caps of the public demo.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from openai import OpenAI
from pydantic import BaseModel

from faulttwin.config import REPO_ROOT, get_secret

Role = Literal["triage", "triage_fallback", "agent", "report", "report_fallback"]
Thinking = Literal["default", "off", "low"]
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """A Token Factory call failed after the SDK's own retries."""


class BudgetExceeded(RuntimeError):
    """A spend cap would be exceeded; the app switches to replay mode."""


@dataclass
class LLMCall:
    """Metering record of one model call (also a trace step)."""

    label: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str | None = None
    ok: bool = True
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    content: str                    # answer text with any reasoning removed
    reasoning: str | None           # reasoning text, for the collapsed trace view
    tool_calls: list[ToolCall]
    call: LLMCall
    assistant_message: dict[str, Any] = field(default_factory=dict)   # to append to the history


def tier_of(model_id: str) -> str | None:
    low = model_id.lower()
    return next((t for t in ("ultra", "super", "lightning", "nano") if t in low), None)


def badge_of(model_id: str) -> str:
    """Short model name for the UI (Lightning / Nano / Super / Ultra)."""
    return (tier_of(model_id) or model_id).title()


_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "‑": "-", "‐": "-"})


def number_text(data: Any) -> str:
    """Text for number checks: JSON without \\u escapes (\\u2013 followed by 2.5 would read as 20132.5) and
    Unicode minus signs and dashes as '-'."""
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
    return text.translate(_DASHES)


_LEDGER_LOCKS: dict[Path, threading.Lock] = {}
_LEDGER_LOCKS_GUARD = threading.Lock()


def _retry_windows_lock(fn: Callable[[], Any], *, cleanup: Path | None = None, what: str | None = None,
                        attempts: int = 40, wait_s: float = 0.025) -> Any:
    """Run a file operation, retrying while another process holds the file open (Windows sharing violation).

    A reader in another process (the app, a CLI run) can briefly block the rename onto the ledger.
    """
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                if cleanup is not None:
                    cleanup.unlink(missing_ok=True)
                if what:
                    raise BudgetExceeded(what) from None
                raise
            time.sleep(wait_s)


class SpendLedger:
    """Daily and total spend in USD, kept in a JSON file and shared by all sessions of the app.

    Writes are atomic (temporary file + rename) under one lock per ledger file. An unreadable
    ledger fails closed: spending is refused until the file is fixed, never treated as $0.
    """

    def __init__(self, path: Path, daily_cap_usd: float, total_cap_usd: float) -> None:
        self.path = path
        self.daily_cap_usd = daily_cap_usd
        self.total_cap_usd = total_cap_usd
        with _LEDGER_LOCKS_GUARD:
            self._lock = _LEDGER_LOCKS.setdefault(path.resolve(), threading.Lock())

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SpendLedger:
        budget = cfg["budget"]
        daily = get_secret("DEMO_DAILY_BUDGET_USD") or budget["daily_usd"]
        total = get_secret("DEMO_TOTAL_BUDGET_USD") or budget["total_usd"]
        return cls(REPO_ROOT / budget["ledger_path"], float(daily), float(total))

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(_retry_windows_lock(lambda: self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return {"total_usd": 0.0, "days": {}}
        except (json.JSONDecodeError, OSError) as exc:
            raise BudgetExceeded(f"the spend ledger {self.path.name} is unreadable ({exc}); live calls are paused")
        if not isinstance(data, dict) or not isinstance(data.get("days", {}), dict):
            raise BudgetExceeded(f"the spend ledger {self.path.name} is malformed; live calls are paused")
        return data

    def add(self, cost_usd: float) -> None:
        with self._lock:
            data = self._read()
            today = date.today().isoformat()
            data["total_usd"] = float(data.get("total_usd", 0.0)) + cost_usd
            data.setdefault("days", {})[today] = float(data["days"].get(today, 0.0)) + cost_usd
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            _retry_windows_lock(lambda: os.replace(tmp, self.path), cleanup=tmp,
                                what=f"could not record spend in {self.path.name}; live calls are paused")

    def spent(self) -> tuple[float, float]:
        """(spent today, spent in total) in USD."""
        with self._lock:
            data = self._read()
        return float(data.get("days", {}).get(date.today().isoformat(), 0.0)), float(data.get("total_usd", 0.0))

    def remaining_usd(self) -> float:
        today, total = self.spent()
        return max(0.0, min(self.daily_cap_usd - today, self.total_cap_usd - total))

    def check(self, needed_usd: float = 0.0) -> None:
        today, total = self.spent()
        if today + needed_usd > self.daily_cap_usd:
            raise BudgetExceeded(f"daily demo budget of ${self.daily_cap_usd:.2f} reached")
        if total + needed_usd > self.total_cap_usd:
            raise BudgetExceeded(f"total demo budget of ${self.total_cap_usd:.2f} reached")


class TokenFactory:
    """Role-based access to the Nemotron models on Nebius Token Factory."""

    def __init__(self, cfg: dict[str, Any], *, ledger: SpendLedger | None = None,
                 client: Any | None = None) -> None:
        llm = cfg["llm"]
        self.cfg = cfg
        self.models: dict[str, str] = {k: v for k, v in llm["models"].items() if isinstance(v, str)}
        self.prices: dict[str, list[float]] = llm["prices_usd_per_1m_tokens"]
        self.sampling: dict[str, list[float]] = llm["sampling"]
        self.base_url: str = get_secret("NEBIUS_BASE_URL") or llm["base_url"]
        self.timeout_s: float = float(llm["timeout_s"])
        self.ledger = ledger
        self.calls: list[LLMCall] = []
        self._lock = threading.Lock()
        api_key = get_secret("NEBIUS_API_KEY")
        if client is not None:
            self._client = client
        elif api_key:
            self._client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=self.timeout_s,
                                  max_retries=int(llm["max_retries"]))
        else:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def missing_models(self) -> list[str]:
        """Configured model IDs that Token Factory does not list (exact case). Empty means all are served."""
        if self._client is None:
            return list(self.models.values())
        listed = {m.id for m in self._client.models.list()}
        return sorted({m for m in self.models.values() if m not in listed})

    def model_for(self, role: Role | str) -> str:
        return self.models.get(role, role)

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price_in, price_out = self.prices.get(tier_of(model) or "", [0.0, 0.0])
        return (prompt_tokens * price_in + completion_tokens * price_out) / 1e6

    def spent_usd(self) -> float:
        with self._lock:
            return sum(c.cost_usd for c in self.calls)

    def _sampling(self, model: str) -> tuple[float, float]:
        temperature, top_p = self.sampling.get(tier_of(model) or "", [0.6, 0.95])
        return float(temperature), float(top_p)

    @staticmethod
    def _thinking_body(thinking: Thinking) -> dict[str, Any]:
        if thinking == "off":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if thinking == "low":
            return {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
        return {}

    def _book(self, call: LLMCall) -> None:
        with self._lock:
            self.calls.append(call)
        if self.ledger is not None and call.cost_usd:
            self.ledger.add(call.cost_usd)

    def chat(self, role: Role | str, messages: list[dict[str, Any]], *, label: str,
             tools: list[dict[str, Any]] | None = None, tool_choice: str | None = None,
             response_format: dict[str, Any] | None = None, thinking: Thinking = "default",
             max_tokens: int = 2048, timeout_s: float | None = None, max_retries: int | None = None) -> ChatResult:
        """One non-streaming chat completion. Raises LLMError or BudgetExceeded."""
        if self._client is None:
            raise LLMError("NEBIUS_API_KEY is not set")
        if self.ledger is not None:
            self.ledger.check()
        model = self.model_for(role)
        temperature, top_p = self._sampling(model)
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens,
                                  "temperature": temperature, "top_p": top_p}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if response_format:
            kwargs["response_format"] = response_format
        extra = self._thinking_body(thinking)
        if extra:
            kwargs["extra_body"] = extra
        client = self._client
        if timeout_s is not None or max_retries is not None:
            client = client.with_options(**({"timeout": timeout_s} if timeout_s is not None else {}),
                                         **({"max_retries": max_retries} if max_retries is not None else {}))
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # the SDK has already retried 429, 5xx and timeouts
            call = LLMCall(label, model, latency_s=time.perf_counter() - t0, ok=False,
                           error=f"{type(exc).__name__}: {str(exc)[:200]}")
            self._book(call)
            raise LLMError(call.error) from exc
        latency = time.perf_counter() - t0
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        call = LLMCall(
            label, model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            latency_s=round(latency, 3),
            finish_reason=resp.choices[0].finish_reason,
        )
        call.cost_usd = self.cost_usd(model, call.prompt_tokens, call.completion_tokens)
        self._book(call)
        msg = resp.choices[0].message
        content, reasoning = split_reasoning(msg)
        tool_calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
                      for tc in (msg.tool_calls or [])]
        assistant: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant["tool_calls"] = [{"id": t.id, "type": "function",
                                        "function": {"name": t.name, "arguments": t.arguments}} for t in tool_calls]
        return ChatResult(content, reasoning, tool_calls, call, assistant)

    def stream_text(self, role: Role | str, messages: list[dict[str, Any]], *, label: str,
                    thinking: Thinking = "off", max_tokens: int = 800, timeout_s: float = 120.0,
                    first_token: threading.Event | None = None, cancel: threading.Event | None = None) -> ChatResult:
        """Streamed text completion (used for the work order). Sets first_token when text starts.

        call.finish_reason is "length" when the text was cut off at max_tokens, and "cancelled"
        when the cancel event stopped it.
        """
        if self._client is None:
            raise LLMError("NEBIUS_API_KEY is not set")
        if self.ledger is not None:
            self.ledger.check()
        model = self.model_for(role)
        temperature, top_p = self._sampling(model)
        client = self._client.with_options(timeout=timeout_s, max_retries=0)
        t0 = time.perf_counter()
        parts: list[str] = []
        usage = None
        finish = None
        try:
            stream = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                stream=True, stream_options={"include_usage": True}, extra_body=self._thinking_body(thinking))
            for chunk in stream:
                if cancel is not None and cancel.is_set():
                    finish = "cancelled"
                    break
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        parts.append(delta.content)
                        if first_token is not None:
                            first_token.set()
                    finish = chunk.choices[0].finish_reason or finish
                if chunk.usage:
                    usage = chunk.usage
        except Exception as exc:
            call = LLMCall(label, model, latency_s=time.perf_counter() - t0, ok=False,
                           error=f"{type(exc).__name__}: {str(exc)[:200]}")
            self._book(call)
            raise LLMError(call.error) from exc
        call = LLMCall(label, model, prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                       completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                       latency_s=round(time.perf_counter() - t0, 3), finish_reason=finish)
        if usage is None and parts:   # cancelled before the usage chunk: estimate from text length
            call.completion_tokens = sum(len(p) for p in parts) // 4
        call.cost_usd = self.cost_usd(model, call.prompt_tokens, call.completion_tokens)
        self._book(call)
        text = THINK_RE.sub("", "".join(parts)).strip()
        return ChatResult(text, None, [], call, {"role": "assistant", "content": text})


def split_reasoning(message: Any) -> tuple[str, str | None]:
    """Separate the answer from the reasoning (Token Factory uses reasoning_content or reasoning)."""
    content = message.content or ""
    extra = getattr(message, "model_extra", None) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    think = THINK_RE.findall(content)
    if think:
        reasoning = (reasoning or "") + "\n".join(t[7:-8] for t in think)
        content = THINK_RE.sub("", content)
    return content.strip(), reasoning or None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def environment_ready() -> dict[str, bool]:
    """Which external services have keys configured (never reveals the keys)."""
    return {"nebius": bool(get_secret("NEBIUS_API_KEY")), "tavily": bool(get_secret("TAVILY_API_KEY")),
            "pure_python": bool(os.environ.get("FAULTTWIN_PURE_PYTHON"))}
