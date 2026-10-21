"""Offline stand-ins for the Token Factory (OpenAI SDK) and Tavily clients."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Callable


def chat_response(content: str = "", tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
                  reasoning: str | None = None, prompt: int = 120, completion: int = 40) -> SimpleNamespace:
    calls = [SimpleNamespace(id=f"call{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
             for i, (name, args) in enumerate(tool_calls or [])]
    msg = SimpleNamespace(content=content, tool_calls=calls or None,
                          model_extra={"reasoning_content": reasoning} if reasoning else {})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if calls else "stop")],
                           usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                                                 completion_tokens_details=None))


def stream_chunks(text: str, delay_s: float = 0.0, first_delay_s: float = 0.0) -> Any:
    if first_delay_s:
        time.sleep(first_delay_s)
    for word in text.split(" "):
        if delay_s:
            time.sleep(delay_s)
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=word + " "), finish_reason=None)],
                              usage=None)
    yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=300, completion_tokens=len(text) // 4))


class FakeOpenAI:
    """Answers chat.completions.create with a responder(kwargs) function."""

    def __init__(self, responder: Callable[[dict[str, Any]], Any]) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.responder(kwargs)

    def with_options(self, **_: Any) -> FakeOpenAI:
        return self


class FakeTavily:
    def __init__(self, pages: dict[str, str] | None = None, results: list[dict[str, str]] | None = None,
                 fail_basic_pdf: bool = False) -> None:
        self.pages = pages or {}
        self.results = results or []
        self.fail_basic_pdf = fail_basic_pdf
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search", {"query": query, **kwargs}))
        return {"results": [dict(r, content=r.get("content", f"about {query}")) for r in self.results],
                "usage": {"credits": 1}}

    def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("extract", {"urls": urls, **kwargs}))
        ok, failed = [], []
        for u in urls:
            if u not in self.pages or (self.fail_basic_pdf and kwargs.get("extract_depth") == "basic" and ".pdf" in u):
                failed.append({"url": u, "error": "Failed to fetch url"})
            else:
                ok.append({"url": u, "raw_content": self.pages[u]})
        return {"results": ok, "failed_results": failed, "usage": {"credits": 1 if ok else 0}}
