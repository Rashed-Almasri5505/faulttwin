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


class FakeBoardSerial:
    """A serial port with the edge-node firmware behind it, played by faulttwin.edge.BoardModel."""

    def __init__(self, *_: Any, heartbeat: bool = True, alarms: bool = True, **__: Any) -> None:
        import queue
        import struct

        import numpy as np

        from faulttwin import edge
        from faulttwin.config import load_config
        from faulttwin.monitor import MonitorThresholds

        self._np, self._struct, self._edge = np, struct, edge
        self.th = MonitorThresholds.from_config(load_config())
        self.model = edge.BoardModel(self.th)
        self.lines: queue.Queue[bytes] = queue.Queue()
        self.buf = bytearray()
        self.display: list[tuple[str, str]] = []
        self.cycles: list[int] = []
        self.closed = False
        self.heartbeat = heartbeat
        self.alarms = alarms                   # False: the board's alarm lines never arrive
        self._last_hb = time.time()
        if heartbeat:
            self.lines.put(b"HB FAULTTWIN-EDGE 1 UNO-R4-WIFI 1000\n")

    def press(self) -> None:
        self.lines.put(b"M\n")

    def write(self, data: bytes) -> int:
        self.buf.extend(data)
        while True:
            start = self.buf.find(b"\xa5\x5a")
            if start < 0 or len(self.buf) < start + 5:
                return len(data)
            kind, length = self.buf[start + 2], self._struct.unpack_from("<H", self.buf, start + 3)[0]
            end = start + 5 + length + 2
            if len(self.buf) < end:
                return len(data)
            body = bytes(self.buf[start + 2:end - 2])
            ok = self._edge.fletcher16(body) == self._struct.unpack_from("<H", self.buf, end - 2)[0]
            del self.buf[:end]
            if ok:
                self._handle(chr(kind), body[3:])
            else:
                self.lines.put(b"ERR checksum\n")

    def _handle(self, kind: str, p: bytes) -> None:
        if kind == "K":
            v = self._struct.unpack(self._edge.CONFIG_FORMAT, p)
            self.th = self.th.model_copy(update={
                "mu_abs_max": v[0], "mu_consecutive_cycles": v[1], "r_kcl_dc_max": v[2], "r_kcl_w1_max": v[3],
                "amp_ratio_band": (v[4], v[5]), "low_rms_frac": v[6], "i_s_step_frac": v[7], "warmup_cycles": v[8],
                "min_i_s_A": v[9], "clear_ratio": v[10], "clear_cycles": v[11], "window_cycles": v[12]})
            self.model = self._edge.BoardModel(self.th, phase_offset=v[13])
        elif kind == "C":
            cycle, m_milli, flags, n = self._struct.unpack_from("<IHBB", p)
            samples = self._np.frombuffer(p[8:8 + 6 * n], dtype="<i2").reshape(n, 3) * self._edge.SAMPLE_LSB_A
            f, events = self.model.process(cycle, m_milli / 1000.0, flags, samples)
            self.cycles.append(cycle)
            vals = [f.i_s_A, *f.mu, f.r_kcl_dc, f.r_kcl_w1, *f.amp_ratio, *f.rms_A]
            self.lines.put((f"F {cycle} {int(f.ready)} " + " ".join(f"{x:.5f}" for x in vals) + "\n").encode())
            for e in events if self.alarms else []:
                self.lines.put(f"E {e.cycle} {e.type} {e.severity}\n".encode())
                if e.shown:
                    self.lines.put(f"S {e.shown}\n".encode())
        elif kind == "D":
            self.display.append((chr(p[0]), p[1:].decode("ascii")))

    def readline(self) -> bytes:
        import queue

        try:
            return self.lines.get(timeout=0.05)
        except queue.Empty:
            if self.heartbeat and time.time() - self._last_hb >= 1.0:      # every second, as the firmware does
                self._last_hb = time.time()
                return b"HB FAULTTWIN-EDGE 1 UNO-R4-WIFI 2000\n"
            return b""

    def close(self) -> None:
        self.closed = True


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
