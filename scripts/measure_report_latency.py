"""Measure report-writer latency on Token Factory.

Settings chosen to make the call as fast as possible: streaming, reasoning off
(chat_template_kwargs enable_thinking = false), a short prompt, and max_tokens = 800.
The diagnosis in the prompt is a synthetic example used only for timing.

Runs N rounds (one call per model per round, optionally pausing between rounds),
prints time to first token, total time, output tokens and cost, and appends every
run to data/cache/latency_log.jsonl.

Usage:  python scripts/measure_report_latency.py --models report agent --runs 3
        python scripts/measure_report_latency.py --models report agent --runs 12 --pause-s 600
        python scripts/measure_report_latency.py --summary      (markdown table of the log)
Models are role names from config.yaml or full model IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from faulttwin.config import REPO_ROOT, get_secret, load_config  # noqa: E402

LOG_PATH = REPO_ROOT / "data" / "cache" / "latency_log.jsonl"
MAX_TOKENS = 800
TIMEOUT_S = 240

SYSTEM = (
    "You write short technician work orders for inverter faults. Use markdown with the headings "
    "Diagnosis, Evidence, Actions taken, Next steps, Sources. At most 250 words. Use only the facts given."
)
DIAGNOSIS = {
    "asset": "INV-01, two-level three-phase inverter, 400 V DC link",
    "label": "AU_open",
    "confidence": 0.9,
    "summary": "Upper switch of leg A (S1) is open.",
    "evidence": [
        "normalized mean currents mu = [-0.42, +0.21, +0.21]",
        "KCL residual r_KCL,dc = 0.001, so the current sensors agree",
        "twin match error 3.1 % for AU_open vs 41 % for sensor_offset_A",
        "mu_a stayed at -0.42 when the modulation index was stepped from 0.8 to 0.4",
    ],
    "actions": ["derated to m = 0.4, approved by the safety interlock"],
    "citations": [{"title": "A1P50S65M2 datasheet", "url": "https://www.st.com/resource/en/datasheet/a1p50s65m2.pdf"}],
}


def tier_of(model_id: str) -> str | None:
    low = model_id.lower()
    return next((t for t in ("ultra", "super", "lightning", "nano") if t in low), None)


def measure(client: OpenAI, model: str, prices: dict[str, list[float]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "settings": {"stream": True, "enable_thinking": False, "max_tokens": MAX_TOKENS},
    }
    t0 = time.perf_counter()
    first_chunk = first_text = None
    usage = None
    finish = None
    text = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": json.dumps(DIAGNOSIS)}],
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            top_p=0.95,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for chunk in stream:
            now = time.perf_counter() - t0
            if first_chunk is None:
                first_chunk = now
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    if first_text is None:
                        first_text = now
                    text += delta.content
                finish = chunk.choices[0].finish_reason or finish
            if chunk.usage:
                usage = chunk.usage
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    total = time.perf_counter() - t0
    p_tok = usage.prompt_tokens if usage else None
    c_tok = usage.completion_tokens if usage else None
    price_in, price_out = prices.get(tier_of(model) or "", [0.0, 0.0])
    gen_s = total - first_text if first_text is not None else None
    row.update(
        ttft_s=round(first_text, 2) if first_text is not None else None,
        first_chunk_s=round(first_chunk, 2) if first_chunk is not None else None,
        total_s=round(total, 2),
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
        output_tok_per_s=round(c_tok / gen_s, 1) if c_tok and gen_s and gen_s > 0.05 else None,
        finish_reason=finish,
        words=len(text.split()),
        cost_usd=round(((p_tok or 0) * price_in + (c_tok or 0) * price_out) / 1e6, 6),
    )
    return row


def print_summary() -> None:
    rows = [json.loads(ln) for ln in LOG_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print("| Time (UTC) | Model | TTFT (s) | Total (s) | Output tokens | Output tok/s | Note |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        note = r.get("error", "") or ("" if r.get("finish_reason") == "stop" else f"finish={r.get('finish_reason')}")
        print(f"| {r['run_at_utc']} | `{r['model']}` | {r.get('ttft_s', '-')} | {r['total_s']} | "
              f"{r.get('completion_tokens', '-')} | {r.get('output_tok_per_s', '-')} | {note} |")
    by_model: dict[str, list[float]] = {}
    for r in rows:
        if "error" not in r and r.get("ttft_s") is not None:
            by_model.setdefault(r["model"], []).append(r["total_s"])
    print()
    for model, totals in by_model.items():
        totals.sort()
        print(f"- `{model}`: {len(totals)} runs, total time min {totals[0]} s, "
              f"median {totals[len(totals) // 2]} s, max {totals[-1]} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["report", "agent"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pause-s", type=float, default=0.0, help="pause between rounds")
    parser.add_argument("--summary", action="store_true", help="print the log as a markdown table and exit")
    args = parser.parse_args()
    if args.summary:
        print_summary()
        return 0

    cfg = load_config()
    api_key = get_secret("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2
    base_url = get_secret("NEBIUS_BASE_URL") or cfg["llm"]["base_url"]
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT_S, max_retries=0)
    prices = cfg["llm"]["prices_usd_per_1m_tokens"]
    models = [cfg["llm"]["models"].get(m, m) for m in args.models]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"{'model':<40} {'run':>3} {'TTFT s':>7} {'total s':>8} {'out tok':>7} {'tok/s':>6} {'finish':>7} {'cost $':>8}")
    total_cost = 0.0
    for i in range(1, args.runs + 1):
        if i > 1 and args.pause_s > 0:
            time.sleep(args.pause_s)
        for model in models:
            row = measure(client, model, prices)
            row["base_url"] = base_url
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            total_cost += row["cost_usd"]
            if "error" in row:
                print(f"{model:<40} {i:>3} ERROR after {row['total_s']} s: {row['error']}", flush=True)
                continue
            print(f"{model:<40} {i:>3} {row['ttft_s'] or '-':>7} {row['total_s']:>8} {row['completion_tokens'] or '-':>7} "
                  f"{row['output_tok_per_s'] or '-':>6} {row['finish_reason'] or '-':>7} {row['cost_usd']:>8.5f}", flush=True)
    print(f"Total cost: ${total_cost:.4f}. Log: {LOG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
