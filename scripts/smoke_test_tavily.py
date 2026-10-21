"""Phase 0 smoke test for the Tavily API.

1. One basic search for six-pack IGBT-module datasheets.
2. One basic extract on an HTML result plus a manufacturer PDF datasheet.
3. If the PDF failed or came back as an anti-bot page, one advanced extract of it.

Reports whether the PDF text is usable (datasheet symbols such as VCES appear in it)
and the credits each call used. Uses about 2-3 Tavily credits.
Saves the report to data/cache/smoke_tavily.json.

Usage:  python scripts/smoke_test_tavily.py
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

from tavily import TavilyClient  # noqa: E402

from faulttwin.config import REPO_ROOT, get_secret  # noqa: E402

REPORT_PATH = REPO_ROOT / "data" / "cache" / "smoke_tavily.json"
QUERY = "650 V 50 A six-pack IGBT module datasheet pdf"
# Manufacturer datasheet used to test PDF extraction (distributor PDFs are often bot-blocked).
PDF_URL = "https://www.st.com/resource/en/datasheet/a1p50s65m2.pdf"
DATASHEET_MARKERS = re.compile(r"V\s*_?\s*CES|I\s*_?\s*C\s*nom|V\s*_?\s*CE\s*\(?sat|collector-emitter", re.IGNORECASE)
BLOCKED_MARKERS = re.compile(r"access denied|automation tools|captcha|enable javascript", re.IGNORECASE)


def is_pdf(url: str) -> bool:
    return ".pdf" in url.lower()


def describe(result: dict[str, Any]) -> dict[str, Any]:
    text = result.get("raw_content") or ""
    return {
        "url": result.get("url"),
        "pdf": is_pdf(result.get("url", "")),
        "chars": len(text),
        "blocked": bool(BLOCKED_MARKERS.search(text[:3000])) and len(text) < 5000,
        "datasheet_markers_found": sorted({m.group(0) for m in DATASHEET_MARKERS.finditer(text)})[:8],
        "preview": text[:300],
    }


def extract(client: TavilyClient, urls: list[str], depth: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = client.extract(urls=urls, extract_depth=depth, format="markdown", include_usage=True, timeout=60)
    return {
        "depth": depth,
        "latency_s": round(time.perf_counter() - t0, 2),
        "usage": resp.get("usage"),
        "results": [describe(r) for r in resp.get("results", [])],
        "failed_results": resp.get("failed_results", []),
    }


def main() -> int:
    api_key = get_secret("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 2
    client = TavilyClient(api_key=api_key)
    report: dict[str, Any] = {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "query": QUERY}

    t0 = time.perf_counter()
    search = client.search(QUERY, search_depth="basic", max_results=5, include_answer=False,
                           include_raw_content=False, topic="general", include_usage=True)
    report["search"] = {
        "latency_s": round(time.perf_counter() - t0, 2),
        "usage": search.get("usage"),
        "results": [{"title": r.get("title"), "url": r.get("url"), "score": r.get("score"),
                     "content_preview": (r.get("content") or "")[:160]} for r in search.get("results", [])],
    }
    html_urls = [r["url"] for r in search.get("results", []) if not is_pdf(r["url"])]
    report["extracts"] = [extract(client, html_urls[:1] + [PDF_URL], "basic")]

    pdf_ok = any(r["pdf"] and r["datasheet_markers_found"] and not r["blocked"] for r in report["extracts"][0]["results"])
    if not pdf_ok:
        report["extracts"].append(extract(client, [PDF_URL], "advanced"))
        pdf_ok = any(r["pdf"] and r["datasheet_markers_found"] and not r["blocked"]
                     for r in report["extracts"][-1]["results"])
    report["pdf_extraction_usable"] = pdf_ok

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Search: {len(report['search']['results'])} results in {report['search']['latency_s']} s, "
          f"usage={report['search']['usage']}")
    for r in report["search"]["results"]:
        print(f"  [{r['score'] or 0:.2f}] {r['title']}\n         {r['url']}")
    for ext in report["extracts"]:
        print(f"Extract ({ext['depth']}): {len(ext['results'])} ok, {len(ext['failed_results'])} failed, "
              f"{ext['latency_s']} s, usage={ext['usage']}")
        for r in ext["results"]:
            kind = "PDF " if r["pdf"] else "HTML"
            flag = "  BLOCKED PAGE" if r["blocked"] else ""
            print(f"  {kind} {r['chars']:>7} chars  markers={r['datasheet_markers_found']}{flag}  {r['url']}")
        for f in ext["failed_results"]:
            print(f"  FAILED {f.get('url')}: {f.get('error')}")
    print(f"PDF datasheet text usable: {pdf_ok}")
    print(f"Full report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0 if pdf_ok else 1


if __name__ == "__main__":
    sys.exit(main())
