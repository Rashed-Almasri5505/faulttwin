"""Phase 0 smoke test for the Tavily API.

Runs one basic search for IGBT-module datasheets, then one basic extract on an
HTML result and a PDF result, and reports whether the PDF text is usable
(datasheet symbols such as V_CES and I_C appear in it).
Uses 2-3 Tavily credits (one more search if the first finds no PDF). Saves the report to data/cache/smoke_tavily.json.

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
PDF_DOMAINS = ["infineon.com", "semikron-danfoss.com", "onsemi.com", "mitsubishielectric.com", "fujielectric.com"]
DATASHEET_MARKERS = re.compile(r"V\s*_?\s*CES|I\s*_?\s*C\s*nom|V\s*_?\s*CE\s*\(?sat|collector-emitter", re.IGNORECASE)


def is_pdf(url: str) -> bool:
    return ".pdf" in url.lower()


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
        "response_time": search.get("response_time"),
        "usage": search.get("usage"),
        "results": [{"title": r.get("title"), "url": r.get("url"), "score": r.get("score"),
                     "content_preview": (r.get("content") or "")[:160]} for r in search.get("results", [])],
    }
    urls = [r["url"] for r in search.get("results", [])]
    pdf_urls = [u for u in urls if is_pdf(u)]
    if not pdf_urls:
        # One more basic search restricted to manufacturer sites, which usually serve datasheets as PDF.
        retry = client.search(QUERY, search_depth="basic", max_results=5, include_domains=PDF_DOMAINS,
                              include_usage=True)
        pdf_urls = [r["url"] for r in retry.get("results", []) if is_pdf(r["url"])]
        report["pdf_retry"] = {"urls": [r["url"] for r in retry.get("results", [])], "usage": retry.get("usage")}
    html_urls = [u for u in urls if not is_pdf(u)]
    targets = html_urls[:1] + pdf_urls[:1]
    report["extract_targets"] = targets

    if targets:
        t0 = time.perf_counter()
        extract = client.extract(urls=targets, extract_depth="basic", format="markdown", include_usage=True)
        rows = []
        for r in extract.get("results", []):
            text = r.get("raw_content") or ""
            rows.append({"url": r.get("url"), "pdf": is_pdf(r.get("url", "")), "chars": len(text),
                         "datasheet_markers_found": sorted({m.group(0) for m in DATASHEET_MARKERS.finditer(text)})[:8],
                         "preview": text[:300]})
        report["extract"] = {"latency_s": round(time.perf_counter() - t0, 2), "results": rows,
                             "failed_results": extract.get("failed_results", []), "usage": extract.get("usage")}
    else:
        report["extract"] = {"skipped": "search returned no URLs"}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Search: {len(report['search']['results'])} results in {report['search']['latency_s']} s")
    for r in report["search"]["results"]:
        print(f"  [{r['score'] or 0:.2f}] {r['title']}\n         {r['url']}")
    ext = report["extract"]
    if "results" in ext:
        print(f"Extract: {len(ext['results'])} ok, {len(ext['failed_results'])} failed, {ext['latency_s']} s")
        for r in ext["results"]:
            kind = "PDF " if r["pdf"] else "HTML"
            print(f"  {kind} {r['chars']:>7} chars  markers={r['datasheet_markers_found']}  {r['url']}")
        for f in ext["failed_results"]:
            print(f"  FAILED {f}")
        if not pdf_urls:
            print("  Note: no PDF in the search results, so PDF extraction was not tested.")
    else:
        print(f"Extract skipped: {ext['skipped']}")
    print(f"Full report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
