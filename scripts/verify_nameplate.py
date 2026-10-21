"""Verify the nameplate part numbers through Tavily and store them in config.yaml.

For the IGBT module: a live Tavily search must name the part, its datasheet must be readable,
and the ratings read from it (quotes checked against the page) must fit a 600/650 V, ~50 A
six-pack. For the gate driver: a live search must name the part and its datasheet page must
mention it and describe a gate driver. Only then, with --write, the verified part numbers and
datasheet URLs are written to the nameplate section of config.yaml.

Usage:  python scripts/verify_nameplate.py --module A1P50S65M2 --driver STDRIVE601 [--write]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faulttwin.config import CONFIG_PATH, load_config  # noqa: E402
from faulttwin.llm import SpendLedger, TokenFactory  # noqa: E402
from faulttwin.web import TavilyWeb, _squash  # noqa: E402


def check_module(web: TavilyWeb, llm: TokenFactory, pn: str) -> tuple[bool, str | None]:
    facts = web.datasheet_facts(pn, llm, cache=False)
    print(f"Module {pn}: source {facts.source_url} ({facts.via})")
    for name in ("V_CES", "I_C_nom", "V_CE_sat_typ", "V_F_typ"):
        f = getattr(facts, name)
        print(f"  {name:<13} {f.value} {f.unit} {f.condition or ''}  verified={f.verified}  quote={f.quote!r}")
    for note in facts.notes:
        print(f"  note: {note}")
    ok = (facts.V_CES.verified and facts.V_CES.value in (600, 650)
          and facts.I_C_nom.verified and 30 <= (facts.I_C_nom.value or 0) <= 75)
    print(f"  -> {'OK' if ok else 'NOT VERIFIED'}")
    return ok, facts.source_url


def check_driver(web: TavilyWeb, pn: str) -> tuple[bool, str | None]:
    key = _squash(pn)
    hits = web.datasheet_candidates(pn, "gate driver datasheet")
    print(f"Driver {pn}: candidates {[h.url for h in hits]}")
    for page in web.extract([h.url for h in hits[:2]]):
        names_part = key in _squash(page.text[:30000])
        is_driver = bool(re.search(r"gate driver|half[- ]bridge|three[- ]phase", page.text[:30000], re.IGNORECASE))
        print(f"  {page.url}  blocked={page.blocked} names_part={names_part} gate_driver_text={is_driver}")
        if not page.blocked and names_part and is_driver:
            print("  -> OK")
            return True, page.url
    print("  -> NOT VERIFIED")
    return False, None


def write_nameplate(module: str, module_url: str, driver: str, driver_url: str, rating: str) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    block = (
        "nameplate:\n"
        f"  # Verified through Tavily with scripts/verify_nameplate.py on {date.today().isoformat()}.\n"
        f"  inverter_rating: \"{rating}\"\n"
        f"  igbt_module_part_number: \"{module}\"\n"
        f"  igbt_module_datasheet_url: \"{module_url}\"\n"
        f"  gate_driver_part_number: \"{driver}\"\n"
        f"  gate_driver_datasheet_url: \"{driver_url}\"\n"
        f"  verified_on: \"{date.today().isoformat()}\"\n"
    )
    new, n = re.subn(r"^nameplate:\n(?:(?:[ \t].*|)\n)*", block, text, flags=re.MULTILINE)
    if n != 1:
        raise SystemExit("could not find the nameplate section in config.yaml")
    CONFIG_PATH.write_text(new, encoding="utf-8")
    print("config.yaml nameplate updated.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--rating", default="Two-level three-phase inverter, 400 V DC link, 27 A peak phase current")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    web = TavilyWeb(cfg)
    llm = TokenFactory(cfg, ledger=SpendLedger.from_config(cfg))
    if not (web.available and llm.available):
        print("Both TAVILY_API_KEY and NEBIUS_API_KEY are needed.", file=sys.stderr)
        return 2
    web.begin_diagnosis()   # the first search is live
    module_ok, module_url = check_module(web, llm, args.module)
    driver_ok, driver_url = check_driver(web, args.driver)
    print(f"Tavily: {web.live_calls} live calls, {web.credits_used:g} credits. Nemotron: ${llm.spent_usd():.4f}")
    if module_ok and driver_ok and args.write:
        write_nameplate(args.module, module_url or "", args.driver, driver_url or "", args.rating)
    return 0 if module_ok and driver_ok else 1


if __name__ == "__main__":
    sys.exit(main())
