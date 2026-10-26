"""Walk the edge node through what the demo shows, to check the board before filming (about 40 s).

    python scripts/edge_selftest.py               # finds the board
    python scripts/edge_selftest.py --port COM5

1. A healthy plant streams to the board: the matrix shows a check mark.
2. Upper switch AU (S1) opens: the board raises the alarm itself and shows 'A' with a down arrow.
3. Investigating: '?' alternates with 'A' and the arrow.
4. The answer scrolls: 'S1 OPEN DERATED'.
5. Touch a jumper wire from D2 to GND (or press the button): the script reports it.
6. Back to the check mark.
No model or web calls are made. Close the app first: only one program can use the board's port.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FAULTTWIN_OFFLINE", "1")          # no model or Tavily client with real keys

from faulttwin import edge  # noqa: E402
from faulttwin.config import load_config  # noqa: E402
from faulttwin.plant import Condition  # noqa: E402
from faulttwin.runtime import build_session, plant_clock  # noqa: E402
from faulttwin.scenarios import SCENARIOS  # noqa: E402

SEED = 6302          # not a calibration or evaluation seed


def pause(text: str, seconds: float) -> None:
    print(text, flush=True)
    time.sleep(seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", help="serial port (default: find the board)")
    args = ap.parse_args()
    link = edge.EdgeLink.open(args.port, "edge node") if args.port else edge.find_board()
    if link is None:
        print("No FaultTwin edge node found. Flash firmware/faulttwin_edge, connect the board, and close the app.")
        return 2
    ok = True
    try:
        print(f"{link.board} on {link.port}: {link.rx.heartbeat}")
        s = build_session(load_config(), seed=SEED)
        link.attach(s)
        link.show("H")
        with plant_clock(s, period_s=0.1, cycles=2):          # 20 cycles/s to the board
            pause("1/6 Healthy plant streaming: check mark.", 4)
            seen = link.events_since(0)[1]
            with s.lock:
                s.plant.set_condition(SCENARIOS["AU_open"].condition)
            deadline, alarm = time.time() + 10, None
            while alarm is None and time.time() < deadline:
                got, seen = link.events_since(seen)
                alarm = next((e for e in got if e.severity in ("medium", "high")), alarm)
                time.sleep(0.05)
            if alarm is None:
                print("   The board raised no alarm within 10 s: FAIL")
                ok = False
            else:
                f = link.features_for(alarm.cycle)
                mu = ", ".join(f"{x:+.3f}" for x in f.mu) if f else "?"
                pause(f"2/6 The board raised '{alarm.type}' ({alarm.severity}) at cycle {alarm.cycle}, "
                      f"mu = [{mu}]: 'A' with a down arrow.", 4)
            link.show("I")
            pause("3/6 Investigating: '?' alternating with 'A' and the arrow.", 5)
            text = edge.display_text("AU_open", [{"request": "derate", "approved": True}])
            link.show("T", text)
            pause(f"4/6 The answer scrolls: {text}.", 8)
            link.take_presses()
            print("5/6 Touch a jumper wire from D2 to GND (or press the button) within 30 s...", flush=True)
            deadline = time.time() + 30
            while not link.take_presses() and time.time() < deadline:
                time.sleep(0.05)
            if time.time() < deadline:
                print("   D2 press received.")
            else:
                print("   No press received: FAIL")
                ok = False
            with s.lock:
                s.plant.set_condition(Condition())
                s.monitor.reset()
            link.show("H")
            pause("6/6 Fault cleared: check mark again.", 2)
        print(f"Board errors: {list(link.rx.errors) or 'none'}")
        ok = ok and not link.rx.errors
    finally:
        link.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
