"""Record the Mystery-pool diagnoses as replays for the public demo (live Token Factory and Tavily calls).

Each scenario runs once through scripts/run_cli_demo.py with --save-replay and --wait-for-ultra.
A replay is saved only when the diagnosis is correct; the console says so for every scenario, and
every trace (correct or not) stays in data/traces/. Cost: about $0.02 per scenario.

Usage:  python scripts/record_replays.py                 (all of MYSTERY_POOL, seed 4343)
        python scripts/record_replays.py AU_open BL_open
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faulttwin.scenarios import MYSTERY_POOL  # noqa: E402

SEED = 4343   # reserved for replay recordings; the evaluation uses other seeds


def main() -> int:
    names = sys.argv[1:] or MYSTERY_POOL
    failed = []
    for name in names:
        print(f"=== {name}", flush=True)
        code = subprocess.call([sys.executable, str(ROOT / "scripts" / "run_cli_demo.py"), "--scenario", name,
                                "--seed", str(SEED), "--wait-for-ultra", "--save-replay"], cwd=ROOT)
        if code != 0:
            failed.append(name)
    print(f"\nRecorded {len(names) - len(failed)} of {len(names)}." + (f" Not saved: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
