"""Start the app as if no API keys were configured (replay mode), to check the deployment fallback.

Usage:  python scripts/run_app_without_keys.py [--server.port 8502]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    os.environ["FAULTTWIN_OFFLINE"] = "1"
    os.chdir(ROOT)
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true", *sys.argv[1:]]
    sys.exit(cli.main())
