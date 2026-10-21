"""Start the app from any working directory (the app's .streamlit/config.toml is used).

Usage:  python scripts/run_app.py [--server.port 8501]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    os.chdir(ROOT)
    from streamlit.web import cli

    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true", *sys.argv[1:]]
    sys.exit(cli.main())
