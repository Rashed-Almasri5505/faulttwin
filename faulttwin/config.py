"""Configuration and secrets.

Secrets are read from Streamlit secrets when deployed, then from environment
variables (a local .env file is loaded into the environment). Secret values are
never logged or printed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

load_dotenv(REPO_ROOT / ".env", override=False)


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st
        from streamlit import runtime

        if runtime.exists() and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # No Streamlit runtime or no secrets.toml: fall back to the environment.
        pass
    return None


def get_secret(name: str) -> str | None:
    """Return a secret by name, or None if it is unset or empty."""
    value = _streamlit_secret(name) or os.environ.get(name)
    return value.strip() if value and value.strip() else None


def has_secret(name: str) -> bool:
    return get_secret(name) is not None
