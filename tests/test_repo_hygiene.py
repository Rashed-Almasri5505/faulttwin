"""Repository hygiene: secrets and local-only files stay out of git, config is well formed."""

import math
import re
from pathlib import Path

import pytest

from faulttwin.config import REPO_ROOT, get_secret, load_config

REQUIRED_GITIGNORE = [
    ".env",
    ".streamlit/secrets.toml",
    "CLAUDE.md",
    "PROGRESS.md",
    "submission/",
    "data/cache/",
    ".venv/",
]
KEY_PATTERNS = [
    re.compile(r"tvly-[A-Za-z0-9_-]{10,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"']?[A-Za-z0-9_\-.]{16,}"),
]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}
SKIP_PREFIXES = (("data", "cache"),)   # gitignored; data/replays is committed, so it is scanned
SKIP_FILES = {".env", "CLAUDE.md", "PROGRESS.md"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json", ".example"}


def test_gitignore_has_required_entries() -> None:
    lines = {ln.strip() for ln in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    missing = [entry for entry in REQUIRED_GITIGNORE if entry not in lines]
    assert not missing, f"missing from .gitignore: {missing}"


def test_env_example_has_names_only() -> None:
    for ln in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        name, _, value = ln.partition("=")
        assert name.isupper(), ln
        assert value == "", f"{name} must have no value in .env.example"


def test_license_is_mit() -> None:
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Rashed Almasri" in text


def test_config_loads_with_expected_sections() -> None:
    cfg = load_config()
    for section in ("plant", "features", "monitor", "llm", "agent", "budget", "tavily", "nameplate"):
        assert section in cfg, section
    plant = cfg["plant"]
    samples_per_period = 2 * math.pi / (plant["w1_rad_s"] * plant["dt_s"])
    assert samples_per_period == pytest.approx(1000)


def test_model_ids_are_resolved_and_priced() -> None:
    llm = load_config()["llm"]
    assert llm["models_verified_on"], "run scripts/smoke_test_nebius.py and record the date"
    for role in ("triage", "triage_fallback", "agent", "report", "report_fallback"):
        model_id = llm["models"][role]
        assert isinstance(model_id, str) and model_id.startswith("nvidia/"), role
        assert any(tier in model_id.lower() for tier in llm["prices_usd_per_1m_tokens"]), f"no price tier for {model_id}"


def scanned_text_files() -> list[Path]:
    files = []
    for path in REPO_ROOT.rglob("*"):
        parts = path.relative_to(REPO_ROOT).parts
        if SKIP_DIRS.intersection(parts) or any(parts[:len(p)] == p for p in SKIP_PREFIXES):
            continue
        if path.name in SKIP_FILES or not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}:
            files.append(path)
    return files


def test_no_api_keys_in_repo_files() -> None:
    for path in scanned_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in KEY_PATTERNS:
            assert not pattern.search(text), f"key-like string in {path.relative_to(REPO_ROOT)}"


def test_real_secret_values_appear_in_no_file() -> None:
    """Uses the actual keys from .env when present; values are never printed."""
    secrets = [s for s in (get_secret("NEBIUS_API_KEY"), get_secret("TAVILY_API_KEY")) if s and len(s) >= 8]
    for path in scanned_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        leaked = any(s in text for s in secrets)
        assert not leaked, f"a configured API key appears in {path.relative_to(REPO_ROOT)}"
