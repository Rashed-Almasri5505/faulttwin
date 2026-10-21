"""Repository hygiene: secrets and local-only files stay out of git, config is well formed."""

import math
import re

import pytest

from faulttwin.config import REPO_ROOT, load_config

REQUIRED_GITIGNORE = [
    ".env",
    ".streamlit/secrets.toml",
    "CLAUDE.md",
    "PROGRESS.md",
    "submission/",
    "data/cache/",
    ".venv/",
]
KEY_PATTERNS = [re.compile(r"tvly-[A-Za-z0-9_-]{10,}"), re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")]
SKIP_DIRS = {".git", ".venv", "data", "__pycache__", ".pytest_cache"}
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


def test_no_api_keys_in_repo_files() -> None:
    for path in REPO_ROOT.rglob("*"):
        if SKIP_DIRS.intersection(path.relative_to(REPO_ROOT).parts) or path.name in SKIP_FILES:
            continue
        if not path.is_file() or (path.suffix not in TEXT_SUFFIXES and path.name not in {".gitignore", "LICENSE"}):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in KEY_PATTERNS:
            assert not pattern.search(text), f"key-like string in {path.relative_to(REPO_ROOT)}"
