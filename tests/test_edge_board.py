"""The real edge node against the Python monitor (|delta mu| < 0.01, the same alarms). Skipped without a board."""

import importlib.util

import pytest

from faulttwin import edge
from faulttwin.config import REPO_ROOT, load_config

_spec = importlib.util.spec_from_file_location("edge_compare", REPO_ROOT / "scripts" / "edge_compare.py")
edge_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(edge_compare)  # type: ignore[union-attr]


@pytest.fixture(scope="module")
def board():
    link = edge.find_board(wait_s=2.5)
    if link is None:
        pytest.skip("no FaultTwin edge node connected (or its port is in use by the app)")
    yield link
    link.close()


@pytest.mark.parametrize("scenario", ["AU_open", "leg_C_open", "AU_BU_open", "sensor_offset_A", "sensor_gain_B"])
def test_the_board_matches_the_python_monitor(board, scenario: str, monkeypatch) -> None:
    monkeypatch.setenv("FAULTTWIN_OFFLINE", "1")
    r = edge_compare.compare(board, scenario, load_config())
    assert r["missing"] == 0 and r["ready_mismatch"] == 0 and r["compared"] > 20
    assert r["max_abs_d_mu"] < 0.01
    assert r["alarms_board"] == r["alarms_python"]
    assert not r["board_errors"]
