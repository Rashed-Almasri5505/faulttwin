"""Shared test setup."""

import numpy as np
import pytest

from faulttwin.plant import InverterPlant


@pytest.fixture(autouse=True)
def assert_kcl_on_every_plant_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLAUDE.md 5.1: the true currents of every plant run must satisfy |i_a + i_b + i_c| < 1e-6 A."""
    original = InverterPlant.run_cycles

    def checked(self: InverterPlant, n_cycles: int = 1):  # type: ignore[no-untyped-def]
        chunk = original(self, n_cycles)
        worst = float(np.max(np.abs(chunk.i_true_A.sum(axis=1))))
        assert worst < 1e-6, f"KCL violated in a plant run: max |i_a + i_b + i_c| = {worst:.3e} A"
        return chunk

    monkeypatch.setattr(InverterPlant, "run_cycles", checked)
