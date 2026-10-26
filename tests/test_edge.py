"""Edge node (Phase 4b): protocol, the firmware's algorithm in Python, and the laptop side of the link.

All offline: the board is played by tests.fakes.FakeBoardSerial. The real board is compared in
tests/test_edge_board.py, which is skipped when no board is connected.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from faulttwin import edge
from faulttwin.config import load_config
from faulttwin.runtime import build_session
from faulttwin.scenarios import SCENARIOS
from tests.fakes import FakeBoardSerial

CFG = load_config()
SEED = 6201   # a Phase 3-5 test seed, not a calibration or evaluation seed


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    monkeypatch.setenv("FAULTTWIN_OFFLINE", "1")          # no model or Tavily client is ever built with real keys


def _session():
    return build_session(CFG, seed=SEED)


def _run(s, scenario: str, healthy: int = 10, after: int = 15) -> None:
    """Healthy cycles, then the scenario; at the end an operating-point change and a monitor reset, as the app does."""
    sc = SCENARIOS[scenario]
    with s.lock:
        s.plant.set_modulation_index(sc.m)
        s.advance(healthy)
        s.plant.set_condition(sc.condition)
        s.advance(after)
        s.plant.set_modulation_index(max(0.3, sc.m - 0.3))       # like the agent's modulation-step test
        s.monitor.expect_operating_point_change()
        s.advance(6)
        s.plant.set_modulation_index(sc.m)
        s.monitor.expect_operating_point_change()
        s.advance(6)
        s.monitor.expect_operating_point_change()                 # "derate (not needed)": a notice, no change of m
        s.advance(4)
        s.monitor.reset()                                          # Clear faults
        s.advance(8)


# --- the firmware's algorithm against the Python monitor ------------------------------------------

@pytest.mark.parametrize("scenario", ["healthy", "AU_open", "AL_open", "BL_open", "leg_A_open", "AU_BU_open",
                                      "sensor_offset_A", "sensor_gain_B", "load_unbalance_A"])
def test_the_board_algorithm_matches_the_python_monitor(scenario: str) -> None:
    """Block-averaged, int16 samples on the board give the same features and the same alarms (|delta mu| < 0.01)."""
    s = _session()
    model = edge.BoardModel(s.monitor.th)
    seen: list[tuple[int, int]] = []
    pairs, board_events = [], []

    def on_cycle(chunk, status):
        now = (s.monitor.resets, s.monitor.operating_point_notices)
        flags = (edge.FLAG_RESET if not seen or now[0] != seen[-1][0] else 0) | \
                (edge.FLAG_OP_CHANGE if seen and now[1] != seen[-1][1] else 0)
        seen.append(now)
        samples = edge.quantize(edge.decimate(chunk.i_meas_A)) * edge.SAMPLE_LSB_A
        f, events = model.process(chunk.first_cycle, chunk.m, flags, samples)
        pairs.append((f, status))
        board_events.extend(events)

    s.cycle_listeners.append(on_cycle)
    _run(s, scenario)
    compared = 0
    for f, status in pairs:
        assert f.ready == (status.features is not None), f.cycle
        if not f.ready:
            continue
        p = status.features
        compared += 1
        assert np.allclose(f.mu, p.mu, atol=0.01), (f.cycle, f.mu, p.mu)
        assert abs(f.r_kcl_dc - p.r_kcl_dc) < 0.005 and abs(f.r_kcl_w1 - p.r_kcl_w1) < 0.005
        assert np.allclose(f.amp_ratio, p.amp_ratio, atol=0.01)
        # Block averages remove most of the sensor noise, so a dead phase reads about 0.04 A RMS on the board and
        # 0.13 A here; both are far below the phase-loss limit (5 % of the other phases), so the alarms agree.
        assert np.allclose(f.rms_A, p.rms_A, rtol=0.01, atol=0.01 * p.i_s_A)
        assert f.i_s_A == pytest.approx(p.i_s_A, rel=0.005, abs=0.01)
    assert compared > 20
    assert [(e.cycle, e.type, e.severity) for e in board_events] == \
        [(e.cycle_index, e.type, e.severity) for e in s.events]


# --- the laptop side, with a fake board -------------------------------------------------------------

def _wait(cond, timeout_s: float = 3.0) -> None:
    deadline = time.time() + timeout_s
    while not cond():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


def test_the_link_streams_every_cycle_with_reset_and_hold_flags() -> None:
    s = _session()
    board = FakeBoardSerial()
    link = edge.EdgeLink(board, port="FAKE", board="Arduino UNO R4 WiFi")
    try:
        _wait(link.alive)
        link.attach(s)
        _run(s, "AU_open")
        last = s.cycle - 1
        _wait(lambda: (f := link.latest_features()) is not None and f.cycle == last)
        assert board.cycles == list(range(board.cycles[0], last + 1))             # every cycle, in order
        events, n = link.events_since(0)
        assert [(e.cycle, e.type) for e in events] == [(e.cycle_index, e.type) for e in s.events] and n == len(events)
        assert link.features_for(last) is not None
        link.show("I")
        link.show("T", edge.display_text("AU_open", [{"request": "derate", "approved": True}]))
        _wait(lambda: len(board.display) == 2)
        assert board.display == [("I", ""), ("T", "S1 OPEN DERATED")]
        board.press()
        _wait(lambda: link.take_presses() == 1)
        link.detach()
        assert link._on_cycle not in s.cycle_listeners
    finally:
        link.close()


def test_old_presses_and_presses_made_for_another_session_are_dropped(monkeypatch) -> None:
    board = FakeBoardSerial()
    link = edge.EdgeLink(board, port="FAKE", board="test")
    try:
        board.press()
        _wait(lambda: len(link.rx.presses) == 1)
        clock = time.time() + edge.PRESS_MAX_AGE_S + 1          # nobody read it for too long
        monkeypatch.setattr(edge.time, "time", lambda: clock)
        assert link.take_presses() == 0 and not link.rx.presses
        monkeypatch.undo()
        board.press()
        _wait(lambda: len(link.rx.presses) == 1)
        link.attach(_session())                                  # a new visitor takes the board
        assert link.take_presses() == 0
    finally:
        link.close()


def test_a_dead_port_marks_the_link_down_and_never_stops_the_plant() -> None:
    class Unplugged(FakeBoardSerial):
        def write(self, data: bytes) -> int:
            raise OSError("device disconnected")

    s = _session()
    link = edge.EdgeLink(Unplugged(), port="FAKE", board="test")
    try:
        link.attach(s)
        s.advance(3)                                    # the listener only queues; the plant keeps running
        _wait(lambda: link.dead)
        assert not link.alive() and s.cycle >= 3
    finally:
        link.close()


def test_find_board_needs_a_known_usb_id_and_the_firmware_heartbeat(monkeypatch) -> None:
    monkeypatch.delenv("FAULTTWIN_EDGE_PORT", raising=False)
    ports = [SimpleNamespace(device="COM3", vid=0x0BDA, pid=0x4853),       # a Bluetooth adapter: never opened
             SimpleNamespace(device="COM5", vid=0x2341, pid=0x1002)]
    opened = []

    def opener(port, baud, **_):
        opened.append((port, baud))
        return FakeBoardSerial()

    link = edge.find_board(opener=opener, ports=ports, wait_s=1.0)
    try:
        assert link is not None and link.port == "COM5" and link.board == "Arduino UNO R4 WiFi"
        assert opened == [("COM5", edge.BAUD)]
    finally:
        link.close()
    silent = edge.find_board(opener=lambda *a, **k: FakeBoardSerial(heartbeat=False), ports=ports[1:], wait_s=0.3)
    assert silent is None                                            # some other device on the port
    monkeypatch.setenv("FAULTTWIN_EDGE_PORT", "off")
    assert edge.find_board(opener=opener, ports=ports, wait_s=0.3) is None


# --- protocol details ----------------------------------------------------------------------------------

def test_frames_checksums_and_sample_encoding() -> None:
    assert edge.fletcher16(b"abcde") == 0xC8F0                        # the standard Fletcher-16 check value
    f = edge.frame("C", b"\x01\x02")
    assert f[:2] == b"\xa5\x5a" and f[2] == ord("C") and len(f) == 2 + 3 + 2 + 2
    i_A = np.random.default_rng(1).normal(0, 20, size=(1000, 3))
    blocks = edge.decimate(i_A)
    assert blocks.shape == (100, 3) and np.allclose(blocks.mean(axis=0), i_A.mean(axis=0))   # mu is preserved
    q = edge.quantize(np.array([[0.0, 65.6, -70.0], [0.0011, -0.0011, 1.0]]))
    assert q.dtype == np.dtype("<i2") and q.tolist() == [[0, 32767, -32768], [1, -1, 500]]
    assert len(edge.cycle_frame(7, 0.8, 0, edge.quantize(blocks))) == 2 + 3 + 8 + 600 + 2
    assert len(edge.config_frame(_session().monitor.th)) == 2 + 3 + 44 + 2
    assert edge.display_frame("T", "s1 open: derated!")[5:-2] == b"TS1 OPEN DERATED"


def test_board_lines_are_parsed_and_junk_is_ignored() -> None:
    kind, f = edge.parse_line("F 12 1 22.5 -0.42 0.21 0.21 0.0001 0.0004 0.67 1.19 1.14 14.2 18.5 17.6")
    assert kind == "F" and f.cycle == 12 and f.ready and f.mu == [-0.42, 0.21, 0.21] and f.rms_A[2] == 17.6
    assert edge.parse_line("E 12 mean_current high")[1].type == "mean_current"
    assert edge.parse_line("M") == ("M", None)
    assert edge.parse_line("HB FAULTTWIN-EDGE 1 UNO-R4-WIFI 1000")[0] == "HB"
    for junk in ("", "F 1 2", "E 3 not_a_check high", "F x 1 a b c d e f g h i j k l"):
        assert edge.parse_line(junk) is None


@pytest.mark.parametrize("label, actions, text", [
    ("AU_open", [{"request": "derate", "approved": True}], "S1 OPEN DERATED"),
    ("CL_open", [{"request": "derate (not needed)", "approved": True}], "S2 OPEN"),
    ("leg_B_open", [{"request": "stop", "approved": True}], "LEG B OPEN STOPPED"),
    ("AU_BU_open", [], "S1 S3 OPEN"),
    ("sensor_offset_A", [{"request": "active test: modulation step", "approved": True}], "SENSOR A - CONVERTER OK"),
    ("sensor_gain_C", [], "SENSOR C - CONVERTER OK"),
    ("load_unbalance_B", [], "LOAD B - CONVERTER OK"),
    ("load_step", [], "LOAD STEP - CONVERTER OK"),
    ("healthy", [], "NO FAULT - CONVERTER OK"),
    ("inconclusive", [{"request": "derate", "approved": True}], "INCONCLUSIVE DERATED"),
    ("BL_open", [{"request": "derate", "approved": False}], "S6 OPEN"),
    ("AU_open", [{"request": "stop", "approved": True}, {"request": "derate", "approved": False},
                 {"request": "resume", "approved": True, "applied": {"m": 0.45, "fault": "AU_open"}}],
     "S1 OPEN DERATED"),                                      # the last approved action is what the plant does
    ("healthy", [{"request": "stop", "approved": True}, {"request": "resume", "approved": True,
                                                        "applied": {"m": 0.8}}], "NO FAULT - CONVERTER OK"),
    ("CU_open", [{"request": "derate", "approved": True}, {"request": "stop", "approved": True}],
     "S5 OPEN STOPPED"),
])
def test_the_matrix_text_names_the_cause_and_the_action(label, actions, text) -> None:
    out = edge.display_text(label, actions)
    assert out == text and set(out) <= edge.DISPLAY_CHARS
