"""Edge node: the edge monitor running on a microcontroller board (Arduino UNO R4 WiFi).

The laptop streams the plant's measured currents to the board over USB serial, one fundamental cycle per
frame (block-averaged to 100 samples per cycle, int16). The board computes the monitor's features and
threshold checks itself and reports them back; its 12x8 LED matrix shows the state, and a button (or a
jumper wire to GND) on pin D2 triggers a Mystery fault. Without a board the app uses the Python monitor
(faulttwin.monitor.EdgeMonitor), as before; the plant stays simulated either way.

Protocol, laptop to board, little-endian:
    0xA5 0x5A | type u8 | len u16 | payload | Fletcher-16 over (type, len, payload) u16
    'K' config   the monitor thresholds (CONFIG_FORMAT), sent on connect
    'C' cycle    cycle u32 | m_milli u16 | flags u8 | n u8 | n x (i_a, i_b, i_c) int16 in SAMPLE_LSB_A units
                 flags: 1 = the monitor was reset, 2 = the operating point was changed (hold new alarms)
    'D' display  mode u8 ('H' healthy, 'I' investigating, 'T' scroll text) | ASCII text
Board to laptop, one text line each:
    HB FAULTTWIN-EDGE <version> <board> <ms>       heartbeat, every second
    F <cycle> <ready> <i_s> <mu_a> <mu_b> <mu_c> <r_dc> <r_w1> <ar_a> <ar_b> <ar_c> <rms_a> <rms_b> <rms_c>
    E <cycle> <type> <severity>                    a new monitor event (the rising edge of a check)
    S <phase><sign>                                what the matrix now shows for that event, e.g. "S A-"
    M                                              the Mystery-fault trigger on D2
    ERR <text>                                     a frame the board rejected

BoardModel mirrors the firmware's arithmetic in Python. It is the oracle in the tests and the fake board
for app tests without hardware; the comparison test checks the real board against it and against the
full-rate Python monitor.
"""

from __future__ import annotations

import math
import os
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from faulttwin.hypotheses import SPECS, Hypothesis
from faulttwin.monitor import CHECKS, SEVERITY, Event, MonitorThresholds

try:                                           # optional: the deployed demo never has a board
    import serial
    from serial.tools import list_ports
except ImportError:                            # pragma: no cover - pyserial is in requirements.txt
    serial = None
    list_ports = None

BAUD = 460800              # measured on the board: 39.6 kB/s with no errors; one cycle frame is 615 bytes
PACE_BYTES_S = 30_000      # the USB bridge has no flow control, so the laptop never sends faster than this
DECIMATION = 10            # 1000 plant samples per cycle -> 100 block averages per cycle
PHASE_OFFSET = (DECIMATION - 1) / (2 * DECIMATION)   # a block average sits at the block's centre time
SAMPLE_LSB_A = 0.002       # int16 range: +/-65.5 A
FLAG_RESET, FLAG_OP_CHANGE = 1, 2
HEARTBEAT = "HB FAULTTWIN-EDGE"
ALIVE_S = 3.0              # the link counts as down after this long without a line from the board
PRESS_MAX_AGE_S = 10.0     # a D2 press nobody read within this long (a closed tab, a paused page) is dropped
# USB vendor:product IDs of the boards the app looks for (None: any product of that vendor).
KNOWN_USB_IDS: dict[tuple[int, int | None], str] = {
    (0x2341, 0x1002): "Arduino UNO R4 WiFi",
    (0x10C4, 0xEA60): "ESP32 DevKit (CP210x)",
    (0x1A86, 0x7523): "ESP32 DevKit (CH340)",
    (0x303A, None): "ESP32 (native USB)",
}
CONFIG_FORMAT = "<fBffffffBffBBf"   # see config_frame
DISPLAY_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -?.=")
SWITCH_NUMBER = {"AU": "S1", "AL": "S4", "BU": "S3", "BL": "S6", "CU": "S5", "CL": "S2"}
_TINY = 1e-9


# --- framing ----------------------------------------------------------------------------------------

def fletcher16(data: bytes) -> int:
    s1 = s2 = 0
    for b in data:
        s1 = (s1 + b) % 255
        s2 = (s2 + s1) % 255
    return (s2 << 8) | s1


def frame(kind: str, payload: bytes) -> bytes:
    body = struct.pack("<BH", ord(kind), len(payload)) + payload
    return b"\xa5\x5a" + body + struct.pack("<H", fletcher16(body))


def decimate(i_A: np.ndarray, factor: int = DECIMATION) -> np.ndarray:
    """Block averages of factor samples: the means (and so mu) stay exactly those of the full-rate cycle."""
    n = len(i_A) // factor
    return i_A[:n * factor].reshape(n, factor, 3).mean(axis=1)


def quantize(i_A: np.ndarray) -> np.ndarray:
    return np.clip(np.round(i_A / SAMPLE_LSB_A), -32768, 32767).astype("<i2")


def cycle_frame(cycle: int, m: float, flags: int, samples_i16: np.ndarray) -> bytes:
    head = struct.pack("<IHBB", cycle & 0xFFFFFFFF, max(0, min(65535, round(m * 1000))), flags, len(samples_i16))
    return frame("C", head + samples_i16.astype("<i2").tobytes())


def config_frame(th: MonitorThresholds, phase_offset: float = PHASE_OFFSET) -> bytes:
    lo, hi = th.amp_ratio_band
    return frame("K", struct.pack(CONFIG_FORMAT, th.mu_abs_max, th.mu_consecutive_cycles, th.r_kcl_dc_max,
                                  th.r_kcl_w1_max, lo, hi, th.low_rms_frac, th.i_s_step_frac, th.warmup_cycles,
                                  th.min_i_s_A, th.clear_ratio, th.clear_cycles, th.window_cycles, phase_offset))


def display_frame(mode: str, text: str = "") -> bytes:
    clean = "".join(c for c in text.upper() if c in DISPLAY_CHARS)[:60]
    return frame("D", mode.encode("ascii") + clean.encode("ascii"))


def display_text(label: str, actions: list[dict[str, Any]] | None = None) -> str:
    """What the matrix scrolls at the end of a diagnosis, e.g. 'S1 OPEN DERATED', 'SENSOR A - CONVERTER OK'."""
    if label not in {h.value for h in Hypothesis}:          # 'inconclusive' (or anything unexpected)
        text = "INCONCLUSIVE"
    elif label == "healthy":
        text = "NO FAULT - CONVERTER OK"
    else:
        spec = SPECS[Hypothesis(label)]
        leg = "ABC"[spec.leg] if spec.leg is not None else ""
        if spec.family == "switch_open":
            text = f"{SWITCH_NUMBER[next(iter(spec.switches))]} OPEN"
        elif spec.family == "leg_open":
            text = f"LEG {leg} OPEN"
        elif spec.family == "two_switch_open":
            text = " ".join(SWITCH_NUMBER[s] for s in sorted(spec.switches, key=label.index)) + " OPEN"
        elif spec.family in ("sensor_offset", "sensor_gain"):
            text = f"SENSOR {leg} - CONVERTER OK"
        elif spec.family == "load_unbalance":
            text = f"LOAD {leg} - CONVERTER OK"
        else:
            text = "LOAD STEP - CONVERTER OK"
    plant = ""                                  # the plant's state after the last approved action, in order
    for a in actions or []:
        request = a.get("request", "")
        if not a.get("approved"):
            continue
        if request.startswith("stop"):
            plant = " STOPPED"
        elif request.startswith("resume"):      # a confirmed fault resumes derated, otherwise at full m
            plant = " DERATED" if (a.get("applied") or {}).get("fault") else ""
        elif request.startswith("derate") and not request.endswith("(not needed)"):
            plant = " DERATED"
    return text + plant


# --- the firmware's algorithm, in Python ------------------------------------------------------------

@dataclass
class BoardFeatures:
    cycle: int
    ready: bool
    i_s_A: float
    mu: list[float]
    r_kcl_dc: float
    r_kcl_w1: float
    amp_ratio: list[float]
    rms_A: list[float]


@dataclass
class BoardEvent:
    cycle: int
    type: str
    severity: str
    shown: str | None = None       # the matrix's phase and arrow for this event, e.g. "A-" (None for a transient)


@dataclass
class _CycleSums:
    n: int
    sum: np.ndarray        # per phase
    sumsq: np.ndarray
    re: np.ndarray         # sum of x cos(theta)
    im: np.ndarray         # sum of x sin(theta)
    total_sum: float
    total_re: float
    total_im: float
    hyp: float             # sum of the Park's vector modulus

    @property
    def i_s_A(self) -> float:
        return self.hyp / self.n


def matrix_fault(check: str, f: BoardFeatures, window: list[_CycleSums]) -> str:
    """The phase the matrix names for a new alarm and its arrow, e.g. 'A-' (the firmware's pick_fault).

    Each alarm points to the phase by its own physics: a sensor offset and an open switch by the mean current, a
    gain error by the direction of the w1 KCL residual (in line with the scaled phase), an open leg by the odd
    amplitude.
    """
    if check == "kcl_w1":
        tre, tim = sum(w.total_re for w in window), sum(w.total_im for w in window)
        re, im = sum(w.re for w in window), sum(w.im for w in window)
        mag = np.hypot(re, im)
        c = np.where(mag > _TINY, (tre * re + tim * im) / np.maximum(mag, _TINY), 0.0)
        p = int(np.argmax(np.abs(c)))
        return "ABC"[p] + ("-" if c[p] < 0 else "+")
    p = max(range(3), key=lambda j: abs(f.mu[j]))
    down = f.mu[p] < 0
    if abs(f.mu[p]) < 0.05 and check != "kcl_dc":         # no mean current (an open leg): the odd amplitude
        p = max(range(3), key=lambda j: abs(f.amp_ratio[j] - 1.0))
        down = f.amp_ratio[p] < 1.0
    return "ABC"[p] + ("-" if down else "+")


def _cycle_sums(i_A: np.ndarray, phase_offset: float) -> _CycleSums:
    n = len(i_A)
    theta = 2.0 * math.pi * (np.arange(n) + phase_offset) / n
    c, s = np.cos(theta), np.sin(theta)
    total = i_A.sum(axis=1)
    i_alpha = (2.0 / 3.0) * (i_A[:, 0] - 0.5 * i_A[:, 1] - 0.5 * i_A[:, 2])
    i_beta = (i_A[:, 1] - i_A[:, 2]) / math.sqrt(3.0)
    return _CycleSums(n, i_A.sum(axis=0), (i_A ** 2).sum(axis=0), c @ i_A, s @ i_A, float(total.sum()),
                      float(c @ total), float(s @ total), float(np.hypot(i_alpha, i_beta).sum()))


class BoardModel:
    """The edge node's monitor: EdgeMonitor's rules on block-averaged samples, as the firmware computes them."""

    def __init__(self, th: MonitorThresholds, phase_offset: float = PHASE_OFFSET) -> None:
        self.th = th
        self.phase_offset = phase_offset
        self.reset()

    def reset(self) -> None:
        self._window: deque[_CycleSums] = deque(maxlen=self.th.window_cycles)
        self._seen = 0
        self._mu_streak = 0
        self._prev_i_s_A: float | None = None
        self._hold = 0
        self._active = dict.fromkeys(CHECKS, False)
        self._clear_streak = dict.fromkeys(CHECKS, 0)

    def _features(self, cycle: int) -> BoardFeatures:
        w = list(self._window)
        n = sum(x.n for x in w)
        i_s = sum(x.hyp for x in w) / n
        scale = i_s if i_s > _TINY else math.inf
        mean = sum(x.sum for x in w) / n
        amp = 2.0 * np.abs(sum(x.re for x in w) - 1j * sum(x.im for x in w)) / n
        a1 = float(amp.mean())
        total = 2.0 * abs(sum(x.total_re for x in w) - 1j * sum(x.total_im for x in w)) / n
        return BoardFeatures(cycle, True, i_s, (mean / scale).tolist(), abs(sum(x.total_sum for x in w) / n) / scale,
                             total / scale, (amp / a1).tolist() if a1 > _TINY else [1.0, 1.0, 1.0],
                             np.sqrt(sum(x.sumsq for x in w) / n).tolist())

    def process(self, cycle: int, m: float, flags: int, i_A: np.ndarray) -> tuple[BoardFeatures, list[BoardEvent]]:
        """One cycle of block-averaged currents i_A[n, 3] (amperes); mirrors EdgeMonitor.update_cycle."""
        th = self.th
        if flags & FLAG_RESET:
            self.reset()
        if flags & FLAG_OP_CHANGE:
            self._hold = max(self._hold, th.window_cycles + 1)
        sums = _cycle_sums(i_A, self.phase_offset)
        self._window.append(sums)
        self._seen += 1
        cycle_i_s = sums.i_s_A
        low_current = cycle_i_s < th.min_i_s_A
        stopped = m == 0.0
        ready = self._seen > th.warmup_cycles and len(self._window) == th.window_cycles
        f = self._features(cycle) if ready else BoardFeatures(cycle, False, 0.0, [0.0] * 3, 0.0, 0.0, [1.0] * 3,
                                                               [0.0] * 3)
        flagged = dict.fromkeys(CHECKS, False)
        clear = dict.fromkeys(CHECKS, True)
        holding = self._hold > 0
        if stopped or not ready:
            self._mu_streak = 0
        elif low_current:
            flagged["no_current"], clear["no_current"] = True, False
            self._mu_streak = 0
        else:
            if not holding:
                cycle_mu = np.abs(sums.sum / sums.n) / cycle_i_s
                self._mu_streak = self._mu_streak + 1 if cycle_mu.max() > th.mu_abs_max else 0
            mu_max = max(abs(x) for x in f.mu)
            flagged["mean_current"] = self._mu_streak >= th.mu_consecutive_cycles
            clear["mean_current"] = mu_max < th.clear_ratio * th.mu_abs_max
            flagged["kcl_dc"] = f.r_kcl_dc > th.r_kcl_dc_max
            clear["kcl_dc"] = f.r_kcl_dc < th.clear_ratio * th.r_kcl_dc_max
            flagged["kcl_w1"] = f.r_kcl_w1 > th.r_kcl_w1_max
            clear["kcl_w1"] = f.r_kcl_w1 < th.clear_ratio * th.r_kcl_w1_max
            lo, hi = th.amp_ratio_band
            flagged["amplitude_ratio"] = any(not lo <= r <= hi for r in f.amp_ratio)
            lo_c, hi_c = 1.0 - th.clear_ratio * (1.0 - lo), 1.0 + th.clear_ratio * (hi - 1.0)
            clear["amplitude_ratio"] = all(lo_c <= r <= hi_c for r in f.amp_ratio)
            flagged["phase_loss"] = _phase_loss(f.rms_A, th.low_rms_frac, th.min_i_s_A)
            clear["phase_loss"] = not _phase_loss(f.rms_A, th.low_rms_frac / th.clear_ratio, th.min_i_s_A)
            if self._prev_i_s_A and not holding:
                change = abs(cycle_i_s - self._prev_i_s_A) / self._prev_i_s_A
                flagged["transient"] = change > th.i_s_step_frac
                clear["transient"] = change < th.clear_ratio * th.i_s_step_frac
        self._prev_i_s_A = None if stopped or low_current else cycle_i_s
        events: list[BoardEvent] = []
        if ready and not holding and not stopped:
            for check in CHECKS:
                if flagged[check] and not self._active[check]:
                    shown = None if check == "transient" else matrix_fault(check, f, list(self._window))
                    events.append(BoardEvent(cycle, check, SEVERITY[check], shown))
                    self._active[check] = True
                    self._clear_streak[check] = 0
                elif self._active[check] and not flagged[check] and clear[check]:
                    self._clear_streak[check] += 1
                    if self._clear_streak[check] >= th.clear_cycles:
                        self._active[check] = False
                elif self._active[check]:
                    self._clear_streak[check] = 0
        self._hold = max(0, self._hold - 1)
        return f, events


def _phase_loss(rms_A: list[float], frac: float, min_ref_A: float) -> bool:
    for k in range(3):
        ref_A = sum(rms_A[j] for j in range(3) if j != k) / 2.0
        if ref_A > min_ref_A and rms_A[k] < frac * ref_A:
            return True
    return False


def as_monitor_event(session: Any, e: BoardEvent) -> Event:
    """The board's alarm as the monitor event the agent starts from: the board's check and cycle, with the
    full-rate features the laptop computed for that cycle (the board reports only its own summary)."""
    same = next((x for x in reversed(session.events) if x.cycle_index == e.cycle and x.type == e.type), None)
    if same is not None:
        message, features = same.message, same.features
    else:
        status = next((s for s in reversed(session.statuses) if s.cycle_index == e.cycle), None)
        message, features = f"{e.type.replace('_', ' ')} check outside its limit", status.features if status else None
    return Event(type=e.type, severity=e.severity, cycle_index=e.cycle, message=f"{message} (raised by the edge node)",
                 features=features)


def parse_line(line: str) -> tuple[str, Any] | None:
    """One line from the board: ('F', BoardFeatures), ('E', BoardEvent), ('M', None), ('HB', text), ('ERR', text)."""
    parts = line.split()
    if not parts:
        return None
    try:
        if parts[0] == "F" and len(parts) == 15:
            v = [float(x) for x in parts[3:]]
            return "F", BoardFeatures(int(parts[1]), parts[2] == "1", v[0], v[1:4], v[4], v[5], v[6:9], v[9:12])
        if parts[0] == "E" and len(parts) == 4 and parts[2] in CHECKS:
            return "E", BoardEvent(int(parts[1]), parts[2], parts[3])
        if parts[0] == "S" and len(parts) == 2 and len(parts[1]) == 2 and parts[1][0] in "ABC" and parts[1][1] in "+-":
            return "S", parts[1]
        if parts[0] == "M":
            return "M", None
        if line.startswith(HEARTBEAT):
            return "HB", line
        if parts[0] == "ERR":
            return "ERR", line[4:]
    except ValueError:
        return None
    return None


# --- the serial link ------------------------------------------------------------------------------

@dataclass
class _Received:
    features: deque[BoardFeatures] = field(default_factory=lambda: deque(maxlen=1024))
    events: list[BoardEvent] = field(default_factory=list)
    presses: list[float] = field(default_factory=list)       # arrival times of D2 presses not taken yet
    shown: deque[str] = field(default_factory=lambda: deque(maxlen=256))   # the matrix's phase and arrow, in order
    errors: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    heartbeat: str = ""


class EdgeLink:
    """The board on a serial port: streams the plant's cycles, collects its features, events and button presses."""

    def __init__(self, ser: Any, *, port: str, board: str) -> None:
        self.ser = ser
        self.port = port
        self.board = board
        self.rx = _Received()
        self._lock = threading.Lock()
        self._out: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self._last_rx = 0.0
        self.dead = False
        self._session: Any = None
        self._seen: tuple[int, int] | None = None     # the monitor's (resets, operating_point_notices) last sent
        self._stop = threading.Event()
        for target in (self._read_loop, self._write_loop):
            threading.Thread(target=target, name=f"faulttwin-edge-{target.__name__}", daemon=True).start()

    # -- lifecycle --
    @classmethod
    def open(cls, port: str, board: str = "edge node", *, opener: Callable[..., Any] | None = None,
             wait_s: float = 3.0) -> EdgeLink | None:
        """Open the port and wait for the firmware's heartbeat; None if it is not a FaultTwin edge node."""
        opener = opener or (serial.Serial if serial is not None else None)
        if opener is None:
            return None
        try:
            ser = opener(port, BAUD, timeout=0.2, write_timeout=2)
        except Exception:  # noqa: BLE001 - a busy or vanished port: no board
            return None
        link = cls(ser, port=port, board=board)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if link.rx.heartbeat:
                return link
            time.sleep(0.05)
        link.close()
        return None

    def close(self) -> None:
        self._stop.set()
        self.detach()
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass

    def alive(self) -> bool:
        return not self.dead and time.time() - self._last_rx < ALIVE_S

    @property
    def session(self) -> Any:
        """The plant session whose cycles are streamed to the board, if any."""
        return self._session

    # -- plant side --
    def attach(self, session: Any) -> None:
        """Stream this session's cycles to the board (and stop streaming another session's)."""
        if self._session is session:
            return
        self.detach()
        self._session = session
        self._seen = None
        with self._lock:                      # a new session counts its cycles from 0 again
            self.rx.features.clear()
            self.rx.shown.clear()
            self.rx.presses.clear()           # a press made for another session is not this one's
        self.send(config_frame(session.monitor.th))
        session.cycle_listeners.append(self._on_cycle)

    def detach(self) -> None:
        if self._session is not None and self._on_cycle in self._session.cycle_listeners:
            self._session.cycle_listeners.remove(self._on_cycle)
        self._session = None

    def _on_cycle(self, chunk: Any, status: Any) -> None:
        monitor = self._session.monitor if self._session is not None else None
        if monitor is None:
            return
        now = (monitor.resets, monitor.operating_point_notices)
        flags = FLAG_RESET if self._seen is None or now[0] != self._seen[0] else 0
        if self._seen is not None and now[1] != self._seen[1]:
            flags |= FLAG_OP_CHANGE
        self._seen = now
        self.send(cycle_frame(chunk.first_cycle, chunk.m, flags, quantize(decimate(chunk.i_meas_A))))

    # -- display --
    def show(self, mode: str, text: str = "") -> None:
        self.send(display_frame(mode, text))

    # -- io --
    def send(self, data: bytes) -> None:
        try:
            self._out.put_nowait(data)
        except queue.Full:                    # the board stopped reading: drop rather than block the plant
            pass

    def _write_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._out.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.ser.write(data)
            except Exception:  # noqa: BLE001 - unplugged
                self.dead = True
                return
            time.sleep(len(data) / PACE_BYTES_S)

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self.ser.readline()
            except Exception:  # noqa: BLE001 - unplugged
                self.dead = True
                return
            if not raw:
                continue
            parsed = parse_line(raw.decode("ascii", "replace").strip())
            if parsed is None:
                continue
            self._last_rx = time.time()
            kind, value = parsed
            with self._lock:
                if kind == "F":
                    self.rx.features.append(value)
                elif kind == "E":
                    self.rx.events.append(value)
                elif kind == "S":
                    self.rx.shown.append(value)
                elif kind == "M":
                    self.rx.presses.append(self._last_rx)
                elif kind == "HB":
                    self.rx.heartbeat = value
                else:
                    self.rx.errors.append(value)

    # -- what the app reads --
    def events_since(self, index: int) -> tuple[list[BoardEvent], int]:
        with self._lock:
            return list(self.rx.events[index:]), len(self.rx.events)

    def latest_features(self) -> BoardFeatures | None:
        with self._lock:
            return self.rx.features[-1] if self.rx.features else None

    def features_for(self, cycle: int) -> BoardFeatures | None:
        with self._lock:
            return next((f for f in reversed(self.rx.features) if f.cycle == cycle), None)

    def take_presses(self, max_age_s: float = PRESS_MAX_AGE_S) -> int:
        """D2 presses since the last call that are at most max_age_s old (older ones are dropped)."""
        now = time.time()
        with self._lock:
            fresh = sum(1 for t in self.rx.presses if now - t <= max_age_s)
            self.rx.presses.clear()
            return fresh


def find_board(opener: Callable[..., Any] | None = None, ports: list[Any] | None = None,
               wait_s: float = 3.0) -> EdgeLink | None:
    """The first FaultTwin edge node on a known USB serial port (FAULTTWIN_EDGE_PORT forces one; 'off' disables)."""
    forced = os.environ.get("FAULTTWIN_EDGE_PORT", "").strip()
    if forced.lower() == "off" or (serial is None and opener is None):
        return None
    if forced:
        return EdgeLink.open(forced, "edge node", opener=opener, wait_s=wait_s)
    for p in ports if ports is not None else (list_ports.comports() if list_ports else []):
        name = KNOWN_USB_IDS.get((p.vid, p.pid)) or KNOWN_USB_IDS.get((p.vid, None))
        if name:
            link = EdgeLink.open(p.device, name, opener=opener, wait_s=wait_s)
            if link is not None:
                return link
    return None
