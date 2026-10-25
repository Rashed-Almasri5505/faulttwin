"""Serial throughput check for the edge node (firmware/serial_bench).

Streams frames of n samples x 3 phases (int16) with a Fletcher-16 checksum as fast as the link takes them,
for a fixed time, and reads the board's per-second STATS lines. Passes if the board's received rate stays at
or above the target in every full second, with no checksum errors and no sequence gaps.

Usage:  python scripts/edge_bench.py --port COM5 --baud 460800 [--samples 100] [--seconds 10] [--target 12000]
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import threading
import time

import serial


def fletcher16(data: bytes) -> int:
    s1 = s2 = 0
    for b in data:
        s1 = (s1 + b) % 255
        s2 = (s2 + s1) % 255
    return (s2 << 8) | s1


def frame(seq: int, samples: list[tuple[int, int, int]]) -> bytes:
    body = struct.pack("<HB", seq & 0xFFFF, len(samples)) + b"".join(struct.pack("<hhh", *s) for s in samples)
    return b"\xa5\x5a" + body + struct.pack("<H", fletcher16(body))


def cycle(n: int, amp: int = 27000) -> list[tuple[int, int, int]]:
    """One fundamental cycle of three-phase currents, scaled to int16."""
    return [tuple(int(amp * math.sin(2 * math.pi * k / n - p)) for p in (0, 2 * math.pi / 3, 4 * math.pi / 3))
            for k in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, required=True)
    ap.add_argument("--samples", type=int, default=100, help="samples per phase per frame (one cycle)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--target", type=float, default=12000.0, help="required bytes per second at the board")
    ap.add_argument("--rate", type=float, default=None,
                    help="send rate in bytes per second (default 1.25 x target; 0 = as fast as the link takes)")
    args = ap.parse_args()
    rate = args.target * 1.25 if args.rate is None else args.rate

    stats: list[tuple[int, ...]] = []
    stop = threading.Event()
    with serial.Serial(args.port, args.baud, timeout=0.2, write_timeout=5) as port:
        time.sleep(2.0)                                       # the board may reset when the port opens
        port.reset_input_buffer()

        def reader() -> None:
            while not stop.is_set():
                line = port.readline().decode("ascii", "replace").strip()
                if line.startswith("STATS "):
                    stats.append(tuple(int(x) for x in line.split()[1:]))

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        port.write(frame(0, []))                              # control frame: reset the board's counters
        time.sleep(1.2)                                       # one STATS line after the reset
        first = len(stats)
        data = cycle(args.samples)
        seq, sent, t0 = 1, 0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            f = frame(seq, data)
            port.write(f)
            sent += len(f)
            seq += 1
            if rate > 0:                                      # pace to the send rate
                ahead = sent / rate - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        elapsed = time.perf_counter() - t0
        time.sleep(1.5)                                       # let the last STATS lines arrive
        stop.set()
        t.join(timeout=1)

    run = stats[first:]
    if len(run) < 3:
        print(f"FAIL: only {len(run)} STATS lines from the board (is serial_bench flashed at {args.baud} baud?)")
        return 1
    rates = [(b[3] - a[3]) * 1000.0 / max(1, b[0] - a[0]) for a, b in zip(run, run[1:])]
    steady = rates[1:-1] or rates                             # drop the partial first and last seconds
    last = run[-1]
    good, bad, good_bytes, gaps, junk = last[1:6]
    frame_len = 2 + 3 + 6 * args.samples + 2
    ok = min(steady) >= args.target and bad == 0 and gaps == 0
    print(f"baud {args.baud}, {args.samples} samples/cycle ({frame_len} B per frame), {elapsed:.1f} s, "
          f"send rate {'as fast as possible' if rate <= 0 else f'{rate:.0f} B/s'}")
    print(f"  laptop sent {sent} B = {sent / elapsed:.0f} B/s; board received {good} good frames = {good_bytes} B")
    print(f"  board rate per second: min {min(steady):.0f}, median {sorted(steady)[len(steady) // 2]:.0f} B/s"
          f" (target {args.target:.0f})")
    print(f"  checksum errors {bad}, sequence gaps {gaps}, junk bytes {junk}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
