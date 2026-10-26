# Edge node vs Python monitor

Measured 2026-10-25 16:40 UTC on Arduino UNO R4 WiFi (COM5), firmware `1`, by `python scripts/edge_compare.py`.

The board receives each simulated cycle as 100 block-averaged int16 samples per phase and computes
the monitor's features in float32; the Python monitor uses all 1000 samples per cycle in float64.
Pass: |Δμ| < 0.01 on every compared cycle and the same alarms on the same cycles.

| Scenario | Cycles compared | max \|Δμ\| | max \|Δr_KCL,dc\| | max \|Δr_KCL,ω₁\| | max \|Δ amplitude ratio\| | Alarms (board = Python) | Result |
|---|---|---|---|---|---|---|---|
| healthy | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: none | pass |
| AU_open | 39 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, mean_current | pass |
| BL_open | 39 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, mean_current | pass |
| leg_C_open | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | same: amplitude_ratio, phase_loss, transient | pass |
| AU_BU_open | 39 | 0.0031 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, mean_current, transient | pass |
| sensor_offset_A | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: kcl_dc, mean_current | pass |
| sensor_gain_B | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, kcl_w1 | pass |
| load_unbalance_A | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: none | pass |
| load_step | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: transient | pass |
| low_load | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: none | pass |

10 of 10 scenarios pass. Largest |Δμ| against the firmware's arithmetic run on the laptop (edge.BoardModel): 5.0e-06.
