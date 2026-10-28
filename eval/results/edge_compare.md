# Edge node vs Python monitor

Measured 2026-10-28 12:01 UTC on Arduino UNO R4 WiFi (COM5), firmware `3`, by `python scripts/edge_compare.py`.

The board receives each simulated cycle as 100 block-averaged int16 samples per phase and computes
the monitor's features in float32; the Python monitor uses all 1000 samples per cycle in float64.
Pass: |Δμ| < 0.01 on every compared cycle, the same alarms on the same cycles, and the LED matrix
showing the phase and arrow that the firmware's rule, run on the laptop (edge.matrix_fault), gives.

| Scenario | Cycles compared | max \|Δμ\| | max \|Δr_KCL,dc\| | max \|Δr_KCL,ω₁\| | max \|Δ amplitude ratio\| | Alarms (board = Python) | Matrix (board = laptop) | Result |
|---|---|---|---|---|---|---|---|---|
| healthy | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: none | same: check mark only | pass |
| AU_open | 39 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, mean_current | same: A- A- A- A- | pass |
| BL_open | 39 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, mean_current | same: B+ B+ B+ B+ | pass |
| leg_C_open | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | same: amplitude_ratio, phase_loss, transient | same: C- C- C- C- | pass |
| AU_BU_open | 39 | 0.0031 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, mean_current, transient | same: C+ C+ C+ C+ | pass |
| sensor_offset_A | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: kcl_dc, mean_current | same: A+ A+ A+ | pass |
| sensor_offset_C | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: kcl_dc, mean_current | same: C- C- C- | pass |
| sensor_gain_A | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: kcl_w1 | same: A+ A+ | pass |
| sensor_gain_B | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: amplitude_ratio, kcl_w1 | same: B- B- B- B- | pass |
| load_unbalance_A | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: none | same: check mark only | pass |
| load_step | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: transient | same: check mark only | pass |
| low_load | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | same: none | same: check mark only | pass |

12 of 12 scenarios pass. Largest |Δμ| against the firmware's arithmetic run on the laptop (edge.BoardModel): 5.0e-06.
