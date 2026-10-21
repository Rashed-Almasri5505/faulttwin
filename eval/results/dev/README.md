# Development runs (not the evaluation)

Seeds 6700 + n, used to try the harness and to check fixes; never the evaluation seeds 8101-8124.

- First dev run (2026-10-21, before the full run): AL_open_low_load, leg_A_open, AU_BU_open,
  sensor_offset_B_minus_3A, sensor_gain_C_0.82, load_unbalance_B_0.75, load_step_0.75, healthy_low_load_1.
- Second dev run (2026-10-22, after the full run, with the full-ranking check and the one-retry rule for
  wording): AL_CL_open, leg_B_open, leg_C_open. These three rows in `results.json` and `results.md` come from
  the fixed agent; the others from the agent of the full run.
