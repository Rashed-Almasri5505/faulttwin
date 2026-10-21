## What this run showed, and what changed afterwards

The tables above are the first and only full run, unchanged. Two of the full agent's answers were wrong for reasons
the evaluation exposed in the design. Both were fixed after the run and checked on development seeds (6700 + n) only;
the evaluation seeds were not run again. Any later full run must use fresh seeds (for example 8201-8224), because
these failures guided the fixes, and it will be reported next to this one. To reproduce these numbers, apply
`eval/results/pre_fix_agent.patch`, which removes the post-run changes from the agent (the report code changed
afterwards only in wording and in the device-drop experiment's window, which makes no model calls).

1. **AL_CL_open was called BU_open.** The agent tested only the 20 default causes; the two-switch causes are tested
   only when asked for. Its best tested cause, BU_open, had a match error of 107 %, so it explained nothing, and no
   check looked at the size of the error. The safety interlock's own ranking of all 32 causes, computed for its derate
   decision, had AL_CL_open first and derated for it. Fix: the twin tool now reports an untested cause that
   matches far better, and the final-answer check sends back an answer whose best match is above 25 % or more than
   twice the error of the interlock's best. On the development seed the agent then tested all 32 causes and found
   AL_CL_open (3.1 %).
2. **leg_B_open ended inconclusive.** The twin had leg_B_open first (6.2 %). The agent also ran the modulation test,
   which cannot decide for an open leg (there is no mean current to compare). Its first answer drew a conclusion from
   that test and was sent back. Its second answer said the test could not separate the causes, but the check matched an
   outcome word elsewhere in the answer and sent it back again (a false positive). The third model turn started 0.5 s
   before the 90 s budget ran out, got the client's 5 s minimum timeout and timed out: our own time budget, not a
   Token Factory outage. Fixes: wording problems are sent back once, then code removes the sentences concerned; the
   test check now works sentence by sentence; and a model turn is started only with at least 8 s of budget left. On
   the development seeds leg_B_open and leg_C_open were then diagnosed correctly.

Other results, stated plainly:

- **No alarm.** The edge monitor raised no medium or high alarm for the two load unbalances (±25 % on one phase), the
  two load steps and the two healthy low-load cases (load_step_0.75 logged one low-severity transient). The pipeline
  therefore reported "healthy" for all six: correct for the healthy converter and no false alarm on the look-alikes,
  but no exact label for them. Larger single-phase unbalances (beyond about ±35-50 %) do raise an alarm.
- **The textbook baseline's two false alarms** were the 3 A sensor offsets, called BU_open and CL_open because a 3 A
  offset puts |μ| above its 0.10 threshold. FaultTwin's KCL residual and modulation test (μ ratio 1.59 and 1.60, where
  an offset predicts 1.60 and a switch fault 1.0) identified both as sensor offsets. The 1.5 A offset and the two gain
  errors the baseline called healthy (misses, not false alarms). It also cannot see an open leg: μ stays near 0, and it
  called all three healthy.
- **Super in one call** named all 5 sensor errors and raised no false alarm. It missed 4 of the 13 faults (AL_open and
  BL_open called sensor gain errors, leg A a load unbalance, AL_CL_open inconclusive), called open leg C a two-switch
  fault, and called both healthy low-load cases load unbalances.
- **Triage.** Lightning named the right family in 17 of the 18 alarmed runs; it called the -3 A offset on phase B a
  switch or leg fault although the code-evaluated rule for a KCL residual held. The agent was not misled, because the
  twin and the modulation test decide.
- **What Tavily changed.** With Tavily the interlock's current limit came from the verified datasheet rating
  (I_C,nom = 50 A, from a Tavily extract cached on 2026-10-21) in all 8 single-switch runs; without it the interlock
  used its conservative default. That changed the derate in 6 of 8 runs (m = 0.66 instead of 0.40 at nominal load); at
  low load no derate was needed either way. Every diagnosis with Tavily cited sources; the labels were the same
  without it. Tavily changes the safety limits and the evidence, not the label.
- **The datasheet's device drops did not improve the twin.** On a plant that drops the datasheet's V_CE(sat) and V_F,
  the true cause's match error went down in 4 of 8 scenarios and up in 4, with a window 10 cycles after the alarm (a
  first version right after the alarm gave the same split). On an ideal plant the drops are matched exactly, so they
  are modelled correctly; on the realistic plant they (about 2 V on a 400 V DC link) are small against the ±5 % R/L
  tolerance and the noise.
- **Low load.** At m = 0.35 a modulation step cannot go down (steps of at least 0.15, and m may not fall below 0.3),
  and the interlock refused the agent's request to raise m to 0.50 because a converter fault was plausible. Both
  low-load faults were diagnosed from the twin alone, both correctly. (This waiver and the reading of datasheet Typ
  values were added before the full run, after the first development run.)
