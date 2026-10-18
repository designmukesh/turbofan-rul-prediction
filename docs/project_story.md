# Project Story — Turbofan RUL Prediction

## The problem I wanted to solve

I work around industrial operations, and the most expensive question in any
asset-heavy business is always some version of: *how much life is left in
this machine, and when do we intervene?* Too early wastes money; too late
risks failure. I wanted to build the real thing — not a tutorial — for the
canonical version of that problem: predicting remaining useful life of a
turbofan engine from sensor data.

## What surprised me in the data

Two things. First, how *clean* the degradation signal is once you look at
the right sensors — plot HPC outlet temperature against cycle for any engine
and the upward drift in the last ~100 cycles is unmistakable. Second, how
many sensors are dead weight: 7 of 21 never move at all in FD001. The EDA
basically told me which instruments matter before any model was trained.

## The first baseline

Ridge regression on the windowed features: RMSE ~17.4 on the test engines.
Respectable, and it set the bar honestly — anything fancier had to beat a
two-minute linear model by a margin worth its complexity.

## Problems I found

- **The 100% accuracy trap (from project #1, applied here):** near-perfect
  scores usually mean leakage. Here I split by *engine*, not by window,
  because overlapping windows from the same engine would leak trivially.
- **Single-split model selection:** picking the winner on one validation
  split is how you fool yourself. I ran repeated grouped CV (3 folds × 3
  seeds) to check the ranking was real.
- **Impurity importance lies by omission:** it told me temperatures matter
  but couldn't say what happens without them. The leave-one-sensor-out
  ablation — retraining 14 times — gave the causal answer.

## What I changed

The big methodological change was treating the test set as *untouchable*:
model selection on validation engines, CV on train engines, one final
scoring. I also moved from "the model says RUL = 42" to a decision framing —
risk bands, recommended actions, and a Streamlit demo a planner could react to.

## How validation affected my conclusions

The CV confirmed HGB > RF > Ridge with non-overlapping distributions — the
ranking is real, not split luck. But it also humbled the headline: the
spread across folds (±~1 cycle) is the honest uncertainty bar, wider than a
single test RMSE suggests. And training the final artifact on all 100 train
engines scored *slightly worse* (13.44) than the 80-engine model (12.74) —
a reminder that a 100-point test set is a noisy ruler.

## What the model learned

It rediscovered the physics: exhaust/HPC temperatures and rotor speeds
dominate, exactly the signals a thermodynamicist would expect from HPC
degradation. The ablation showed no single sensor is load-bearing — the
information is distributed, which is good news for robustness and bad news
for anyone wanting a 3-sensor story.

## Where it failed

Errors concentrate at high RUL — the capped "healthy" region where sensors
show no degradation trend. The model is essentially guessing "healthy," which
is the right failure mode: uncertain early, precise near failure. The PHM
decomposition confirmed late (dangerous) errors are the minority.

## What I would deploy

Batch scoring daily per engine, RUL + trend in the reliability engineer's
existing dashboard, shadow mode first, thresholds as *advisory* with human
sign-off. Never a fully automated pull decision — aviation doesn't work that
way and neither should the product story.

## What I would not claim

Savings. Field accuracy. Multi-condition generalization. Certification
readiness. This is simulated, single-condition data, and I say so everywhere.

## What I learned

That the unglamorous parts — the split design, the ablation, the decision
framing — are the actual project. The model took an afternoon; making the
result *believable* took the rest.

## What I would build next

FD002/FD004 (multiple operating conditions — the real generalization test),
prediction intervals instead of point estimates, and a lead-time-vs-accuracy
curve scored over full test trajectories.
