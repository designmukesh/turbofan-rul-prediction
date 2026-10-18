# Social Kit — Turbofan RUL Project

LinkedIn-safe: these posts document *learning and building* with public data.
No job-seeking signals, no "open to work" framing, nothing a manager would
read as a departure announcement.

---

## Post 1 — Launch

I just finished a project I've been wanting to build properly: predicting
remaining useful life of turbofan engines from sensor data, using NASA's
public C-MAPSS dataset.

The part I'm most pleased with isn't the RMSE (~13 cycles on unseen engines).
It's the validation discipline: I split by engine instead of by data row
(overlapping time windows would leak), confirmed the model ranking with
repeated cross-validation, and stress-tested it with sensor noise and a
simulated sensor failure.

I also wrote up where the model fails — errors concentrate where sensors
show no signal yet, which is the honest failure mode.

Repo + full write-up: [github link]

## Post 2 — Technical (validation)

A lesson this project reinforced: your validation split design matters more
than your model choice.

Engine sensor data comes as overlapping 30-cycle windows. Split rows randomly
and near-duplicate windows land on both sides — the model "predicts" by
recognition and your score is fiction.

The fix is unglamorous: split by engine (GroupKFold), repeat over seeds,
report mean ± std. My model ranking (boosting > forest > linear) survived
that; the headline number got a wider, more honest uncertainty bar.

If your time-series validation doesn't respect the grouping structure of the
data, the leaderboard is lying to you.

## Post 3 — What I learned

Three things this project taught me:

1. Feature importance is not an instrumentation plan. My impurity rankings
said temperatures matter — but only the leave-one-sensor-out ablation
(retrain without it, measure the damage) told me what each sensor is
actually worth. No single sensor turned out to be load-bearing.

2. The metric should encode the cost. RMSE treats all errors equally; the
PHM score punishes over-predicting remaining life exponentially harder —
because "50 cycles left" when the truth is 10 is the dangerous mistake.

3. Simulated data is a sandbox, not a proof. C-MAPSS is clean, single
condition, single fault mode. Everything I claim is conditional on that,
and the limitations doc is the longest file in the repo after the code.

## Post 4 — Business/product insight

A prediction is not a product. "RUL = 42 cycles" helps nobody until it
becomes: who sees it, what threshold triggers what action, who signs off,
and what a false alarm costs.

For this project I built the decision layer as a demo: RUL → risk bands →
illustrative actions → human review → work order. The thresholds are
placeholders, clearly labeled. In aviation the model advises and certified
processes decide — and I think that's the right shape for most industrial AI:
decision support with a human in the loop, shadow mode first, prospective
measurement before any ROI claim.

## Article outline — "I predicted engine failure. Here's what actually mattered."

1. The problem: RUL and why the industry cares (hook: $ cost of unplanned removal)
2. The dataset: C-MAPSS, what it is and isn't
3. EDA: the sensors that never move, the drift you can see by eye
4. The leakage trap: why windowing breaks random splits
5. Baselines before boosters: Ridge → RF → HGB
6. Validation upgrade: repeated grouped CV
7. Explainability done honestly: permutation + ablation
8. The product layer: from prediction to decision
9. Limitations (the longest section)
10. What I'd do next
