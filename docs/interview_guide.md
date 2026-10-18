# Interview Guide — Turbofan RUL Prediction

How to talk about this project at every depth, and how to survive hostile
questions. All numbers below are executed results from this repo.

## 30-second explanation

"I predicted remaining useful life of turbofan engines from sensor data —
NASA's C-MAPSS FD001 set, 100 run-to-failure engines. Thirty-cycle sliding
windows, simple statistical features, gradient boosting. Test RMSE about
13 cycles, evaluated with a leak-free split by engine, and the model's top
signals are the temperatures and speeds that thermodynamics says degradation
should move."

## 90-second explanation

Add: "The framing is piecewise-linear RUL capped at 125 cycles, because early
life shows no degradation signal. I compared Ridge, Random Forest and
HistGradientBoosting with proper validation — repeated grouped CV over
engines, since overlapping windows from one engine would leak. HGB won on
both RMSE and the PHM asymmetric score, which penalizes late predictions
harder because over-predicting remaining life is the dangerous error. Then I
stress-tested it: permutation importance, leave-one-sensor-out ablation,
noise injection, and a simulated sensor failure. Errors concentrate at high
RUL — the healthy region with no signal — which is the right failure mode."

## 5-minute technical walkthrough

1. Data: FD001, 100 train engines (20,631 cycles), 100 truncated test engines,
   21 sensors + 3 operating settings. Simulated, single condition, single
   fault mode (HPC degradation).
2. EDA: 7 sensors constant → dropped; 14 carry trend signal; degradation
   visible in last ~100 cycles.
3. Features: 30-cycle windows × 6 stats (mean/std/min/max/last/slope) × 14
   sensors = 84 features. Label = min(125, cycles remaining).
4. Split: 80/20 by engine (seed 42). Test set touched once, at the end.
5. Models: Ridge 17.43 → RF 13.48 → HGB 12.74 test RMSE; PHM 484 → 307 → 247.
6. Validation upgrade: repeated grouped CV (3 folds × 3 seeds) confirmed the
   ranking; see `reports/metrics_validation.json`.
7. Explainability: permutation importance + sensor ablation (no single sensor
   is load-bearing).
8. Robustness: seed-stable, graceful under 2× noise, survives worst-sensor
   failure with mean imputation.
9. Product: Streamlit demo mapping RUL → risk bands → illustrative actions;
   business case with asymmetric failure costs.

## 10-minute architecture/business walkthrough

Add the decision flow (sensor stream → windows → features → model → RUL +
trend → rules + human review → work order → feedback), the business case
(false-negative cost dominates: unplanned removal vs early shop visit),
deployment design (batch daily scoring, drift monitoring, shadow mode first),
limitations (simulated, single condition, no calibration), and roadmap
(FD002–FD004, prediction intervals, prospective trial).

---

## Q&A

**Q: What did YOU personally do?**
A: Everything in the repo: the EDA, the feature pipeline design, the
validation methodology (including catching the window-leakage issue and
designing the engine-grouped splits), model comparison, the ablation and
robustness experiments, the Streamlit demo, and all documentation. The dataset
is NASA's; every result was computed by code I wrote and executed.

**Q: Your accuracy looks strong. Why should I believe it?**
A: Don't take my word — check the design: split by engine so no overlapping
window leaks; model selection on validation engines; test engines scored
exactly once; repeated grouped CV confirming the ranking isn't split luck.
And the model rediscovers the physics — temperatures and speeds, not
artifacts. Skepticism is welcome; the repo is built to survive it.

**Q: How do you know there is no leakage?**
A: Three layers: (1) windows from one engine can never be on both sides of a
split — the grouping key is enforced in code and asserted; (2) the test set is
separate trajectories, scored once after all decisions; (3) CV folds are built
from permuted engine lists. Residual risk: engines share one operating
condition, so random-split optimism *across conditions* isn't tested — that's
why FD002–FD004 is the stated next step.

**Q: Why didn't you use deep learning?**
A: 17k windows × 84 tabular features is gradient boosting's home turf. An LSTM
would model within-window order more richly, but that only earns its place by
beating HGB out-of-sample on the same grouped splits. It's listed as future
work, not skipped from ignorance.

**Q: Why RMSE and PHM score?**
A: RMSE is comparable and interpretable (cycles). PHM encodes the operational
truth: over-predicting RUL (late) risks failure; under-predicting wastes a
shop visit. The exponential asymmetry makes the metric match the business cost.

**Q: What would happen on another machine / operating condition?**
A: Honestly: unknown — FD001 is one condition. The pipeline is condition-aware
in structure (operating settings are columns), but multi-condition validation
needs FD002/FD004. I don't claim transfer.

**Q: What if a sensor fails in the field?**
A: Tested: mean-impute its features at inference, no retraining — degradation
is graceful, and the ablation shows no single sensor is load-bearing. A real
deployment would add a data-quality gate that flags the dropout to the
engineer rather than silently imputing forever.

**Q: Why the 125 cap? Isn't that cheating?**
A: It's the opposite of cheating — it's refusing to ask the model to predict
the unpredictable. Early-life sensors are flat; without the cap you'd train
on noise and *look* worse for being honest. The cap is documented as a
modeling choice and every metric is conditional on it.

**Q: How would this generate actual business value?**
A: The model outputs a per-engine risk estimate that replaces fleet-average
scheduling: inspect the right engines first, defer healthy ones, provision
spares. Value = avoided unplanned removals (3–10× planned cost) minus false
alarms and platform cost. But: value is *framework*, not claim — proving it
needs shadow mode and a prospective trial, which is exactly the rollout plan
in the business case.

**Q: Correlated features — doesn't that break your importance analysis?**
A: It breaks *impurity* importance, which is why I don't rely on it. I used
permutation importance (measures actual reliance) *and* leave-one-sensor-out
ablation with retraining (the causal test). They agree: temperatures and
speeds matter, and they're mutually redundant enough that no single sensor
is critical.

**Q: What would you redesign with another month?**
A: (1) Prediction intervals via quantile regression — planners need
uncertainty, not point estimates. (2) FD002/FD004 multi-condition training.
(3) Full-trajectory scoring for a lead-time-vs-accuracy curve. (4) A proper
calibration analysis. (5) Shadow-mode evaluation protocol on live data.

**Q: What are the five limitations you'd volunteer?**
A: Simulated data; single condition and fault mode; 100-point test set is a
noisy ruler; no uncertainty quantification; never validated prospectively.
(All in `docs/limitations.md`, each with its next experiment.)
