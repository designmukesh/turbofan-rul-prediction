# Limitations — and what each one demands next

Stated before any recruiter asks. Each limitation ships with the experiment
that would address it.

## 1. Simulated data, not field data
C-MAPSS trajectories come from a thermodynamic model. Sensor noise is
well-behaved, fault progression is monotonic, and there are no maintenance
events, aborted flights, or mixed faults.
**Next:** retrain/evaluate on real fleet data; at minimum, inject realistic
noise and dropout (partially done in notebook 03 robustness tests).

## 2. FD001 only — one operating condition, one fault mode
All results are for a single cruise-like condition and HPC degradation.
FD002/FD004 add six operating conditions and fan+HPC faults.
**Next:** extend the pipeline to FD002–FD004; expect operating-condition
normalization to become the main modeling problem.

## 3. Single train/test split for the headline numbers
Model selection used a validation split, and the test set was touched once —
but it is still one split. Random-split optimism is reduced (not eliminated)
by engine grouping.
**Next:** repeated GroupKFold over engines with confidence intervals —
implemented in notebook 03.

## 4. The 125-cycle RUL cap is a modeling choice
Early-life RUL is unknowable from these sensors (no degradation trend), so
we cap. A different cap changes every reported metric.
**Next:** sensitivity analysis over cap values (100/125/150) — planned.

## 5. Window statistics, not sequence modeling
Mean/std/slope over 30 cycles discard order *within* the window beyond the
slope term. LSTMs/TCNs might capture richer dynamics.
**Next:** benchmark a sequence model on the same engine-grouped splits;
only adopt if it beats HGB out-of-sample, not on complexity appeal.

## 6. Feature importance ≠ instrumentation advice
Impurity importance is unstable under correlated sensors (s3/s4, s8/s9 move
together). Importance identifies *candidates* for ablation, not sensors to buy.
**Next:** leave-one-sensor-out retraining — implemented in notebook 03.

## 7. No calibration analysis
We report point predictions. A planner needs "70% chance RUL < 40 cycles".
**Next:** quantile regression or calibrated prediction intervals.

## 8. No concept-drift handling
A deployed model faces new operating envelopes, sensor recalibrations,
and fleet aging.
**Next:** input-drift monitors + scheduled retraining (see deployment
design in README).

## 9. Test engines contribute one point each
The scored test set is 100 single predictions — a thin basis for
slice analysis (e.g., accuracy specifically near failure).
**Next:** score full test trajectories (every valid window) for a
lead-time vs accuracy curve.

## 10. Not validated prospectively
Everything is retrospective. Real credibility needs shadow-mode scoring
on live engines with logged decisions vs outcomes.
**Next:** the rollout plan in `docs/business_case.md`.
