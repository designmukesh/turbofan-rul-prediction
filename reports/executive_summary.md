# Executive Summary — Turbofan Engine RUL Prediction

## The problem
Airlines schedule engine overhauls on fixed intervals (wastes healthy life) or
react to failures (dangerous, 3–10× the cost of a planned shop visit). The
decision to improve: *when to pull an engine* — late enough to use its life,
early enough to avoid failure.

## The data
NASA C-MAPSS FD001 (Saxena & Goebel, 2008): 100 run-to-failure turbofan
engines (20,631 cycles) for training, 100 truncated engines for testing,
21 sensors + 3 operating settings. **Simulated** data, one operating
condition, one fault mode (HPC degradation).

## The approach
- **Features:** 30-cycle sliding windows; 6 statistics (mean, std, min, max,
  last, slope) × 14 informative sensors = 84 features. Label: piecewise-linear
  RUL capped at 125 cycles.
- **Validation:** split by *engine* (overlapping windows would leak); model
  selection on 20 validation engines; test set of 100 engines scored once;
  repeated grouped CV (3 folds × 3 seeds) to confirm rankings.
- **Models:** Ridge → Random Forest → HistGradientBoosting.

## Verified results
| Model | Test RMSE | Test PHM score | CV RMSE (mean ± std) |
|---|---|---|---|
| Ridge | 17.43 | 484 | 19.06 ± 0.76 |
| Random Forest | 13.48 | 307 | 15.14 ± 0.96 |
| **HistGradientBoosting** | **12.74** | **247** | **14.40 ± 0.93** |

The honest expectation for new engines is **~14–15 cycles**, not the 12.74
headline — the test set was a kinder-than-average draw, and we say so.

## What the model learned
- It keys on **rates of change**: 7 of the top 8 features are `_slope`
  features — it learned degradation as a *process*.
- Top channels (HPC outlet pressure/temperature, core speeds) match the
  thermodynamics of HPC degradation — the model rediscovers the physics.
- **No single sensor is load-bearing** (leave-one-out ablation: worst loss
  +0.64 RMSE) — the signal is distributed, which means graceful degradation.

## Robustness
- Seed-stable (13.49–14.18 over 5 seeds).
- **Noise-sensitive**: +5.2 RMSE at 0.5× sensor noise — the top deployment
  risk, flagged for noise-augmented training.
- Survives worst-sensor failure with simple imputation (+1.37).
- Errors concentrate at high RUL (4.83 RMSE in the 0–30 band — precise where
  it matters); predictions lean optimistic (late errors dominate PHM 2.6×),
  so a deployment needs a conservative margin.

## Product framing
A Streamlit demo maps RUL → risk bands → illustrative actions → human review,
showing the decision flow a reliability engineer would use. The business case
documents asymmetric failure costs, rollout stages (shadow mode first), and
an ROI framework — all labeled as framework, not claimed savings.

## Limitations (volunteered)
Simulated single-condition data; 100-point test set; point predictions only
(no intervals); never validated prospectively. Each has a next experiment in
`docs/limitations.md`.

## Bottom line
RUL can be estimated to ~14 cycles on this data with a leak-free evaluation,
the model's signals are physically plausible, and its failure modes are the
right ones (uncertain when healthy, precise near failure, flagged where
fragile). What it does *not* prove — field accuracy, savings, certification —
is stated explicitly.
