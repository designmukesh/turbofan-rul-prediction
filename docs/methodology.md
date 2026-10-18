# Methodology — Turbofan RUL Prediction

## Problem statement

> **Business problem → Data → Prediction → Decision → Impact:**
> Unplanned engine removals are 3–10× more expensive than planned shop visits
> → per-cycle multivariate sensor streams from 100 run-to-failure engines
> → remaining-useful-life (RUL) in cycles per engine → reliability engineer
> prioritizes inspections and shop-visit timing → fewer surprises, less wasted
> healthy life. (Full business case: `docs/business_case.md`. All value
> figures are frameworks/assumptions, not claims.)

## Data

NASA C-MAPSS FD001 (Saxena & Goebel, 2008): 100 train engines (20,631 cycles),
100 truncated test engines, 21 sensors + 3 operating settings. **Simulated**,
single operating condition, single fault mode (HPC degradation). Full audit:
`notebooks/01_eda.ipynb`; column documentation: `docs/data_dictionary.md`.

## Target

Piecewise-linear RUL: `min(125, cycles_remaining)`. The 125-cycle cap is a
modeling choice (early life shows no degradation signal); all metrics are
conditional on it.

## Features

30-cycle sliding windows; per informative sensor: mean, std, min, max, last,
slope → 84 features. 14 of 21 sensors used (7 constant in FD001, screened in
EDA). Rationale: transparent, physics-readable statistics — a rising
temperature mean or steepening slope means something physical. Canonical
implementation: `src/features.py`.

## Baselines first

Ridge (α=1.0): test RMSE 17.43, PHM 484. Everything fancier had to beat this
by a margin worth its complexity.

## Model comparison

| Model | Test RMSE | Test PHM | Val RMSE |
|---|---|---|---|
| Ridge | 17.43 | 484 | — |
| RandomForest (200 trees) | 13.48 | 307 | — |
| **HistGradientBoosting** | **12.74** | **247** | 13.88 |

No deep learning: 17k windows × 84 tabular features is gradient boosting's
territory; sequence models are future work, to be adopted only on
out-of-sample evidence.

## Validation design (the part that makes it believable)

1. **Split by engine, not by window.** Overlapping windows from one engine
   would leak; the grouping key is enforced in code.
2. **Model selection on 20 validation engines** (seed 42, same split as
   notebook 02 for comparability).
3. **Repeated grouped CV** (3 folds × 3 seeds = 9 fits/model) over the 80
   train engines — see Q1 of notebook 03. Results: Ridge 19.06 ± 0.76, RF 15.14 ± 0.96, **HGB 14.40 ± 0.93** (ranges: HGB 12.95–16.23, RF 14.21–17.40). The ranking HGB > RF > Ridge holds with clearly separated means; tails touch, so single-split selection would be fragile. Note the CV means are worse than the 12.74 test headline — the test set was a kinder-than-average draw; ~14–15 is the honest expectation.
4. **Test set: 100 engines, scored once**, after every decision was frozen.
   Each engine contributes one prediction (its final window).

## Metrics

- **RMSE** (cycles): comparable, interpretable.
- **PHM08 asymmetric score**: late (dangerous) over-predictions penalized
  exponentially harder than early ones — the metric encodes the operational
  cost. Implementation: `src/evaluation.py`.

## Explainability

- **Permutation importance** (5 repeats, val engines): measures actual
  reliance, not impurity bias. Top: temperature/speed statistics.
- **Leave-one-sensor-out ablation**: retrain without each sensor → ΔRMSE.
  No single sensor is load-bearing; signal is distributed.
- **Physics check:** top signals (HPC/LPT outlet temps, rotor speeds) match
  what HPC degradation should move thermodynamically. The model rediscovers
  the physics — see notebook 03 Q2–Q3. Top permutation features are dominated by `_slope` (7 of top 8): the model keys on rates of change. Sensor level: s11 (Ps30), s14 (corrected core speed), s3 (HPC outlet temp). Ablation: worst single-sensor loss +0.64 RMSE (s14); no sensor is load-bearing.

## Robustness

- Seed stability (5 seeds): 5 seeds: 13.49–14.18, tight — not seed luck.
- Noise injection (0.5×/1×/2× feature std): 0.5×/1×/2× feature-std noise → RMSE 19.12 / 26.26 / 34.91 (+5.2 / +12.4 / +21.0). Steep: the top deployment risk; noise-augmented training is the next hardening step.
- Worst-sensor failure at inference (mean imputation): worst sensor (s14) mean-imputed at inference: 15.25 (+1.37) — graceful.
- Residual analysis: errors concentrate at high RUL (capped healthy region —
  the honest failure mode); PHM late/early decomposition in notebook 03 Q5.

## What this does not establish

Simulated single-condition data; 100-point test set; point predictions only
(no intervals); no prospective validation. See `docs/limitations.md`.

## Reproducibility

`python data/download_data.py` → `scripts/train_final.py` →
`jupyter nbconvert --execute notebooks/*.ipynb`. Seeds fixed (42 throughout;
CV seeds 0,1,2). Requirements pinned in `requirements.txt`. Unit tests:
`pytest tests/`.
