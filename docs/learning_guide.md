# Learning Guide — Turbofan RUL Prediction

Written to be understood from first principles by an engineer moving into
hands-on AI. Every concept: what it is, why we used it, how it works, the
practical math, where it lives in this repo, and what an interviewer asks.

---

## 1. Remaining Useful Life (RUL) as a regression problem

**What:** Predict a number — cycles until failure — from sensor history.
**Why:** Classification ("will fail in 30 cycles: yes/no") forces an arbitrary
threshold; regression gives a number a planner can combine with costs.
**Where:** `src/data.py::piecewise_rul`, `src/features.py::window_features`.
**Goes wrong when:** the target is unknowable from the inputs (early life —
hence the cap).

## 2. Piecewise-linear RUL labels (the 125-cycle cap)

**What:** `label = min(125, cycles_remaining)`.
**Why:** Sensors show no degradation trend in early life; asking the model to
distinguish "200 vs 180 cycles left" from flat traces teaches it noise.
**Math:** trivial — a min() function. The subtlety is that it changes every
metric: errors in the capped region are measured against an artificial ceiling.
**Where:** `src/data.py` (`RUL_CAP = 125`).
**Interviewer:** "Why 125 and not 130?" — Honest answer: it's the community
standard for C-MAPSS (from the PHM08 challenge literature); results are
conditional on it. A sensitivity analysis over cap values is listed as future work.

## 3. Sliding windows + statistical features

**What:** Each training row = 30 consecutive cycles, summarized per sensor as
mean, std, min, max, last value, slope.
**Why:** One cycle has no trend information. The window turns a sequence into a
fixed-size vector of *physics-readable* numbers: a rising `s4_mean` means the
turbine runs hotter; a steep `s4_slope` means it's *getting* hotter fast.
**Math (practical):**
- mean/std/min/max — distribution of the sensor within the window.
- `last` — the most recent reading (recency matters).
- `slope` — least-squares line slope: `polyfit(arange(30), v, 1)[0]`.
  Positive slope on temperature = degradation drift.
**Where:** `src/features.py::window_features` (6 stats × 14 sensors = 84 features).
**Goes wrong when:** windows overlap across train/val — we split by engine,
never by window (see §7).

## 4. Ridge regression (the baseline)

**What:** Linear regression with an L2 penalty: minimize
`||y − Xw||² + α||w||²`.
**Why used:** Two-minute baseline. If a fancy model can't beat it clearly, the
fancy model isn't earning its complexity. Ours: test RMSE 17.43.
**Intuition:** The penalty `α||w||²` shrinks weights, taming the 84 correlated
features (temperatures move together; unpenalized linear models thrash).
**Interviewer:** "Why a baseline at all?" — Without one, you can't tell whether
12.74 is good. (It is: ~27% better than linear.)

## 5. Random Forest

**What:** 200 decision trees, each trained on a bootstrap sample with random
feature subsets; prediction = average.
**Why:** Captures non-linear interactions (e.g., "hot AND fast-spooling"),
robust to feature scale, gives impurity importance.
**Practical math:** Each tree splits to minimize variance in the leaves;
averaging 200 de-correlated trees kills variance. Our config:
`n_estimators=200, random_state=42` — test RMSE 13.48.
**Goes wrong when:** you read impurity importance as truth (biased toward
high-cardinality features; spreads credit across correlated sensors).

## 6. HistGradientBoosting (the winner)

**What:** Trees built *sequentially*, each correcting the previous trees'
residuals; histogram binning makes it fast.
**Why it won:** Test RMSE 12.74, PHM 247. Boosting typically beats bagging on
structured tabular data because each tree focuses on the current errors.
**Practical math:** `F_m = F_{m-1} + ν·h_m` where `h_m` fits the negative
gradient of the squared-error loss. Histogram binning (256 bins) makes
split-finding O(bins) instead of O(samples).
**Why not deep learning:** 17k samples × 84 tabular features is gradient
boosting's home turf; an LSTM adds complexity without demonstrated gain here.
**Where:** `src/models.py::get_models`.

## 7. Grouped splitting — the leakage lesson

**What:** Split by *engine*, never by window.
**Why:** Windows from one engine overlap (window ending at cycle 50 shares 29
cycles with the window ending at 51). A random row split puts near-duplicates
on both sides → the model "predicts" by recognition, and your validation
score is a lie.
**Where:** notebooks 02/03 (`rng.shuffle(engines)` → 80/20).
**Interviewer:** "How do you know there's no leakage?" — Windows can't cross
the engine boundary by construction; `units` array is asserted aligned; and
the test engines are entirely unseen trajectories scored once.

## 8. Repeated grouped cross-validation

**What:** 3 folds × 3 seeds = 9 fits per model, groups = engines.
**Why:** One validation split can crown a lucky winner. CV measures the
*distribution* of performance — mean ± std is the honest headline.
**Where:** `notebooks/03_validation.ipynb` Q1.
**Output looks like:** boxplots per model; HGB's box sits below RF's with
little overlap → ranking is real.

## 9. RMSE and the PHM08 asymmetric score

**RMSE:** `sqrt(mean((pred − true)²))` — in cycles, directly interpretable.
Punishes large errors quadratically.
**PHM score:** `Σ exp(−d/13)−1` for early (d<0), `Σ exp(d/10)−1` for late
(d>0), where d = pred − true. **Late predictions (saying 50 when truth is 10)
cost exponentially more** — because in aviation, late = potential failure.
**Why both:** RMSE is comparable across projects; PHM encodes the operational
cost asymmetry. A model can win on RMSE and lose on PHM — that disagreement
is itself information.
**Where:** `src/evaluation.py`.

## 10. Permutation importance

**What:** Shuffle one feature's values, measure how much validation RMSE
*worsens*. Big worsening = the model really uses it.
**Why over impurity importance:** it measures actual reliance, on held-out
data, without the cardinality bias.
**Math:** `importance_j = RMSE(X with column j shuffled) − RMSE(X)`, averaged
over 5 shuffles.
**Where:** notebook 03 Q2 (`sklearn.inspection.permutation_importance`).
**Goes wrong when:** features are correlated — shuffling one of two twins
barely hurts because the twin covers for it. That's why we *also* ablate.

## 11. Leave-one-sensor-out ablation

**What:** Delete a sensor's 6 features, retrain from scratch, measure ΔRMSE.
**Why:** The causal version of importance. Tells you what you'd actually lose —
the number an instrumentation engineer needs.
**Where:** notebook 03 Q3. Finding: no single sensor is load-bearing (ΔRMSE
small across the board) — the signal is distributed, which is good for
robustness.

## 12. Robustness tests

- **Seed stability:** same data, 5 model seeds → RMSE spread. If the spread is
  wide, the headline is fragile.
- **Noise injection:** add 0.5×/1×/2× feature-std Gaussian noise at inference.
  Simulates worse field sensors. Expect graceful, not cliff-edge, degradation.
- **Sensor failure:** mean-impute the most critical sensor's features at
  inference (no retraining) — what a real pipeline does on dropout.
**Where:** notebook 03 Q4.

## 13. Residual analysis

**What:** Plot `predicted − true` vs true RUL.
**Why:** Tells you *where* the model fails. Ours fails most at high RUL —
the capped healthy region with no signal. That's the *right* failure mode:
uncertain early, precise near failure.
**Where:** notebook 03 Q5.

## 14. What "the model rediscovered the physics" means

The top features are HPC/LPT outlet temperatures and rotor speeds — exactly
the channels thermodynamics says HPC degradation moves. If the top features
had been, say, operating-setting artifacts, we'd suspect the model learned a
dataset quirk instead of degradation. Always ask this question; a model that
can't pass it isn't deployable no matter its RMSE.
