# Turbofan Engine Remaining-Useful-Life Prediction

Predict **how many operating cycles remain before a turbofan engine fails** —
so maintenance is planned before failure, not after. NASA C-MAPSS FD001:
100 run-to-failure engines, 100 truncated test engines, 21 sensor channels.

**Headline result:** HistGradientBoosting, **test RMSE 12.74 cycles / PHM 247**,
validated with engine-grouped splits and repeated cross-validation
(**CV 14.40 ± 0.93** — the honest band is ~14–15). The model keys on *rates of
change* (slopes), rediscovers the degradation physics, and is most precise near
failure (4.83 RMSE in the 0–30 cycle band).

> **Independence note:** independent work on a public research dataset.
> No employer data, systems, or proprietary information were used.

## Why it matters

Unplanned engine removals cost 3–10× a planned shop visit. The decision that
needs improving: *when to pull an engine* — late enough to use its life, early
enough to avoid failure. This project turns sensor streams into a per-engine
RUL number and judges itself the way maintenance economics demands:
**late predictions cost more than early ones** (PHM asymmetric score).

## How it works

```
per-cycle sensors → 30-cycle windows → 84 statistical features
→ HistGradientBoosting → predicted RUL + trend
→ risk band → illustrative action → human review → work order
```

## Dataset

NASA C-MAPSS FD001 (Saxena & Goebel, 2008), via the PHM Society mirror.
**Simulated** run-to-failure trajectories — one operating condition, one fault
mode (HPC degradation). 20,631 train cycles. Full audit: `docs/data_dictionary.md`.

## Key EDA findings (`notebooks/01_eda.ipynb`)

- 7 of 21 sensors are constant in FD001 → dropped; 14 carry degradation signal.
- Degradation becomes visible in the last ~100 cycles (temperatures rise,
  speeds shift as the controller compensates).
- Engine lifespans vary widely (128–362 cycles) — RUL is not inferable from age.

## Method

1. **Features** (`src/features.py`): 30-cycle sliding windows; per sensor —
   mean, std, min, max, last value, within-window slope. 84 features, all
   physics-readable.
2. **Target**: piecewise-linear RUL, `min(125, cycles_remaining)`. The cap is a
   documented modeling choice: early-life sensors show no trend.
3. **Models** (`src/models.py`): Ridge baseline → Random Forest →
   HistGradientBoosting. No deep learning — 17k windows × 84 tabular features
   is gradient boosting's home turf.
4. **Metrics** (`src/evaluation.py`): RMSE (cycles) + PHM08 asymmetric score
   (late over-predictions penalized exponentially harder).

## Evaluation design (why you can believe it)

- **Split by engine, never by window** — overlapping windows would leak.
- Model selection on 20 validation engines; **test set (100 engines) scored once**.
- **Repeated grouped CV** (3 folds × 3 seeds, engine groups) confirms the ranking:
  HGB 14.40 ± 0.93 > RF 15.14 ± 0.96 > Ridge 19.06 ± 0.76.
- Unit tests: `pytest tests/` (10 tests: windowing geometry, label correctness,
  metric properties).

## Verified results

| Model | Test RMSE | Test PHM | CV RMSE |
|---|---|---|---|
| **HistGradientBoosting** | **12.74** | **247** | **14.40 ± 0.93** |
| Random Forest | 13.48 | 307 | 15.14 ± 0.96 |
| Ridge | 17.43 | 484 | 19.06 ± 0.76 |

Honest framing: the test set was a kinder-than-average draw — **~14–15 cycles
is the defensible expectation** for new engines. See `notebooks/03_validation.ipynb`.

## Interpretability (`notebooks/03_validation.ipynb`)

- **Permutation importance**: 7 of top 8 features are `_slope` — the model
  watches rates of change, i.e. degradation as a process.
- **Leave-one-sensor-out ablation**: worst single loss +0.64 RMSE — no sensor
  is load-bearing; the signal is distributed (good for robustness).
- **Physics check**: top channels (HPC outlet pressure/temperature, core speeds)
  match HPC-degradation thermodynamics. The model rediscovers the physics.

## Robustness

- Seed-stable (13.49–14.18 over 5 seeds).
- **Noise-sensitive** (+5.2 RMSE at 0.5× sensor noise) — flagged as the top
  deployment risk; noise-augmented training is next.
- Survives worst-sensor failure with mean imputation (+1.37).
- Errors concentrate at high RUL; **4.83 RMSE in the 0–30 band** — precise where
  it matters. Predictions lean optimistic (late PHM penalty 2.6× early) →
  deployment needs a conservative margin.

## Demo

`app/streamlit_app.py` — pick a test engine: sensor traces, predicted-vs-true
RUL over its life, risk band, and an illustrative recommended action
(thresholds labeled as placeholders, not policy). Run: `streamlit run app/streamlit_app.py`
(model artifact in `models/`; train it yourself with `scripts/train_final.py`).

## Limitations (volunteered)

Simulated, single-condition data · 100-point test set · point predictions only
(no intervals) · never validated prospectively. Each ships with its next
experiment: `docs/limitations.md`. No savings, deployments, or field accuracy
are claimed anywhere in this repo.

## Repository structure

```
├── README.md
├── data/               # download_data.py + README (raw *.txt not committed)
├── notebooks/
│   ├── 01_eda.ipynb            # executed: audit, sensor screening
│   ├── 02_modeling.ipynb       # executed: features, model comparison, test eval
│   └── 03_validation.ipynb     # executed: CV, importance, ablation, robustness
├── src/                # data.py, features.py, models.py, evaluation.py (tested)
├── scripts/            # notebook generators, train_final.py
├── tests/              # pytest suite (10 tests)
├── app/                # Streamlit decision demo
├── models/             # trained HGB artifact + model card (512K)
├── reports/            # figures/, metrics.json, metrics_validation.json, executive_summary.md
├── docs/               # data dictionary, methodology, business case, limitations,
│                       # interview guide, learning guide, project story, social kit
└── presentation/       # slide deck
```

## Reproduce

```bash
python data/download_data.py        # fetch FD001 (~43 MB)
pip install -r requirements.txt
pytest tests/                       # 10 tests
python scripts/train_final.py       # train + score the artifact
jupyter nbconvert --execute --to notebook notebooks/01_eda.ipynb --output /tmp/01.ipynb
# ... same for 02_modeling, 03_validation (03 takes ~20 min: repeated CV)
streamlit run app/streamlit_app.py  # decision demo
```

Seeds fixed (42; CV seeds 0,1,2). All results in `reports/` are regenerated by the above.

## Future work

FD002–FD004 (multi-condition) · prediction intervals (quantile regression) ·
noise-augmented training · full-trajectory lead-time curves · shadow-mode protocol.

## Citation

Saxena, A. & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data
Set.* NASA Ames Prognostics Data Repository.
