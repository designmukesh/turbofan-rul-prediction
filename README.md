# Turbofan Engine Remaining-Useful-Life Prediction

Predict **how many operating cycles remain before a turbofan engine fails** —
so maintenance is planned before failure, not after. Built on the NASA C-MAPSS
turbofan degradation dataset (FD001): 100 training engines, 100 truncated test
engines, 21 sensor channels per cycle.

> **Independence note:** this is independent work on a public research dataset.
> No employer data, systems, or proprietary information were used.

## The product question

A maintenance planner doesn't need a sensor dashboard — they need a number:
*"Engine 42 has ~30 cycles left; schedule the shop visit next week."*
This project turns raw sensor streams into that number, and judges itself the
way maintenance economics demands: **late predictions cost more than early ones.**

## Approach

1. **EDA** (`notebooks/01_eda.ipynb`) — screened all 21 sensors across 100 engines:
   7 are constant (dropped), 14 trend with engine age. FD001 runs a single
   operating condition, so no regime normalization is needed.
2. **Features** — 30-cycle sliding windows → 6 statistics per informative sensor
   (mean, std, min, max, last value, within-window slope) = 84 features.
   Split **by engine** (80/20), never by window, to avoid leakage.
3. **Target** — piecewise-linear RUL capped at 125 cycles: engines spend most of
   life healthy with no degradation signal, so the model isn't asked to predict
   noise in the healthy phase.
4. **Models** — Ridge (baseline), Random Forest, HistGradientBoosting, judged on
   **RMSE** and the **PHM asymmetric score** (late predictions penalized
   exponentially harder — because an unplanned failure costs more than an early
   inspection).

## Results (held-out test: 100 engines, one prediction each)

| Model | Test RMSE (cycles) | Test PHM score |
|---|---|---|
| **HistGradientBoosting** | **12.74** | **247** |
| Random Forest | 13.48 | 307 |
| Ridge | 17.43 | 484 |

Predictions track the diagonal well, tightest at low RUL — the region where the
decision actually matters. Full error analysis (error by RUL band, sensor
importance) is in `notebooks/02_modeling.ipynb`.

## Repo structure

```
├── data/download_data.py      # fetch C-MAPSS from the PHM Society mirror
├── notebooks/
│   ├── 01_eda.ipynb           # executed: sensor screening, RUL label design
│   └── 02_modeling.ipynb      # executed: windows, model comparison, test eval
├── reports/
│   ├── metrics.json           # machine-readable results
│   └── figures/               # all charts (also embedded in the notebooks)
├── src/                       # notebook generators + feature code
└── requirements.txt
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python data/download_data.py          # ~43 MB, PHM Society mirror
# then open the notebooks in order
```

## Limitations (stated plainly)

- **Simulated** degradation, not real engines — cleaner signal than field data.
- **FD001 only**: one operating condition, one fault mode. FD002–FD004 add
  operating regimes and a second fault mode (regime normalization required).
- Fixed-window statistics, not a learned sequence model (LSTM/Transformer is the
  natural next step).
- The 125-cycle RUL cap is a modeling choice, not physics.

## Bridge

Project #3 in this portfolio turns RUL into decisions — *replace now / inspect
later / keep operating* — under a cost model, which is where the maintenance
planner actually lives.
