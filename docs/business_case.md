# Business Case — Turbofan RUL Prediction

> Every number below is either **observed** (computed from the FD001 data)
> or a **labeled assumption** for a hypothetical rollout. No savings,
> deployments, or customer impacts are claimed — this is a portfolio
> project on simulated data.

## 1. Business problem

Airlines and MRO (maintenance, repair, overhaul) providers schedule engine
overhauls on **fixed intervals** (conservative, wastes remaining life) or
react to **in-service failures** (dangerous, extremely expensive). The
decision that needs improving: *when to pull an engine for maintenance* —
early enough to avoid failure, late enough to use its useful life.

## 2. Current process (typical)

1. Engine flies; sensor data streams to the ground (ACARS/telemetry).
2. Maintenance is scheduled by flight-hour/cycle limits set by the OEM.
3. Boroscope inspections at intervals; overhaul when limits hit or
   findings demand it.
4. Unplanned removals happen when degradation outruns the schedule.

**Pain:** conservative limits scrap healthy life; surprises cause
unscheduled removals, spare-engine leasing, and schedule disruption.

## 3. Proposed AI workflow

```
Sensor stream (per flight cycle)
  → 30-cycle sliding window
  → statistical features (mean/std/min/max/last/slope per sensor)
  → RUL regression model (HistGradientBoosting, RMSE ~13 cycles on FD001 test)
  → predicted RUL + trend
  → maintenance-planning rules (thresholds + human review)
  → work order in MRO system → outcome feedback → retrain
```

The model does **not** decide alone. It produces a *RUL estimate with
uncertainty* that a reliability engineer combines with inspection findings
and operational constraints.

## 4. Users and decisions

| User | Decision enabled |
|---|---|
| Reliability engineer | Which engines to boroscope first; whether a trend is real |
| Maintenance planner | Pull-forward vs defer an overhaul within the safe window |
| Fleet manager | Spare-engine provisioning, route assignment for flagged engines |
| MRO shop | Work-scope planning (parts, labor) weeks ahead |

## 5. Value drivers (framework, not a claim)

```
Expected failure exposure  = P(failure before next check) × cost of unplanned removal
Expected intervention value = avoided expected failure exposure − planned intervention cost
```

The model's contribution is a **better P(failure before next check)** —
a per-engine, per-week risk estimate replacing a fleet-average schedule.

## 6. Failure costs — why the metric is asymmetric

- **False negative** (predict healthy, engine fails): unscheduled removal,
  possible in-flight shutdown, safety investigation, regulatory exposure.
  Cost: orders of magnitude above a planned overhaul.
- **False positive** (predict failure, engine healthy): an early shop
  visit, wasted remaining life, inspection labor.

This is exactly why the project uses the **PHM08 asymmetric score**
(late predictions penalized exponentially harder) alongside RMSE.

## 7. Hypothetical ROI sketch (assumptions, not observed)

Assume a mid-size fleet, per-engine overhaul cost **C**, unplanned-removal
premium **k×C** (industry rule of thumb: k ≈ 3–10×):

```
Annual value ≈ (unplanned removals avoided) × (k−1) × C
             − (false-alarm inspections) × inspection cost
             − (platform + integration + monitoring cost)
```

The model is worth deploying only if the first term dominates — which
requires measuring **false alarms per 100 engine-cycles** and **warning
lead time** in a prospective trial, not in a notebook.

## 8. Implementation costs and risks

- **Data pipeline:** per-cycle sensor ingestion, windowed feature store,
  model serving (batch daily is enough — RUL moves slowly).
- **Integration:** MRO/ERP work-order systems, alert routing, engineer UI.
- **Monitoring:** input drift (new operating conditions), prediction
  calibration drift, sensor dropout handling.
- **Adoption risk:** engineers override or ignore the model if false
  alarms are frequent or explanations are opaque — hence the
  explainability work in this project (permutation importance, sensor
  ablation).
- **Certification:** aviation is regulated; any decision-support tool
  needs a safety case. The model advises; certified processes decide.

## 9. Rollout stages

1. **Shadow mode:** score historical + live engines, no actions taken.
   Measure agreement with actual removals.
2. **Advisory:** reliability engineers see RUL + trend; actions logged
   against model suggestions.
3. **Constrained automation:** low-risk decisions (inspection scheduling)
   follow model priority; overhauls stay human-approved.
4. **Fleet expansion:** extend beyond FD001-like conditions only after
   retraining on multi-condition data (FD002/FD004 analogues).

## 10. What this project does and does not prove

- **Proves:** RUL can be estimated from windowed sensor statistics with
  ~13-cycle RMSE on single-condition simulated data, using a leak-free
  engine-grouped evaluation, and the model's top signals are physically
  plausible (temperatures, speeds).
- **Does not prove:** savings, field accuracy, multi-condition
  generalization, or readiness for certification. Those need real fleet
  data and prospective trials (see `docs/limitations.md`).
