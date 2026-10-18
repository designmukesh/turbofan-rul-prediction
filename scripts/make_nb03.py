"""Generate notebooks/03_validation.ipynb — validation, explainability, robustness.

This notebook imports the canonical logic from src/ (data, features, models,
evaluation), which also validates the refactor against the original analysis.
Each section follows: Question -> Why it matters -> Method -> Code -> Output
-> Interpretation -> Decision. Interpretation cells carry an anchor
<!-- INTERP:name --> patched with findings after execution.
"""
import sys
sys.path.insert(0, "..")

import nbformat as nbf

C = []

C.append(nbf.v4.new_markdown_cell(
    "# 03 — Validation, Explainability & Robustness\n\n"
    "Notebook 02 reported **test RMSE 12.74 cycles (HGB)** on an untouched test set — "
    "but model *selection* rested on a single validation split, importance came from "
    "impurity alone, and nothing was stress-tested. This notebook closes those gaps:\n\n"
    "1. **Repeated grouped cross-validation** — are the model rankings stable, or split luck?\n"
    "2. **Permutation importance** — which features drive the winner, honestly measured?\n"
    "3. **Sensor ablation** — what does each sensor *causally* contribute (retrain without it)?\n"
    "4. **Robustness** — seeds, sensor noise, and a failed sensor at inference time.\n"
    "5. **Error analysis** — where does the model fail, and does the failure make sense?\n\n"
    "All code imports from `src/` (`data`, `features`, `models`, `evaluation`). "
    "The 100-engine test set stays **untouched** throughout — every experiment here "
    "uses only the 80 train engines and 20 validation engines."))

C.append(nbf.v4.new_code_cell(
    'import json, os, sys, time\n'
    'sys.path.insert(0, "..")\n'
    'import numpy as np, pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'plt.rcParams.update({"figure.dpi": 110, "font.size": 10})\n'
    'from sklearn.base import clone\n'
    'from sklearn.inspection import permutation_importance\n'
    'from src.data import load_subset, INFORMATIVE_SENSORS, WINDOW, RUL_CAP\n'
    'from src.features import window_features\n'
    'from src.models import get_models, WINNER\n'
    'from src.evaluation import rmse, phm_score\n'
    'DATA, FIG = "../data", "../reports/figures"\n'
    'os.makedirs(FIG, exist_ok=True)\n'
    'train, test, rul_test = load_subset("FD001", DATA)\n'
    't0 = time.time()\n'
    'X_all, y_all, units = window_features(train)\n'
    'print(f"windows: {len(X_all):,} | features: {X_all.shape[1]} | built in {time.time()-t0:.0f}s")\n'
    '# identical 80/20 engine split as notebook 02 (seed 42) for comparability\n'
    'rng = np.random.default_rng(42)\n'
    'engines = train.unit.unique(); rng.shuffle(engines)\n'
    'n_tr = int(0.8 * len(engines))\n'
    'tr_eng, va_eng = set(engines[:n_tr]), set(engines[n_tr:])\n'
    'tr_m = np.isin(units, list(tr_eng)); va_m = np.isin(units, list(va_eng))\n'
    'Xtr, ytr = X_all[tr_m].reset_index(drop=True), y_all[tr_m]\n'
    'Xva, yva = X_all[va_m].reset_index(drop=True), y_all[va_m]\n'
    'gtr = units[tr_m]\n'
    'print(f"train engines: {len(tr_eng)} -> {len(Xtr):,} windows")\n'
    'print(f"val   engines: {len(va_eng)} -> {len(Xva):,} windows")\n'
    'print("winner model:", WINNER)'))

C.append(nbf.v4.new_markdown_cell(
    "## Q1 — Are the model rankings stable, or single-split luck?\n\n"
    "**Why it matters.** Notebook 02 picked HGB over RF on one validation split "
    "(80/20 engines, seed 42). If the ranking flips on a different split, the "
    "'winner' is noise and every downstream conclusion is shaky.\n\n"
    "**Method.** Repeated grouped cross-validation over the 80 train engines: "
    "3 folds × 3 seeds = 9 fits per model, groups = engine ids (no window of one "
    "engine ever trains and validates together). Metric: RMSE on held-out engines. "
    "We report mean ± std and the full distribution."))

C.append(nbf.v4.new_code_cell(
    'def repeated_grouped_cv(X, y, groups, models, n_splits=3, seeds=(0, 1, 2)):\n'
    '    """Manual repeated GroupKFold: permute engines per seed, then fold."""\n'
    '    res = {n: [] for n in models}\n'
    '    U = np.unique(groups)\n'
    '    for s in seeds:\n'
    '        r = np.random.default_rng(s)\n'
    '        perm = r.permutation(U)\n'
    '        fold_of = {e: i % n_splits for i, e in enumerate(perm)}\n'
    '        folds = np.array([fold_of[e] for e in groups])\n'
    '        for f in range(n_splits):\n'
    '            tr, va = folds != f, folds == f\n'
    '            for name, mdl in models.items():\n'
    '                m = clone(mdl).fit(X[tr], y[tr])\n'
    '                res[name].append(rmse(y[va], m.predict(X[va])))\n'
    '    return res\n'
    'models = get_models()\n'
    'Xn = Xtr.to_numpy()\n'
    't0 = time.time()\n'
    'cv_res = repeated_grouped_cv(Xn, ytr, gtr, models)\n'
    'print(f"CV done in {time.time()-t0:.0f}s: {len(cv_res[\"Ridge\"])} fits/model")\n'
    'for name, v in cv_res.items():\n'
    '    print(f"{name:22s} RMSE {np.mean(v):.2f} ± {np.std(v):.2f}  (range {min(v):.2f}–{max(v):.2f})")\n'
    '# checkpoint: CV is expensive; persist immediately\n'
    'json.dump({n: [float(x) for x in v] for n, v in cv_res.items()},\n'
    '          open("../reports/cv_checkpoint.json", "w"))\n'
    'print("checkpointed ../reports/cv_checkpoint.json")'))

C.append(nbf.v4.new_code_cell(
    'fig, ax = plt.subplots(figsize=(7, 4))\n'
    'order = ["Ridge", "RandomForest", "HistGradientBoosting"]\n'
    'ax.boxplot([cv_res[n] for n in order], tick_labels=["Ridge", "RF", "HGB"])\n'
    'ax.set_ylabel("RMSE (cycles) — held-out engines")\n'
    'ax.set_title("Repeated grouped CV: 3 folds x 3 seeds = 9 fits per model")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/09_cv_stability.png"); plt.show()\n'
    'print("saved 09_cv_stability.png")'))

C.append(nbf.v4.new_markdown_cell("<!-- INTERP:cv -->"))

C.append(nbf.v4.new_markdown_cell(
    "## Q2 — Which features actually drive the winner? (permutation importance)\n\n"
    "**Why it matters.** Notebook 02 used impurity importance, which is biased: "
    "it favors high-cardinality features and spreads credit arbitrarily across "
    "correlated sensors (s3/s4 temperatures move together, as do s8/s9 speeds). "
    "Permutation importance measures the real thing — how much validation RMSE "
    "*degrades* when a feature's signal is destroyed.\n\n"
    "**Method.** Train HGB on the 80 train engines; permute each of the 84 features "
    "(5 repeats) and measure RMSE increase on the 20 validation engines. "
    "Then aggregate to sensor level (sum over the 6 stats per sensor)."))

C.append(nbf.v4.new_code_cell(
    'hgb = get_models()["HistGradientBoosting"].fit(Xtr, ytr)\n'
    'base = rmse(yva, hgb.predict(Xva))\n'
    'print(f"HGB val RMSE (train engines -> val engines): {base:.2f}")\n'
    't0 = time.time()\n'
    'perm = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=42,\n'
    '                             scoring="neg_root_mean_squared_error", n_jobs=-1)\n'
    'print(f"permutation importance done in {time.time()-t0:.0f}s")\n'
    'imp = pd.Series(perm.importances_mean, index=Xva.columns).sort_values(ascending=False)\n'
    'print("top 12 individual features:")\n'
    'print(imp.head(12).to_string())\n'
    '# sensor-level: sum the 6 stats per sensor\n'
    'by_sensor = imp.groupby(lambda c: c.rsplit("_", 1)[0]).sum().sort_values(ascending=False)\n'
    'print("\\nby sensor:")\n'
    'print(by_sensor.to_string())'))

C.append(nbf.v4.new_code_cell(
    'fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n'
    'imp.head(15)[::-1].plot.barh(ax=axes[0], color="steelblue")\n'
    'axes[0].set_xlabel("RMSE increase when permuted (cycles)")\n'
    'axes[0].set_title("Top 15 features — permutation importance (HGB, val engines)")\n'
    'by_sensor[::-1].plot.barh(ax=axes[1], color="darkorange")\n'
    'axes[1].set_xlabel("summed RMSE increase (cycles)")\n'
    'axes[1].set_title("Sensor-level importance (6 stats summed)")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/10_permutation_importance.png"); plt.show()\n'
    'print("saved 10_permutation_importance.png")'))

C.append(nbf.v4.new_markdown_cell("<!-- INTERP:perm -->"))

C.append(nbf.v4.new_markdown_cell(
    "## Q3 — What does each sensor *causally* contribute? (ablation)\n\n"
    "**Why it matters.** Importance scores are observational — they say which "
    "signals the model *uses*, not what would happen without them. Correlated "
    "sensors can cover for each other. The honest instrumentation question is "
    "causal: retrain *without* the sensor and measure the damage. Only then can "
    "we discuss sensor redundancy.\n\n"
    "**Method.** Leave-one-sensor-out: drop each sensor's 6 features, retrain HGB "
    "from scratch on train engines, evaluate on val engines. Report ΔRMSE vs the "
    "full 84-feature model. (14 retrains — HGB is fast enough to afford honesty.)"))

C.append(nbf.v4.new_code_cell(
    'full_rmse = base\n'
    'abl = {}\n'
    't0 = time.time()\n'
    'for s in INFORMATIVE_SENSORS:\n'
    '    cols = [c for c in Xtr.columns if not c.startswith(s + "_")]\n'
    '    m = get_models()["HistGradientBoosting"].fit(Xtr[cols], ytr)\n'
    '    r = rmse(yva, m.predict(Xva[cols]))\n'
    '    abl[s] = r - full_rmse\n'
    '    print(f"drop {s:4s}: val RMSE {r:.2f}  (Δ {abl[s]:+.2f})")\n'
    'print(f"ablation done in {time.time()-t0:.0f}s")\n'
    'abl = pd.Series(abl).sort_values(ascending=False)'))

C.append(nbf.v4.new_code_cell(
    'fig, ax = plt.subplots(figsize=(8, 4))\n'
    'colors = ["crimson" if v > 0.5 else "steelblue" for v in abl.values]\n'
    'abl[::-1].plot.barh(ax=ax, color=colors[::-1])\n'
    'ax.axvline(0, color="k", lw=0.8)\n'
    'ax.set_xlabel("Δ val RMSE (cycles) when sensor removed & model retrained")\n'
    'ax.set_title("Leave-one-sensor-out ablation (HGB retrained each time)")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/11_ablation.png"); plt.show()\n'
    'print("saved 11_ablation.png")'))

C.append(nbf.v4.new_markdown_cell("<!-- INTERP:ablation -->"))

C.append(nbf.v4.new_markdown_cell(
    "## Q4 — How robust is the winner?\n\n"
    "**Why it matters.** A model that collapses when the random seed changes, "
    "when sensors get noisier (field sensors are worse than sim sensors), or when "
    "one sensor drops out is not deployable — however good its headline RMSE.\n\n"
    "**Method.** Three stress tests on the validation engines:\n"
    "- (a) **Seed stability:** retrain HGB with 5 seeds, spread of val RMSE.\n"
    "- (b) **Noise injection:** add Gaussian noise at 0.5×/1×/2× each feature's "
    "train std to val features (simulating noisier field instrumentation).\n"
    "- (c) **Sensor failure at inference:** mean-impute the most critical "
    "sensor's 6 features (what a pipeline would do on dropout) without retraining."))

C.append(nbf.v4.new_code_cell(
    '# (a) seed stability\n'
    'seed_rmses = []\n'
    'for s in [0, 1, 2, 3, 4]:\n'
    '    m = get_models(random_state=s)["HistGradientBoosting"].fit(Xtr, ytr)\n'
    '    seed_rmses.append(rmse(yva, m.predict(Xva)))\n'
    'print(f"(a) seed stability: {np.mean(seed_rmses):.2f} ± {np.std(seed_rmses):.2f} over 5 seeds")\n'
    '# (b) noise injection\n'
    'feat_std = Xtr.std()\n'
    'rng_n = np.random.default_rng(7)\n'
    'noise_res = {}\n'
    'for k in [0.5, 1.0, 2.0]:\n'
    '    Xn_ = Xva + rng_n.normal(0, 1, Xva.shape) * (k * feat_std.values)\n'
    '    noise_res[k] = rmse(yva, hgb.predict(Xn_))\n'
    '    print(f"(b) noise {k}x feature-std: val RMSE {noise_res[k]:.2f} (Δ {noise_res[k]-base:+.2f})")\n'
    '# (c) worst sensor fails at inference -> mean imputation\n'
    'crit = abl.index[0]\n'
    'crit_cols = [c for c in Xva.columns if c.startswith(crit + "_")]\n'
    'Xmiss = Xva.copy()\n'
    'Xmiss[crit_cols] = Xtr[crit_cols].mean().values  # what a pipeline would impute\n'
    'miss_rmse = rmse(yva, hgb.predict(Xmiss))\n'
    'print("(c) sensor " + crit + " failed at inference, val RMSE %.2f" % miss_rmse)\n'))

C.append(nbf.v4.new_code_cell(
    'fig, ax = plt.subplots(figsize=(8, 4))\n'
    'labels = ["clean"] + [f"noise {k}x" for k in [0.5, 1.0, 2.0]] + [f"{crit} failed"]\n'
    'vals = [base] + [noise_res[k] for k in [0.5, 1.0, 2.0]] + [miss_rmse]\n'
    'bars = ax.bar(labels, vals, color=["steelblue"] + ["darkorange"]*3 + ["crimson"])\n'
    'ax.axhline(base, color="k", ls="--", lw=0.8)\n'
    'for b, v in zip(bars, vals):\n'
    '    ax.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.1f}", ha="center", fontsize=9)\n'
    'ax.set_ylabel("val RMSE (cycles)")\n'
    'ax.set_title("Robustness: noise injection & single-sensor failure (HGB)")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/12_robustness.png"); plt.show()\n'
    'print("saved 12_robustness.png")'))

C.append(nbf.v4.new_markdown_cell("<!-- INTERP:robust -->"))

C.append(nbf.v4.new_markdown_cell(
    "## Q5 — Where does the model fail? (error analysis)\n\n"
    "**Why it matters.** A single RMSE hides *where* errors concentrate. If errors "
    "are worst at high RUL (the capped 'healthy' region where sensors show no "
    "degradation trend), that is expected and honest — the signal simply is not "
    "there. If errors spike near failure (low RUL), that is operationally dangerous "
    "and must be fixed.\n\n"
    "**Method.** Residuals (pred − true) vs true RUL on val engines; PHM score "
    "decomposed into late-prediction vs early-prediction penalty."))

C.append(nbf.v4.new_code_cell(
    'pred_va = hgb.predict(Xva)\n'
    'resid = pred_va - yva\n'
    'fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n'
    'axes[0].scatter(yva, resid, s=6, alpha=0.35)\n'
    'axes[0].axhline(0, color="k", lw=0.8)\n'
    'axes[0].set_xlabel("true RUL (cycles)")\n'
    'axes[0].set_ylabel("residual = predicted − true (cycles)")\n'
    'axes[0].set_title("Residuals vs true RUL (val engines)")\n'
    'late = resid[resid > 0]; early = resid[resid < 0]\n'
    'late_pen = float(np.sum(np.exp(late/10) - 1)); early_pen = float(np.sum(np.exp(-early/13) - 1))\n'
    'axes[1].bar(["late\\n(over-predict)", "early\\n(under-predict)"], [late_pen, early_pen],\n'
    '            color=["crimson", "steelblue"])\n'
    'axes[1].set_ylabel("PHM penalty contribution")\n'
    'axes[1].set_title(f"PHM asymmetry on val: late={late_pen:.0f} vs early={early_pen:.0f}")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/13_residuals.png"); plt.show()\n'
    'bins = pd.cut(yva, [0, 30, 60, 125], labels=["0-30", "30-60", "60-125"])\n'
    'print("RMSE by true-RUL band (val):")\n'
    'print(pd.Series(resid**2).groupby(bins, observed=True).mean().pow(0.5).to_string())\n'
    'print(f"\\nshare of |residual| > 30 cycles: {(np.abs(resid) > 30).mean():.1%}")'))

C.append(nbf.v4.new_markdown_cell("<!-- INTERP:errors -->"))

C.append(nbf.v4.new_markdown_cell(
    "## Decision — what this notebook changes\n\n"
    "<!-- INTERP:summary -->"))

C.append(nbf.v4.new_code_cell(
    'metrics_val = {\n'
    '    "cv_rmse": {n: {"mean": float(np.mean(v)), "std": float(np.std(v)),\n'
    '                     "min": float(min(v)), "max": float(max(v))} for n, v in cv_res.items()},\n'
    '    "cv_design": "repeated grouped CV: 3 folds x 3 seeds = 9 fits/model, groups=engines",\n'
    '    "hgb_val_rmse": float(base),\n'
    '    "permutation_top12": {k: float(v) for k, v in imp.head(12).items()},\n'
    '    "sensor_importance": {k: float(v) for k, v in by_sensor.items()},\n'
    '    "ablation_d_rmse": {k: float(v) for k, v in abl.items()},\n'
    '    "seed_stability_rmse": [float(v) for v in seed_rmses],\n'
    '    "noise_rmse": {str(k): float(v) for k, v in noise_res.items()},\n'
    '    "sensor_failure_rmse": {"sensor": crit, "rmse": float(miss_rmse)},\n'
    '    "resid_rmse_by_band": {str(k): float(v) for k, v in\n'
    '        pd.Series(resid**2).groupby(bins, observed=True).mean().pow(0.5).items()},\n'
    '    "phm_val": {"late": float(late_pen), "early": float(early_pen)},\n'
    '    "note": "All experiments on train/val engines only. The 100-engine test set remains untouched."\n'
    '}\n'
    'with open("../reports/metrics_validation.json", "w") as f:\n'
    '    json.dump(metrics_val, f, indent=2)\n'
    'print("wrote ../reports/metrics_validation.json")'))

nb = nbf.v4.new_notebook()
nb.cells = C
with open("../notebooks/03_validation.ipynb", "w") as f:
    nbf.write(nb, f)
print(f"wrote notebook with {len(C)} cells -> ../notebooks/03_validation.ipynb")
