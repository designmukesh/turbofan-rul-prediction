"""Generate notebooks/02_modeling.ipynb for Project #2 (NASA C-MAPSS RUL).

Sliding-window features -> piecewise-linear RUL -> model comparison
(RMSE + PHM asymmetric score) -> test-set evaluation -> sensor importance.
"""
import nbformat as nbf

P = "../notebooks/02_modeling.ipynb"
C = []

C.append(nbf.v4.new_markdown_cell(
    "# Turbofan RUL Prediction — Modeling\n\n"
    "From the EDA: 14 of 21 sensors carry degradation signal, FD001 has a single "
    "operating condition, and piecewise-linear RUL (cap 125) is the honest target.\n\n"
    "Plan: sliding windows of 30 cycles → statistical features per sensor → "
    "regressors predicting RUL. We judge with **RMSE** and the **PHM asymmetric "
    "score** — which punishes *late* predictions (predicting an engine has longer "
    "than it does) harder than early ones. In maintenance, a late prediction can "
    "mean an unplanned failure; an early one just means a slightly early inspection."))

C.append(nbf.v4.new_code_cell(
    'import os, json, time\n'
    'import numpy as np\n'
    'import pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'plt.rcParams.update({"figure.dpi": 110, "font.size": 10})\n'
    'from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor\n'
    'from sklearn.linear_model import Ridge\n'
    'from sklearn.metrics import mean_squared_error\n'
    'DATA, FIG = "../data", "../reports/figures"\n'
    'os.makedirs(FIG, exist_ok=True)\n'
    'COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]\n'
    '# informative sensors from the EDA screen (|trend corr| > 0.5, non-constant)\n'
    'SENS = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]\n'
    'RUL_MAX, WIN = 125, 30\n'
    'def load(name):\n'
    '    return pd.read_csv(f"{DATA}/{name}", sep=r"\\s+", header=None, names=COLS)\n'
    'train = load("train_FD001.txt"); test = load("test_FD001.txt")\n'
    'rul_test = pd.read_csv(f"{DATA}/RUL_FD001.txt", header=None, names=["rul"])\n'
    'print("train rows:", len(train), "| engines:", train.unit.nunique())\n'
    'print("sensors used:", len(SENS))\n'
    'def phm_score(y_true, y_pred):\n'
    '    """PHM08 asymmetric scoring: late predictions (d>0) cost more."""\n'
    '    d = np.asarray(y_pred) - np.asarray(y_true)\n'
    '    return float(np.sum(np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)))\n'
    'def rmse(y_true, y_pred):\n'
    '    return float(np.sqrt(mean_squared_error(y_true, y_pred)))\n'))

C.append(nbf.v4.new_markdown_cell(
    "## 1. Sliding-window features\n\n"
    "One row per cycle can't capture *trend* — so we feed each model a 30-cycle "
    "window and summarize every informative sensor with six statistics: mean, std, "
    "min, max, last value, and slope (linear trend within the window). "
    "The label is the piecewise-linear RUL at the window's end."))

C.append(nbf.v4.new_code_cell(
    'def window_features(df, win=WIN, stride=1):\n'
    '    """Build one feature row per sliding window. Label = RUL at window end."""\n'
    '    rows, labels = [], []\n'
    '    for u, d in df.groupby("unit"):\n'
    '        d = d.sort_values("cycle").reset_index(drop=True)\n'
    '        life = d.cycle.max()\n'
    '        vals = d[SENS].values\n'
    '        for end in range(win, len(d)+1, stride):\n'
    '            w = vals[end-win:end]\n'
    '            last_cycle = d.cycle.iloc[end-1]\n'
    '            feat = {}\n'
    '            for j, s in enumerate(SENS):\n'
    '                v = w[:, j]\n'
    '                feat[f"{s}_mean"] = v.mean()\n'
    '                feat[f"{s}_std"] = v.std()\n'
    '                feat[f"{s}_min"] = v.min()\n'
    '                feat[f"{s}_max"] = v.max()\n'
    '                feat[f"{s}_last"] = v[-1]\n'
    '                feat[f"{s}_slope"] = np.polyfit(np.arange(win), v, 1)[0]\n'
    '            rows.append(feat)\n'
    '            labels.append(min(RUL_MAX, life - last_cycle))\n'
    '    X = pd.DataFrame(rows)\n'
    '    return X, np.array(labels)\n'
    't0 = time.time()\n'
    'X_all, y_all = window_features(train)\n'
    'print(f"windows: {len(X_all):,} | features: {X_all.shape[1]} | built in {time.time()-t0:.0f}s")\n'
    'print(f"label: mean RUL={y_all.mean():.1f}, share capped at {RUL_MAX}: {(y_all==RUL_MAX).mean():.1%}")\n'
    '# engine ids per window (for a leak-free split by engine)\n'
    'units_per_window = []\n'
    'for u, d in train.groupby("unit"):\n'
    '    n = (len(d) - WIN)//1 + 1\n'
    '    units_per_window += [u]*n\n'
    'units_per_window = np.array(units_per_window)\n'
    'assert len(units_per_window) == len(X_all)\n'
    'X_all.head(3).to_string()'))

C.append(nbf.v4.new_markdown_cell(
    "## 2. Leak-free split: by engine, not by window\n\n"
    "Windows from the same engine overlap — splitting rows randomly would leak. "
    "We split 80/20 on *engines* instead."))

C.append(nbf.v4.new_code_cell(
    'rng = np.random.default_rng(42)\n'
    'engines = train.unit.unique()\n'
    'rng.shuffle(engines)\n'
    'n_tr = int(0.8*len(engines))\n'
    'tr_eng, va_eng = set(engines[:n_tr]), set(engines[n_tr:])\n'
    'tr_m = np.isin(units_per_window, list(tr_eng))\n'
    'Xtr, ytr = X_all[tr_m], y_all[tr_m]\n'
    'Xva, yva = X_all[~tr_m], y_all[~tr_m]\n'
    'print(f"train engines: {len(tr_eng)} -> {len(Xtr):,} windows")\n'
    'print(f"val   engines: {len(va_eng)} -> {len(Xva):,} windows")'))

C.append(nbf.v4.new_markdown_cell(
    "## 3. Model comparison\n\n"
    "Three regressors: a regularized linear baseline (Ridge), random forest, and "
    "histogram gradient boosting. Judged on RMSE and the PHM asymmetric score."))

C.append(nbf.v4.new_code_cell(
    'models = {\n'
    '    "ridge": Ridge(alpha=1.0),\n'
    '    "rf": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),\n'
    '    "hgb": HistGradientBoostingRegressor(random_state=42),\n'
    '}\n'
    'lb = []\n'
    'fitted = {}\n'
    'for name, m in models.items():\n'
    '    t0 = time.time()\n'
    '    m.fit(Xtr, ytr)\n'
    '    p = m.predict(Xva)\n'
    '    lb.append({"model": name, "rmse": rmse(yva, p),\n'
    '               "phm": phm_score(yva, p), "secs": time.time()-t0})\n'
    '    fitted[name] = m\n'
    'lb = pd.DataFrame(lb).sort_values("rmse")\n'
    'print(lb.round(2).to_string(index=False))\n'
    'WINNER = lb.iloc[0]["model"]\n'
    'print("\\nwinner:", WINNER)'))

C.append(nbf.v4.new_markdown_cell(
    "## 4. Reading the winner's errors\n\n"
    "A single RMSE hides *where* the model fails. For maintenance planning, errors "
    "near end-of-life matter most — so we look at predicted-vs-true and the error "
    "distribution."))

C.append(nbf.v4.new_code_cell(
    'm = fitted[WINNER]\n'
    'pva = m.predict(Xva)\n'
    'err = pva - yva\n'
    'fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n'
    'axes[0].scatter(yva, pva, s=4, alpha=0.25, color="#2a7f62")\n'
    'axes[0].plot([0, RUL_MAX], [0, RUL_MAX], "r--", lw=1.2)\n'
    'axes[0].set_title(f"{WINNER}: predicted vs true RUL (val)", fontweight="bold")\n'
    'axes[0].set_xlabel("true RUL"); axes[0].set_ylabel("predicted RUL")\n'
    'axes[1].hist(err, bins=40, color="#2a7f62", edgecolor="white")\n'
    'axes[1].axvline(0, color="black", lw=1)\n'
    'axes[1].set_title("Error distribution (pred - true)", fontweight="bold")\n'
    'axes[1].set_xlabel("cycles")\n'
    '# error by true-RUL band: the business-critical view\n'
    'bands = pd.cut(yva, bins=[0, 25, 50, 75, 100, 125],\n'
    '               labels=["0-25", "25-50", "50-75", "75-100", "100-125"])\n'
    'band_rmse = pd.DataFrame({"band": bands, "se": (pva-yva)**2}).groupby("band", observed=True).se.mean().pow(0.5)\n'
    'band_rmse.plot.bar(ax=axes[2], color="#8e6f3e", edgecolor="white")\n'
    'axes[2].set_title("RMSE by true-RUL band", fontweight="bold")\n'
    'axes[2].set_xlabel("true RUL band"); axes[2].set_ylabel("RMSE")\n'
    'fig.suptitle("Validation error analysis — where the model struggles", fontsize=13,\n'
    '             fontweight="bold")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/06_val_errors.png", dpi=120); plt.show()\n'
    'print(f"val RMSE: {rmse(yva, pva):.2f} | val PHM score: {phm_score(yva, pva):.0f}")\n'
    'print(f"share of predictions within ±10 cycles: {(np.abs(err) <= 10).mean():.1%}")\n'
    'print(band_rmse.round(2).to_string())'))

C.append(nbf.v4.new_markdown_cell(
    "## 5. Test set: the real exam\n\n"
    "Each test engine contributes exactly **one** scored point: the window ending "
    "at its last observed (truncated) cycle, against the ground-truth RUL file. "
    "This mirrors deployment — one prediction per engine, made mid-life."))

C.append(nbf.v4.new_code_cell(
    'Xte_all, _ = window_features(test)\n'
    '# last window per test engine\n'
    'last_idx = []\n'
    'pos = 0\n'
    'for u, d in test.groupby("unit"):\n'
    '    n = (len(d) - WIN)//1 + 1\n'
    '    last_idx.append(pos + n - 1)\n'
    '    pos += n\n'
    'Xte = Xte_all.iloc[last_idx].reset_index(drop=True)\n'
    'yte = rul_test.rul.clip(upper=RUL_MAX).values\n'
    'assert len(Xte) == len(yte) == 100\n'
    'res = []\n'
    'for name, mm in fitted.items():\n'
    '    p = mm.predict(Xte)\n'
    '    res.append({"model": name, "test_rmse": rmse(yte, p), "test_phm": phm_score(yte, p)})\n'
    'res = pd.DataFrame(res).sort_values("test_rmse")\n'
    'print(res.round(2).to_string(index=False))\n'
    'pw = fitted[WINNER].predict(Xte)\n'
    'fig, ax = plt.subplots(figsize=(9, 4.5))\n'
    'ax.scatter(yte, pw, s=30, alpha=0.7, color="#2a7f62", edgecolors="white")\n'
    'ax.plot([0, RUL_MAX], [0, RUL_MAX], "r--", lw=1.2, label="perfect")\n'
    'ax.set_title(f"Test set: predicted vs true RUL ({WINNER}, 100 engines, one point each)",\n'
    '             fontweight="bold")\n'
    'ax.set_xlabel("true RUL at truncation"); ax.set_ylabel("predicted RUL"); ax.legend()\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/07_test_rul.png", dpi=120); plt.show()\n'
    'print(f"\\ntest RMSE ({WINNER}): {rmse(yte, pw):.2f} | test PHM score: {phm_score(yte, pw):.0f}")'))

C.append(nbf.v4.new_markdown_cell(
    "## 6. Which sensors does the model lean on?\n\n"
    "Aggregating feature importances back to sensor level — the same "
    "'where to spend instrumentation budget' view as Project #1."))

C.append(nbf.v4.new_code_cell(
    'rf = fitted["rf"]\n'
    'imp = pd.Series(rf.feature_importances_, index=X_all.columns)\n'
    'by_sensor = imp.groupby(lambda f: f.rsplit("_", 1)[0]).sum().sort_values()\n'
    'fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n'
    'by_sensor.plot.barh(ax=axes[0], color="#2a7f62", edgecolor="white")\n'
    'axes[0].set_title("Sensor-level importance (random forest)", fontweight="bold")\n'
    'axes[0].set_xlabel("summed importance")\n'
    'imp.nlargest(12).sort_values().plot.barh(ax=axes[1], color="#8e6f3e", edgecolor="white")\n'
    'axes[1].set_title("Top 12 individual features", fontweight="bold")\n'
    'axes[1].set_xlabel("importance")\n'
    'fig.suptitle("What the model leans on", fontsize=13, fontweight="bold")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/08_importance.png", dpi=120); plt.show()\n'
    'print("top 5 sensors:", by_sensor.nlargest(5).index.tolist())'))

C.append(nbf.v4.new_markdown_cell(
    "## 7. Results, limitations, next steps\n\n"
    "**What we built:** sliding-window RUL regression on FD001, judged by RMSE and "
    "the business-weighted PHM score.\n\n"
    "**Limitations (stated plainly):**\n"
    "- *Simulated* degradation, not real engines — signal shapes are cleaner than field data.\n"
    "- FD001 only: single operating condition, single fault mode. FD002–FD004 add "
    "regimes and a second fault mode (regime normalization needed).\n"
    "- Windows are fixed-length summaries, not a learned sequence model (LSTM/Transformer "
    "is the natural next step).\n"
    "- Piecewise-linear cap (125) is a modeling choice, not physics.\n\n"
    "**Next:** Project #3 turns RUL into decisions — *replace now / inspect later / "
    "keep operating* — with a cost model, which is where the maintenance planner "
    "actually lives."))

C.append(nbf.v4.new_code_cell(
    '# persist a machine-readable summary for the README\n'
    'summary = {\n'
    '    "dataset": "NASA C-MAPSS FD001",\n'
    '    "window": WIN, "rul_cap": RUL_MAX, "sensors_used": SENS,\n'
    '    "winner": WINNER,\n'
    '    "val": {"rmse": rmse(yva, fitted[WINNER].predict(Xva)),\n'
    '            "phm": phm_score(yva, fitted[WINNER].predict(Xva))},\n'
    '    "test": {"rmse": float(res[res.model==WINNER].test_rmse.iloc[0]),\n'
    '             "phm": float(res[res.model==WINNER].test_phm.iloc[0])},\n'
    '    "leaderboard_val": lb.round(3).to_dict("records"),\n'
    '}\n'
    'os.makedirs("../reports", exist_ok=True)\n'
    'json.dump(summary, open("../reports/metrics.json", "w"), indent=2)\n'
    'print(json.dumps({k: v for k, v in summary.items() if k != "leaderboard_val"}, indent=2))'))

nb = nbf.v4.new_notebook()
nb.cells = C
nbf.write(nb, P)
print("wrote", P, "—", len(C), "cells")
