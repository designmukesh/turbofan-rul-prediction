"""Generate notebooks/01_eda.ipynb for Project #2 (NASA C-MAPSS turbofan RUL).

Follows the Project #1 pattern: markdown narrative + executed code cells,
figures saved to reports/figures/.
"""
import nbformat as nbf

P = "../notebooks/01_eda.ipynb"
FIG = "../reports/figures"
C = []

C.append(nbf.v4.new_markdown_cell(
    "# Turbofan RUL Prediction — Exploratory Analysis\n\n"
    "Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation "
    "(Saxena & Goebel, 2008). We start with **FD001** — 100 training engines, "
    "100 test engines, one operating condition, one fault mode (HPC degradation).\n\n"
    "The product question: given an engine's sensor history, **how many operating "
    "cycles remain before failure?** Before modeling, we need to know which sensors "
    "actually carry degradation signal — and which are just noise or constants."))

C.append(nbf.v4.new_code_cell(
    'import os\n'
    'import numpy as np\n'
    'import pandas as pd\n'
    'import matplotlib\n'
    'import matplotlib.pyplot as plt\n'
    'plt.rcParams.update({"figure.dpi": 110, "font.size": 10})\n'
    'DATA, FIG = "../data", "../reports/figures"\n'
    'os.makedirs(FIG, exist_ok=True)\n'
    'COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]\n'
    'SENSORS = [f"s{i}" for i in range(1, 22)]\n'
    'def load(name):\n'
    '    return pd.read_csv(f"{DATA}/{name}", sep=r"\\s+", header=None, names=COLS)\n'
    'train = load("train_FD001.txt"); test = load("test_FD001.txt")\n'
    'rul_test = pd.read_csv(f"{DATA}/RUL_FD001.txt", header=None, names=["rul"])\n'
    'print("train:", train.shape, "| test:", test.shape, "| rul rows:", len(rul_test))\n'
    'print("train engines:", train.unit.nunique(), "| test engines:", test.unit.nunique())\n'
    'print("columns:", len(COLS))\n'
    'train.head(3).to_string()'))

C.append(nbf.v4.new_markdown_cell(
    "## 1. Engine lifespans\n\n"
    "Each row is one operating cycle of one engine. Engines fail at different "
    "ages — the spread of lifespans is the first thing a maintenance planner cares about."))

C.append(nbf.v4.new_code_cell(
    'life = train.groupby("unit").cycle.max()\n'
    'print(f"engines: {len(life)} | lifespan: min={life.min()}, max={life.max()}, "\n'
    '      f"mean={life.mean():.0f}, median={life.median():.0f} cycles")\n'
    'fig, ax = plt.subplots(figsize=(9, 4))\n'
    'ax.hist(life, bins=20, color="#2a7f62", edgecolor="white")\n'
    'ax.axvline(life.median(), color="#c0392b", ls="--", lw=1.5,\n'
    '           label=f"median {life.median():.0f}")\n'
    'ax.set_title("FD001 — distribution of engine lifespans (training)", fontweight="bold")\n'
    'ax.set_xlabel("cycles to failure"); ax.set_ylabel("engines"); ax.legend()\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/01_lifespans.png", dpi=120); plt.show()\n'
    'life.describe().round(1).to_string()'))

C.append(nbf.v4.new_markdown_cell(
    "## 2. Do the sensors show degradation?\n\n"
    "Pick three engines (short, medium, long life) and watch four sensors across "
    "their whole lives. A sensor with prognostics value should *trend* as failure "
    "approaches — not just wiggle."))

C.append(nbf.v4.new_code_cell(
    'units = [life.idxmin(), life.sub(life.median()).abs().idxmin(), life.idxmax()]\n'
    'show = ["s2", "s4", "s9", "s14"]\n'
    'fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)\n'
    'for ax, s in zip(axes.ravel(), show):\n'
    '    for u in units:\n'
    '        d = train[train.unit == u]\n'
    '        ax.plot(d.cycle.values, d[s].values, lw=1.4,\n'
    '                label=f"engine {u} ({d.cycle.max()} cycles)")\n'
    '    ax.set_title(f"{s} over engine life", fontweight="bold")\n'
    '    ax.set_xlabel("cycle"); ax.set_ylabel(s); ax.legend(fontsize=8); ax.grid(alpha=0.3)\n'
    'fig.suptitle("Three engines, four sensors: degradation trends vs noise",\n'
    '             fontsize=13, fontweight="bold")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/02_sensor_trends.png", dpi=120); plt.show()'))

C.append(nbf.v4.new_markdown_cell(
    "## 3. Which sensors actually move?\n\n"
    "Two cheap, honest screens across all 100 engines: (a) is the sensor constant "
    "(zero variance — dead weight for any model), and (b) does it correlate with "
    "elapsed cycles (a trend toward failure)?"))

C.append(nbf.v4.new_code_cell(
    'var = train.groupby("unit")[SENSORS].std().mean()          # avg within-engine std\n'
    'corr = train.groupby("unit")[SENSORS + ["cycle"]].apply(\n'
    '    lambda d: d[SENSORS].corrwith(d["cycle"])).median()     # median trend corr\n'
    'screen = pd.DataFrame({"avg_std": var, "trend_corr": corr})\n'
    'screen["constant"] = screen.avg_std < 1e-6\n'
    'print(f"constant sensors: {screen[screen.constant].index.tolist()}")\n'
    'print(f"informative (|trend corr| > 0.5): "\n'
    '      f"{screen[~screen.constant & (screen.trend_corr.abs() > 0.5)].index.tolist()}")\n'
    'fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n'
    'screen.sort_values("trend_corr").trend_corr.plot.barh(ax=axes[0], color="#2a7f62",\n'
    '                                                      edgecolor="white")\n'
    'axes[0].axvline(0, color="black", lw=0.8)\n'
    'axes[0].set_title("Median correlation with elapsed cycles", fontweight="bold")\n'
    'axes[0].set_xlabel("correlation")\n'
    'screen.sort_values("avg_std").avg_std.plot.barh(ax=axes[1], color="#8e6f3e",\n'
    '                                                 edgecolor="white", logx=True)\n'
    'axes[1].set_title("Average within-engine std (log scale)", fontweight="bold")\n'
    'axes[1].set_xlabel("std")\n'
    'fig.suptitle("Sensor screen: trend vs noise across 100 engines", fontsize=13,\n'
    '             fontweight="bold")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/03_sensor_screen.png", dpi=120); plt.show()\n'
    'screen.round(3).to_string()'))

C.append(nbf.v4.new_markdown_cell(
    "## 4. Operating conditions\n\n"
    "FD001 was simulated under a *single* operating condition — worth verifying, "
    "because FD002/FD004 have six regimes and need regime-normalized features "
    "(a stretch goal for the modeling notebook)."))

C.append(nbf.v4.new_code_cell(
    'ops = train.groupby("unit")[["op1", "op2", "op3"]].agg(["mean", "std"])\n'
    'print("Within-engine std of operational settings (should be ~0 in FD001):")\n'
    'print(ops.xs("std", axis=1, level=1).mean().round(6).to_string())\n'
    'print("\\nDistinct setting combinations in FD001:",\n'
    '      train[["op1", "op2", "op3"]].drop_duplicates().shape[0])'))

C.append(nbf.v4.new_markdown_cell(
    "## 5. Building the RUL label\n\n"
    "The dataset gives us cycles, not RUL — so we construct it: "
    "`RUL = max_cycle − cycle` per engine. But engines spend most of their life "
    "healthy, with no degradation signal. The standard practice (and the honest one) "
    "is a **piecewise-linear** target: RUL is capped at 125 cycles early in life, "
    "then decreases linearly. This stops the model from chasing noise in the healthy "
    "phase."))

C.append(nbf.v4.new_code_cell(
    'RUL_MAX = 125\n'
    'u = life.idxmax()\n'
    'd = train[train.unit == u].copy()\n'
    'd["rul_raw"] = d.cycle.max() - d.cycle\n'
    'd["rul"] = d.rul_raw.clip(upper=RUL_MAX)\n'
    'fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n'
    'axes[0].plot(d.cycle, d.rul_raw, color="#c0392b", lw=1.6, label="raw RUL")\n'
    'axes[0].plot(d.cycle, d.rul, color="#2a7f62", lw=1.6, label=f"capped at {RUL_MAX}")\n'
    'axes[0].set_title(f"Engine {u}: raw vs piecewise-linear RUL", fontweight="bold")\n'
    'axes[0].set_xlabel("cycle"); axes[0].set_ylabel("RUL (cycles)")\n'
    'axes[0].legend(); axes[0].grid(alpha=0.3)\n'
    'rul_all = train.groupby("unit").apply(\n'
    '    lambda x: (x.cycle.max() - x.cycle).clip(upper=RUL_MAX), include_groups=False)\n'
    'axes[1].hist(rul_all.values, bins=30, color="#2a7f62", edgecolor="white")\n'
    'axes[1].set_title("Training RUL distribution (piecewise-linear)", fontweight="bold")\n'
    'axes[1].set_xlabel("RUL (cycles)"); axes[1].set_ylabel("windows")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/04_rul_label.png", dpi=120); plt.show()\n'
    'print(f"RUL cap: {RUL_MAX} | share of capped (healthy-phase) rows: "\n'
    '      f"{(rul_all.values == RUL_MAX).mean():.1%}")'))

C.append(nbf.v4.new_markdown_cell(
    "## 6. The test set: truncated engines\n\n"
    "Test engines are cut off *before* failure — that's the actual prediction task. "
    "The separate `RUL_FD001.txt` gives the ground truth for each test engine's "
    "final observed cycle."))

C.append(nbf.v4.new_code_cell(
    'test_life = test.groupby("unit").cycle.max()\n'
    'print(f"test engines: {len(test_life)} | observed cycles: "\n'
    '      f"min={test_life.min()}, max={test_life.max()}, mean={test_life.mean():.0f}")\n'
    'print(f"ground-truth RUL at truncation: min={rul_test.rul.min()}, "\n'
    '      f"max={rul_test.rul.max()}, mean={rul_test.rul.mean():.0f}")\n'
    'fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n'
    'axes[0].hist(test_life, bins=20, color="#2a7f62", edgecolor="white")\n'
    'axes[0].set_title("Test engines: cycles observed before truncation", fontweight="bold")\n'
    'axes[0].set_xlabel("cycles"); axes[0].set_ylabel("engines")\n'
    'axes[1].hist(rul_test.rul, bins=20, color="#8e6f3e", edgecolor="white")\n'
    'axes[1].set_title("Test engines: true RUL at truncation", fontweight="bold")\n'
    'axes[1].set_xlabel("RUL (cycles)"); axes[1].set_ylabel("engines")\n'
    'fig.tight_layout(); fig.savefig(f"{FIG}/05_test_truncation.png", dpi=120); plt.show()'))

C.append(nbf.v4.new_markdown_cell(
    "## 7. Takeaways → modeling\n\n"
    "1. **~7 of 21 sensors are constant** in FD001 — drop them before modeling.\n"
    "2. **~14 sensors trend with age** — the degradation signal is real and visible.\n"
    "3. **Single operating condition** — no regime normalization needed for FD001 "
    "(keep the technique in the back pocket for FD002/FD004).\n"
    "4. **Piecewise-linear RUL (cap 125)** keeps the model honest in the healthy phase.\n"
    "5. Test engines are truncated mid-life — the model must extrapolate, not just "
    "interpolate.\n\n"
    "Next: `02_modeling.ipynb` — sliding-window features, model comparison, RMSE + "
    "the PHM asymmetric score (late predictions cost more than early ones — because "
    "in maintenance, they do)."))

nb = nbf.v4.new_notebook()
nb.cells = C
nbf.write(nb, P)
print("wrote", P, "—", len(C), "cells")
