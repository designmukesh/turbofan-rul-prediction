"""Dataset loading and RUL label construction for NASA C-MAPSS.

Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation
(Saxena & Goebel, 2008), FD001 subset — single operating condition,
single fault mode (HPC degradation). Simulated data, not field data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Raw file layout (all four FD subsets share it)
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
OPERATING_COLUMNS = ["op1", "op2", "op3"]

# Sensors carrying degradation signal, screened in notebook 01_eda
# (|trend correlation with cycle| > 0.5 and non-constant)
INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11",
    "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]
# Constant across all FD001 cycles — carry no signal, excluded
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Piecewise-linear RUL cap (cycles). Standard C-MAPSS practice: an engine with
# 200+ cycles of life left is "healthy"; only the last 125 cycles show
# measurable degradation. This is a modeling choice, not a measured threshold.
RUL_CAP = 125
WINDOW = 30


def load_subset(subset="FD001", data_dir=None):
    """Load train / test trajectories and the hidden RUL truth for one subset.

    Returns (train_df, test_df, rul_test) where rul_test[i] is the true
    remaining cycles for test engine (i + 1) at its truncation point.
    """
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    train = pd.read_csv(data_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(data_dir / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul_test = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None, names=["rul"])
    return train, test, rul_test


def piecewise_rul(df, cap=RUL_CAP):
    """Piecewise-linear RUL label per row: min(cap, cycles remaining).

    Early life is capped because sensor traces show no degradation trend
    until roughly the last `cap` cycles (see notebook 01_eda).
    """
    life = df.groupby("unit")["cycle"].transform("max")
    return np.minimum(cap, life - df["cycle"]).to_numpy()
