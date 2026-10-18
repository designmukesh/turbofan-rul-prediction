"""Sliding-window statistical feature extraction.

One raw cycle cannot capture *trend*, so each model input is a window of
consecutive cycles. Every informative sensor in the window is summarized with
six statistics: mean, std, min, max, last value, and within-window slope
(linear trend). These are transparent, physics-readable features — a rising
mean exhaust temperature or a steepening slope means something physical.
"""
import numpy as np
import pandas as pd

from .data import INFORMATIVE_SENSORS, WINDOW, RUL_CAP

STATS = ["mean", "std", "min", "max", "last", "slope"]


def window_features(df, sensors=None, win=WINDOW, stride=1, cap=RUL_CAP):
    """Build one feature row per sliding window, per engine.

    Label = piecewise-linear RUL at the window's end.

    Returns (X, y, unit_ids):
      X        DataFrame, one row per window, 6 stats x len(sensors) columns
      y        ndarray of RUL labels
      unit_ids ndarray of engine id per row (for engine-grouped validation)
    """
    sensors = list(sensors) if sensors else list(INFORMATIVE_SENSORS)
    rows, labels, units = [], [], []
    for unit, d in df.groupby("unit"):
        d = d.sort_values("cycle").reset_index(drop=True)
        life = d["cycle"].max()
        vals = d[sensors].to_numpy()
        xs = np.arange(win)
        for end in range(win, len(d) + 1, stride):
            w = vals[end - win:end]
            last_cycle = d["cycle"].iloc[end - 1]
            feat = {}
            for j, s in enumerate(sensors):
                v = w[:, j]
                feat[f"{s}_mean"] = v.mean()
                feat[f"{s}_std"] = v.std()
                feat[f"{s}_min"] = v.min()
                feat[f"{s}_max"] = v.max()
                feat[f"{s}_last"] = v[-1]
                feat[f"{s}_slope"] = np.polyfit(xs, v, 1)[0]
            rows.append(feat)
            labels.append(min(cap, life - last_cycle))
            units.append(unit)
    X = pd.DataFrame(rows)
    return X, np.asarray(labels), np.asarray(units)


def last_window_per_engine(df, sensors=None, win=WINDOW, cap=RUL_CAP):
    """Feature row for the final window of each engine (the scored point)."""
    X, _, units = window_features(df, sensors=sensors, win=win, cap=cap)
    # keep the last row of each engine, in engine order
    keep = []
    seen_from_end = set()
    for i in range(len(units) - 1, -1, -1):
        if units[i] not in seen_from_end:
            seen_from_end.add(units[i])
            keep.append(i)
    keep = sorted(keep)
    return X.iloc[keep].reset_index(drop=True), np.asarray(sorted(set(units)))
