"""Tests for src/features.py — window geometry and label correctness."""
import numpy as np
import pandas as pd

from src.data import INFORMATIVE_SENSORS, RUL_CAP, WINDOW
from src.features import last_window_per_engine, window_features


def _toy_df(n_units=3, cycles=50, seed=0):
    rng = np.random.default_rng(seed)
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    frames = []
    for u in range(1, n_units + 1):
        n = cycles
        d = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
        d["unit"] = u
        d["cycle"] = np.arange(1, n + 1)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_window_count_and_shape():
    df = _toy_df(n_units=2, cycles=40)
    X, y, units = window_features(df, win=10)
    # (40 - 10) + 1 = 31 windows per engine
    assert len(X) == 62 and len(y) == 62 and len(units) == 62
    assert X.shape[1] == len(INFORMATIVE_SENSORS) * 6


def test_label_is_capped_rul_at_window_end():
    df = _toy_df(n_units=1, cycles=200)
    _, y, _ = window_features(df, win=WINDOW, cap=RUL_CAP)
    # first window ends at cycle 30 of a 200-cycle life -> capped
    assert y[0] == RUL_CAP
    # last window ends at cycle 200 -> 0 remaining
    assert y[-1] == 0


def test_unit_ids_align_with_rows():
    df = _toy_df(n_units=3, cycles=40)
    _, _, units = window_features(df, win=10)
    assert set(units) == {1, 2, 3}
    # each engine's block is contiguous and equal-sized
    assert (units == 1).sum() == (units == 2).sum() == 31


def test_last_window_per_engine_returns_one_row_each():
    df = _toy_df(n_units=3, cycles=40)
    X_last, engines = last_window_per_engine(df, win=10)
    assert len(X_last) == 3
    assert list(engines) == [1, 2, 3]
