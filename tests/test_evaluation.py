"""Tests for src/evaluation.py — the metrics everything else is judged by."""
import numpy as np

from src.evaluation import evaluate, mae, phm_score, rmse


def test_rmse_perfect_is_zero():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmse_known_value():
    # errors [1, -1] -> mean square 1 -> rmse 1
    assert rmse([0, 0], [1, -1]) == 1.0


def test_phm_score_perfect_is_zero():
    assert phm_score([10, 20], [10, 20]) == 0.0


def test_phm_score_penalizes_late_more_than_early():
    # same absolute error, opposite direction: late (over-predict) must cost more
    late = phm_score([10], [20])    # predicted 20, truth 10 -> dangerous
    early = phm_score([10], [0])    # predicted 0, truth 10 -> conservative
    assert late > early > 0


def test_phm_score_matches_phm08_formula():
    d = 5.0
    assert phm_score([10], [15]) == float(np.exp(d / 10) - 1)
    assert phm_score([10], [5]) == float(np.exp(d / 13) - 1)


def test_evaluate_bundle_schema():
    m = evaluate([10, 20, 30], [11, 19, 32])
    assert set(m) == {"rmse", "mae", "r2", "phm_score", "n"}
    assert m["n"] == 3
    assert m["rmse"] > 0 and m["phm_score"] > 0
