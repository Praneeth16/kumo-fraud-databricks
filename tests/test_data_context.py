import numpy as np
import pandas as pd
import pytest

from kumo_fraud.context import correct, enriched_context
from kumo_fraud.data import TARGET, split


def test_split_is_stratified_disjoint_and_complete():
    y = (np.random.default_rng(0).random(10_000) < 0.02).astype(np.int64)
    parts = split(pd.DataFrame({TARGET: y}), seed=3)
    rows = np.concatenate(list(parts.values()))
    assert len(np.unique(rows)) == len(rows) == len(y)
    for name, frac in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        idx = parts[name]
        assert np.all(np.diff(idx) > 0)
        assert abs(len(idx) / len(y) - frac) < 1e-3
        assert abs(y[idx].sum() - frac * y.sum()) <= 1


def test_enriched_context_composition():
    rng = np.random.default_rng(0)
    y = (rng.random(5_000) < 0.05).astype(np.int64)
    rows = np.arange(1_000, 5_000)
    ctx = enriched_context(y, rows, 101, 0.3, rng)
    assert len(ctx) == len(np.unique(ctx)) == 101
    assert np.all(np.diff(ctx) > 0) and np.isin(ctx, rows).all()
    assert y[ctx].sum() == round(101 * 0.3)
    blind = enriched_context(y, rows, 200, None, rng)
    assert len(np.unique(blind)) == 200 and np.isin(blind, rows).all()


def test_correct_matches_hand_calculation():
    # factor = (0.3 / 0.7) / (0.02 / 0.98) = 21; p = 0.5 -> odds 1 / 21 -> 1 / 22
    assert correct(np.array([0.5]), 0.3, 0.02)[0] == pytest.approx(1 / 22)
    assert correct(np.array([0.0, 1.0]), 0.3, 0.02).tolist() == [0.0, 1.0]
    p = np.array([0.1, 0.7])
    assert np.array_equal(correct(p, None, 0.02), p)
    assert correct(np.array([0.3]), 0.3, 0.0167)[0] == pytest.approx(0.0167)
