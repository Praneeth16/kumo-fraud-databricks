"""Fraud-enriched context sampling and the prior correction that undoes it."""

from __future__ import annotations

import numpy as np


def enriched_context(
    y: np.ndarray, rows: np.ndarray, k: int, pos_frac: float | None, rng: np.random.Generator
) -> np.ndarray:
    """`k` sorted row positions from `rows` without replacement: `round(k * pos_frac)` frauds and the rest
    non-frauds. `pos_frac=None` draws label-blind."""
    rows = np.asarray(rows)
    if pos_frac is None:
        return np.sort(rng.choice(rows, size=k, replace=False))
    is_pos = y[rows] == 1
    n_pos = round(k * pos_frac)
    pos = rng.choice(rows[is_pos], size=n_pos, replace=False)
    neg = rng.choice(rows[~is_pos], size=k - n_pos, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def correct(p_fraud: np.ndarray, pos_frac: float | None, prior: float) -> np.ndarray:
    """Map P(fraud) under a context with fraud share `pos_frac` back to base rate `prior`: the odds are divided
    by the enrichment factor `(pos_frac / (1 - pos_frac)) / (prior / (1 - prior))`."""
    p = np.asarray(p_fraud, dtype=np.float64)
    if pos_frac is None:
        return p
    factor = (pos_frac / (1 - pos_frac)) / (prior / (1 - prior))
    return p / (p + factor * (1 - p))
