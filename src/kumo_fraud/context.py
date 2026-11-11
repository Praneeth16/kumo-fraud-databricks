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


def build_contexts(
    y: np.ndarray, rows: np.ndarray, k: int, pos_frac: float, members: int, seed: int = 0, include=()
) -> list[np.ndarray]:
    """`members` enriched contexts of `k` rows from `rows`. The fraud positions `include` go into every member and
    count toward its `round(k * pos_frac)` frauds, so the fraud share, and with it `correct`, is unchanged."""
    include = np.unique(np.asarray(include, dtype=np.int64))
    n_pos, m = round(k * pos_frac), len(include)
    if m > n_pos or (y[include] != 1).any():
        raise ValueError(f"include must be at most {n_pos} fraud rows")
    rng = np.random.default_rng(seed)
    pool = np.setdiff1d(rows, include)
    return [
        np.sort(np.concatenate([include, enriched_context(y, pool, k - m, (n_pos - m) / (k - m), rng)]))
        for _ in range(members)
    ]
