"""Features from the written text of each value.

The competition data is synthetic: a generative model wrote every field as text. The exact text of a value holds
signal that a float loses, e.g. "250.0" and "250" are the same number and different texts. A key groups rows by a
piece of that text: the whole written value of a column, or one digit part of a number. Each key gives one numeric
feature, the fraud rate of the train rows that share the key's value. For train rows the rate is computed out of
fold, so a row's own label never reaches its feature.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kumo_fraud.data import CATEGORICAL, NUMERIC

DIGIT_PARTS = ("int", "frac", "ndec", "last1", "last2", "first1", "first2")


def load_written(data_dir: str, name: str = "train.csv") -> pd.DataFrame:
    """The feature columns of `<data_dir>/<name>` as written text, in the row order of `data.load`."""
    frame = pd.read_csv(os.path.join(data_dir, name), dtype=str, keep_default_na=False)
    return frame[NUMERIC + CATEGORICAL].reset_index(drop=True)


def digit_parts(written: pd.Series) -> dict[str, pd.Series]:
    """Seven texts per number: the part before the decimal point, the part after it, the number of written
    decimals, the last one and two digits, and the first one and two digits."""
    split = written.str.split(".", n=1, expand=True)
    whole = split[0]
    frac = split[1].fillna("") if split.shape[1] > 1 else pd.Series("", index=written.index)
    digits = whole + frac
    return {"int": whole, "frac": frac, "ndec": frac.str.len().astype(str), "last1": digits.str[-1:],
            "last2": digits.str[-2:], "first1": whole.str[:1], "first2": whole.str[:2]}  # fmt: skip


def key_codes(written: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer codes per key, named `<column>_value` or `<column>_<digit part>`. A key with one value on every
    row, or with the same grouping as an earlier key, is left out."""
    texts = {f"{c}_value": written[c] for c in NUMERIC + CATEGORICAL}
    for c in NUMERIC:
        texts.update({f"{c}_{part}": s for part, s in digit_parts(written[c]).items()})
    codes: dict[str, np.ndarray] = {}
    seen: dict[bytes, str] = {}
    for name, text in texts.items():
        c = pd.factorize(text)[0].astype(np.int32)  # numbered by first appearance: equal groupings give equal codes
        if c.max() == 0 or c.tobytes() in seen:
            continue
        seen[c.tobytes()] = name
        codes[name] = c
    return codes


def fraud_rate(codes: np.ndarray, y: np.ndarray, fit: np.ndarray, apply: np.ndarray, smoothing: float, prior: float):
    """Smoothed fraud rate of each `apply` row's key value among the `fit` rows."""
    n = int(codes.max()) + 1
    count = np.bincount(codes[fit], minlength=n)
    frauds = np.bincount(codes[fit], weights=y[fit], minlength=n)
    return ((frauds + smoothing * prior) / (count + smoothing))[codes[apply]].astype(np.float32)


def encode(codes: dict[str, np.ndarray], y: np.ndarray, train: np.ndarray, smoothing: float = 20.0, folds: int = 5,
           seed: int = 0) -> pd.DataFrame:  # fmt: skip
    """One column `rate_<key>` per key for every row. Rows in `train` are encoded out of fold; every other row
    is encoded from all of `train`. Only `y[train]` is read."""
    train = np.asarray(train, dtype=np.int64)
    n = len(next(iter(codes.values())))
    other = np.setdiff1d(np.arange(n), train)
    labels = np.zeros(n, dtype=np.float64)
    labels[train] = y[train]
    prior = float(labels[train].mean())
    parts = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(train, labels[train]))
    out = {}
    for name, c in codes.items():
        col = np.empty(n, dtype=np.float32)
        for fit, apply in parts:
            col[train[apply]] = fraud_rate(c, labels, train[fit], train[apply], smoothing, prior)
        col[other] = fraud_rate(c, labels, train, other, smoothing, prior)
        out[f"rate_{name}"] = col
    return pd.DataFrame(out)
