"""Kaggle playground-series-s6e11 card-fraud data: loading and a stratified train/val/test split."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

KEY = "id"
TARGET = "is_fraud"
NUMERIC = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
CATEGORICAL = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def _read(path: str, target: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    out = pd.DataFrame({KEY: frame[KEY].astype(np.int64)})
    for c in NUMERIC:
        out[c] = frame[c].astype(np.float64)
    for c in CATEGORICAL:
        out[c] = frame[c].astype(str)
    if target:
        out[TARGET] = frame[TARGET].astype(np.int64)
    return out.reset_index(drop=True)


def load(data_dir: str) -> pd.DataFrame:
    """`<data_dir>/train.csv` with float64 numerics, str categoricals and an int64 target."""
    return _read(os.path.join(data_dir, "train.csv"), target=True)


def load_test(data_dir: str) -> pd.DataFrame:
    """`<data_dir>/test.csv`, typed like `load`, without a target."""
    return _read(os.path.join(data_dir, "test.csv"), target=False)


def split(frame: pd.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
    """Sorted, disjoint row positions for train (80%), val (10%) and test (10%), stratified by the target."""
    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    y = frame[TARGET].to_numpy()
    for label in np.unique(y):
        rows = rng.permutation(np.flatnonzero(y == label))
        a, b = round(0.8 * len(rows)), round(0.9 * len(rows))
        parts["train"].append(rows[:a])
        parts["val"].append(rows[a:b])
        parts["test"].append(rows[b:])
    return {name: np.sort(np.concatenate(rows)) for name, rows in parts.items()}
