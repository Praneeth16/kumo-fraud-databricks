import numpy as np
import pandas as pd

from kumo_fraud import features
from kumo_fraud.data import CATEGORICAL, NUMERIC


def written(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(np.int64)
    out = {}
    for i, c in enumerate(NUMERIC):
        v = np.round(rng.gamma(2.0, 30.0, n), 2)
        text = pd.Series(v).map(lambda x: f"{x:.2f}")
        if i % 2 == 0:  # frauds are written with a last digit of 7 more often
            text = text.where(~((y == 1) & (rng.random(n) < 0.6)), text.str[:-1] + "7")
        out[c] = text.to_numpy()
    for c in CATEGORICAL:
        out[c] = rng.choice(["a", "b", "c"], n)
    return pd.DataFrame(out), y


def test_digit_parts():
    parts = features.digit_parts(pd.Series(["1234.50", "7", "0.05"]))
    assert parts["int"].tolist() == ["1234", "7", "0"] and parts["frac"].tolist() == ["50", "", "05"]
    assert parts["ndec"].tolist() == ["2", "0", "2"] and parts["last1"].tolist() == ["0", "7", "5"]
    assert parts["last2"].tolist() == ["50", "7", "05"] and parts["first2"].tolist() == ["12", "7", "0"]


def test_key_codes_drop_constant_and_duplicate_keys():
    frame, _ = written()
    frame["card_type"] = "x"  # one value on every row
    codes = features.key_codes(frame)
    assert "card_type_value" not in codes and "amount_usd_value" in codes and "amount_usd_last1" in codes
    assert "amount_usd_ndec" not in codes  # every number has two decimals
    assert all(len(c) == len(frame) for c in codes.values())
    assert len({c.tobytes() for c in codes.values()}) == len(codes)


def test_a_train_row_is_encoded_without_its_own_label():
    frame, y = written()
    codes = features.key_codes(frame)
    train = np.arange(0, 400)
    base = features.encode(codes, y, train)
    counts = frame["amount_usd"].iloc[train].value_counts()
    row = int(np.flatnonzero(frame["amount_usd"].iloc[train].map(counts).to_numpy() == 1)[0])  # a value seen once
    flipped = y.copy()
    flipped[row] = 1 - flipped[row]
    again = features.encode(codes, flipped, train)
    # Out of fold, the row's rate is the smoothed prior of the other rows, which its label moves by 1/400.
    # Encoded with its own label, the rate would move by 1/21 (one row, smoothing 20).
    shift = abs(float(again["rate_amount_usd_value"][row]) - float(base["rate_amount_usd_value"][row]))
    assert shift < 0.005 < 1 / 21
    assert base.shape == (len(frame), len(codes)) and list(base.columns) == [f"rate_{k}" for k in codes]


def test_rows_outside_train_do_not_change_the_encoding_and_their_labels_are_not_read():
    frame, y = written()
    codes = features.key_codes(frame)
    train = np.arange(0, 400)
    base = features.encode(codes, y, train)
    other = y.astype(np.float64)
    other[400:] = np.nan  # e.g. the Kaggle test file has no labels
    assert np.array_equal(features.encode(codes, other, train).to_numpy(), base.to_numpy())


def test_the_encoding_carries_the_planted_signal():
    from sklearn.metrics import roc_auc_score

    frame, y = written(4000)
    train, held = np.arange(0, 3000), np.arange(3000, 4000)
    enc = features.encode(features.key_codes(frame), y, train)
    assert roc_auc_score(y[held], enc["rate_amount_usd_last1"][held]) > 0.6
    assert roc_auc_score(y[train], enc["rate_amount_usd_last1"][train]) > 0.6  # out of fold, still informative
    assert abs(roc_auc_score(y[held], enc["rate_card_type_value"][held]) - 0.5) < 0.08  # no signal planted
