"""Final comparison on the held-out test split (10% of train.csv), plus an optional Kaggle submission file.

One GPU job. Environment: DATA_DIR, CHECKPOINT (base weights), WEIGHTS (fine-tuned weights), OUT_DIR, and
optionally SIZE, POS_FRAC, SEED, SUBMISSION=1. Contexts always come from the train split; the test split is
only scored. Prints one `SUMMARY {json}` line and writes summary.json (and submission.csv) to OUT_DIR.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from kumo_fraud import data
from kumo_fraud.context import correct, enriched_context
from kumo_fraud.scorer import Scorer, load_model

E = os.environ
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POS_FRAC, SEED = float(E.get("POS_FRAC", "0.3")), int(E.get("SEED", "0"))
# (label, fraud share of the context, rows per context, members)
SETTINGS = [
    ("random 8 x 1,024", None, 1024, 8),
    ("enriched 1 x 1,024", POS_FRAC, 1024, 1),
    ("enriched 8 x 1,024", POS_FRAC, 1024, 8),
    ("enriched 8 x 8,192", POS_FRAC, 8192, 8),
]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))}  # fmt: skip


def main() -> None:
    start = time.time()
    frame = data.load(E["DATA_DIR"])
    parts = data.split(frame)
    y = frame[data.TARGET].to_numpy()
    prior = float(y[parts["train"]].mean())
    contexts = {}
    rng = np.random.default_rng(SEED)
    for label, share, k, members in SETTINGS:  # the same contexts for both models, so the comparison is paired
        contexts[label] = [enriched_context(y, parts["train"], k, share, rng) for _ in range(members)]

    results, timings = {}, {}
    models = [("frozen", None)] + ([("fine-tuned", E["WEIGHTS"])] if E.get("WEIGHTS") else [])
    for name, weights in models:
        model = load_model(E.get("SIZE", "small"), DEVICE, checkpoint=E.get("CHECKPOINT"), weights=weights)
        scorer = Scorer(frame, model, device=DEVICE, autocast=True)
        for label, share, _, _ in SETTINGS:
            t0 = time.time()
            p = np.mean([correct(scorer.predict(c, parts["test"], view=i), share, prior)
                         for i, c in enumerate(contexts[label])], axis=0)  # fmt: skip
            results[f"{name} | {label}"] = metrics(y[parts["test"]], p)
            timings[f"{name} | {label}"] = round(time.time() - t0, 1)
            print(name, label, json.dumps(results[f"{name} | {label}"]), flush=True)

    os.makedirs(E["OUT_DIR"], exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # write locally, then copy: a volume rejects seeks
        if E.get("SUBMISSION"):
            label, share = "enriched 8 x 8,192", POS_FRAC
            kaggle = data.load_test(E["DATA_DIR"])
            queries = scorer.encode_queries(kaggle)  # the last model loaded: fine-tuned when WEIGHTS is set
            p = np.mean([correct(scorer.predict(c, queries, view=i), share, prior)
                         for i, c in enumerate(contexts[label])], axis=0)  # fmt: skip
            pd.DataFrame({"id": kaggle[data.KEY], "is_fraud": p}).to_csv(f"{tmp}/submission.csv", index=False)
            shutil.copy(f"{tmp}/submission.csv", os.path.join(E["OUT_DIR"], "submission.csv"))
        summary = {"device": torch.cuda.get_device_name() if DEVICE == "cuda" else DEVICE, "prior": prior,
                   "test_rows": len(parts["test"]), "test_frauds": int(y[parts["test"]].sum()), "pos_frac": POS_FRAC,
                   "results": results, "score_seconds": timings, "seconds": round(time.time() - start, 1)}  # fmt: skip
        with open(f"{tmp}/summary.json", "w") as fh:
            json.dump(summary, fh, indent=1)
        shutil.copy(f"{tmp}/summary.json", os.path.join(E["OUT_DIR"], "summary.json"))
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
