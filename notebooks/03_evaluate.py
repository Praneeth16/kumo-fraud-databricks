# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Evaluate on the test part
# MAGIC
# MAGIC This notebook scores the test part with the released model and the fine-tuned model. Both models get the
# MAGIC same contexts, so the comparison is paired. It then shows how the fraud share of the context changes the
# MAGIC scores, adds a LightGBM baseline, adds features built from the written text of each value, and writes
# MAGIC submission files for Kaggle.
# MAGIC
# MAGIC Run it on AI Runtime with one GPU. It also runs headless with `databricks air run --file jobs/evaluate.yaml`.
# MAGIC Every parameter can then be set as an environment variable with the same name in upper case. Needs notebook 2.

# COMMAND ----------

# MAGIC %pip install -q "structured-data-models==0.2.1" "lightgbm>=4.5"

# COMMAND ----------

if "dbutils" in globals():
    dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import shutil
import sys
import tempfile
import time

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # a workspace notebook has no __file__ and runs in its own folder
    HERE = os.getcwd()
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "src")))

import numpy as np
import pandas as pd
import torch

from kumo_fraud import data, workspace
from kumo_fraud.context import correct, enriched_context
from kumo_fraud.finetune import metrics
from kumo_fraud.scorer import Scorer, load_model

P = workspace.params(
    {
        "data_dir": "/Volumes/<catalog>/<schema>/artifacts/raw/s6e11",
        "checkpoint": "/Volumes/<catalog>/<schema>/artifacts/checkpoints/kumo-tabular-v1.0.1/small/classifier.pt",
        "weights": "/Volumes/<catalog>/<schema>/artifacts/runs/finetune/finetuned.pt",
        "out_dir": "/Volumes/<catalog>/<schema>/artifacts/runs/evaluate",
        "size": "small",
        "pos_frac": "0.3",
        "seed": "0",
        "submission": "1",
        "lightgbm": "1",
        "sweep": "1",
        "written": "1",
        "written_keys": "20",
        "val_rows": "all",
        "test_rows": "all",
        "kaggle_rows": "all",
        "context_sizes": "1024,8192",
    },
    dbutils=globals().get("dbutils"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = P["out_dir"]
POS_FRAC, SEED = float(P["pos_frac"]), int(P["seed"])
SMALL, LARGE = (int(k) for k in P["context_sizes"].split(","))  # rows per context


def show(frame: pd.DataFrame) -> None:
    """A table in a notebook, plain text in a script."""
    if "display" in globals():
        display(frame)
    else:
        print(frame.to_string(index=False), flush=True)


def write(name: str, fill) -> None:
    """Write locally, then copy: the output directory may be a volume, and a volume rejects seeks."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        fill(path)
        shutil.copy(path, os.path.join(OUT_DIR, name))


def subsample(rows: np.ndarray, n: str) -> np.ndarray:
    """All of `rows`, or a fixed random `n` of them."""
    if n == "all":
        return rows
    return np.sort(np.random.default_rng(0).choice(rows, size=int(n), replace=False))


print(DEVICE, torch.cuda.get_device_name() if DEVICE == "cuda" else "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load, split and build the contexts
# MAGIC
# MAGIC Contexts always come from the train part. The test part is only scored. A context with extra fraud gives
# MAGIC scores that are too high, so every score is corrected back to the fraud rate of the train part.

# COMMAND ----------

start = time.time()
frame = data.load(P["data_dir"])
parts = data.split(frame)
y = frame[data.TARGET].to_numpy()
prior = float(y[parts["train"]].mean())
val_rows, test_rows = subsample(parts["val"], P["val_rows"]), subsample(parts["test"], P["test_rows"])

# (fraud share of the context, rows per context, members); a share of None draws rows at random
SETTINGS = [
    (None, SMALL, 8),
    (POS_FRAC, SMALL, 1),
    (POS_FRAC, SMALL, 8),
    (POS_FRAC, LARGE, 8),
]
rng = np.random.default_rng(SEED)
contexts = {}
for share, k, members in SETTINGS:
    label = f"{'random' if share is None else 'enriched'} {members} x {k:,}"
    contexts[label] = (share, [enriched_context(y, parts["train"], k, share, rng) for _ in range(members)])
print(f"{len(test_rows):,} test rows with {int(y[test_rows].sum()):,} frauds; fraud rate of the train part {prior:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test part: released and fine-tuned model

# COMMAND ----------


def score(scorer: Scorer, ctxs: list, share, queries) -> np.ndarray:
    return np.mean([correct(scorer.predict(c, queries, view=i), share, prior) for i, c in enumerate(ctxs)], axis=0)


checkpoint = P["checkpoint"] or None  # blank: download the base weights from Hugging Face
scorers = {"released": Scorer(frame, load_model(P["size"], DEVICE, checkpoint), device=DEVICE, autocast=True)}
if P["weights"]:
    tuned = load_model(P["size"], DEVICE, checkpoint, weights=P["weights"])
    scorers["fine-tuned"] = Scorer(frame, tuned, device=DEVICE, autocast=True)

results, timings, rows = {}, {}, []
for name, scorer in scorers.items():
    for label, (share, ctxs) in contexts.items():
        t0 = time.time()
        m = metrics(y[test_rows], score(scorer, ctxs, share, test_rows))
        results[f"{name} | {label}"] = m
        timings[f"{name} | {label}"] = round(time.time() - t0, 1)
        rows.append({"model": name, "context": label, **m})
        print(name, label, json.dumps(m), flush=True)
show(pd.DataFrame(rows).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fraud share of the context
# MAGIC
# MAGIC The released model, 8 contexts of 1,024 rows, on the validation part. Only the fraud share of the contexts
# MAGIC changes. A random context holds about 1.7% fraud. The log-loss is computed after the correction.

# COMMAND ----------

sweep = []
if P["sweep"] == "1":
    sweep_rng = np.random.default_rng(SEED + 3)
    for share in (None, 0.05, 0.2, 0.3, 0.5, 0.8):
        ctxs = [enriched_context(y, parts["train"], SMALL, share, sweep_rng) for _ in range(8)]
        m = metrics(y[val_rows], score(scorers["released"], ctxs, share, val_rows))
        sweep.append({"fraud_share": "natural" if share is None else share, **m})
    show(pd.DataFrame(sweep).round(4))
else:
    print("Skipped: sweep is not 1.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LightGBM baseline
# MAGIC
# MAGIC LightGBM trains on the train part with fixed settings and stops early on the validation part. It adds one row
# MAGIC to the test table. The cell is skipped if `lightgbm` is not installed or the `lightgbm` parameter is 0.

# COMMAND ----------

lightgbm = None
if P["lightgbm"] != "1":
    print("Skipped: lightgbm is 0.")
else:
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
        print("Skipped: the lightgbm package is not installed.")
    if lgb is not None:
        x = frame[data.NUMERIC + data.CATEGORICAL].copy()
        for col in data.CATEGORICAL:
            x[col] = x[col].astype("category")
        gbm = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)  # fmt: skip
        gbm.fit(x.iloc[parts["train"]], y[parts["train"]], eval_set=[(x.iloc[val_rows], y[val_rows])],
                eval_metric="auc", callbacks=[lgb.early_stopping(150, verbose=False)])  # fmt: skip
        m = metrics(y[test_rows], gbm.predict_proba(x.iloc[test_rows])[:, 1])
        lightgbm = {"trees": int(gbm.best_iteration_), **m}
        results["LightGBM | trained on the train part"] = m
        rows.append({"model": "LightGBM", "context": "trained on the train part", **m})
        show(pd.DataFrame(rows).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features from the written text
# MAGIC
# MAGIC The data is synthetic, and the exact text of a value carries signal that a float loses. This step builds one
# MAGIC fraud-rate feature per written value and per digit part of each number (see `kumo_fraud/features.py`). The
# MAGIC rates are fitted on the train part, and out of fold for the rows of the train part.
# MAGIC
# MAGIC LightGBM is trained again with all of these features. Kumo-Tabular gets the `written_keys` features that
# MAGIC LightGBM used most, on the same contexts as above. The fine-tuned weights were trained on the 24 raw
# MAGIC columns, so the released model can do better here. The validation part decides which of the two models
# MAGIC writes the Kaggle file. The cell needs the LightGBM cell above.

# COMMAND ----------

written, rates, top, scorers_w, best_w = None, None, [], {}, None
if P["written"] != "1" or lightgbm is None:
    print("Skipped: written is not 1, or the LightGBM cell did not run.")
else:
    from kumo_fraud import features

    text = features.load_written(P["data_dir"])
    if P["submission"] == "1":  # the Kaggle rows get the same features, fitted on the train part
        text = pd.concat([text, features.load_written(P["data_dir"], "test.csv")], ignore_index=True)
    labels = np.concatenate([y, np.zeros(len(text) - len(y), dtype=y.dtype)])  # only the train part is read
    rates = features.encode(features.key_codes(text), labels, parts["train"])
    del text
    xw = pd.concat([x, rates.iloc[: len(y)]], axis=1)
    gbm_w = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)  # fmt: skip
    gbm_w.fit(xw.iloc[parts["train"]], y[parts["train"]], eval_set=[(xw.iloc[val_rows], y[val_rows])],
              eval_metric="auc", callbacks=[lgb.early_stopping(150, verbose=False)])  # fmt: skip
    m = metrics(y[test_rows], gbm_w.predict_proba(xw.iloc[test_rows])[:, 1])
    results["LightGBM | with written-text features"] = m
    rows.append({"model": "LightGBM", "context": "with written-text features", **m})
    print("LightGBM with written-text features", json.dumps(m), flush=True)

    gain = pd.Series(gbm_w.booster_.feature_importance("gain"), index=xw.columns)[rates.columns]
    top = gain.sort_values(ascending=False).index[: int(P["written_keys"])].tolist()
    frame_w = pd.concat([frame, rates.iloc[: len(y)][top]], axis=1)
    for name, scorer in scorers.items():
        scorers_w[name] = Scorer(frame_w, scorer.model, device=DEVICE, autocast=True, extra=tuple(top))
        for label in (f"enriched 8 x {SMALL:,}", f"enriched 8 x {LARGE:,}"):
            share, ctxs = contexts[label]
            t0 = time.time()
            m = metrics(y[test_rows], score(scorers_w[name], ctxs, share, test_rows))
            results[f"{name} + written-text features | {label}"] = m
            timings[f"{name} + written-text features | {label}"] = round(time.time() - t0, 1)
            rows.append({"model": f"{name} + written-text features", "context": label, **m})
            print(name, "+ written-text features", label, json.dumps(m), flush=True)
    share, ctxs = contexts[f"enriched 8 x {LARGE:,}"]
    val_auc = {name: metrics(y[val_rows], score(s, ctxs, share, val_rows))["roc_auc"] for name, s in scorers_w.items()}
    best_w = max(val_auc, key=val_auc.get)
    print("validation ROC-AUC with written-text features:", json.dumps(val_auc), "-> Kaggle file from", best_w)
    written = {"keys": len(rates.columns), "used_by_kumo": top, "lightgbm_trees": int(gbm_w.best_iteration_),
               "validation_roc_auc": val_auc, "submission_model": best_w}  # fmt: skip
    del xw, frame_w
    show(pd.DataFrame(rows).round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kaggle submission
# MAGIC
# MAGIC The fine-tuned model scores `test.csv` with 8 contexts of 8,192 rows. If no fine-tuned weights are given, the
# MAGIC released model is used. If the written-text step ran, a second file `submission_written.csv` comes from the
# MAGIC model that did best on the validation part with those features, on the same contexts.

# COMMAND ----------

if P["submission"] == "1":
    name = "fine-tuned" if "fine-tuned" in scorers else "released"
    kaggle = data.load_test(P["data_dir"])
    if P["kaggle_rows"] != "all":
        kaggle = kaggle.head(int(P["kaggle_rows"]))
    label = f"enriched 8 x {LARGE:,}"
    share, ctxs = contexts[label]
    p = score(scorers[name], ctxs, share, scorers[name].encode_queries(kaggle))
    submission = pd.DataFrame({data.KEY: kaggle[data.KEY], data.TARGET: p})
    write("submission.csv", lambda path: submission.to_csv(path, index=False))
    print(f"{name} model, {len(submission):,} rows written to", os.path.join(OUT_DIR, "submission.csv"))
    if written is not None:
        extra = rates.iloc[len(y) : len(y) + len(kaggle)][top].reset_index(drop=True)
        kaggle_w = pd.concat([kaggle.reset_index(drop=True), extra], axis=1)
        p = score(scorers_w[best_w], ctxs, share, scorers_w[best_w].encode_queries(kaggle_w))
        submission_w = pd.DataFrame({data.KEY: kaggle[data.KEY], data.TARGET: p})
        write("submission_written.csv", lambda path: submission_w.to_csv(path, index=False))
        print(f"{best_w} model with written-text features, {len(submission_w):,} rows written to",
              os.path.join(OUT_DIR, "submission_written.csv"))  # fmt: skip
else:
    print("Skipped: submission is not 1.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

summary = {
    "device": torch.cuda.get_device_name() if DEVICE == "cuda" else DEVICE,
    "prior": prior,
    "test_rows": len(test_rows),
    "test_frauds": int(y[test_rows].sum()),
    "val_rows": len(val_rows),
    "pos_frac": POS_FRAC,
    "results": results,
    "score_seconds": timings,
    "fraud_share_sweep": sweep,
    "lightgbm": lightgbm,
    "written": written,
    "seconds": round(time.time() - start, 1),
}


def dump(path: str) -> None:
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)


write("summary.json", dump)
print("SUMMARY " + json.dumps(summary))
