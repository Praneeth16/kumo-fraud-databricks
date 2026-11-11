# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Fine-tune Kumo-Tabular
# MAGIC
# MAGIC Each step draws a context and a separate batch of rows to predict from the train part, and both hold the
# MAGIC same share of fraud. The loss is plain cross-entropy on the batch. The correction for the extra fraud is
# MAGIC applied only when scoring. The model starts to overfit after about 2,000 steps, so the notebook keeps the
# MAGIC weights with the best validation score.
# MAGIC
# MAGIC Run it on AI Runtime with one GPU. It also runs headless with `databricks air run --file jobs/finetune.yaml`.
# MAGIC Every parameter can then be set as an environment variable with the same name in upper case. Needs notebook 0.

# COMMAND ----------

# MAGIC %pip install -q "structured-data-models==0.2.1"

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
from dataclasses import asdict

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
from kumo_fraud.finetune import FinetuneConfig, finetune, metrics, save
from kumo_fraud.scorer import Scorer, load_model

P = workspace.params(
    {
        "data_dir": "/Volumes/<catalog>/<schema>/artifacts/raw/s6e11",
        "checkpoint": "/Volumes/<catalog>/<schema>/artifacts/checkpoints/kumo-tabular-v1.0.1/small/classifier.pt",
        "out_dir": "/Volumes/<catalog>/<schema>/artifacts/runs/finetune",
        "size": "small",
        "steps": "2000",
        "lr": "5e-5",
        "eval_every": "250",
        "pos_frac": "0.3",
        "k_choices": "512,1024,2048",
        "q": "1024",
        "warmup": "50",
        "train": "all",
        "amp": "bf16",
        "seed": "0",
        "val_rows": "all",
        "context_sizes": "1024,8192",
        "mlflow_experiment": "",
    },
    dbutils=globals().get("dbutils"),
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = P["out_dir"]
POS_FRAC = float(P["pos_frac"])
CONTEXT_SIZES = [int(k) for k in P["context_sizes"].split(",")]  # rows per context in the validation tables
cfg = FinetuneConfig(
    steps=int(P["steps"]),
    lr=float(P["lr"]),
    k_choices=tuple(int(k) for k in P["k_choices"].split(",")),
    q=int(P["q"]),
    pos_frac=POS_FRAC,
    warmup=int(P["warmup"]),
    train=P["train"],
    eval_every=int(P["eval_every"]),
    seed=int(P["seed"]),
    amp=P["amp"],
)


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


print(DEVICE, torch.cuda.get_device_name() if DEVICE == "cuda" else "")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load and split
# MAGIC
# MAGIC The training file is split into train (80%), validation (10%) and test (10%). This notebook never scores the
# MAGIC test part.

# COMMAND ----------

start = time.time()
frame = data.load(P["data_dir"])
parts = data.split(frame)
val_rows = parts["val"]
if P["val_rows"] != "all":
    val_rows = np.sort(np.random.default_rng(0).choice(val_rows, size=int(P["val_rows"]), replace=False))
model = load_model(P["size"], DEVICE, checkpoint=P["checkpoint"] or None)
scorer = Scorer(frame, model, device=DEVICE, autocast=True)
prior = float(scorer.labels[parts["train"]].mean())
print(f"{len(parts['train']):,} train rows, {len(val_rows):,} validation rows, fraud rate {prior:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The released model on the validation part
# MAGIC
# MAGIC Contexts of 1,024 and 8,192 rows, each with 30% fraud, scored with 1 context and with the mean of 8. The same
# MAGIC contexts are used again for the fine-tuned model below.


# COMMAND ----------


def evaluate(scorer: Scorer) -> dict:
    y = scorer.labels
    rng = np.random.default_rng(cfg.seed + 2)
    out = {}
    for k in CONTEXT_SIZES:
        ctxs = [enriched_context(y, parts["train"], k, POS_FRAC, rng) for _ in range(8)]
        p = [correct(scorer.predict(c, val_rows, view=i), POS_FRAC, prior) for i, c in enumerate(ctxs)]
        out[f"1 x {k:,}"] = metrics(y[val_rows], p[0])
        out[f"8 x {k:,}"] = metrics(y[val_rows], np.mean(p, axis=0))
    return out


def table(result: dict) -> pd.DataFrame:
    return pd.DataFrame([{"context": name, **m} for name, m in result.items()]).round(4)


released = evaluate(scorer)
show(table(released))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tune
# MAGIC
# MAGIC The loop scores the validation part every `eval_every` steps and remembers the best weights. One step takes
# MAGIC about a quarter of a second on an A10 GPU.

# COMMAND ----------


def progress(entry: dict) -> None:
    loss = "" if entry["train_loss"] is None else f"  train loss {entry['train_loss']:.4f}"
    print(f"step {entry['step']:>5}  ROC-AUC {entry['roc_auc']:.4f}  log-loss {entry['logloss']:.4f}{loss}", flush=True)


history, best = finetune(scorer, parts["train"], val_rows, cfg, log=progress)
best_step = max(history, key=lambda h: (h["roc_auc"], -h["logloss"]))["step"]
print("best step:", best_step)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation curve

# COMMAND ----------

curve = pd.DataFrame(history)[["step", "roc_auc", "pr_auc", "logloss", "train_loss"]].round(4)
show(curve)
try:
    import matplotlib

    if "dbutils" not in globals():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
if plt is not None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(curve["step"], curve["roc_auc"], marker="o")
    ax.axvline(best_step, color="grey", linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("validation ROC-AUC")
    fig.tight_layout()
    write("validation_curve.png", fig.savefig)
    if "dbutils" in globals():
        plt.show()
    else:
        plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the best weights
# MAGIC
# MAGIC `finetuned.pt` holds the weights from the best step, with the settings and the curve. Notebooks 3 and 4 read it.

# COMMAND ----------

model.models["classification"].load_state_dict(best)
write("finetuned.pt", lambda path: save(path, best, cfg, history))
print("saved", os.path.join(OUT_DIR, "finetuned.pt"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## The fine-tuned model on the validation part

# COMMAND ----------

finetuned = evaluate(scorer)
comparison = table(released).merge(table(finetuned), on="context", suffixes=("_released", "_finetuned"))
show(comparison[["context", "roc_auc_released", "roc_auc_finetuned", "logloss_released", "logloss_finetuned"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC `summary.json` in the output folder holds the settings, both validation tables and the curve. If
# MAGIC `mlflow_experiment` is set, the same numbers are logged to an MLflow run.

# COMMAND ----------

summary = {
    "config": asdict(cfg),
    "device": torch.cuda.get_device_name() if DEVICE == "cuda" else DEVICE,
    "val_rows": len(val_rows),
    "prior": prior,
    "best_step": best_step,
    "released": released,
    "finetuned": finetuned,
    "history": history,
    "seconds": round(time.time() - start, 1),
}


def dump(path: str) -> None:
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=1)


write("summary.json", dump)
print("SUMMARY " + json.dumps({k: summary[k] for k in ("best_step", "released", "finetuned", "seconds")}))

if P["mlflow_experiment"]:
    import mlflow

    mlflow.set_experiment(P["mlflow_experiment"])
    with mlflow.start_run(run_name=f"finetune-{cfg.train}-{cfg.steps}"):
        mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})
        for h in history:
            mlflow.log_metrics({f"val_{m}": h[m] for m in ("roc_auc", "pr_auc", "logloss")}, step=h["step"])
        for name, result in (("released", released), ("finetuned", finetuned)):
            for cell, m in result.items():
                cell = cell.replace(" x ", "x").replace(",", "")
                mlflow.log_metrics({f"{name}_{cell}_{key}": value for key, value in m.items()})
        mlflow.log_artifact(os.path.join(OUT_DIR, "summary.json"))
