"""One fine-tuning job: frozen baseline, episodic fine-tune, best weights re-scored on val. Prints `SUMMARY {json}`.

Environment: DATA_DIR (train.csv), CHECKPOINT (base weights; unset = Hugging Face), OUT_DIR, SIZE, STEPS, LR,
POS_FRAC, TRAIN ("all" or "icl"), SEED. Optional: K_CHOICES, Q, WARMUP, EVAL_EVERY, AMP, VAL_ROWS,
MLFLOW_EXPERIMENT. The test split is not scored here.
"""

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict

import numpy as np
import torch

from kumo_fraud import data
from kumo_fraud.context import correct, enriched_context
from kumo_fraud.finetune import FinetuneConfig, finetune, metrics, save
from kumo_fraud.scorer import Scorer, load_model

E = os.environ
OUT_DIR = E["OUT_DIR"]
POS_FRAC = float(E.get("POS_FRAC", "0.3"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = FinetuneConfig(
    steps=int(E.get("STEPS", "1000")),
    lr=float(E.get("LR", "2e-5")),
    k_choices=tuple(int(k) for k in E.get("K_CHOICES", "512,1024,2048").split(",")),
    q=int(E.get("Q", "1024")),
    pos_frac=POS_FRAC,
    warmup=int(E.get("WARMUP", "50")),
    train=E.get("TRAIN", "all"),
    eval_every=int(E.get("EVAL_EVERY", "100")),
    seed=int(E.get("SEED", "0")),
    amp=E.get("AMP", "bf16"),
)


def evaluate(scorer: Scorer, train_rows: np.ndarray, val_rows: np.ndarray, prior: float) -> dict:
    """Val metrics at context sizes 1,024 and 8,192 with 1 and 8 members; the same contexts on every call."""
    y = scorer.labels
    rng = np.random.default_rng(cfg.seed + 2)
    out = {}
    for k in (1024, 8192):
        ctxs = [enriched_context(y, train_rows, k, POS_FRAC, rng) for _ in range(8)]
        p = [correct(scorer.predict(c, val_rows, view=i), POS_FRAC, prior) for i, c in enumerate(ctxs)]
        out[f"k{k}_m1"] = metrics(y[val_rows], p[0])
        out[f"k{k}_m8"] = metrics(y[val_rows], np.mean(p, axis=0))
    return out


def write(name: str, fill) -> None:
    """Write locally, then copy: the output directory may be a volume that rejects seeks."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        fill(path)
        shutil.copy(path, os.path.join(OUT_DIR, name))


def main() -> None:
    start = time.time()
    frame = data.load(E["DATA_DIR"])
    parts = data.split(frame)
    val_rows = parts["val"]
    if "VAL_ROWS" in E:
        val_rows = np.sort(np.random.default_rng(0).choice(val_rows, size=int(E["VAL_ROWS"]), replace=False))
    model = load_model(size=E.get("SIZE", "small"), device=DEVICE, checkpoint=E.get("CHECKPOINT"))
    scorer = Scorer(frame, model, device=DEVICE, autocast=True)
    prior = float(scorer.labels[parts["train"]].mean())

    frozen = evaluate(scorer, parts["train"], val_rows, prior)
    print("FROZEN", json.dumps(frozen), flush=True)
    history, best = finetune(scorer, parts["train"], parts["val"], cfg, log=lambda e: print("EVAL", json.dumps(e)))
    model.models["classification"].load_state_dict(best)
    tuned = evaluate(scorer, parts["train"], val_rows, prior)

    os.makedirs(OUT_DIR, exist_ok=True)
    write("finetuned.pt", lambda path: save(path, best, cfg, history))
    best_step = max(history, key=lambda h: (h["roc_auc"], -h["logloss"]))["step"]
    summary = {
        "config": asdict(cfg),
        "device": torch.cuda.get_device_name() if DEVICE == "cuda" else DEVICE,
        "val_rows": len(val_rows),
        "prior": prior,
        "best_step": best_step,
        "frozen": frozen,
        "finetuned": tuned,
        "history": history,
        "seconds": time.time() - start,
    }

    def dump(path: str) -> None:
        with open(path, "w") as fh:
            json.dump(summary, fh, indent=1)

    write("summary.json", dump)
    print("SUMMARY " + json.dumps(summary), flush=True)

    if E.get("MLFLOW_EXPERIMENT"):
        import mlflow

        mlflow.set_experiment(E["MLFLOW_EXPERIMENT"])
        with mlflow.start_run(run_name=f"finetune-{cfg.train}-{cfg.steps}"):
            mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})
            for h in history:
                mlflow.log_metrics({f"val_{m}": h[m] for m in ("roc_auc", "pr_auc", "logloss")}, step=h["step"])
            for name, result in (("frozen", frozen), ("finetuned", tuned)):
                for cell, m in result.items():
                    mlflow.log_metrics({f"{name}_{cell}_{key}": value for key, value in m.items()})
            mlflow.log_artifact(os.path.join(OUT_DIR, "summary.json"))


if __name__ == "__main__":
    main()
