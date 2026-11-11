"""Episodic fine-tuning of Kumo-Tabular on fraud-enriched contexts.

Each step draws a context of k train rows and a disjoint query batch of q train rows, both enriched to the same
fraud share, and minimises cross-entropy on the query labels. Context and queries share one distribution, which is
what the model assumes; the prior correction is applied only at inference.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import sdm
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from kumo_fraud.context import enriched_context
from kumo_fraud.scorer import Scorer, view_recipe


@dataclass
class FinetuneConfig:
    steps: int = 1000
    lr: float = 2e-5
    k_choices: tuple[int, ...] = (512, 1024, 2048)
    q: int = 1024
    pos_frac: float = 0.3
    warmup: int = 50
    weight_decay: float = 0.01
    train: str = "all"  # "all" or "icl" (row embedding frozen)
    eval_every: int = 100
    eval_rows: int = 10_000
    eval_k: int = 1024
    eval_members: int = 4
    seed: int = 0
    amp: str = "none"  # "none", "bf16" or "fp16"; CUDA only


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def episode_loss(scorer: Scorer, ctx: np.ndarray, qry: np.ndarray, labels: torch.Tensor, view: int) -> torch.Tensor:
    """Cross-entropy of the query labels from one training-mode forward on context `ctx`."""
    c, q = scorer._rows(ctx), scorer._rows(qry)
    gen = torch.Generator(device=scorer.device).manual_seed(view)
    recipe = view_recipe(scorer.model, view, logits=True)
    out = scorer.model(scorer.x[c], scorer.y[c], scorer.x[q], recipe=recipe, num_estimators=1, generator=gen)
    names = [int(n) for n in out.columns[sdm.Stype.numerical]]
    logits = out.numerical[..., [names.index(0), names.index(1)]].float()
    return torch.nn.functional.cross_entropy(logits, labels)


def finetune(
    scorer: Scorer, train_rows: np.ndarray, val_rows: np.ndarray, config: FinetuneConfig, log=print
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    """Fine-tune `scorer.model` in place. Returns the eval history and the state dict with the best val ROC-AUC
    (ties: lower log-loss); step 0 is the starting model. The model is left with the last-step weights, in eval
    mode."""
    cfg = config
    model, device = scorer.model, scorer.device
    net = model.models["classification"]
    y = scorer.labels
    prior = float(y[train_rows].mean())

    eval_rng = np.random.default_rng(cfg.seed + 1)
    eval_q = np.sort(eval_rng.choice(val_rows, size=min(cfg.eval_rows, len(val_rows)), replace=False))
    eval_ctx = [enriched_context(y, train_rows, cfg.eval_k, cfg.pos_frac, eval_rng) for _ in range(cfg.eval_members)]

    def evaluate() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            p = scorer.score(eval_ctx, eval_q, cfg.pos_frac, prior)
        return metrics(y[eval_q], p)

    frozen = [] if cfg.train == "all" else [p for n, p in net.named_parameters() if n.startswith("row_embedding.")]
    for p in frozen:
        p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() > 1], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() <= 1], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg.lr)

    def lr_factor(step: int) -> float:
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    cuda = torch.device(device).type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp) if cuda else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    def snapshot() -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    rng = np.random.default_rng(cfg.seed)
    history = [{"step": 0, **evaluate(), "train_loss": None, "seconds_per_step": None}]
    log(history[-1])
    best_key, best = (history[0]["roc_auc"], -history[0]["logloss"]), snapshot()
    losses: list[float] = []
    clock = time.perf_counter()
    try:
        for step in range(1, cfg.steps + 1):
            model.train()
            k = int(rng.choice(cfg.k_choices))
            ctx = enriched_context(y, train_rows, k, cfg.pos_frac, rng)
            qry = enriched_context(y, np.setdiff1d(train_rows, ctx, assume_unique=True), cfg.q, cfg.pos_frac, rng)
            labels = torch.as_tensor(y[qry], device=device)
            view = int(rng.integers(0, 2**31 - 1))
            amp = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
            with amp:
                loss = episode_loss(scorer, ctx, qry, labels, view)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

            if step % cfg.eval_every == 0 or step == cfg.steps:
                seconds = (time.perf_counter() - clock) / len(losses)
                entry = {"step": step, **evaluate(), "train_loss": float(np.mean(losses)), "seconds_per_step": seconds}
                history.append(entry)
                log(entry)
                key = (entry["roc_auc"], -entry["logloss"])
                if key > best_key:
                    best_key, best = key, snapshot()
                losses = []
                clock = time.perf_counter()
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.eval()
    return history, best


def save(path: str, state: dict[str, torch.Tensor], config: FinetuneConfig, history: list[dict]) -> None:
    """Weights for `scorer.load_model(weights=path)`, with the config and history that produced them."""
    torch.save({"state": state, "config": asdict(config), "history": history}, path)
