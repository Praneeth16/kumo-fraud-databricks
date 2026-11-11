"""Score card transactions with Kumo-Tabular from one or more fitted contexts of training rows.

The whole training frame is encoded once with fixed semantic types and rows are taken by position. This is
exact because the recipe's `AlignCategories` re-fits every categorical vocabulary on the context rows and maps
queries by value, so queries encoded from another frame (`encode_queries`) give the same predictions.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import sdm
import sdm.processing as sp
import torch
from torch.nn import ModuleList

from kumo_fraud.context import correct
from kumo_fraud.data import CATEGORICAL, NUMERIC, TARGET

STYPES = {**{c: "numerical" for c in NUMERIC}, **{c: "categorical" for c in CATEGORICAL}}


def load_model(size: str = "small", device: str = "cpu", checkpoint: str | None = None, weights: str | None = None):
    """Pretrained Kumo-Tabular classifier in eval mode. `checkpoint`: base weights from a local file instead of
    Hugging Face. `weights`: fine-tuned weights saved by `finetune.save`, loaded on top."""
    if checkpoint is None:
        model = sdm.models.KumoTabular(task="classification", size=size, device=device)
    else:
        model = sdm.models.KumoTabular(task="classification", size=size, pretrained=False, device=device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.models["classification"].load_state_dict(state, assign=True)
    if weights is not None:
        saved = torch.load(weights, map_location=device, weights_only=True)
        model.models["classification"].load_state_dict(saved["state"])
    return model.eval()


def view_recipe(model, view: int, logits: bool = False) -> sp.Recipe:
    """Default recipe with every `Choice` rotated so a single-estimator fit takes numeric transform
    `view % len(options)`. `logits=True` drops the final softmax so the output is the raw class scores."""
    recipe = model.default_recipe()
    for mod in recipe.features.modules():
        if isinstance(mod, sp.Choice):
            options = list(mod.options)
            shift = view % len(options)
            mod.options = ModuleList(options[shift:] + options[:shift])
    if logits:
        recipe.output = sp.TaskDispatch(classification=[sp.AverageEstimators()])
    return recipe


def fraud_column(pred: sdm.TableTensor) -> torch.Tensor:
    """The class-1 column of a model output; the recipe shuffles class order, columns are named by class."""
    names = [int(c) for c in pred.columns[sdm.Stype.numerical]]
    return pred.numerical[..., names.index(1)]


def encode_features(frame: pd.DataFrame, device: str = "cpu") -> sdm.TableTensor:
    return sdm.TableTensor.from_pandas(df=frame[list(STYPES)], stypes=STYPES, device=device)


class Scorer:
    """Encodes `frame` once; contexts and queries are row positions into it (or queries from `encode_queries`)."""

    def __init__(self, frame: pd.DataFrame, model, device: str = "cpu", autocast: bool = False, query_batch=20_000):
        self.model, self.device, self.query_batch = model, device, query_batch
        self.autocast = autocast and torch.device(device).type == "cuda"
        self.x = encode_features(frame, device)
        y = sdm.TableTensor.from_pandas(df=frame[[TARGET]], stypes={TARGET: "categorical"}, device=device)
        self.y = y[:, TARGET]
        self.labels = frame[TARGET].to_numpy()

    def _amp(self):
        if self.autocast:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _rows(self, rows) -> torch.Tensor:
        return torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device)

    def encode_queries(self, frame: pd.DataFrame) -> sdm.TableTensor:
        """Features of rows from another frame (Kaggle test file, a serving request) to pass as `query_rows`."""
        return encode_features(frame, self.device)

    def predict(self, context_rows: np.ndarray, query_rows, view: int = 0) -> np.ndarray:
        """Raw P(fraud) for `query_rows` (positions in the frame, or a table from `encode_queries`) from one
        single-estimator fit on `context_rows` with view `view`."""
        if self.model.training:
            raise RuntimeError("Scorer.predict needs the model in eval mode")
        queries = query_rows if isinstance(query_rows, sdm.TableTensor) else None
        n = len(queries) if queries is not None else len(query_rows)
        out = np.zeros(n, dtype=np.float64)
        ctx = self._rows(context_rows)
        gen = torch.Generator(device=self.device).manual_seed(int(view))
        try:
            with self._amp():
                self.model.fit(
                    self.x[ctx], self.y[ctx], recipe=view_recipe(self.model, view), num_estimators=1, generator=gen
                )
                for lo in range(0, n, self.query_batch):
                    hi = min(lo + self.query_batch, n)
                    if queries is None:
                        xq = self.x[self._rows(query_rows[lo:hi])]
                    else:
                        xq = queries[torch.arange(lo, hi, device=self.device)]
                    out[lo:hi] = fraud_column(self.model.predict(xq)).double().cpu().numpy()
        finally:
            self.model.clear()
        if not np.isfinite(out).all():
            raise FloatingPointError("non-finite probabilities from the model (check autocast)")
        return out

    def score(self, contexts: list[np.ndarray], query_rows, pos_frac: float | None, prior: float) -> np.ndarray:
        """Prior-corrected P(fraud) averaged over `contexts`; context `i` uses view `i`."""
        return np.mean(
            [correct(self.predict(c, query_rows, view=i), pos_frac, prior) for i, c in enumerate(contexts)], axis=0
        )
