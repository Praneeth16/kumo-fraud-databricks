"""MLflow pyfunc (models from code) that serves Kumo-Tabular from contexts fitted once at load.

The labelled context rows travel inside the model artifact; a request carries only the transactions to score
(on Databricks, only their ids: Feature Engineering looks the features up in the online store). Member `i` is one
single-estimator fit with view `i`, so the scores equal `Scorer.score` on the same contexts.
"""

from __future__ import annotations

import json
import os
import shutil
import threading

import mlflow
import numpy as np
import pandas as pd

from kumo_fraud.context import correct
from kumo_fraud.data import CATEGORICAL, KEY, NUMERIC, TARGET

FEATURES = NUMERIC + CATEGORICAL
QUERY_BATCH = 8192


def _category(value) -> str | None:
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    return {"true": "True", "false": "False"}.get(text.lower(), text)


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """`id` and the features in a fixed order and type: float64 numerics (strings, ints and Decimals parsed,
    anything else missing) and str categoricals (bools as "True"/"False", missing as None). Extra columns are
    dropped; absent columns count as missing."""
    n = len(frame)
    out = pd.DataFrame(index=pd.RangeIndex(n))
    out[KEY] = frame[KEY].to_numpy() if KEY in frame else np.full(n, None, dtype=object)
    for c in NUMERIC:
        col = frame[c] if c in frame else pd.Series(np.nan, index=frame.index)
        out[c] = pd.to_numeric(col.astype(object), errors="coerce").astype(np.float64).to_numpy()
    for c in CATEGORICAL:
        out[c] = [_category(v) for v in frame[c]] if c in frame else [None] * n
    return out


def export_artifacts(
    out_dir: str,
    frame: pd.DataFrame,
    contexts: list[np.ndarray],
    *,
    pos_frac: float,
    prior: float,
    weights: str | None = None,
    base_checkpoint: str | None = None,
    size: str = "small",
    version: str = "v1",
) -> dict[str, str]:
    """Write `contexts.parquet` (rows of `frame` at each context's positions, with a `member` column) and
    `meta.json`, and copy the weight files, into `out_dir`. Returns the artifacts mapping for `log_model`.
    `base_checkpoint=None` makes the served model download the base weights from Hugging Face at load."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [frame.iloc[np.asarray(c)][FEATURES + [TARGET]].assign(member=i) for i, c in enumerate(contexts)]
    paths = {"contexts": os.path.join(out_dir, "contexts.parquet"), "meta": os.path.join(out_dir, "meta.json")}
    pd.concat(rows, ignore_index=True).to_parquet(paths["contexts"], index=False)
    meta = {
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "pos_frac": pos_frac,
        "prior": prior,
        "size": size,
        "version": version,
        "n_members": len(contexts),
        "context_rows": [len(c) for c in contexts],
    }
    with open(paths["meta"], "w") as fh:
        json.dump(meta, fh, indent=1)
    # Real copies: the artifact upload fails on symlinks (a Hugging Face cache is all symlinks).
    for name, src in (("checkpoint", base_checkpoint), ("weights", weights)):
        if src is not None:
            paths[name] = os.path.join(out_dir, f"{name}.pt")
            shutil.copyfile(src, paths[name])
    return paths


class FraudModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import sdm
        import torch

        from kumo_fraud.scorer import encode_features, load_model, view_recipe

        arts = context.artifacts
        with open(arts["meta"]) as fh:
            self.meta = json.load(fh)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(
            self.meta["size"], self.device, checkpoint=arts.get("checkpoint"), weights=arts.get("weights")
        )
        self.lock = threading.Lock()
        # One fitted cache per member, kept on the device and swapped into the model under the lock. A
        # single-estimator fit is never offloaded to host memory (SDM does that only for E > 1 on CUDA).
        self.caches = []
        rows = pd.read_parquet(arts["contexts"])
        for member, ctx in rows.groupby("member", sort=True):
            x = encode_features(normalize(ctx), self.device)
            y = sdm.TableTensor.from_pandas(
                df=ctx[[TARGET]].astype(np.int64).reset_index(drop=True),
                stypes={TARGET: "categorical"},
                device=self.device,
            )[:, TARGET]
            gen = torch.Generator(device=self.device).manual_seed(int(member))
            recipe = view_recipe(self.model, int(member))
            with self.lock:
                self.model.fit(x, y, recipe=recipe, num_estimators=1, generator=gen)
                self.caches.append(self.model._cache)
                self.model.clear()
        self.predict(context, rows.head(2))

    def _member(self, cache, x, n: int) -> np.ndarray:
        import torch

        from kumo_fraud.scorer import fraud_column

        out = np.empty(n, dtype=np.float64)
        self.model._cache = cache
        try:
            for lo in range(0, n, QUERY_BATCH):
                idx = torch.arange(lo, min(lo + QUERY_BATCH, n), device=self.device)
                out[lo : lo + len(idx)] = fraud_column(self.model.predict(x[idx])).double().cpu().numpy()
        finally:
            self.model.clear()
        return out

    def predict(self, context, model_input, params=None):
        from kumo_fraud.scorer import encode_features

        frame = normalize(pd.DataFrame(model_input))
        n = len(frame)
        p = np.zeros(n, dtype=np.float64)
        if n:
            x = encode_features(frame, self.device)
            with self.lock:
                raw = [self._member(cache, x, n) for cache in self.caches]
            p = np.mean([correct(r, self.meta["pos_frac"], self.meta["prior"]) for r in raw], axis=0)
            if not np.isfinite(p).all():
                raise FloatingPointError("non-finite probabilities from the model")
        return pd.DataFrame({KEY: frame[KEY].to_numpy(), "p_fraud": p, "model_version": self.meta["version"]})


mlflow.models.set_model(FraudModel())
