"""LightGBM on the same split, for an honest reference. CPU, a few minutes:
`PYTHONPATH=src python jobs/baseline_lightgbm.py <data_dir>`.

Needs `pip install lightgbm`. Keep it out of any process that imports torch (the two can clash on macOS).
"""

from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from kumo_fraud import data

frame = data.load(sys.argv[1])
parts = data.split(frame)
x = frame[list(data.NUMERIC) + list(data.CATEGORICAL)].copy()
for col in data.CATEGORICAL:
    x[col] = x[col].astype("category")
y = frame[data.TARGET].to_numpy()

model = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)  # fmt: skip
model.fit(x.iloc[parts["train"]], y[parts["train"]], eval_set=[(x.iloc[parts["val"]], y[parts["val"]])],
          eval_metric="auc", callbacks=[lgb.early_stopping(150, verbose=False)])  # fmt: skip
out = {"trees": int(model.best_iteration_)}
for name in ("val", "test"):
    p = model.predict_proba(x.iloc[parts[name]])[:, 1]
    out[name] = {"roc_auc": float(roc_auc_score(y[parts[name]], p)),
                 "pr_auc": float(average_precision_score(y[parts[name]], p)),
                 "logloss": float(log_loss(y[parts[name]], np.clip(p, 1e-7, 1 - 1e-7)))}  # fmt: skip
print(json.dumps(out, indent=1))
