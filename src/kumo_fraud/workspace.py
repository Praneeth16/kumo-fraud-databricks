"""Workspace helpers for the notebooks: names, parameters, Feature Engineering, model logging, endpoint REST calls
and chunked scoring. HTTP goes through `do(method, path, body=None)` (`WorkspaceClient().api_client.do` on
Databricks) so everything here can be tested with a fake."""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kumo_fraud.data import CATEGORICAL, KEY, NUMERIC, TARGET

FEATURES = NUMERIC + CATEGORICAL
MAX_REQUEST_BYTES = 16 * 2**20  # Model Serving request limit


@dataclass(frozen=True)
class Names:
    catalog: str
    schema: str
    volume: str = ""  # default /Volumes/<catalog>/<schema>/artifacts
    size: str = "small"

    def _t(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    root = property(lambda self: self.volume.rstrip("/") or f"/Volumes/{self.catalog}/{self.schema}/artifacts")
    raw_dir = property(lambda self: f"{self.root}/raw/s6e11")
    checkpoint = property(lambda self: f"{self.root}/checkpoints/kumo-tabular-v1.0.1/{self.size}/classifier.pt")
    landing_dir = property(lambda self: f"{self.root}/landing")
    feature_table = property(lambda self: self._t("transaction_features"))
    label_table = property(lambda self: self._t("transaction_labels"))
    online_table = property(lambda self: self._t("transaction_features_online"))
    scores_table = property(lambda self: self._t("batch_scores"))
    model = property(lambda self: self._t("kumo_fraud"))


def params(defaults: dict[str, str], dbutils=None) -> dict[str, str]:
    """Notebook parameters. With `dbutils` (a workspace notebook) every key is a text widget with its default and
    the widget values are returned. Without it (the notebook run as a script) every key comes from the upper-case
    environment variable, or else the default."""
    if dbutils is not None:
        for key, value in defaults.items():
            dbutils.widgets.text(key, value)
        return {key: dbutils.widgets.get(key) for key in defaults}
    return {key: os.environ.get(key.upper(), value) for key, value in defaults.items()}


def csv_columns() -> list[str]:
    """`selectExpr` casts that type the features of a header-only Spark CSV read like `data.load`."""
    return [f"CAST({KEY} AS BIGINT) AS {KEY}", *[f"CAST({c} AS DOUBLE) AS {c}" for c in NUMERIC], *CATEGORICAL]


def label_table_ddl(names: Names) -> str:
    return (
        f"CREATE OR REPLACE TABLE {names.label_table} ({KEY} BIGINT NOT NULL, {TARGET} BIGINT, split STRING, "
        f"CONSTRAINT transaction_labels_pk PRIMARY KEY ({KEY})) TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )


def label_frame(frame: pd.DataFrame, parts: dict[str, np.ndarray]) -> pd.DataFrame:
    """`id`, `is_fraud`, `split` for every labelled row."""
    split = np.empty(len(frame), dtype=object)
    for name, rows in parts.items():
        split[rows] = name
    return pd.DataFrame({KEY: frame[KEY].to_numpy(), TARGET: frame[TARGET].to_numpy(), "split": split})


def held_out_frauds(y: np.ndarray, rows: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Sorted positions of `n` frauds among `rows`: labels that are confirmed only after the first deployment."""
    rows = np.asarray(rows)
    frauds = rows[y[rows] == 1]
    return np.sort(np.random.default_rng(seed).choice(frauds, size=n, replace=False))


def ensure_online_store(fe, name: str, capacity: str = "CU_1", timeout_s: int = 1800, poll_s: int = 30):
    """The Lakebase online store `name`, created if absent, once it is AVAILABLE."""
    store = fe.get_online_store(name=name)  # None when absent
    if store is None:
        fe.create_online_store(name=name, capacity=capacity)
    deadline = time.time() + timeout_s
    while True:
        store = fe.get_online_store(name=name)
        if store is not None and "AVAILABLE" in str(getattr(store, "state", "")).upper():
            return store
        if time.time() > deadline:
            raise TimeoutError(f"online store {name} not AVAILABLE after {timeout_s}s: {store}")
        time.sleep(poll_s)


def training_set(fe, spark, names: Names):
    """Labels of the train split joined to every feature by `id`; `id` stays a model input."""
    from databricks.feature_engineering import FeatureLookup

    keys = spark.table(names.label_table).where("split = 'train'").select(KEY, TARGET)
    lookup = FeatureLookup(table_name=names.feature_table, lookup_key=KEY, feature_names=FEATURES)
    return fe.create_training_set(df=keys, feature_lookups=[lookup], label=TARGET)


def signature():
    """`id` required, every feature optional (looked up when absent, overriding the lookup when present)."""
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, Schema

    inputs = Schema(
        [ColSpec("long", KEY)]
        + [ColSpec("double", c, required=False) for c in NUMERIC]
        + [ColSpec("string", c, required=False) for c in CATEGORICAL]
    )
    outputs = Schema([ColSpec("long", KEY), ColSpec("double", "p_fraud"), ColSpec("string", "model_version")])
    return ModelSignature(inputs=inputs, outputs=outputs)


def serving_env() -> dict:
    """What the GPU serving container installs. Python is pinned because the model library needs 3.11 or newer,
    whatever the notebook that logs the model runs on. On a T4 (GPU_SMALL) pip resolves torch 2.7.1+cu118."""
    import mlflow

    pip = [
        "structured-data-models==0.2.1",
        "torch>=2.7,<2.8",
        "numpy>=1.26",
        "pandas>=2.2",
        "pyarrow>=16",
        f"mlflow=={mlflow.__version__}",
    ]
    return {"name": "kumo-fraud", "channels": ["conda-forge"], "dependencies": ["python=3.12", "pip", {"pip": pip}]}


def log_model(fe, train_set, artifacts: dict[str, str], model_name: str, run_name: str) -> int:
    """Log `serving.py` as code with the artifacts through Feature Engineering, register it in Unity Catalog and
    return the new version."""
    import mlflow

    from kumo_fraud import serving

    with mlflow.start_run(run_name=run_name) as run:
        fe.log_model(
            model=serving.__file__,
            artifact_path="model",
            flavor=mlflow.pyfunc,
            training_set=train_set,
            registered_model_name=model_name,
            artifacts=artifacts,
            code_paths=[os.path.dirname(serving.__file__)],
            signature=signature(),  # no input example: the notebook never loads the model, so it needs no torch
            conda_env=serving_env(),
        )
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(f"name='{model_name}'")
    return max(int(v.version) for v in versions if v.run_id == run.info.run_id)


def served_name(version: int) -> str:
    return f"kumo-fraud-{version}"


def endpoint_config(model_name: str, version: int, workload_type: str = "GPU_SMALL", scale_to_zero: bool = False):
    """One served entity taking all traffic."""
    entity = {
        "name": served_name(version),
        "entity_name": model_name,
        "entity_version": str(version),
        "workload_type": workload_type,
        "workload_size": "Small",
        "scale_to_zero_enabled": scale_to_zero,
    }
    route = {"served_model_name": entity["name"], "traffic_percentage": 100}
    return {"served_entities": [entity], "traffic_config": {"routes": [route]}}


def deploy(do, name: str, config: dict, inference_table: dict | None = None) -> str:
    """Create endpoint `name`, or replace its config (serving keeps the old one until the new one is ready).
    `inference_table` ({"catalog_name", "schema_name", "table_name_prefix"}) applies only at creation; a prefix
    cannot be reused after its endpoint is deleted."""
    existing = {e["name"] for e in do("GET", "/api/2.0/serving-endpoints").get("endpoints", [])}
    if name in existing:
        do("PUT", f"/api/2.0/serving-endpoints/{name}/config", body=config)
        return "updated"
    body = {"name": name, "config": config, "tags": [{"key": "project", "value": "kumo-fraud"}]}
    if inference_table:
        body["ai_gateway"] = {"inference_table_config": {**inference_table, "enabled": True}}
    do("POST", "/api/2.0/serving-endpoints", body=body)
    return "created"


def wait_ready(do, name: str, version: int | None = None, timeout_s: int = 3600, poll_s: int = 30, sleep=time.sleep):
    """Poll until the endpoint is READY, not updating and (if given) serving `version`. A first GPU build takes
    about 20 minutes."""
    waited = 0
    while True:
        ep = do("GET", f"/api/2.0/serving-endpoints/{name}")
        state = ep.get("state", {})
        if state.get("config_update") in ("UPDATE_FAILED", "UPDATE_CANCELED"):
            raise RuntimeError(f"endpoint {name}: {state}")
        served = {e.get("entity_version") for e in ep.get("config", {}).get("served_entities", [])}
        if (
            state.get("ready") == "READY"
            and state.get("config_update") == "NOT_UPDATING"
            and (version is None or str(version) in served)
        ):
            return ep
        if waited >= timeout_s:
            raise TimeoutError(f"endpoint {name} not ready after {timeout_s}s: {state}")
        sleep(poll_s)
        waited += poll_s


def invoker(do, name: str):
    """`invoke(body) -> response` for the endpoint's invocations URL."""
    return lambda body: do("POST", f"/serving-endpoints/{name}/invocations", body=body)


def records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: Python scalars, missing values as None."""
    obj = frame.astype(object)
    return obj.where(frame.notna(), None).to_dict(orient="records")


def request_bytes(rows: list[dict]) -> int:
    return len(json.dumps({"dataframe_records": rows}))


def rows_per_request(sample: list[dict], limit: int = MAX_REQUEST_BYTES, fill: float = 0.5, cap: int = 2000) -> int:
    """Rows per request from the largest row in `sample`, so a request uses at most `fill` of `limit`."""
    per_row = max(len(json.dumps(r)) + 2 for r in sample)
    return max(1, min(cap, math.floor((limit * fill - request_bytes([])) / per_row)))


def score_records(
    invoke, rows: list[dict], chunk: int, workers: int = 4, retries: int = 3, sleep=time.sleep
) -> list[dict]:
    """Predictions for `rows` in their order, `chunk` rows per request on `workers` threads. Each request is
    checked against the payload limit; a failed request is retried with backoff."""

    def call(part: list[dict]) -> list[dict]:
        body = {"dataframe_records": part}
        size = request_bytes(part)
        if size >= MAX_REQUEST_BYTES:
            raise ValueError(f"request of {size} bytes is over the {MAX_REQUEST_BYTES}-byte limit")
        for attempt in range(retries + 1):
            try:
                preds = invoke(body)["predictions"]
                break
            except Exception:
                if attempt == retries:
                    raise
                sleep(2**attempt)
        if len(preds) != len(part):
            raise RuntimeError(f"{len(preds)} predictions for {len(part)} rows")
        return preds

    parts = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [p for preds in pool.map(call, parts) for p in preds]


def timed(invoke, body: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    out = invoke(body)
    return out, (time.perf_counter() - t0) * 1000


def latency_summary(ms) -> dict[str, float]:
    ms = np.asarray(ms, dtype=np.float64)
    return {"n": len(ms), **{f"p{q}": round(float(np.percentile(ms, q)), 1) for q in (50, 95, 99)}}


def ai_query_sql(endpoint: str, source: str) -> str:
    """The same batch in SQL: one `ai_query` call per row, full features in the request."""
    fields = ", ".join(f"'{c}', {c}" for c in [KEY, *FEATURES])
    return (
        f"SELECT {KEY}, r.p_fraud, r.model_version, current_timestamp() AS scored_at FROM (\n"
        f"  SELECT {KEY}, ai_query('{endpoint}', named_struct({fields}),\n"
        f"    returnType => 'STRUCT<{KEY}: BIGINT, p_fraud: DOUBLE, model_version: STRING>') AS r\n"
        f"  FROM {source})"
    )
