import json
import os
import threading
import time
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from kumo_fraud import data, workspace
from kumo_fraud.context import build_contexts
from kumo_fraud.serving import FEATURES, export_artifacts, normalize

DATA_DIR = os.environ.get("KF_DATA_DIR", "/tmp/kaggle_s6e11")
CHECKPOINT = os.environ.get("KF_CHECKPOINT", "/tmp/gk_vol/checkpoints/kumo-tabular-v1.0.1/small/classifier.pt")
needs_model = pytest.mark.skipif(
    not (os.path.exists(os.path.join(DATA_DIR, "train.csv")) and os.path.exists(CHECKPOINT)),
    reason="local data or checkpoint missing (set KF_DATA_DIR and KF_CHECKPOINT)",
)


def synthetic(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({data.KEY: np.arange(n, dtype=np.int64) + 1000})
    for c in data.NUMERIC:
        frame[c] = rng.normal(size=n).round(2)
    frame["txn_count_last_24h"] = rng.integers(0, 30, size=n).astype(np.float64)
    for c in data.CATEGORICAL:
        frame[c] = rng.choice(["True", "False"], size=n)
    frame[data.TARGET] = (rng.random(n) < 0.2).astype(np.int64)
    return frame


def messy(frame: pd.DataFrame) -> pd.DataFrame:
    """The same rows as a client might send them: shuffled columns, extras, numbers as str/int/Decimal, flags
    as bools or lower-case strings."""
    out = frame[[data.KEY, *FEATURES]].copy()
    out["amount_usd"] = out["amount_usd"].map(str)
    out["velocity_score"] = out["velocity_score"].map(lambda v: Decimal(repr(v)))
    out["txn_count_last_24h"] = out["txn_count_last_24h"].astype(np.int64)
    out["used_vpn"] = out["used_vpn"] == "True"
    out["is_new_merchant"] = out["is_new_merchant"].str.lower()
    out["channel"] = " " + out["channel"] + " "
    out["extra"] = "ignored"
    return out[list(reversed(out.columns))]


def test_build_contexts_includes_and_keeps_fraud_share():
    y = (np.random.default_rng(1).random(5_000) < 0.05).astype(np.int64)
    rows = np.arange(4_000)
    late = workspace.held_out_frauds(y, rows, 20, seed=2)
    assert len(late) == 20 and y[late].all() and np.isin(late, rows).all()
    first = build_contexts(y, np.setdiff1d(rows, late), 200, 0.3, 3, seed=0)
    again = build_contexts(y, rows, 200, 0.3, 3, seed=0, include=late)
    for a, b in zip(first, again, strict=True):
        assert len(a) == len(b) == len(np.unique(b)) == 200
        assert y[a].sum() == y[b].sum() == 60
        assert not np.isin(late, a).any() and np.isin(late, b).all()
    assert not np.array_equal(first[0], first[1])
    with pytest.raises(ValueError):
        build_contexts(y, rows, 200, 0.3, 1, include=np.flatnonzero(y == 0)[:3])


def test_export_artifacts_round_trip(tmp_path):
    frame = synthetic()
    contexts = [np.array([3, 1, 7]), np.array([2, 5])]
    src = tmp_path / "src.pt"
    src.write_bytes(b"weights")
    link = tmp_path / "link.pt"
    link.symlink_to(src)
    arts = export_artifacts(
        str(tmp_path / "out"), frame, contexts, pos_frac=0.3, prior=0.02, weights=str(link), version="v9"
    )
    assert set(arts) == {"contexts", "meta", "weights"}
    assert not os.path.islink(arts["weights"]) and open(arts["weights"], "rb").read() == b"weights"
    rows = pd.read_parquet(arts["contexts"])
    assert list(rows.columns) == [*FEATURES, data.TARGET, "member"]
    for i, ctx in enumerate(contexts):
        got = rows[rows.member == i].drop(columns="member").reset_index(drop=True)
        pd.testing.assert_frame_equal(got, frame.iloc[ctx][[*FEATURES, data.TARGET]].reset_index(drop=True))
    meta = json.load(open(arts["meta"]))
    assert meta["n_members"] == 2 and meta["version"] == "v9" and meta["pos_frac"] == 0.3
    assert meta["numeric"] == data.NUMERIC and meta["categorical"] == data.CATEGORICAL


def test_normalize_is_robust_to_client_formats():
    frame = synthetic(20)
    clean = normalize(frame)
    assert list(clean.columns) == [data.KEY, *FEATURES]
    pd.testing.assert_frame_equal(normalize(messy(frame)), clean)
    holes = frame.astype(object)
    holes.loc[0, "amount_usd"] = None
    holes.loc[1, "amount_usd"] = "not a number"
    holes.loc[2, "card_type"] = np.nan
    holes = holes.drop(columns=["prior_disputes"])
    out = normalize(holes)
    assert np.isnan(out.loc[:1, "amount_usd"]).all() and pd.isna(out.loc[2, "card_type"])
    assert out["prior_disputes"].isna().all() and out["prior_disputes"].dtype == np.float64
    assert normalize(frame.drop(columns=data.KEY))[data.KEY].isna().all()


def test_rows_per_request_keeps_requests_under_the_limit():
    rows = workspace.records(normalize(synthetic(50)))
    json.dumps(rows)  # JSON-safe
    big = [{**r, "merchant_category": "x" * 3000} for r in rows]
    for sample, limit in ((rows, workspace.MAX_REQUEST_BYTES), (big, 200_000)):
        n = workspace.rows_per_request(sample, limit=limit, fill=0.5, cap=10**9)
        request = [sample[i % len(sample)] for i in range(n)]
        assert 0.8 * limit * 0.5 < workspace.request_bytes(request) <= limit * 0.5
    assert workspace.rows_per_request(rows) == 2000


def test_score_records_keeps_order_and_retries():
    rows = [{data.KEY: i} for i in range(1_003)]
    failed, lock, sleeps = set(), threading.Lock(), []

    def invoke(body):
        part = body["dataframe_records"]
        first = part[0][data.KEY]
        with lock:
            if first % 200 == 0 and first not in failed:
                failed.add(first)
                raise ConnectionError("503")
        time.sleep(np.random.default_rng(first).random() / 100)
        return {"predictions": [{data.KEY: r[data.KEY], "p_fraud": r[data.KEY] / 1e4} for r in part]}

    preds = workspace.score_records(invoke, rows, chunk=100, workers=4, sleep=sleeps.append)
    assert [p[data.KEY] for p in preds] == list(range(1_003))
    assert failed == {0, 200, 400, 600, 800, 1000} and sleeps == [1] * 6

    def broken(body):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        workspace.score_records(broken, rows[:5], chunk=5, retries=2, sleep=sleeps.append)
    with pytest.raises(ValueError, match="limit"):
        workspace.score_records(invoke, [{"x": "y" * workspace.MAX_REQUEST_BYTES}], chunk=1)


def test_deploy_and_wait_with_fake_http():
    calls, states = [], iter(["IN_PROGRESS", "IN_PROGRESS", "NOT_UPDATING"])

    def do(method, path, body=None):
        calls.append((method, path, body))
        if path == "/api/2.0/serving-endpoints":
            return {"endpoints": [{"name": "old"}]} if method == "GET" else {}
        if method == "GET":
            update = next(states)
            return {
                "state": {"ready": "READY", "config_update": update},
                "config": {"served_entities": [{"entity_version": "3" if update == "NOT_UPDATING" else "2"}]},
            }
        return {}

    config = workspace.endpoint_config("c.s.kumo_fraud", 3, scale_to_zero=True)
    assert config["served_entities"][0]["workload_type"] == "GPU_SMALL"
    assert config["traffic_config"]["routes"] == [{"served_model_name": "kumo-fraud-3", "traffic_percentage": 100}]
    table = {"catalog_name": "c", "schema_name": "s", "table_name_prefix": "kumo_fraud_v1"}
    assert workspace.deploy(do, "new", config, inference_table=table) == "created"
    post = calls[-1]
    assert post[:2] == ("POST", "/api/2.0/serving-endpoints") and post[2]["config"] == config
    assert post[2]["ai_gateway"]["inference_table_config"] == {**table, "enabled": True}
    assert workspace.deploy(do, "old", config) == "updated"
    assert calls[-1] == ("PUT", "/api/2.0/serving-endpoints/old/config", config)
    sleeps = []
    workspace.wait_ready(do, "old", version=3, poll_s=5, sleep=sleeps.append)
    assert sleeps == [5, 5]


def test_names_and_sql():
    n = workspace.Names("c", "s")
    assert n.feature_table == "c.s.transaction_features" and n.raw_dir == "/Volumes/c/s/artifacts/raw/s6e11"
    assert workspace.Names("c", "s", "/Volumes/x/y/z/").checkpoint.startswith("/Volumes/x/y/z/checkpoints/")
    sql = workspace.ai_query_sql("kumo-fraud", "new_transactions")
    assert all(f"'{c}', {c}" in sql for c in [data.KEY, *FEATURES]) and "ai_query('kumo-fraud'" in sql
    assert len(workspace.csv_columns()) == len(FEATURES) + 1
    assert "PRIMARY KEY (id)" in workspace.label_table_ddl(n)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    import mlflow
    import torch

    from kumo_fraud import serving
    from kumo_fraud.scorer import Scorer, load_model

    frame = data.load(DATA_DIR).iloc[:50_000].reset_index(drop=True)
    parts = data.split(frame)
    y = frame[data.TARGET].to_numpy()
    prior = float(y[parts["train"]].mean())
    contexts = build_contexts(y, parts["train"], 256, 0.3, 3, seed=0)
    tmp = tmp_path_factory.mktemp("serving")
    tuned = load_model(checkpoint=CHECKPOINT)
    net = tuned.models["classification"]
    with torch.no_grad():
        next(p for p in net.parameters() if p.dim() > 1).mul_(1.05)
    torch.save({"state": net.state_dict()}, tmp / "tuned.pt")

    models = {}
    for name, weights in (("frozen", None), ("tuned", str(tmp / "tuned.pt"))):
        arts = export_artifacts(
            str(tmp / name), frame, contexts, pos_frac=0.3, prior=prior, weights=weights, base_checkpoint=CHECKPOINT
        )
        mlflow.pyfunc.save_model(
            path=str(tmp / f"{name}-model"),
            python_model=serving.__file__,
            artifacts=arts,
            code_paths=[os.path.dirname(serving.__file__)],
            signature=workspace.signature(),
        )
        models[name] = mlflow.pyfunc.load_model(str(tmp / f"{name}-model"))
    scorers = {
        "frozen": Scorer(frame, load_model(checkpoint=CHECKPOINT)),
        "tuned": Scorer(frame, load_model(checkpoint=CHECKPOINT, weights=str(tmp / "tuned.pt"))),
    }
    return frame, parts, contexts, prior, models, scorers


@needs_model
@pytest.mark.parametrize("name", ["frozen", "tuned"])
def test_served_model_matches_scorer(served, name):
    frame, parts, contexts, prior, models, scorers = served
    q = parts["val"][:300]
    queries = frame.iloc[q].drop(columns=data.TARGET).reset_index(drop=True)
    out = models[name].predict(queries)
    assert list(out.columns) == [data.KEY, "p_fraud", "model_version"]
    assert out[data.KEY].tolist() == queries[data.KEY].tolist() and (out.model_version == "v1").all()
    expected = scorers[name].score(contexts, q, 0.3, prior)
    np.testing.assert_allclose(out.p_fraud.to_numpy(), expected, atol=1e-5, rtol=0)


@needs_model
def test_served_model_robustness_and_serving_requests(served):
    from mlflow.pyfunc.scoring_server import invocations

    frame, parts, _, _, models, _ = served
    model = models["frozen"]
    queries = frame.iloc[parts["val"][:40]].drop(columns=data.TARGET).reset_index(drop=True)
    clean = model.predict(queries).p_fraud.to_numpy()
    raw = model.unwrap_python_model()
    np.testing.assert_allclose(raw.predict(None, messy(queries)).p_fraud, clean, atol=1e-12, rtol=0)

    holes = queries.copy()
    holes.loc[:9, "amount_usd"] = np.nan
    holes.loc[5:14, "merchant_category"] = None
    p = raw.predict(None, holes).p_fraud.to_numpy()
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p[15:], clean[15:], atol=1e-6, rtol=0)
    assert len(raw.predict(None, queries.iloc[:0])) == 0

    def serve(rows):
        body = json.dumps({"dataframe_records": rows})
        res = invocations(body, "application/json", model, model.metadata.get_input_schema())
        assert res.status == 200, res.response
        return pd.DataFrame(json.loads(res.response)["predictions"])

    full = serve(workspace.records(queries))
    np.testing.assert_allclose(full.p_fraud, clean, atol=1e-12, rtol=0)
    assert full[data.KEY].tolist() == queries[data.KEY].tolist()
    keys_only = serve([{data.KEY: int(i)} for i in queries[data.KEY][:3]])
    assert np.isfinite(keys_only.p_fraud).all() and len(keys_only) == 3
