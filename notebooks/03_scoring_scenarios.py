# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Three scoring scenarios on one endpoint
# MAGIC
# MAGIC (a) real time with keys only, (b) a batch upload scored in chunks with full feature rows, (c) a context
# MAGIC refresh: newly confirmed fraud labels go into the contexts with no training, as a new model version.
# MAGIC Needs notebooks 1 and 2. Keep `k`, `members`, `pos_frac`, `late_frauds`, `seed` and `weights` equal to
# MAGIC notebook 2.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.13"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))  # the repo's package, straight from the checked-out folder

dbutils.widgets.text("catalog", "<catalog>")
dbutils.widgets.text("schema", "<schema>")
dbutils.widgets.text("volume", "", "volume path (blank: /Volumes/<catalog>/<schema>/artifacts)")
dbutils.widgets.text("endpoint", "kumo-fraud")
dbutils.widgets.text("online_store", "kumo-fraud-store")
dbutils.widgets.text("weights", "frozen", "weights: frozen, or a finetuned.pt path on the volume")
dbutils.widgets.text("k", "1024", "context rows per member")
dbutils.widgets.text("members", "4")
dbutils.widgets.text("pos_frac", "0.3", "fraud share of each context")
dbutils.widgets.text("late_frauds", "50", "train frauds held out in notebook 2")
dbutils.widgets.text("seed", "0")
dbutils.widgets.text("refresh_version", "v2-refresh", "model_version string of the refreshed model")
dbutils.widgets.dropdown("workload", "GPU_SMALL", ["GPU_SMALL", "GPU_MEDIUM"])
dbutils.widgets.dropdown("scale_to_zero", "false", ["false", "true"])
dbutils.widgets.text("batch_rows", "20000", "rows of test.csv in the simulated upload")
dbutils.widgets.text("workers", "4", "concurrent batch requests")
dbutils.widgets.text("experiment", "", "MLflow experiment (blank: /Users/<you>/kumo-fraud-serving)")

import os
import tempfile

import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient

from kumo_fraud import data, workspace
from kumo_fraud.context import build_contexts
from kumo_fraud.serving import export_artifacts, normalize

w = dbutils.widgets.get
names = workspace.Names(w("catalog"), w("schema"), w("volume"))
invoke = workspace.invoker(WorkspaceClient().api_client.do, w("endpoint"))
KEY = data.KEY

# COMMAND ----------

# MAGIC %md ## (a) Real time: keys only
# MAGIC Features come from the online store; a request is `{"dataframe_records": [{"id": ...}]}`. Latency is
# MAGIC measured from this notebook, so it includes the network and the serving proxy.

# COMMAND ----------

test = data.load_test(names.raw_dir)
ids = test[KEY].head(20).tolist()
invoke({"dataframe_records": [{KEY: ids[0]}]})  # warm-up
latencies, rows = [], []
for i in ids:
    out, ms = workspace.timed(invoke, {"dataframe_records": [{KEY: int(i)}]})
    latencies.append(ms)
    rows += out["predictions"]
print("batch-1 latency ms", workspace.latency_summary(latencies))
print("request bytes per id", workspace.request_bytes([{KEY: int(ids[0])}]))
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md ## (b) Batch upload: full feature rows in chunks
# MAGIC A CSV of new transactions lands in the volume (here: rows of `test.csv`). Each request carries full rows,
# MAGIC whose values override the lookup, so the rows need not be in the feature table yet. The chunk size comes
# MAGIC from the measured bytes per row and every request is checked against the 16 MB limit.

# COMMAND ----------

landing = f"{names.landing_dir}/new_transactions.csv"
os.makedirs(names.landing_dir, exist_ok=True)
pd.read_csv(f"{names.raw_dir}/test.csv", nrows=int(w("batch_rows"))).to_csv(landing, index=False)

new = normalize(pd.read_csv(landing))
records = workspace.records(new)
chunk = workspace.rows_per_request(records[:1000])
print(f"{len(records)} rows, {chunk} rows per request, first request {workspace.request_bytes(records[:chunk])} bytes")
preds = workspace.score_records(invoke, records, chunk, workers=int(w("workers")))
scores = pd.DataFrame(preds)[[KEY, "p_fraud", "model_version"]].assign(scored_at=pd.Timestamp.now(tz="UTC"))
assert scores[KEY].tolist() == new[KEY].tolist()
spark.createDataFrame(scores).write.mode("append").saveAsTable(names.scores_table)
display(spark.table(names.scores_table).orderBy("scored_at", ascending=False).limit(10))

# COMMAND ----------

# MAGIC %md Full rows and keys only give the same score for transactions that are in the online store.

# COMMAND ----------

keys_only = workspace.score_records(invoke, [{KEY: r[KEY]} for r in records[:5]], chunk=5)
print([round(a["p_fraud"] - b["p_fraud"], 9) for a, b in zip(keys_only, preds[:5], strict=True)])

# COMMAND ----------

# MAGIC %md Alternative: the same batch from SQL with `ai_query` (one call per row; shown on 100 rows).

# COMMAND ----------

spark.createDataFrame(new.head(100)).createOrReplaceTempView("new_transactions")
display(spark.sql(workspace.ai_query_sql(w("endpoint"), "new_transactions")))

# COMMAND ----------

# MAGIC %md ## (c) Context refresh: newly confirmed frauds, no training
# MAGIC The frauds held out in notebook 2 are confirmed now. They go into every context member (the fraud share
# MAGIC stays at `pos_frac`, so the prior correction is unchanged), the weights stay as they are, and the result
# MAGIC is logged as a new version. The endpoint keeps serving the old version until the new one is ready
# MAGIC (about 20 minutes on GPU).

# COMMAND ----------

fe = FeatureEngineeringClient()
client = mlflow.MlflowClient(registry_uri="databricks-uc")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(w("experiment") or f"/Users/{spark.sql('SELECT current_user()').first()[0]}/kumo-fraud-serving")
K, MEMBERS, POS_FRAC, SEED = int(w("k")), int(w("members")), float(w("pos_frac")), int(w("seed"))

frame = data.load(names.raw_dir)
parts = data.split(frame)
y = frame[data.TARGET].to_numpy()
prior = float(y[parts["train"]].mean())
late = workspace.held_out_frauds(y, parts["train"], int(w("late_frauds")), seed=SEED)
contexts = build_contexts(y, parts["train"], K, POS_FRAC, MEMBERS, seed=SEED, include=late)
affected = [{KEY: int(i)} for i in frame[KEY].iloc[late[:8]]]
before = pd.DataFrame(workspace.score_records(invoke, affected, chunk=len(affected)))

# COMMAND ----------

previous = int(client.get_model_version_by_alias(names.model, "champion").version)
artifacts = export_artifacts(
    tempfile.mkdtemp(),
    frame,
    contexts,
    pos_frac=POS_FRAC,
    prior=prior,
    weights=None if w("weights") == "frozen" else w("weights"),
    base_checkpoint=names.checkpoint,
    version=w("refresh_version"),
)
version = workspace.log_model(
    fe,
    workspace.training_set(fe, spark, names),
    artifacts,
    names.model,
    run_name=f"kumo-fraud-{w('refresh_version')}",
)
client.set_registered_model_alias(names.model, "previous", previous)
client.set_registered_model_alias(names.model, "champion", version)
do = WorkspaceClient().api_client.do
config = workspace.endpoint_config(names.model, version, w("workload"), w("scale_to_zero") == "true")
print(workspace.deploy(do, w("endpoint"), config))
workspace.wait_ready(do, w("endpoint"), version)

# COMMAND ----------

after = pd.DataFrame(workspace.score_records(invoke, affected, chunk=len(affected)))
display(before.merge(after, on=KEY, suffixes=("_before", "_after")))
# Roll back: workspace.deploy(do, w("endpoint"), workspace.endpoint_config(names.model, previous, w("workload")))
