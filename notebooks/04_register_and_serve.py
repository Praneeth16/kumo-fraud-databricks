# Databricks notebook source
# MAGIC %md
# MAGIC # 4. Register the model and serve it on a GPU endpoint
# MAGIC
# MAGIC Builds `members` fraud-enriched contexts of `k` rows from the train split, packs them with the
# MAGIC Kumo-Tabular weights into a code-based MLflow model, logs it through Feature Engineering (so requests
# MAGIC carry only `id`), registers it in Unity Catalog with alias `champion` and creates or updates the endpoint.
# MAGIC
# MAGIC `late_frauds` train frauds are held out of these contexts; notebook 5 adds them back as newly confirmed
# MAGIC labels. Needs notebook 1 and the base checkpoint at `<volume>/checkpoints/kumo-tabular-v1.0.1/small/`.
# MAGIC A first GPU build takes about 20 minutes.

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
dbutils.widgets.text("late_frauds", "50", "train frauds held out for notebook 5")
dbutils.widgets.text("seed", "0")
dbutils.widgets.text("version", "v1", "model_version string in responses")
dbutils.widgets.dropdown("workload", "GPU_SMALL", ["GPU_SMALL", "GPU_MEDIUM"])
dbutils.widgets.dropdown("scale_to_zero", "false", ["false", "true"])
dbutils.widgets.text("inference_prefix", "", "inference table prefix (blank: none; never reuse one)")
dbutils.widgets.text("experiment", "", "MLflow experiment (blank: /Users/<you>/kumo-fraud-serving)")

import tempfile

import mlflow
import numpy as np
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient

from kumo_fraud import data, workspace
from kumo_fraud.context import build_contexts
from kumo_fraud.serving import export_artifacts

w = dbutils.widgets.get
names = workspace.Names(w("catalog"), w("schema"), w("volume"))
fe = FeatureEngineeringClient()
do = WorkspaceClient().api_client.do
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(w("experiment") or f"/Users/{spark.sql('SELECT current_user()').first()[0]}/kumo-fraud-serving")
K, MEMBERS, POS_FRAC, SEED = int(w("k")), int(w("members")), float(w("pos_frac")), int(w("seed"))

# COMMAND ----------

frame = data.load(names.raw_dir)
parts = data.split(frame)
y = frame[data.TARGET].to_numpy()
prior = float(y[parts["train"]].mean())
late = workspace.held_out_frauds(y, parts["train"], int(w("late_frauds")), seed=SEED)
contexts = build_contexts(y, np.setdiff1d(parts["train"], late), K, POS_FRAC, MEMBERS, seed=SEED)
print(f"prior {prior:.4f}; {MEMBERS} contexts of {K} rows, {y[contexts[0]].sum()} frauds each; {len(late)} held out")

# COMMAND ----------

weights = None if w("weights") == "frozen" else w("weights")
artifacts = export_artifacts(
    tempfile.mkdtemp(),
    frame,
    contexts,
    pos_frac=POS_FRAC,
    prior=prior,
    weights=weights,
    base_checkpoint=names.checkpoint,
    version=w("version"),
)
version = workspace.log_model(
    fe,
    workspace.training_set(fe, spark, names),
    artifacts,
    names.model,
    run_name=f"kumo-fraud-{w('version')}",
)
mlflow.MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(names.model, "champion", version)
print(f"{names.model} version {version} is @champion")

# COMMAND ----------

config = workspace.endpoint_config(names.model, version, w("workload"), w("scale_to_zero") == "true")
table = None
if w("inference_prefix"):
    table = {"catalog_name": names.catalog, "schema_name": names.schema, "table_name_prefix": w("inference_prefix")}
print(workspace.deploy(do, w("endpoint"), config, inference_table=table))
endpoint = workspace.wait_ready(do, w("endpoint"), version)
print(endpoint["state"], [e["name"] for e in endpoint["config"]["served_entities"]])
