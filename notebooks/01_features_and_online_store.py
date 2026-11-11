# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Feature table, label table and Lakebase online store
# MAGIC
# MAGIC Writes the 24 features of every transaction in `train.csv` and `test.csv` to
# MAGIC `<catalog>.<schema>.transaction_features` (primary key `id`, change data feed on), the labels and
# MAGIC split of the training file to `<catalog>.<schema>.transaction_labels`, creates the online store and
# MAGIC publishes the feature table to it. Run it once; re-running merges the features and replaces the labels.
# MAGIC
# MAGIC Needs `train.csv` and `test.csv` in `<volume>/raw/s6e11/`.

# COMMAND ----------

# MAGIC %pip install -q .. "databricks-feature-engineering>=0.13"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "<catalog>")
dbutils.widgets.text("schema", "<schema>")
dbutils.widgets.text("volume", "", "volume path (blank: /Volumes/<catalog>/<schema>/artifacts)")
dbutils.widgets.text("endpoint", "kumo-fraud")
dbutils.widgets.text("online_store", "kumo-fraud-store")

from databricks.feature_engineering import FeatureEngineeringClient

from kumo_fraud import data, workspace

w = dbutils.widgets.get
names = workspace.Names(w("catalog"), w("schema"), w("volume"))
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md Features of both files, typed by name exactly as `kumo_fraud.data` types them.

# COMMAND ----------


def read(name):
    return spark.read.option("header", True).csv(f"{names.raw_dir}/{name}").selectExpr(*workspace.csv_columns())


features = read("train.csv").unionByName(read("test.csv"))
if spark.catalog.tableExists(names.feature_table):
    fe.write_table(name=names.feature_table, df=features, mode="merge")
else:
    fe.create_table(
        name=names.feature_table,
        primary_keys=[data.KEY],
        df=features,
        description="Card transaction features (Kaggle playground s6e11, train and test)",
    )
spark.sql(f"ALTER TABLE {names.feature_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
display(spark.table(names.feature_table).limit(5))

# COMMAND ----------

# MAGIC %md Labels and the stratified train / val / test split that the model notebooks use.

# COMMAND ----------

frame = data.load(names.raw_dir)
labels = workspace.label_frame(frame, data.split(frame))
spark.sql(workspace.label_table_ddl(names))
spark.createDataFrame(labels).write.mode("append").saveAsTable(names.label_table)
display(spark.table(names.label_table).groupBy("split").agg({"is_fraud": "mean", "id": "count"}))

# COMMAND ----------

# MAGIC %md Online store (a first create takes several minutes), then a triggered publish. The first sync of about
# MAGIC a million rows must finish before notebook 3 sends keys-only requests.

# COMMAND ----------

store = workspace.ensure_online_store(fe, w("online_store"))
fe.publish_table(
    online_store=store,
    source_table_name=names.feature_table,
    online_table_name=names.online_table,
    publish_mode="TRIGGERED",
)
