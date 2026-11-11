# Databricks notebook source
# MAGIC %md
# MAGIC # 0. Setup
# MAGIC
# MAGIC This notebook prepares the volume that the other notebooks read from. It creates the schema and the volume
# MAGIC if they are missing. It downloads the small Kumo-Tabular checkpoint from Hugging Face into the volume. It
# MAGIC then checks that the two Kaggle files are in the volume.
# MAGIC
# MAGIC The catalog must exist already. Run this notebook on serverless compute. It does not need a GPU.

# COMMAND ----------

# MAGIC %pip install -q huggingface_hub

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath("../src"))  # the repo's package, straight from the checked-out folder

from huggingface_hub import hf_hub_download

from kumo_fraud import workspace

P = workspace.params({"catalog": "<catalog>", "schema": "<schema>", "volume": ""}, dbutils=dbutils)
# A blank volume means /Volumes/<catalog>/<schema>/artifacts.
names = workspace.Names(P["catalog"], P["schema"], P["volume"])
print("volume:", names.root)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema and volume

# COMMAND ----------

catalog, schema, volume = names.root.split("/")[2:5]
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume}`")
print(f"{catalog}.{schema}.{volume} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model checkpoint
# MAGIC
# MAGIC The file comes from the Hugging Face repo `nvidia/Kumo-Tabular` at revision `v1.0.1`. It is downloaded to a
# MAGIC local folder first and then copied to the volume as a plain file. The cell does nothing if the file is
# MAGIC already there.

# COMMAND ----------

if os.path.exists(names.checkpoint):
    print("already present:", names.checkpoint)
else:
    with tempfile.TemporaryDirectory() as tmp:
        local = hf_hub_download(
            repo_id="nvidia/Kumo-Tabular", filename=f"{names.size}/classifier.pt", revision="v1.0.1", local_dir=tmp
        )
        os.makedirs(os.path.dirname(names.checkpoint), exist_ok=True)
        shutil.copyfile(local, names.checkpoint)
    print("downloaded:", names.checkpoint)
print(f"{os.path.getsize(names.checkpoint) / 2**20:.1f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kaggle data
# MAGIC
# MAGIC The data cannot be downloaded for you, because Kaggle asks every user to accept the competition rules
# MAGIC first. This cell creates the folder and stops with an error until both files are in it.

# COMMAND ----------

os.makedirs(names.raw_dir, exist_ok=True)
missing = [name for name in ("train.csv", "test.csv") if not os.path.exists(f"{names.raw_dir}/{name}")]
if missing:
    print(
        "Missing: " + ", ".join(missing) + "\n"
        "1. Open https://www.kaggle.com/competitions/playground-series-s6e11 and sign in.\n"
        "2. Accept the competition rules on the Rules tab.\n"
        "3. On the Data tab, download train.csv and test.csv.\n"
        f"4. Upload both files to {names.raw_dir}/. You can upload in Catalog Explorer.\n"
        "5. Run this cell again."
    )
    raise FileNotFoundError(f"{', '.join(missing)} not found in {names.raw_dir}")
print("train.csv and test.csv are in", names.raw_dir)
