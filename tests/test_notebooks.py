import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from kumo_fraud import workspace

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.environ.get("KF_DATA_DIR", "/tmp/kaggle_s6e11")
CHECKPOINT = os.environ.get("KF_CHECKPOINT", "/tmp/gk_vol/checkpoints/kumo-tabular-v1.0.1/small/classifier.pt")
DEFAULTS = {"catalog": "<catalog>", "steps": "2000"}


class FakeWidgets:
    def __init__(self, values):
        self.values, self.created = dict(values), {}

    def text(self, name, default):
        self.created[name] = default
        self.values.setdefault(name, default)

    def get(self, name):
        return self.values[name]


class FakeDbutils:
    def __init__(self, values):
        self.widgets = FakeWidgets(values)


def test_params_reads_the_environment(monkeypatch):
    monkeypatch.setenv("CATALOG", "main")
    monkeypatch.delenv("STEPS", raising=False)
    assert workspace.params(DEFAULTS) == {"catalog": "main", "steps": "2000"}


def test_params_falls_back_to_the_defaults(monkeypatch):
    monkeypatch.delenv("CATALOG", raising=False)
    monkeypatch.delenv("STEPS", raising=False)
    assert workspace.params(DEFAULTS) == DEFAULTS


def test_params_uses_widgets_when_dbutils_is_given(monkeypatch):
    monkeypatch.setenv("CATALOG", "ignored")
    dbutils = FakeDbutils({"steps": "10"})
    assert workspace.params(DEFAULTS, dbutils=dbutils) == {"catalog": "<catalog>", "steps": "10"}
    assert dbutils.widgets.created == DEFAULTS


def run(notebook: str, env: dict, cwd: Path) -> str:
    """Run a notebook as a plain script from another folder and without PYTHONPATH, so it must find `src` itself.
    LIGHTGBM=0 because lightgbm and torch in one process can crash on macOS."""
    base = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    base.update(DATA_DIR=DATA_DIR, CHECKPOINT=CHECKPOINT, LIGHTGBM="0", MPLBACKEND="Agg")
    out = subprocess.run(
        [sys.executable, str(ROOT / "notebooks" / notebook)],
        env={**base, **env},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(DATA_DIR, "train.csv")) and os.path.exists(CHECKPOINT)),
    reason="local data or checkpoint missing (set KF_DATA_DIR and KF_CHECKPOINT)",
)
def test_finetune_and_evaluate_notebooks_run_as_scripts(tmp_path):
    tiny = {"VAL_ROWS": "400", "CONTEXT_SIZES": "512,1024"}
    run(
        "02_finetune.py",
        {**tiny, "OUT_DIR": str(tmp_path / "ft"), "STEPS": "2", "K_CHOICES": "64,128", "Q": "32", "WARMUP": "1"},
        tmp_path,
    )
    finetuned = json.loads((tmp_path / "ft" / "summary.json").read_text())
    assert (tmp_path / "ft" / "finetuned.pt").exists()
    assert [h["step"] for h in finetuned["history"]] == [0, 2]
    rows = ["1 x 512", "8 x 512", "1 x 1,024", "8 x 1,024"]
    assert list(finetuned["released"]) == list(finetuned["finetuned"]) == rows

    weights = str(tmp_path / "ft" / "finetuned.pt")
    env = {**tiny, "OUT_DIR": str(tmp_path / "ev"), "WEIGHTS": weights, "TEST_ROWS": "400", "KAGGLE_ROWS": "300"}
    stdout = run("03_evaluate.py", env, tmp_path)
    evaluated = json.loads((tmp_path / "ev" / "summary.json").read_text())
    settings = ["random 8 x 512", "enriched 1 x 512", "enriched 8 x 512", "enriched 8 x 1,024"]
    assert list(evaluated["results"]) == [f"{m} | {s}" for m in ("released", "fine-tuned") for s in settings]
    assert [row["fraud_share"] for row in evaluated["fraud_share_sweep"]] == ["natural", 0.05, 0.2, 0.3, 0.5, 0.8]
    assert evaluated["lightgbm"] is None and "Skipped: lightgbm is 0." in stdout
    submission = pd.read_csv(tmp_path / "ev" / "submission.csv")
    assert list(submission.columns) == ["id", "is_fraud"] and len(submission) == 300
    assert submission["is_fraud"].between(0, 1).all()
