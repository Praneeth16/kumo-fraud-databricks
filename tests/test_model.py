import os

import numpy as np
import pytest
import torch

from kumo_fraud import data
from kumo_fraud.context import enriched_context
from kumo_fraud.finetune import FinetuneConfig, finetune
from kumo_fraud.scorer import Scorer, load_model

DATA_DIR = os.environ.get("KF_DATA_DIR", "/tmp/kaggle_s6e11")
CHECKPOINT = os.environ.get("KF_CHECKPOINT", "/tmp/gk_vol/checkpoints/kumo-tabular-v1.0.1/small/classifier.pt")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(DATA_DIR, "train.csv")) and os.path.exists(CHECKPOINT)),
    reason="local data or checkpoint missing (set KF_DATA_DIR and KF_CHECKPOINT)",
)


@pytest.fixture(scope="module")
def setup():
    frame = data.load(DATA_DIR).iloc[:50_000].reset_index(drop=True)
    parts = data.split(frame)
    scorer = Scorer(frame, load_model(checkpoint=CHECKPOINT))
    return frame, parts, scorer


def test_scorer_is_deterministic_and_encode_queries_matches(setup):
    frame, parts, scorer = setup
    ctx = enriched_context(scorer.labels, parts["train"], 256, 0.3, np.random.default_rng(0))
    q = parts["val"][:200]
    a = scorer.predict(ctx, q, view=2)
    assert np.array_equal(a, scorer.predict(ctx, q, view=2))
    assert ((a > 0) & (a < 1)).all()
    other = frame.iloc[q].drop(columns=[data.TARGET]).reset_index(drop=True)
    b = scorer.predict(ctx, scorer.encode_queries(other), view=2)
    np.testing.assert_allclose(b, a, atol=1e-5)
    s = scorer.score([ctx, ctx], q, 0.3, 0.0167)
    assert s.shape == (len(q),) and np.isfinite(s).all()


def test_scorer_takes_written_text_features_as_extra_columns(setup):
    from kumo_fraud import features

    frame, parts, scorer = setup
    text = features.load_written(DATA_DIR).iloc[: len(frame)].reset_index(drop=True)
    rates = features.encode(features.key_codes(text), scorer.labels, parts["train"])
    extra = ("rate_amount_usd_frac", "rate_merchant_risk_score_value")
    wide = Scorer(frame.join(rates[list(extra)]), scorer.model, extra=extra)
    ctx = enriched_context(scorer.labels, parts["train"], 256, 0.3, np.random.default_rng(0))
    q = parts["val"][:200]
    a = wide.predict(ctx, q, view=1)
    assert ((a > 0) & (a < 1)).all() and not np.allclose(a, scorer.predict(ctx, q, view=1))
    other = frame.join(rates[list(extra)]).iloc[q].drop(columns=[data.TARGET]).reset_index(drop=True)
    np.testing.assert_allclose(wide.predict(ctx, wide.encode_queries(other), view=1), a, atol=1e-5)


def test_three_finetune_steps_change_weights(setup):
    _, parts, scorer = setup
    net = scorer.model.models["classification"]
    before = {k: v.clone() for k, v in net.state_dict().items()}
    cfg = FinetuneConfig(
        steps=3, lr=1e-4, k_choices=(64, 128), q=32, warmup=1, eval_every=3, eval_rows=200, eval_k=128, eval_members=1
    )
    try:
        history, best = finetune(scorer, parts["train"], parts["val"], cfg, log=lambda _: None)
        assert not scorer.model.training
        assert [h["step"] for h in history] == [0, 3]
        assert np.isfinite(history[-1]["train_loss"])
        changed = [k for k, v in net.state_dict().items() if not torch.equal(v, before[k])]
        assert len(changed) > 0
        assert best.keys() == before.keys()
    finally:
        net.load_state_dict(before)
