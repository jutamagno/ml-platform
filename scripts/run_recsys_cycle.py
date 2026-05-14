"""
Integration demo: recsys-adtech simulator → ml-platform lifecycle.

Generates auction events using the recsys-adtech user/item pools and
auction simulator, ingests them through ml-platform, trains a PointwiseScorer
(LightGBM, 114-dim features) and a DeepFM CTR model, deploys both through
the staged rollout, and demonstrates automated rollback.
"""
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── recsys-adtech paths ────────────────────────────────────────────────────────
_RECSYS = Path(__file__).parent.parent.parent / "recsys-adtech"
for _p in [str(_RECSYS), str(_RECSYS / "05-adtech" / "src"), str(_RECSYS / "02-ranking" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING)

from shared.data.generators import make_item_pool, make_user_pool
from shared.simulator.auction import simulate_impression

from src.config import DeploymentConfig, FeatureStoreConfig, RollbackConfig, TriggerConfig
from src.deployment.engine import DeploymentEngine
from src.deployment.monitor import PredictionLog, RollbackMonitor
from src.feature_store.skew_detector import SkewDetector
from src.feature_store.store import FeatureStore
from src.ingestor.bridge import auction_to_events, build_feature_row
from src.ingestor.ingestor import EventIngestor
from src.registry.registry import ModelRegistry
from src.training.recsys_adapters import DeepFMCTRAdapter, PointwiseScorerAdapter
from src.training.trigger import TrainingTrigger

import fakeredis

# ── Config ─────────────────────────────────────────────────────────────────────
PARQUET_ROOT = "/tmp/ml-platform-recsys/features"
REGISTRY_ROOT = "/tmp/ml-platform-recsys/registry"
N_USERS = 500
N_ITEMS = 200
TRIGGER_THRESHOLD = 2_000
MODEL_NAME = "ctr_model"
rng = np.random.default_rng(42)


def banner(tag: str, msg: str) -> None:
    print(f"[{tag:<12}]  {msg}")


# ── Setup ──────────────────────────────────────────────────────────────────────
users = make_user_pool(n=N_USERS, seed=42)
items = make_item_pool(n=N_ITEMS, seed=42)

redis_client = fakeredis.FakeRedis()
store = FeatureStore(
    redis_client=redis_client,
    config=FeatureStoreConfig(parquet_root=PARQUET_ROOT),
)
ingestor = EventIngestor(feature_store=store)
trigger = TrainingTrigger(config=TriggerConfig(event_count_threshold=TRIGGER_THRESHOLD))
registry = ModelRegistry(root=REGISTRY_ROOT)
deploy_cfg = DeploymentConfig(min_shadow_hours=0.0, min_canary_hours=0.0)
engine = DeploymentEngine(registry=registry, config=deploy_cfg)
skew = SkewDetector()


# ── Auction simulation ─────────────────────────────────────────────────────────
from shared.simulator.auction import Bid, run_auction

def simulate_batch(n_auctions: int, start_ts: datetime) -> tuple[list[dict], list[dict]]:
    """
    Run n_auctions and return (event_dicts, feature_rows).
    Hour for feature extraction is derived from the event timestamp so that
    the hour_sin/hour_cos features have realistic within-batch variation.
    """
    feature_rows: list[dict] = []
    event_dicts: list[dict] = []

    for i in range(n_auctions):
        user = users[int(rng.integers(0, len(users)))]
        item = items[int(rng.integers(0, len(items)))]
        ts = start_ts + timedelta(seconds=i)

        bid = Bid(bidder_id=item.item_id, amount=rng.uniform(0.1, 2.0))
        result = run_auction([bid], floor_price=item.floor_price, rng=rng)
        if result is None:
            continue
        winner_id, clearing_price = result

        auction_result = simulate_impression(user, item, clearing_price, rng)

        for ev in auction_to_events(user, item, auction_result, occurred_at=ts):
            event_dicts.append(ev)

        if auction_result.impression:
            # Use actual event hour — features must reflect when the impression happened
            row = build_feature_row(user, item, hour=ts.hour)
            row.update({
                "user_id": user.user_id,
                "item_id": item.item_id,
                "label": int(auction_result.click),
                "confirmed": True,
                "event_type": "click" if auction_result.click else "impression",
                "timestamp": ts.isoformat(),
                "event_id": str(uuid.uuid4()),
            })
            feature_rows.append(row)

    return event_dicts, feature_rows


def train_and_register(
    adapter,
    feature_rows: list[dict],
    trigger_reason: str,
    version_label: str,
) -> tuple[str, float]:
    """Build training DataFrame from feature rows and register the model."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame(feature_rows)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No feature columns found — bridge may have failed")

    X = df[feat_cols].fillna(0.0)
    y = df["label"]

    if len(y.unique()) < 2:
        raise ValueError("Training labels are all one class — not enough variation")

    # stratify=y preserves click rate in both splits; critical with ~5% positives
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    adapter.fit(X_tr, y_tr)
    preds = adapter.predict_proba(X_val)
    auc = float(roc_auc_score(y_val, preds))

    version = registry.register(
        model_name=MODEL_NAME,
        adapter=adapter,
        metrics={"auc": round(auc, 4)},
        meta={
            "trigger_reason": trigger_reason,
            "version_label": version_label,
            "n_examples": len(df),
            "n_features": len(feat_cols),
        },
    )
    return version, auc


# ── Main ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  recsys-adtech → ml-platform Integration Demo")
print("=" * 65 + "\n")

# Phase 1: Simulate 10k auctions, ingest events
t0 = datetime(2026, 5, 14, 8, 0, 0)
banner("SIMULATOR", f"Running 10,000 auctions ({N_USERS} users × {N_ITEMS} items)…")
event_dicts, feature_rows_v1 = simulate_batch(10_000, start_ts=t0)

ok = sum(1 for ev in event_dicts if ingestor.process(ev).value == "ok")
banner("INGESTOR", f"{ok}/{len(event_dicts)} events ingested; {len(ingestor.dlq)} dead-letter")

# Record feature distribution for skew detection
feat_cols = [c for c in feature_rows_v1[0] if c.startswith("feat_")] if feature_rows_v1 else []
if feat_cols:
    import pandas as pd
    df_v1 = pd.DataFrame(feature_rows_v1)
    # feat_112 = hour_sin = sin(2π·hour/24): captures temporal distribution
    # feat_0 (segment indicator) is hour-independent — wrong signal for drift
    skew.record_training("feat_112_hour_sin", df_v1["feat_112"].values)

# Phase 2: Trigger training → PointwiseScorer (LightGBM, 114-dim)
trigger.record_confirmed_examples(len([r for r in feature_rows_v1 if r.get("confirmed")]))
should, reason = trigger.should_trigger()
banner("TRIGGER", f"Fired: {reason}")
trigger.mark_triggered()

banner("TRAINING", "Training PointwiseScorerAdapter (LightGBM, 114-dim recsys features)…")
v1, auc1 = train_and_register(
    PointwiseScorerAdapter(n_rounds=200),
    feature_rows_v1,
    trigger_reason=reason,
    version_label="lgbm_v1",
)
banner("TRAINING", f"v1 (LightGBM): AUC={auc1:.4f}, n={len(feature_rows_v1)} examples, 114 features")
trigger.mark_complete()

# Phase 3: Deploy v1 SHADOW → CANARY → FULL
engine.promote(MODEL_NAME, v1)
banner("DEPLOY", "v1: SHADOW → evaluating…")
engine.advance(MODEL_NAME)
banner("DEPLOY", "v1: CANARY (10%)…")
engine.advance(MODEL_NAME)
banner("DEPLOY", "v1: FULL")
registry.promote(MODEL_NAME, v1, "full")

# Phase 4: Simulate drift — different hour (prime-time shift)
t1 = t0 + timedelta(hours=12)
banner("SIMULATOR", "Running 5,000 auctions with prime-time distribution (20:00–21:23)…")
_, feature_rows_v2_raw = simulate_batch(5_000, start_ts=t1)

if feature_rows_v2_raw:
    import pandas as pd
    df_v2 = pd.DataFrame(feature_rows_v2_raw)
    skew.record_serving("feat_112_hour_sin", df_v2["feat_112"].values)
    skew_report = skew.report()
    psi = skew_report.get("feat_112_hour_sin", 0.0)
    label = "WARNING" if psi > 0.2 else "OK"
    # Expected: high PSI because sin(2π×10/24)≈0.50 vs sin(2π×20/24)≈-0.87
    banner("SKEW", f"hour_sin PSI={psi:.3f} ({label})")

# Phase 5: Train v2 — DeepFM CTR model on drifted distribution
trigger.record_confirmed_examples(len(feature_rows_v2_raw))
should, reason = trigger.should_trigger()
banner("TRIGGER", f"Fired: {reason}")
trigger.mark_triggered()

banner("TRAINING", "Training DeepFMCTRAdapter (PyTorch, 114-dim recsys features)…")
feature_rows_v2 = feature_rows_v1 + feature_rows_v2_raw  # combined window
v2, auc2 = train_and_register(
    DeepFMCTRAdapter(n_epochs=5, batch_size=256),
    feature_rows_v2,
    trigger_reason=reason,
    version_label="deepfm_v2",
)
banner("TRAINING", f"v2 (DeepFM): AUC={auc2:.4f}, n={len(feature_rows_v2)} examples, 114 features")
trigger.mark_complete()

# Phase 6: Deploy v2 SHADOW → CANARY, then inject latency spike → rollback
engine.promote(MODEL_NAME, v2)
banner("DEPLOY", "v2: SHADOW")
engine.advance(MODEL_NAME)
banner("DEPLOY", "v2: CANARY")

rollback_cfg = RollbackConfig(max_latency_ms=100.0, rollback_auc_drop=0.01, max_error_rate=0.01)
monitor = RollbackMonitor(engine, MODEL_NAME, rollback_cfg)
monitor.set_baseline_auc(auc1)

# Inject high-latency spike (simulates slow model serving)
for i in range(100):
    latency = 143.0 if i < 5 else 15.0
    monitor.log_prediction(PredictionLog(
        user_id=f"u{i}", model_version=v2,
        prediction=0.6, outcome=None, latency_ms=latency,
    ))

result = monitor.check_now()
if result["rollback_triggered"]:
    banner("ROLLBACK", f"Triggered: {result['reason']}")
banner("DEPLOY", "Rolled back to v1")

# ── Final registry state ───────────────────────────────────────────────────────
print()
banner("REGISTRY", "Final state:")
v1_mv = registry.get_version(MODEL_NAME, v1)
v2_mv = registry.get_version(MODEL_NAME, v2)
print(f"\n  v1 (LightGBM)  AUC={v1_mv.metrics['auc']}  stage={v1_mv.stage}")
print(f"  v2 (DeepFM)    AUC={v2_mv.metrics['auc']}  stage={v2_mv.stage}")
print(f"\n  DLQ events:    {len(ingestor.dlq)}")
print()
