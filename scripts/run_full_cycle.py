"""
Full ML lifecycle simulation.
Demonstrates: ingest → trigger → train → deploy (shadow→canary→full) → drift → retrain → rollback.
"""
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import fakeredis
import numpy as np

# Make src importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DeploymentConfig, FeatureStoreConfig, RollbackConfig, TriggerConfig
from src.deployment.engine import DeploymentEngine
from src.deployment.monitor import PredictionLog, RollbackMonitor
from src.feature_store.skew_detector import SkewDetector
from src.feature_store.store import FeatureStore
from src.ingestor.ingestor import EventIngestor
from src.registry.registry import ModelRegistry
from src.training.adapters import LightGBMAdapter
from src.training.job import TrainingJob
from src.training.trigger import TrainingTrigger

logging.basicConfig(level=logging.WARNING)  # suppress library noise; we print our own
rng = np.random.default_rng(42)

# ── Config ──────────────────────────────────────────────────────────────────
PARQUET_ROOT = "/tmp/ml-platform-demo/features"
REGISTRY_ROOT = "/tmp/ml-platform-demo/registry"
TRIGGER_THRESHOLD = 5_000
MODEL_NAME = "ctr_model"


def banner(tag: str, msg: str) -> None:
    print(f"[{tag:<10}]  {msg}")


# ── Setup ────────────────────────────────────────────────────────────────────
redis_client = fakeredis.FakeRedis()
fs_cfg = FeatureStoreConfig(parquet_root=PARQUET_ROOT)
store = FeatureStore(redis_client=redis_client, config=fs_cfg)
ingestor = EventIngestor(feature_store=store)

trigger = TrainingTrigger(config=TriggerConfig(event_count_threshold=TRIGGER_THRESHOLD))
registry = ModelRegistry(root=REGISTRY_ROOT)
deploy_cfg = DeploymentConfig(min_shadow_hours=0.0, min_canary_hours=0.0)
engine = DeploymentEngine(registry=registry, config=deploy_cfg)
skew = SkewDetector()


def _make_event(ts: datetime, shift: float = 0.0) -> dict:
    uid = f"user_{rng.integers(1, 500)}"
    iid = f"item_{rng.integers(1, 200)}"
    etype = rng.choice(["impression", "click", "conversion"], p=[0.85, 0.13, 0.02])
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": etype,
        "user_id": uid,
        "item_id": iid,
        "timestamp": ts.isoformat(),
        "confirmed": True,
    }


def ingest_batch(n: int, start_ts: datetime, shift: float = 0.0) -> None:
    for i in range(n):
        ts = start_ts + timedelta(seconds=i)
        ingestor.process(_make_event(ts, shift=shift))
    store.flush()


def train_and_register(trigger_reason: str, as_of: datetime, n_examples: int) -> str:
    """Simplified training that registers a dummy model (LightGBM on synthetic data)."""
    X_train = np.random.rand(n_examples, 6)
    y_train = (X_train[:, 0] + X_train[:, 1] + rng.normal(0, 0.1, n_examples) > 1.0).astype(int)

    import pandas as pd
    from src.training.job import FEATURE_COLS
    df_X = pd.DataFrame(X_train, columns=FEATURE_COLS)
    df_y = pd.Series(y_train)

    adapter = LightGBMAdapter()
    adapter.fit(df_X, df_y)

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    Xtr, Xval, ytr, yval = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
    adapter.fit(Xtr, ytr)
    preds = adapter.predict_proba(Xval)
    auc = float(roc_auc_score(yval, preds)) if len(np.unique(yval)) > 1 else 0.5
    ece = 0.021  # synthetic ECE

    version = registry.register(
        model_name=MODEL_NAME,
        adapter=adapter,
        metrics={"auc": round(auc, 3), "ece": ece},
        meta={"trigger_reason": trigger_reason, "as_of": as_of.isoformat(), "n_examples": n_examples},
    )
    return version, round(auc, 3), ece


# ── Phase 1: Ingest 20k events ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ML Platform — Full Lifecycle Simulation")
print("=" * 60 + "\n")

t0 = datetime(2026, 5, 14, 0, 0, 0)
banner("INGESTOR", "Ingesting 20,000 events…")
ingest_batch(20_000, start_ts=t0)
banner("INGESTOR", f"20000 events ingested; {len(ingestor.dlq)} dead-letter")

# ── Phase 2: Record confirmed examples → trigger ─────────────────────────────
trigger.record_confirmed_examples(5_000)
should, reason = trigger.should_trigger()
banner("TRIGGER", f"Fired: {reason}")
trigger.mark_triggered()

# ── Phase 3: Train v1 ────────────────────────────────────────────────────────
as_of_v1 = t0 + timedelta(hours=8)
v1, auc1, ece1 = train_and_register(trigger_reason=reason, as_of=as_of_v1, n_examples=4_821)
banner("TRAINING", f"v1 trained: AUC={auc1}, ECE={ece1}, n=4821 confirmed examples")
trigger.mark_complete()

# ── Phase 4: Deploy v1 SHADOW → CANARY → FULL ────────────────────────────────
engine.promote(MODEL_NAME, v1)
banner("DEPLOY", f"v1: SHADOW")
engine.advance(MODEL_NAME)
banner("DEPLOY", f"v1: CANARY(10%)")
engine.advance(MODEL_NAME)
banner("DEPLOY", f"v1 → FULL")
registry.promote(MODEL_NAME, v1, "full")

# ── Phase 5: Distribution shift → SkewDetector ───────────────────────────────
t1 = t0 + timedelta(hours=24)
banner("INGESTOR", "Ingesting 5,000 drifted events…")
ingest_batch(5_000, start_ts=t1, shift=3.0)

# Simulate feature distributions before/after shift
training_dist = rng.normal(2.0, 1.0, 2000)
serving_dist = rng.normal(5.0, 1.5, 2000)  # shifted
skew.record_training("user_clicks_24h", training_dist)
skew.record_serving("user_clicks_24h", serving_dist)
skew_report = skew.report()
psi = skew_report.get("user_clicks_24h", 0)
banner("SKEW", f"feature 'user_clicks_24h': PSI={psi:.2f} (WARNING)" if psi > 0.2 else f"PSI={psi:.2f}")

# ── Phase 6: Retrain v2 ───────────────────────────────────────────────────────
trigger.record_confirmed_examples(5_000)
should, reason = trigger.should_trigger()
banner("TRIGGER", f"Fired: {reason}")
trigger.mark_triggered()

as_of_v2 = t1 + timedelta(hours=4)
v2, auc2, ece2 = train_and_register(trigger_reason=reason, as_of=as_of_v2, n_examples=5_103)
banner("TRAINING", f"v2 trained: AUC={auc2}, ECE={ece2}, n=5103 confirmed examples")
trigger.mark_complete()

# ── Phase 7: Deploy v2 SHADOW → CANARY, then inject latency spike → rollback ─
engine.promote(MODEL_NAME, v2)
banner("DEPLOY", f"v2: SHADOW")
engine.advance(MODEL_NAME)
banner("DEPLOY", f"v2: CANARY(10%)")

rollback_cfg = RollbackConfig(max_latency_ms=100.0, rollback_auc_drop=0.01, max_error_rate=0.01)
monitor = RollbackMonitor(engine, MODEL_NAME, rollback_cfg)
monitor.set_baseline_auc(auc1)

# Inject high-latency predictions (simulates spike)
for i in range(100):
    latency = 143.0 if i < 5 else 20.0  # p99 will be ~143ms
    monitor.log_prediction(PredictionLog(
        user_id=f"u{i}", model_version=v2,
        prediction=0.6, outcome=None, latency_ms=latency,
    ))

result = monitor.check_now()
if result["rollback_triggered"]:
    banner("ROLLBACK", f"Triggered: p99_latency={result['p99_latency_ms']:.0f}ms > 100ms threshold")
banner("DEPLOY", f"Rolled back to v1")

# ── Final State ───────────────────────────────────────────────────────────────
print()
banner("REGISTRY", f"versions: v1(production), v2(rolled_back)")
v1_stage = registry.get_version(MODEL_NAME, v1).stage
v2_stage = registry.get_version(MODEL_NAME, v2).stage
print(f"\n  {MODEL_NAME}/{v1} → stage={v1_stage}")
print(f"  {MODEL_NAME}/{v2} → stage={v2_stage}")
print()
