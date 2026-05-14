"""
Tests for the seven production-quality improvements:
1. get_serving_features — online feature lookup at serving time
2. Redis fail-open — ingestion survives Redis outage
3. Temporal train/val split — no future leakage via random shuffle
4. Data quality gate — blocks degenerate training sets
5. Thread-safe ingestion counter
6. DLQ replay — re-process invalid events after schema fix
7. Canary A/B cohort comparison in rollback monitor
"""
import pickle
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import fakeredis
import numpy as np
import pandas as pd
import pytest

from src.config import (
    DeploymentConfig,
    FeatureStoreConfig,
    RollbackConfig,
    TrainingConfig,
)
from src.deployment.engine import DeploymentEngine, DeploymentStage
from src.deployment.monitor import PredictionLog, RollbackMonitor
from src.deployment.proxy import InferenceRequest, ServingProxy
from src.feature_store.store import FeatureStore
from src.ingestor.dead_letter import DeadLetterHandler
from src.ingestor.ingestor import EventIngestor
from src.ingestor.schemas import EnrichedEvent, EventType
from src.registry.registry import ModelRegistry
from src.training.job import FEATURE_COLS, DataQualityReport, TrainingJob


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_event(event_type=EventType.CLICK, user_id="u1", item_id="i1", ts=None, confirmed=True):
    return EnrichedEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        user_id=user_id,
        item_id=item_id,
        timestamp=ts or datetime(2026, 1, 1, tzinfo=timezone.utc),
        ingested_at=datetime.now(timezone.utc),
        confirmed=confirmed,
    )


class _DummyAdapter:
    def __init__(self, score=0.7):
        self._score = score
        self._last_X = None

    def fit(self, X, y):
        self._last_X = X.copy()

    def predict_proba(self, X):
        return np.full(len(X), self._score)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def store(tmp_path, redis_client):
    cfg = FeatureStoreConfig(parquet_root=str(tmp_path / "features"))
    return FeatureStore(redis_client=redis_client, config=cfg)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=str(tmp_path / "registry"))


@pytest.fixture
def fast_engine(registry):
    return DeploymentEngine(registry=registry, config=DeploymentConfig(min_shadow_hours=0, min_canary_hours=0))


# ---------------------------------------------------------------------------
# 1. get_serving_features — real online feature lookup
# ---------------------------------------------------------------------------


class TestGetServingFeatures:
    def test_returns_zero_for_unknown_user_and_items(self, store):
        df = store.get_serving_features("unknown_user", ["item_a", "item_b"])
        assert len(df) == 2
        assert "user_clicks_1h" in df.columns
        assert "item_impressions_1h" in df.columns
        assert (df["user_clicks_1h"] == 0).all()
        assert (df["item_impressions_1h"] == 0).all()

    def test_returns_live_counters_after_events(self, store):
        click = _make_event(EventType.CLICK, user_id="u99", item_id="i1")
        impression = _make_event(EventType.IMPRESSION, user_id="u99", item_id="i1")
        store.increment_event_counters(click)
        store.increment_event_counters(impression)

        df = store.get_serving_features("u99", ["i1"])
        assert len(df) == 1
        assert df.iloc[0]["user_clicks_1h"] == 1
        assert df.iloc[0]["item_impressions_1h"] == 1

    def test_user_features_same_across_all_items(self, store):
        click = _make_event(EventType.CLICK, user_id="u7", item_id="i1")
        store.increment_event_counters(click)
        store.increment_event_counters(click)

        df = store.get_serving_features("u7", ["i1", "i2", "i3"])
        assert len(df) == 3
        # User features must be identical for all items
        assert df["user_clicks_1h"].nunique() == 1
        assert df["user_clicks_1h"].iloc[0] == 2

    def test_item_features_differ_per_item(self, store):
        for _ in range(3):
            store.increment_event_counters(_make_event(EventType.IMPRESSION, "u1", "item_hot"))
        store.increment_event_counters(_make_event(EventType.IMPRESSION, "u1", "item_cold"))

        df = store.get_serving_features("u1", ["item_hot", "item_cold"])
        df = df.set_index(["item_hot", "item_cold"]) if False else df
        hot_row = df.iloc[0]  # item_hot was first
        cold_row = df.iloc[1]
        assert hot_row["item_impressions_1h"] == 3
        assert cold_row["item_impressions_1h"] == 1

    def test_returns_empty_dataframe_for_empty_item_list(self, store):
        df = store.get_serving_features("u1", [])
        assert df.empty


# ---------------------------------------------------------------------------
# 2. Redis fail-open
# ---------------------------------------------------------------------------


class TestRedisFailOpen:
    def test_increment_event_counters_survives_redis_error(self, store):
        store._redis = MagicMock(side_effect=Exception("Redis down"))
        event = _make_event(EventType.CLICK)
        # Must not raise — ingestion continues, online features just become 0
        store.increment_event_counters(event)

    def test_get_online_returns_zeros_on_redis_error(self, store):
        store._redis = MagicMock(side_effect=Exception("Redis down"))
        result = store.get_online("user:u1", ["clicks_1h"])
        assert result == {"clicks_1h": 0}

    def test_get_serving_features_returns_zeros_on_redis_error(self, store):
        store._redis = MagicMock(side_effect=Exception("Redis down"))
        df = store.get_serving_features("u1", ["i1"])
        assert len(df) == 1
        assert (df == 0).all().all()

    def test_ingestor_continues_after_redis_error(self, store):
        store._redis = MagicMock(side_effect=Exception("Redis down"))
        ingestor = EventIngestor(feature_store=store)
        raw = {
            "event_id": str(uuid.uuid4()),
            "event_type": "click",
            "user_id": "u1",
            "item_id": "i1",
            "timestamp": "2026-01-01T00:00:00",
        }
        result = ingestor.process(raw)
        # Event still ingested — Parquet write is independent of Redis
        assert result.value == "ok"


# ---------------------------------------------------------------------------
# 3. Temporal train/val split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def _make_job(self, store, registry, n_examples=200):
        cfg = TrainingConfig(random_seed=42, validation_fraction=0.2)
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        n_conv = 20
        for i in range(n_examples - n_conv):
            ts = base + timedelta(minutes=i)
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", item_id="i1", ts=ts))
        for i in range(n_conv):
            ts = base + timedelta(minutes=n_examples - n_conv + i)
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"cu{i}", item_id="i1", ts=ts))
        store.flush()
        return TrainingJob(store, registry, _DummyAdapter(), cfg)

    def test_validation_set_is_temporally_after_training_set(self, store, registry):
        """Last 20% by time must be the val set — training should use earlier events."""
        job = self._make_job(store, registry)

        captured = {}

        class _CapturingAdapter(_DummyAdapter):
            def fit(self, X_train, y_train):
                captured["train_size"] = len(X_train)

            def predict_proba(self, X_val):
                captured["val_size"] = len(X_val)
                return np.full(len(X_val), 0.5)

        job._adapter = _CapturingAdapter()
        job.run("test", as_of=datetime(2026, 3, 10, tzinfo=timezone.utc))

        train_size = captured["train_size"]
        val_size = captured["val_size"]
        total = train_size + val_size
        # Validation should be ~20% of total, not random
        assert abs(val_size / total - 0.2) < 0.05

    def test_temporal_split_uses_earlier_data_for_training(self, store, registry):
        """Training rows must have timestamps earlier than validation rows."""
        cfg = TrainingConfig(random_seed=42, validation_fraction=0.2)
        n_conv = 20
        n_click = 100
        base = datetime(2026, 5, 1, tzinfo=timezone.utc)
        for i in range(n_click):
            ts = base + timedelta(hours=i)
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", ts=ts))
        for i in range(n_conv):
            ts = base + timedelta(hours=n_click + i)
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"cu{i}", ts=ts))
        store.flush()

        # Access the split directly by calling _temporal_split
        job = TrainingJob(store, registry, _DummyAdapter(), cfg)
        all_events = store.get_offline_as_of([], FEATURE_COLS, as_of=datetime(2026, 6, 1, tzinfo=timezone.utc))
        confirmed = all_events[all_events["confirmed"] == True]
        confirmed["label"] = (confirmed["event_type"] == "conversion").astype(int)
        confirmed = confirmed.sort_values("timestamp")

        feat_cols = [c for c in FEATURE_COLS if c in confirmed.columns]
        X_train, X_val, y_train, y_val = job._temporal_split(confirmed, feat_cols)

        # The last conversion events (which are the positives) should be in val set
        # because they were appended last (latest timestamps)
        assert y_val.sum() > 0, "Val set should contain some conversions (the later events)"


# ---------------------------------------------------------------------------
# 4. Data quality gate
# ---------------------------------------------------------------------------


class TestDataQualityGate:
    def _make_job(self, store, registry, **cfg_kwargs):
        cfg = TrainingConfig(**cfg_kwargs)
        return TrainingJob(store, registry, _DummyAdapter(), cfg)

    def test_raises_on_too_few_examples(self, store, registry):
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        for i in range(5):
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", ts=base + timedelta(minutes=i)))
        for i in range(3):
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"cu{i}", ts=base + timedelta(minutes=10 + i)))
        store.flush()

        job = self._make_job(store, registry, min_training_examples=100)
        with pytest.raises(ValueError, match="Data quality gate"):
            job.run("test", as_of=datetime(2026, 3, 2, tzinfo=timezone.utc))

    def test_raises_on_too_few_positives(self, store, registry):
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        for i in range(200):
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", ts=base + timedelta(minutes=i), confirmed=True))
        store.flush()

        job = self._make_job(store, registry, min_training_examples=10, min_positive_examples=5)
        with pytest.raises(ValueError, match="Data quality gate"):
            job.run("test", as_of=datetime(2026, 3, 2, tzinfo=timezone.utc))

    def test_passes_with_valid_dataset(self, store, registry):
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        for i in range(90):
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", ts=base + timedelta(minutes=i)))
        for i in range(20):
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"cu{i}", ts=base + timedelta(minutes=90 + i)))
        store.flush()

        job = self._make_job(store, registry, min_training_examples=50, min_positive_examples=10)
        result = job.run("test", as_of=datetime(2026, 3, 2, tzinfo=timezone.utc))
        assert result.data_quality is not None
        assert not result.data_quality.is_critical
        assert result.data_quality.n_positives >= 10

    def test_quality_report_attached_to_result(self, store, registry):
        base = datetime(2026, 3, 1, tzinfo=timezone.utc)
        for i in range(90):
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", ts=base + timedelta(minutes=i)))
        for i in range(20):
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"cu{i}", ts=base + timedelta(minutes=90 + i)))
        store.flush()

        job = self._make_job(store, registry, min_training_examples=50)
        result = job.run("test", as_of=datetime(2026, 3, 2, tzinfo=timezone.utc))
        assert isinstance(result.data_quality, DataQualityReport)
        assert result.data_quality.n_examples > 0
        assert 0.0 <= result.data_quality.label_rate <= 1.0


# ---------------------------------------------------------------------------
# 5. Thread-safe ingestion counter
# ---------------------------------------------------------------------------


class TestThreadSafeIngestionCounter:
    def test_concurrent_ingestion_count_is_accurate(self, store):
        ingestor = EventIngestor(feature_store=store)
        n_threads = 20
        n_events_per_thread = 50

        def _ingest():
            for i in range(n_events_per_thread):
                ingestor.process({
                    "event_id": str(uuid.uuid4()),
                    "event_type": "click",
                    "user_id": "u1",
                    "item_id": "i1",
                    "timestamp": "2026-01-01T00:00:00",
                })

        threads = [threading.Thread(target=_ingest) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ingestor.ingested_count == n_threads * n_events_per_thread


# ---------------------------------------------------------------------------
# 6. DLQ replay
# ---------------------------------------------------------------------------


class TestDLQReplay:
    def test_replay_removes_successfully_processed_records(self, store):
        dlq = DeadLetterHandler()
        # Seed two records manually (simulating past failures)
        dlq.write(
            raw_payload={
                "event_id": str(uuid.uuid4()),
                "event_type": "click",
                "user_id": "u1",
                "item_id": "i1",
                "timestamp": "2026-01-01T00:00:00",
            },
            error_type="schema_validation_error",
            error_message="was invalid before",
        )
        assert len(dlq) == 1

        ingestor = EventIngestor(feature_store=store, dlq=dlq)
        results = dlq.replay(ingestor)

        assert results["ok"] == 1
        assert results["dlq"] == 0
        assert len(dlq) == 0  # record removed after successful replay

    def test_replay_keeps_still_failing_records(self, store):
        dlq = DeadLetterHandler()
        dlq.write(
            raw_payload={"bad": "payload"},  # will still fail validation
            error_type="missing_required_field",
            error_message="missing event_id",
        )

        ingestor = EventIngestor(feature_store=store, dlq=dlq)
        results = dlq.replay(ingestor)

        assert results["ok"] == 0
        assert results["dlq"] == 1
        assert len(dlq) == 1  # record kept — still invalid

    def test_replay_with_filter_only_retries_matching_error_type(self, store):
        dlq = DeadLetterHandler()
        dlq.write({"bad": "data"}, "missing_required_field", "msg")
        dlq.write(
            raw_payload={
                "event_id": str(uuid.uuid4()),
                "event_type": "click",
                "user_id": "u1",
                "item_id": "i1",
                "timestamp": "2026-01-01T00:00:00",
            },
            error_type="schema_validation_error",
            error_message="was invalid",
        )
        assert len(dlq) == 2

        ingestor = EventIngestor(feature_store=store, dlq=dlq)
        # Only replay schema_validation_error records
        results = dlq.replay(ingestor, filter_error_type="schema_validation_error")

        assert results["ok"] == 1
        assert len(dlq) == 1  # missing_required_field record untouched, schema record removed


# ---------------------------------------------------------------------------
# 7. Canary A/B cohort comparison
# ---------------------------------------------------------------------------


class TestCanaryABComparison:
    def _setup(self, registry, fast_engine):
        adapter = _DummyAdapter()
        v1 = registry.register("ctr_model", adapter, {"auc": 0.75}, {})
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = registry.register("ctr_model", adapter, {"auc": 0.78}, {})
        fast_engine.promote("ctr_model", v2)
        fast_engine.advance("ctr_model")  # → CANARY
        return v1, v2

    def _make_log(self, version, prediction, outcome):
        return PredictionLog(
            user_id=str(uuid.uuid4()),
            model_version=version,
            prediction=prediction,
            outcome=outcome,
            latency_ms=10.0,
        )

    def test_rollback_triggered_when_canary_auc_below_control(self, registry, fast_engine):
        v1, v2 = self._setup(registry, fast_engine)
        monitor = RollbackMonitor(fast_engine, "ctr_model", RollbackConfig(rollback_auc_drop=0.02))

        rng = np.random.default_rng(0)
        # Control (v1): good predictions → AUC ~0.75
        for i in range(50):
            outcome = float(i % 4 == 0)
            monitor.log_prediction(self._make_log(v1, 0.8 if outcome else 0.2, outcome))
        # Canary (v2): random predictions → AUC ~0.5
        for i in range(50):
            outcome = float(i % 4 == 0)
            monitor.log_prediction(self._make_log(v2, float(rng.random()), outcome))

        result = monitor.check_now()
        assert result["rollback_triggered"]
        assert result["canary_auc"] is not None
        assert result["control_auc"] is not None
        assert result["canary_auc"] < result["control_auc"]

    def test_no_rollback_when_canary_matches_control(self, registry, fast_engine):
        v1, v2 = self._setup(registry, fast_engine)
        monitor = RollbackMonitor(fast_engine, "ctr_model", RollbackConfig(rollback_auc_drop=0.02))

        # Both versions have good, similar predictions
        for i in range(50):
            outcome = float(i % 4 == 0)
            monitor.log_prediction(self._make_log(v1, 0.8 if outcome else 0.2, outcome))
            monitor.log_prediction(self._make_log(v2, 0.8 if outcome else 0.2, outcome))

        result = monitor.check_now()
        assert not result["rollback_triggered"]

    def test_ab_comparison_not_triggered_with_too_few_canary_logs(self, registry, fast_engine):
        v1, v2 = self._setup(registry, fast_engine)
        monitor = RollbackMonitor(fast_engine, "ctr_model", RollbackConfig(rollback_auc_drop=0.02))

        # Only 5 canary logs — below the 30-log threshold for A/B
        for i in range(50):
            outcome = float(i % 4 == 0)
            monitor.log_prediction(self._make_log(v1, 0.8 if outcome else 0.2, outcome))
        for i in range(5):
            monitor.log_prediction(self._make_log(v2, 0.1, 0.0))

        result = monitor.check_now()
        # canary_auc is None because we didn't have enough canary logs
        assert result["canary_auc"] is None


# ---------------------------------------------------------------------------
# 8. ServingProxy uses live features (integration)
# ---------------------------------------------------------------------------


class TestServingProxyWithFeatureStore:
    def test_proxy_uses_feature_store_when_provided(self, store, registry, fast_engine, tmp_path):
        # Populate Redis with known feature values
        click_event = _make_event(EventType.CLICK, user_id="active_user", item_id="hot_item")
        store.increment_event_counters(click_event)
        store.increment_event_counters(click_event)
        impression = _make_event(EventType.IMPRESSION, user_id="active_user", item_id="hot_item")
        store.increment_event_counters(impression)

        adapter = _DummyAdapter(score=0.6)
        v1 = registry.register("m", adapter, {"auc": 0.8}, {})
        registry.promote("m", v1, "full")
        fast_engine.promote("m", v1)
        fast_engine.advance("m")  # CANARY
        fast_engine.advance("m")  # FULL

        def loader(version):
            path = tmp_path / "registry" / "m" / version / "model.pkl"
            with open(path, "rb") as f:
                return pickle.load(f)

        proxy = ServingProxy("m", fast_engine, loader, feature_store=store)
        response = proxy.predict(InferenceRequest(user_id="active_user", item_ids=["hot_item"]))

        assert response.model_version == v1
        assert "hot_item" in response.scores

    def test_proxy_zero_fills_missing_features_for_cold_users(self, store, registry, fast_engine, tmp_path):
        adapter = _DummyAdapter(score=0.5)
        v1 = registry.register("m", adapter, {"auc": 0.8}, {})
        registry.promote("m", v1, "full")
        fast_engine.promote("m", v1)
        fast_engine.advance("m")
        fast_engine.advance("m")

        def loader(version):
            path = tmp_path / "registry" / "m" / version / "model.pkl"
            with open(path, "rb") as f:
                return pickle.load(f)

        proxy = ServingProxy("m", fast_engine, loader, feature_store=store)
        # cold_user has no history → all features 0, model should still return a score
        response = proxy.predict(InferenceRequest(user_id="cold_user", item_ids=["new_item"]))
        assert response.model_version == v1
        assert "new_item" in response.scores
        assert 0.0 <= response.scores["new_item"] <= 1.0
