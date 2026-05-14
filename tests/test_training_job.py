import pickle
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest

from src.config import FeatureStoreConfig, TrainingConfig
from src.feature_store.store import FeatureStore
from src.ingestor.schemas import EnrichedEvent, EventType
from src.registry.registry import ModelRegistry
from src.training.job import FEATURE_COLS, LABEL_COL, TrainingJob


class _DummyAdapter:
    def __init__(self):
        self._data = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._data = (X.copy(), y.copy())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._data = pickle.load(f)


def _make_event(event_type: EventType, user_id: str, ts: datetime, confirmed: bool = True) -> EnrichedEvent:
    return EnrichedEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        user_id=user_id,
        item_id="item_1",
        timestamp=ts,
        ingested_at=ts,
        confirmed=confirmed,
    )


@pytest.fixture
def tmp_parquet(tmp_path):
    return str(tmp_path / "features")


@pytest.fixture
def store(tmp_parquet):
    redis_client = fakeredis.FakeRedis()
    return FeatureStore(redis_client=redis_client, config=FeatureStoreConfig(parquet_root=tmp_parquet))


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=str(tmp_path / "registry"))


@pytest.fixture
def job(store, registry):
    return TrainingJob(
        feature_store=store,
        registry=registry,
        adapter=_DummyAdapter(),
        config=TrainingConfig(random_seed=42, validation_fraction=0.2),
    )


def _populate_store(store: FeatureStore, n_confirmed: int, n_unconfirmed: int, base_ts: datetime) -> None:
    # ~10% conversions (positives) so the data quality gate passes
    n_conv = max(10, n_confirmed // 10)
    for i in range(n_confirmed - n_conv):
        ts = base_ts + timedelta(seconds=i)
        store.append_to_parquet(_make_event(EventType.CLICK, f"user_{i}", ts, confirmed=True))
    for i in range(n_conv):
        ts = base_ts + timedelta(seconds=n_confirmed - n_conv + i)
        store.append_to_parquet(_make_event(EventType.CONVERSION, f"conv_user_{i}", ts, confirmed=True))
    for i in range(n_unconfirmed):
        ts = base_ts + timedelta(seconds=n_confirmed + i)
        store.append_to_parquet(_make_event(EventType.CLICK, f"unconf_user_{i}", ts, confirmed=False))
    store.flush()


class TestConfirmedOnly:
    def test_training_dataset_contains_only_confirmed_examples(self, job, store, registry):
        base = datetime(2026, 3, 1, 0, 0, 0)
        _populate_store(store, n_confirmed=200, n_unconfirmed=50, base_ts=base)

        adapter = _DummyAdapter()
        job._adapter = adapter
        result = job.run(trigger_reason="test", as_of=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc))

        X_used, y_used = adapter._data
        assert len(X_used) > 0
        # All training rows should have come from confirmed events only
        assert result.n_examples <= 200  # never more than n_confirmed


class TestPointInTimeCorrectness:
    def test_training_dataset_does_not_include_events_after_as_of(self, job, store, registry):
        base = datetime(2026, 3, 1, 0, 0, 0)
        # 90 clicks + 10 conversions (positives) before as_of
        for i in range(90):
            ts = base + timedelta(hours=i)
            store.append_to_parquet(_make_event(EventType.CLICK, f"user_{i}", ts, confirmed=True))
        for i in range(10):
            ts = base + timedelta(hours=90 + i)
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"conv_{i}", ts, confirmed=True))
        # Events after as_of (future leakage if included)
        cutoff = base + timedelta(hours=200)
        for i in range(50):
            ts = cutoff + timedelta(hours=i + 1)
            store.append_to_parquet(_make_event(EventType.CLICK, f"future_user_{i}", ts, confirmed=True))
        store.flush()

        as_of = datetime(2026, 3, 1, tzinfo=timezone.utc) + timedelta(hours=150)
        result = job.run(trigger_reason="test", as_of=as_of)

        # n_examples should not include the 50 future events
        assert result.n_examples <= 100

    def test_future_events_not_present_in_training_set(self, store, registry, tmp_parquet):
        base = datetime(2026, 4, 1, 0, 0, 0)
        as_of_ts = base + timedelta(minutes=100)

        # 110 events before as_of (90 clicks + 20 conversions = enough for quality gate)
        for i in range(90):
            ts = base + timedelta(minutes=i)
            store.append_to_parquet(_make_event(EventType.CLICK, f"u{i}", ts, confirmed=True))
        for i in range(20):
            ts = base + timedelta(minutes=i)
            store.append_to_parquet(_make_event(EventType.CONVERSION, f"uc{i}", ts, confirmed=True))
        # 30 events strictly after as_of — must not be included
        for i in range(30):
            ts = as_of_ts + timedelta(minutes=i + 1)
            store.append_to_parquet(_make_event(EventType.CLICK, f"future_{i}", ts, confirmed=True))
        store.flush()

        # Use a low-threshold config since we're testing PIT correctness, not quality gates
        cfg = TrainingConfig(random_seed=42, min_training_examples=50, min_positive_examples=5)
        job = TrainingJob(store, registry, _DummyAdapter(), cfg)
        adapter = _DummyAdapter()
        job._adapter = adapter
        job.run(trigger_reason="test", as_of=as_of_ts.replace(tzinfo=timezone.utc))

        if adapter._data is not None:
            X_used, _ = adapter._data
            assert len(X_used) <= 110  # never exceeds events before as_of


class TestTrainingResult:
    def test_result_contains_version_string_and_metrics(self, job, store, registry):
        base = datetime(2026, 3, 10, 0, 0, 0)
        _populate_store(store, n_confirmed=150, n_unconfirmed=0, base_ts=base)

        result = job.run(
            trigger_reason="event_count=10000",
            as_of=datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc),
        )

        assert isinstance(result.version, str) and len(result.version) > 0
        assert "auc" in result.metrics
        assert "ece" in result.metrics
        assert 0.0 <= result.metrics["auc"] <= 1.0
        assert result.n_examples > 0
        assert result.trigger_reason == "event_count=10000"

    def test_result_is_registered_in_registry(self, job, store, registry):
        base = datetime(2026, 3, 10, 0, 0, 0)
        _populate_store(store, n_confirmed=150, n_unconfirmed=0, base_ts=base)

        result = job.run(
            trigger_reason="schedule",
            as_of=datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc),
        )

        mv = registry.get_version("ctr_model", result.version)
        assert mv.metrics["auc"] == result.metrics["auc"]


class TestDeterminism:
    def test_same_seed_and_data_produces_same_auc(self, store, registry, tmp_parquet):
        base = datetime(2026, 3, 15, 0, 0, 0)
        _populate_store(store, n_confirmed=200, n_unconfirmed=0, base_ts=base)
        as_of = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)

        cfg = TrainingConfig(random_seed=42)

        job1 = TrainingJob(store, registry, _DummyAdapter(), cfg)
        r1 = job1.run("test", as_of)

        job2 = TrainingJob(store, registry, _DummyAdapter(), cfg)
        r2 = job2.run("test", as_of)

        # DummyAdapter always predicts 0.5 so AUC is deterministic given same data split
        assert r1.metrics["auc"] == r2.metrics["auc"]
        assert r1.n_examples == r2.n_examples
