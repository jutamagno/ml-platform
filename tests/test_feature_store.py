import tempfile
import uuid
from datetime import datetime, timezone

import fakeredis
import pytest

from src.config import FeatureStoreConfig
from src.feature_store.store import FeatureStore
from src.ingestor.schemas import EnrichedEvent, EventType


def _make_enriched(event_type=EventType.CLICK, user_id="u1", item_id="i1", timestamp=None):
    return EnrichedEvent(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        user_id=user_id,
        item_id=item_id,
        timestamp=timestamp or datetime(2026, 1, 15, 12, 0, 0),
        ingested_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def tmp_parquet(tmp_path):
    return str(tmp_path / "features")


@pytest.fixture
def redis_client():
    return fakeredis.FakeRedis()


@pytest.fixture
def store(redis_client, tmp_parquet):
    cfg = FeatureStoreConfig(parquet_root=tmp_parquet)
    return FeatureStore(redis_client=redis_client, config=cfg)


class TestOnlineStore:
    def test_get_online_returns_updated_value(self, store):
        event = _make_enriched(event_type=EventType.CLICK, user_id="u99")
        store.increment_event_counters(event)
        result = store.get_online("user:u99", ["clicks_1h"])
        assert result["clicks_1h"] == 1

    def test_multiple_increments_accumulate(self, store):
        for _ in range(3):
            store.increment_event_counters(_make_enriched(EventType.CLICK, user_id="uacc"))
        result = store.get_online("user:uacc", ["clicks_1h"])
        assert result["clicks_1h"] == 3

    def test_missing_key_returns_zero(self, store):
        result = store.get_online("user:nonexistent", ["clicks_1h"])
        assert result["clicks_1h"] == 0

    def test_redis_keys_have_ttl_for_1h_window(self, redis_client, store):
        event = _make_enriched(EventType.CLICK, user_id="uttl")
        store.increment_event_counters(event)
        ttl = redis_client.ttl("user:uttl:clicks_1h")
        # TTL should be set (not -1 = no expiry, not -2 = missing)
        assert 0 < ttl <= 3600

    def test_redis_keys_have_ttl_for_24h_window(self, redis_client, store):
        event = _make_enriched(EventType.CLICK, user_id="uttl24")
        store.increment_event_counters(event)
        ttl = redis_client.ttl("user:uttl24:clicks_24h")
        assert 0 < ttl <= 86400

    def test_impression_counter_written_to_item_key(self, redis_client, store):
        event = _make_enriched(EventType.IMPRESSION, item_id="item42")
        store.increment_event_counters(event)
        ttl = redis_client.ttl("item:item42:impressions_1h")
        assert ttl > 0


class TestOfflineStore:
    def test_parquet_partitioned_by_date(self, store, tmp_parquet):
        import os
        event = _make_enriched(timestamp=datetime(2026, 3, 10, 8, 0, 0))
        store.append_to_parquet(event)
        store.flush()
        assert os.path.isdir(f"{tmp_parquet}/date=2026-03-10")

    def test_get_offline_as_of_returns_events_before_cutoff(self, store):
        early = _make_enriched(user_id="uearly", timestamp=datetime(2026, 3, 10, 6, 0, 0))
        late = _make_enriched(user_id="ulate", timestamp=datetime(2026, 3, 10, 18, 0, 0))
        store.append_to_parquet(early)
        store.append_to_parquet(late)
        store.flush()

        as_of = datetime(2026, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        df = store.get_offline_as_of(["uearly", "ulate"], [], as_of)
        user_ids = set(df["user_id"].tolist())
        assert "uearly" in user_ids
        assert "ulate" not in user_ids

    def test_get_offline_as_of_excludes_future_partitions(self, store):
        past = _make_enriched(user_id="upast", timestamp=datetime(2026, 3, 9, 12, 0, 0))
        future = _make_enriched(user_id="ufuture", timestamp=datetime(2026, 3, 11, 12, 0, 0))
        store.append_to_parquet(past)
        store.append_to_parquet(future)
        store.flush()

        as_of = datetime(2026, 3, 10, 0, 0, 0, tzinfo=timezone.utc)
        df = store.get_offline_as_of(["upast", "ufuture"], [], as_of)
        if not df.empty:
            assert "ufuture" not in df["user_id"].tolist()

    def test_empty_result_when_no_events_before_cutoff(self, store):
        future = _make_enriched(user_id="ufut2", timestamp=datetime(2026, 5, 1, 12, 0, 0))
        store.append_to_parquet(future)
        store.flush()
        as_of = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        df = store.get_offline_as_of(["ufut2"], [], as_of)
        assert df.empty
