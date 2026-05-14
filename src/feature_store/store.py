import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import redis

from src.config import FeatureStoreConfig
from src.ingestor.schemas import EnrichedEvent, EventType

logger = logging.getLogger(__name__)

_WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 604_800}

# Ordered suffixes for user and item features used at serving time.
# Must stay in sync with FEATURE_COLS in src/training/job.py.
_USER_FEAT_SUFFIXES = ["clicks_1h", "clicks_24h", "clicks_7d"]
_ITEM_FEAT_SUFFIXES = ["impressions_1h", "impressions_24h", "impressions_7d"]


class FeatureStore:
    """
    Online store (Redis): low-latency real-time aggregates.
    Offline store (Parquet): append-only log for point-in-time-correct training.

    The critical invariant: get_offline_as_of(T) never reads data written
    after T, preventing future leakage in training sets.
    """

    def __init__(self, redis_client: redis.Redis, config: FeatureStoreConfig | None = None) -> None:
        self._redis = redis_client
        self._cfg = config or FeatureStoreConfig()
        Path(self._cfg.parquet_root).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._event_rows: list[dict] = []

    # ------------------------------------------------------------------
    # Online store
    # ------------------------------------------------------------------

    def increment_event_counters(self, event: EnrichedEvent) -> None:
        try:
            pipe = self._redis.pipeline()
            if event.event_type == EventType.CLICK:
                for window, ttl in _WINDOWS.items():
                    key = f"user:{event.user_id}:clicks_{window}"
                    pipe.incr(key)
                    pipe.expire(key, ttl)
            if event.event_type == EventType.IMPRESSION:
                for window, ttl in _WINDOWS.items():
                    key = f"item:{event.item_id}:impressions_{window}"
                    pipe.incr(key)
                    pipe.expire(key, ttl)
            if event.event_type == EventType.CONVERSION:
                for window, ttl in _WINDOWS.items():
                    key = f"user:{event.user_id}:conversions_{window}"
                    pipe.incr(key)
                    pipe.expire(key, ttl)
            pipe.execute()
        except Exception as exc:
            logger.error("Redis increment failed (fail-open, event not lost): %s", exc)

    def get_online(self, entity_id: str, feature_names: list[str]) -> dict[str, Any]:
        try:
            pipe = self._redis.pipeline()
            keys = [f"{entity_id}:{name}" for name in feature_names]
            for key in keys:
                pipe.get(key)
            values = pipe.execute()
            return {
                name: int(v) if v is not None else 0
                for name, v in zip(feature_names, values)
            }
        except Exception as exc:
            logger.error("Redis get_online failed (fail-open): %s", exc)
            return {name: 0 for name in feature_names}

    def get_serving_features(self, user_id: str, item_ids: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame (one row per item) with real-time user+item features
        for serving. All lookups are batched in a single Redis pipeline call.

        Key format mirrors increment_event_counters:
          user:{user_id}:clicks_{window}   → user_clicks_{window}
          item:{item_id}:impressions_{window} → item_impressions_{window}
        """
        user_keys = [f"user:{user_id}:{s}" for s in _USER_FEAT_SUFFIXES]
        item_keys = [
            f"item:{iid}:{s}"
            for iid in item_ids
            for s in _ITEM_FEAT_SUFFIXES
        ]
        try:
            pipe = self._redis.pipeline()
            for key in user_keys + item_keys:
                pipe.get(key)
            raw = pipe.execute()
        except Exception as exc:
            logger.error("Redis get_serving_features failed (fail-open): %s", exc)
            raw = [None] * (len(user_keys) + len(item_keys))

        user_vals = raw[: len(_USER_FEAT_SUFFIXES)]
        item_vals_flat = raw[len(_USER_FEAT_SUFFIXES) :]
        n_item_feats = len(_ITEM_FEAT_SUFFIXES)

        user_base = {f"user_{s}": int(v or 0) for s, v in zip(_USER_FEAT_SUFFIXES, user_vals)}
        rows = []
        for i, iid in enumerate(item_ids):
            row = dict(user_base)
            chunk = item_vals_flat[i * n_item_feats : (i + 1) * n_item_feats]
            for s, v in zip(_ITEM_FEAT_SUFFIXES, chunk):
                row[f"item_{s}"] = int(v or 0)
            rows.append(row)

        if not rows:
            cols = [f"user_{s}" for s in _USER_FEAT_SUFFIXES] + [f"item_{s}" for s in _ITEM_FEAT_SUFFIXES]
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Offline store
    # ------------------------------------------------------------------

    def append_to_parquet(self, event: EnrichedEvent) -> None:
        row = event.model_dump()
        row["date"] = event.timestamp.strftime("%Y-%m-%d")
        with self._lock:
            self._event_rows.append(row)
            should_flush = len(self._event_rows) >= 500
        if should_flush:
            self._flush()

    def flush(self) -> None:
        self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._event_rows:
                return
            rows = self._event_rows.copy()
            self._event_rows.clear()

        df = pd.DataFrame(rows)
        for date, group in df.groupby("date"):
            partition_dir = Path(self._cfg.parquet_root) / f"date={date}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            out_path = partition_dir / f"events_{datetime.now(timezone.utc).strftime('%H%M%S%f')}.parquet"
            table = pa.Table.from_pandas(group.drop(columns=["date"]), preserve_index=False)
            pq.write_table(table, out_path)

    def get_offline_as_of(
        self,
        entity_ids: list[str],
        feature_names: list[str],
        as_of: datetime,
    ) -> pd.DataFrame:
        """
        Returns feature values as they existed at `as_of`.
        Reads only partitions whose date <= as_of.date() and filters rows
        with timestamp <= as_of to enforce point-in-time correctness.
        """
        self._flush()

        root = Path(self._cfg.parquet_root)
        as_of_date = as_of.date()
        partitions = sorted(p for p in root.glob("date=*") if p.is_dir())

        frames: list[pd.DataFrame] = []
        for partition in partitions:
            date_str = partition.name.replace("date=", "")
            try:
                part_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if part_date > as_of_date:
                continue  # never read future partitions

            parquet_files = list(partition.glob("*.parquet"))
            if not parquet_files:
                continue
            df = pd.read_parquet(partition, engine="pyarrow")
            if df.empty:
                continue
            # Filter rows within the partition whose timestamp is after as_of
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            as_of_ts = pd.Timestamp(as_of, tz=timezone.utc) if as_of.tzinfo is None else pd.Timestamp(as_of)
            df = df[df["timestamp"] <= as_of_ts]
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        all_events = pd.concat(frames, ignore_index=True)
        # Empty entity_ids means "all entities" (training over full dataset)
        if entity_ids and "user_id" in all_events.columns:
            all_events = all_events[all_events["user_id"].isin(entity_ids)]
        return all_events
