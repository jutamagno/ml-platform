import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone

from src.ingestor.schemas import DLQRecord

logger = logging.getLogger(__name__)


class DeadLetterHandler:
    """
    Persists invalid events with their validation error.
    A separate daily report surfaces error patterns without hiding them.
    Design decision: DLQ over silent drop — silent drops cause training data
    gaps that surface weeks later, making root-cause analysis very hard.
    """

    def __init__(self) -> None:
        self._records: list[DLQRecord] = []

    def write(self, raw_payload: dict, error_type: str, error_message: str) -> None:
        record = DLQRecord(
            raw_payload=raw_payload,
            error_type=error_type,
            error_message=error_message,
            producer_id=raw_payload.get("producer_id"),
            failed_at=datetime.now(timezone.utc),
        )
        self._records.append(record)
        logger.warning("DLQ: %s | %s | payload=%s", error_type, error_message, json.dumps(raw_payload)[:200])

    def daily_report(self) -> dict:
        error_counts: Counter = Counter()
        producer_counts: Counter = Counter()
        samples: dict[str, dict] = {}

        for rec in self._records:
            error_counts[rec.error_type] += 1
            if rec.producer_id:
                producer_counts[rec.producer_id] += 1
            if rec.error_type not in samples:
                samples[rec.error_type] = rec.raw_payload

        return {
            "total_dlq_events": len(self._records),
            "error_type_breakdown": dict(error_counts),
            "top_offending_producers": producer_counts.most_common(10),
            "sample_payloads": samples,
        }

    def replay(self, ingestor, filter_error_type: str | None = None) -> dict[str, int]:
        """
        Re-processes DLQ records through an ingestor (e.g. after a schema fix).
        Records that now pass are removed from the DLQ.
        Records that still fail remain.
        Returns {"ok": N, "dlq": M}.
        """
        pending = []
        results = {"ok": 0, "dlq": 0}
        for record in self._records:
            if filter_error_type and record.error_type != filter_error_type:
                pending.append(record)
                continue
            outcome = ingestor.process(record.raw_payload)
            if outcome.value == "ok":
                results["ok"] += 1
            else:
                results["dlq"] += 1
                pending.append(record)
        self._records = pending
        logger.info("DLQ replay: ok=%d still_dlq=%d", results["ok"], results["dlq"])
        return results

    def __len__(self) -> int:
        return len(self._records)
