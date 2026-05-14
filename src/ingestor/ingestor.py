import logging
import threading
from datetime import datetime, timezone

from pydantic import ValidationError

from src.ingestor.dead_letter import DeadLetterHandler
from src.ingestor.schemas import AdEvent, EnrichedEvent, ProcessResult

logger = logging.getLogger(__name__)


class EventIngestor:
    """
    Validates incoming raw event dicts, enriches them, then fans out to the
    feature store (Redis aggregates) and Parquet event log.

    Unknown fields are silently ignored (forward-compat schema evolution).
    Missing required fields route to DLQ — never silently dropped.
    """

    def __init__(self, feature_store, dlq: DeadLetterHandler | None = None) -> None:
        self._feature_store = feature_store
        self._dlq = dlq if dlq is not None else DeadLetterHandler()
        self._ingested_count = 0
        self._count_lock = threading.Lock()

    def process(self, raw_event: dict) -> ProcessResult:
        try:
            event = AdEvent.model_validate(raw_event)
        except ValidationError as exc:
            self._dlq.write(
                raw_payload=raw_event,
                error_type=self._classify_error(exc),
                error_message=str(exc),
            )
            return ProcessResult.DLQ

        enriched = self.enrich(event)
        self.write_to_feature_store(enriched)
        self.write_to_parquet(enriched)
        with self._count_lock:
            self._ingested_count += 1
        return ProcessResult.OK

    def enrich(self, event: AdEvent) -> EnrichedEvent:
        return EnrichedEvent(
            **event.model_dump(),
            ingested_at=datetime.now(timezone.utc),
        )

    def write_to_feature_store(self, event: EnrichedEvent) -> None:
        self._feature_store.increment_event_counters(event)

    def write_to_parquet(self, event: EnrichedEvent) -> None:
        self._feature_store.append_to_parquet(event)

    @property
    def ingested_count(self) -> int:
        with self._count_lock:
            return self._ingested_count

    @property
    def dlq(self) -> DeadLetterHandler:
        return self._dlq

    def _classify_error(self, exc: ValidationError) -> str:
        for error in exc.errors():
            if error["type"] in ("missing", "value_error.missing"):
                return "missing_required_field"
            if error["type"].startswith("enum"):
                return "invalid_enum_value"
        return "schema_validation_error"
