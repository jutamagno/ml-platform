from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


class EventType(str, Enum):
    IMPRESSION = "impression"
    CLICK = "click"
    CONVERSION = "conversion"


class AdEvent(BaseModel):
    model_config = {"extra": "ignore"}  # forward-compat: unknown fields are silently ignored

    event_id: str
    event_type: EventType
    user_id: str
    item_id: str
    timestamp: datetime
    session_id: str | None = None
    # Optional enrichment fields — producers may omit
    publisher_id: str | None = None
    campaign_id: str | None = None
    revenue: float | None = None  # only on conversions

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, (int, float)):
            return datetime.utcfromtimestamp(v)
        return v


class EnrichedEvent(AdEvent):
    ingested_at: datetime
    confirmed: bool = False  # set True once conversion window has passed


class DLQRecord(BaseModel):
    raw_payload: dict
    error_type: str
    error_message: str
    producer_id: str | None
    failed_at: datetime


class ProcessResult(str, Enum):
    OK = "ok"
    DLQ = "dlq"
