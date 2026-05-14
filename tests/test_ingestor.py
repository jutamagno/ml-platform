import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.ingestor.dead_letter import DeadLetterHandler
from src.ingestor.ingestor import EventIngestor
from src.ingestor.schemas import ProcessResult


def _make_event(**overrides):
    base = {
        "event_id": str(uuid.uuid4()),
        "event_type": "click",
        "user_id": "u1",
        "item_id": "i1",
        "timestamp": "2026-01-01T12:00:00",
    }
    base.update(overrides)
    return base


@pytest.fixture
def fs():
    mock = MagicMock()
    return mock


@pytest.fixture
def dlq():
    return DeadLetterHandler()


@pytest.fixture
def ingestor(fs, dlq):
    return EventIngestor(feature_store=fs, dlq=dlq)


class TestValidEvent:
    def test_valid_event_routes_to_feature_store_and_parquet(self, ingestor, fs):
        result = ingestor.process(_make_event())
        assert result == ProcessResult.OK
        fs.increment_event_counters.assert_called_once()
        fs.append_to_parquet.assert_called_once()

    def test_valid_event_increments_ingested_count(self, ingestor):
        ingestor.process(_make_event())
        ingestor.process(_make_event())
        assert ingestor.ingested_count == 2


class TestMissingRequiredField:
    def test_missing_required_field_routes_to_dlq(self, ingestor, fs, dlq):
        bad = _make_event()
        del bad["user_id"]
        result = ingestor.process(bad)
        assert result == ProcessResult.DLQ
        fs.increment_event_counters.assert_not_called()
        fs.append_to_parquet.assert_not_called()

    def test_missing_field_error_type_is_missing_required_field(self, ingestor, dlq):
        bad = _make_event()
        del bad["event_type"]
        ingestor.process(bad)
        report = dlq.daily_report()
        assert "missing_required_field" in report["error_type_breakdown"]

    def test_dlq_report_contains_correct_error_type_for_invalid_enum(self, ingestor, dlq):
        bad = _make_event(event_type="unknown_type")
        ingestor.process(bad)
        report = dlq.daily_report()
        assert report["total_dlq_events"] == 1
        assert any("invalid_enum" in k or "schema" in k for k in report["error_type_breakdown"])


class TestSchemaEvolution:
    def test_unknown_extra_field_is_accepted(self, ingestor, fs):
        event = _make_event(new_field_from_producer="some_value", another_new_field=42)
        result = ingestor.process(event)
        assert result == ProcessResult.OK
        fs.increment_event_counters.assert_called_once()


class TestDLQReport:
    def test_dlq_report_tracks_multiple_errors(self, ingestor, dlq):
        bad1 = _make_event()
        del bad1["user_id"]
        bad2 = _make_event(event_type="bad_type")
        ingestor.process(bad1)
        ingestor.process(bad2)
        report = dlq.daily_report()
        assert report["total_dlq_events"] == 2
        assert len(report["error_type_breakdown"]) >= 1

    def test_dlq_report_includes_sample_payloads(self, ingestor, dlq):
        bad = _make_event()
        del bad["event_id"]
        ingestor.process(bad)
        report = dlq.daily_report()
        assert len(report["sample_payloads"]) >= 1
