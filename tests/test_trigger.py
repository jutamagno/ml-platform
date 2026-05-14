from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from src.config import TriggerConfig
from src.training.trigger import TrainingTrigger


@pytest.fixture
def trigger():
    cfg = TriggerConfig(event_count_threshold=100, schedule_cron="0 2 * * *")
    return TrainingTrigger(config=cfg)


class TestEventCountTrigger:
    def test_does_not_fire_below_threshold(self, trigger):
        trigger.record_confirmed_examples(99)
        should, reason = trigger.should_trigger()
        assert not should

    def test_fires_at_exactly_threshold(self, trigger):
        trigger.record_confirmed_examples(100)
        should, reason = trigger.should_trigger()
        assert should
        assert "event_count" in reason

    def test_fires_above_threshold(self, trigger):
        trigger.record_confirmed_examples(150)
        should, _ = trigger.should_trigger()
        assert should

    def test_mark_triggered_resets_counter_to_zero(self, trigger):
        trigger.record_confirmed_examples(100)
        trigger.mark_triggered()
        assert trigger.event_counter == 0

    def test_does_not_fire_after_reset(self, trigger):
        trigger.record_confirmed_examples(100)
        trigger.mark_triggered()
        trigger.mark_complete()
        should, _ = trigger.should_trigger()
        assert not should


class TestScheduleTrigger:
    def test_fires_at_scheduled_cron_time(self):
        cfg = TriggerConfig(event_count_threshold=10_000, schedule_cron="0 2 * * *")
        with freeze_time("2026-05-15 02:00:00"):
            trigger = TrainingTrigger(config=cfg)
            # Force next_schedule to be in the past
            trigger._next_schedule = datetime(2026, 5, 15, 1, 59, 0)
            should, reason = trigger.should_trigger()
        assert should
        assert reason == "schedule"

    def test_does_not_fire_before_scheduled_time(self):
        cfg = TriggerConfig(event_count_threshold=10_000, schedule_cron="0 2 * * *")
        with freeze_time("2026-05-15 01:00:00"):
            trigger = TrainingTrigger(config=cfg)
            # next_schedule will be in the future (02:00)
            should, _ = trigger.should_trigger()
        assert not should


class TestConcurrencyLock:
    def test_does_not_fire_when_training_is_running(self, trigger):
        trigger.record_confirmed_examples(100)
        trigger.mark_triggered()  # sets running=True
        should, reason = trigger.should_trigger()
        assert not should
        assert reason == "training_already_running"

    def test_fires_again_after_mark_complete(self, trigger):
        trigger.record_confirmed_examples(100)
        trigger.mark_triggered()
        trigger.mark_complete()
        trigger.record_confirmed_examples(100)  # accumulate again after reset
        should, _ = trigger.should_trigger()
        assert should
