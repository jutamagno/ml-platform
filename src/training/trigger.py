import threading
from datetime import datetime, timezone

from croniter import croniter

from src.config import TriggerConfig


class TrainingTrigger:
    """
    Two trigger modes that can be combined:
    1. EVENT_COUNT: fires after N confirmed new examples since last training.
    2. SCHEDULE: fires on a cron regardless of event volume.

    Both check a lock before triggering — prevents concurrent training runs.
    """

    def __init__(self, config: TriggerConfig | None = None) -> None:
        self._cfg = config or TriggerConfig()
        self._event_counter: int = 0
        self._lock = threading.Lock()
        self._running = False
        self._last_triggered_at: datetime | None = None
        self._cron = croniter(self._cfg.schedule_cron)
        self._next_schedule: datetime = self._cron.get_next(datetime)

    def record_confirmed_example(self) -> None:
        with self._lock:
            self._event_counter += 1

    def record_confirmed_examples(self, n: int) -> None:
        with self._lock:
            self._event_counter += n

    def should_trigger(self) -> tuple[bool, str]:
        with self._lock:
            if self._running:
                return False, "training_already_running"

            if self._event_counter >= self._cfg.event_count_threshold:
                return True, f"event_count={self._event_counter}"

            now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            if now >= self._next_schedule:
                return True, "schedule"

            return False, "no_trigger"

    def mark_triggered(self) -> None:
        with self._lock:
            self._event_counter = 0
            self._running = True
            self._last_triggered_at = datetime.now(timezone.utc)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            while self._next_schedule <= now:
                self._next_schedule = self._cron.get_next(datetime)

    def mark_complete(self) -> None:
        with self._lock:
            self._running = False

    @property
    def event_counter(self) -> int:
        with self._lock:
            return self._event_counter

    @property
    def is_running(self) -> bool:
        return self._running
