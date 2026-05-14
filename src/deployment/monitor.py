import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.config import RollbackConfig
from src.deployment.engine import DeploymentEngine, DeploymentStage

logger = logging.getLogger(__name__)


@dataclass
class PredictionLog:
    user_id: str
    model_version: str
    prediction: float
    outcome: float | None  # None until label is observed
    latency_ms: float
    logged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: bool = False


class RollbackMonitor:
    """
    Periodically evaluates online AUC, error rate, and p99 latency.
    Three independent rollback conditions — a model can degrade on any axis
    without the others being affected.
    """

    def __init__(
        self,
        engine: DeploymentEngine,
        model_name: str,
        config: RollbackConfig | None = None,
    ) -> None:
        self._engine = engine
        self._model_name = model_name
        self._cfg = config or RollbackConfig()
        self._logs: list[PredictionLog] = []
        self._baseline_auc: float | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def log_prediction(self, log: PredictionLog) -> None:
        with self._lock:
            self._logs.append(log)

    def set_baseline_auc(self, auc: float) -> None:
        self._baseline_auc = auc

    def check_now(self) -> dict:
        with self._lock:
            logs = list(self._logs)

        if not logs:
            return {"status": "no_data"}

        latencies = [log.latency_ms for log in logs]
        error_rate = sum(1 for log in logs if log.error) / len(logs)
        p99_latency = float(np.percentile(latencies, 99)) if latencies else 0.0

        labeled = [log for log in logs if log.outcome is not None]
        online_auc: float | None = None
        if len(labeled) >= 50:
            y_true = np.array([log.outcome for log in labeled])
            y_pred = np.array([log.prediction for log in labeled])
            from sklearn.metrics import roc_auc_score
            try:
                online_auc = float(roc_auc_score(y_true, y_pred))
            except ValueError:
                online_auc = None

        stage = self._engine.get_current_stage(self._model_name)
        reasons: list[str] = []
        canary_auc: float | None = None
        control_auc: float | None = None

        if stage in (DeploymentStage.CANARY, DeploymentStage.FULL):
            if error_rate > self._cfg.max_error_rate:
                reasons.append(f"error_rate={error_rate:.3f} > {self._cfg.max_error_rate}")

            if p99_latency > self._cfg.max_latency_ms:
                reasons.append(f"p99_latency={p99_latency:.1f}ms > {self._cfg.max_latency_ms}ms")

            # Canary A/B: compare canary cohort vs control cohort on the same window.
            # This is more accurate than a stale fixed baseline: a model that is 0.79
            # when control is also 0.79 should not roll back.
            if stage == DeploymentStage.CANARY:
                canary_version = self._engine.get_current_version(self._model_name)
                if canary_version:
                    canary_labeled = [l for l in labeled if l.model_version == canary_version]
                    control_labeled = [l for l in labeled if l.model_version != canary_version]
                    if len(canary_labeled) >= 30 and len(control_labeled) >= 30:
                        try:
                            canary_auc = float(roc_auc_score(
                                [l.outcome for l in canary_labeled],
                                [l.prediction for l in canary_labeled],
                            ))
                            control_auc = float(roc_auc_score(
                                [l.outcome for l in control_labeled],
                                [l.prediction for l in control_labeled],
                            ))
                            if canary_auc < control_auc - self._cfg.rollback_auc_drop:
                                reasons.append(
                                    f"canary_auc={canary_auc:.4f} < control_auc={control_auc:.4f}"
                                    f" (gap={control_auc - canary_auc:.4f})"
                                )
                        except ValueError:
                            pass

            # Fallback: absolute AUC check against stale baseline (used when A/B cohorts are too small).
            if online_auc is not None and self._baseline_auc is not None:
                if online_auc < self._baseline_auc - self._cfg.rollback_auc_drop:
                    reasons.append(
                        f"online_auc={online_auc:.4f} < baseline-threshold="
                        f"{self._baseline_auc - self._cfg.rollback_auc_drop:.4f}"
                    )

        should_rollback = bool(reasons)
        reason = "; ".join(reasons)

        if should_rollback and stage not in (DeploymentStage.ROLLED_BACK, None):
            logger.warning("ROLLBACK triggered: %s", reason)
            self._engine.rollback(self._model_name, reason)

        return {
            "error_rate": error_rate,
            "p99_latency_ms": p99_latency,
            "online_auc": online_auc,
            "canary_auc": canary_auc,
            "control_auc": control_auc,
            "rollback_triggered": should_rollback,
            "reason": reason,
        }

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.check_now()
            self._stop_event.wait(self._cfg.check_interval_s)
