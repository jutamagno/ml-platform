import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum

from src.config import DeploymentConfig
from src.registry.registry import ModelRegistry

logger = logging.getLogger(__name__)


class DeploymentStage(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"
    ROLLED_BACK = "rolled_back"


class _Deployment:
    def __init__(self, model_name: str, version: str, stage: DeploymentStage) -> None:
        self.model_name = model_name
        self.version = version
        self.stage = stage
        self.started_at: datetime = datetime.now(timezone.utc)
        self.stage_entered_at: datetime = datetime.now(timezone.utc)
        self.previous_full_version: str | None = None


class DeploymentEngine:
    """
    Manages staged rollout: SHADOW → CANARY → FULL → ROLLED_BACK.

    SHADOW: new model runs in parallel but results are discarded (catches
    crashes before real users are affected).

    CANARY: 10% traffic by deterministic user_id hash — same user always
    sees the same model during canary, preventing inconsistent experiences.

    ROLLBACK target: the last FULL version, not the shadow/canary under test.
    """

    def __init__(self, registry: ModelRegistry, config: DeploymentConfig | None = None) -> None:
        self._registry = registry
        self._cfg = config or DeploymentConfig()
        self._deployments: dict[str, _Deployment] = {}
        self._lock = threading.Lock()

    def promote(self, model_name: str, version: str) -> DeploymentStage:
        with self._lock:
            last_full = self._get_last_full_version(model_name)

            deployment = _Deployment(model_name, version, DeploymentStage.SHADOW)
            deployment.previous_full_version = last_full
            self._deployments[model_name] = deployment
            self._registry.promote(model_name, version, DeploymentStage.SHADOW.value)
            logger.info("DEPLOY %s v%s → SHADOW", model_name, version)
            return DeploymentStage.SHADOW

    def advance(self, model_name: str, online_metrics: dict | None = None) -> DeploymentStage | None:
        """Advance stage if conditions are met. Returns new stage or None if not ready."""
        with self._lock:
            dep = self._deployments.get(model_name)
            if dep is None:
                return None

            elapsed = (datetime.now(timezone.utc) - dep.stage_entered_at).total_seconds() / 3600

            if dep.stage == DeploymentStage.SHADOW:
                if elapsed < self._cfg.min_shadow_hours:
                    return None
                if not self._metrics_ok(model_name, online_metrics):
                    return None
                dep.stage = DeploymentStage.CANARY
                dep.stage_entered_at = datetime.now(timezone.utc)
                self._registry.promote(model_name, dep.version, DeploymentStage.CANARY.value)
                logger.info("DEPLOY %s v%s → CANARY (10%%)", model_name, dep.version)
                return DeploymentStage.CANARY

            if dep.stage == DeploymentStage.CANARY:
                if elapsed < self._cfg.min_canary_hours:
                    return None
                if not self._metrics_ok(model_name, online_metrics):
                    return None
                dep.stage = DeploymentStage.FULL
                dep.stage_entered_at = datetime.now(timezone.utc)
                self._registry.promote(model_name, dep.version, DeploymentStage.FULL.value)
                logger.info("DEPLOY %s v%s → FULL", model_name, dep.version)
                return DeploymentStage.FULL

        return None

    def rollback(self, model_name: str, reason: str) -> None:
        with self._lock:
            dep = self._deployments.get(model_name)
            if dep is None:
                logger.warning("ROLLBACK requested for %s but no active deployment", model_name)
                return

            previous = dep.previous_full_version
            # Keep dep as current deployment but mark it ROLLED_BACK so get_current_stage() reflects it.
            # route() in ROLLED_BACK state serves previous_full_version.
            dep.stage = DeploymentStage.ROLLED_BACK
            self._registry.promote(model_name, dep.version, DeploymentStage.ROLLED_BACK.value)
            logger.warning("ROLLBACK %s v%s → ROLLED_BACK | reason=%s", model_name, dep.version, reason)

            if previous:
                self._registry.promote(model_name, previous, DeploymentStage.FULL.value)
                logger.info("RESTORED %s v%s → FULL", model_name, previous)

    def get_current_stage(self, model_name: str) -> DeploymentStage | None:
        dep = self._deployments.get(model_name)
        return dep.stage if dep else None

    def route(self, model_name: str, user_id: str) -> str | None:
        """
        Returns the model version to serve for this user_id.
        In CANARY: deterministic 10% split by hash — same user always gets same model.
        """
        dep = self._deployments.get(model_name)
        if dep is None:
            return self._fallback_version(model_name)

        if dep.stage == DeploymentStage.FULL:
            return dep.version

        if dep.stage == DeploymentStage.CANARY:
            bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
            if bucket < int(self._cfg.canary_fraction * 100):
                return dep.version
            return dep.previous_full_version or dep.version

        if dep.stage == DeploymentStage.SHADOW:
            # Shadow: always route to production; caller handles shadow inference
            return dep.previous_full_version or dep.version

        # ROLLED_BACK — route to the previous FULL version
        return dep.previous_full_version or dep.version

    def get_shadow_version(self, model_name: str) -> str | None:
        dep = self._deployments.get(model_name)
        if dep and dep.stage == DeploymentStage.SHADOW:
            return dep.version
        return None

    def get_current_version(self, model_name: str) -> str | None:
        dep = self._deployments.get(model_name)
        return dep.version if dep else None

    def get_latest_full_version(self, model_name: str) -> str | None:
        return self._get_last_full_version(model_name)

    def _fallback_version(self, model_name: str) -> str | None:
        return self._get_last_full_version(model_name)

    def _get_last_full_version(self, model_name: str) -> str | None:
        try:
            mv = self._registry.get_latest(model_name, stage="full")
            return mv.version
        except KeyError:
            return None

    def _metrics_ok(self, model_name: str, online_metrics: dict | None) -> bool:
        if online_metrics is None:
            return True  # no metrics yet → optimistic advance
        production_auc = online_metrics.get("production_auc", 0)
        candidate_auc = online_metrics.get("candidate_auc", production_auc)
        return candidate_auc >= production_auc - self._cfg.metric_tolerance
