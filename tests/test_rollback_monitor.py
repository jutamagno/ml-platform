import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import DeploymentConfig, RollbackConfig
from src.deployment.engine import DeploymentEngine, DeploymentStage
from src.deployment.monitor import PredictionLog, RollbackMonitor
from src.registry.registry import ModelRegistry


class _DummyAdapter:
    def fit(self, X, y): pass
    def predict_proba(self, X): return np.zeros(len(X))
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump({}, f)
    def load(self, path): pass


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=str(tmp_path / "registry"))


@pytest.fixture
def engine(registry):
    cfg = DeploymentConfig(min_shadow_hours=0, min_canary_hours=0)
    return DeploymentEngine(registry=registry, config=cfg)


@pytest.fixture
def rollback_cfg():
    return RollbackConfig(rollback_auc_drop=0.01, max_error_rate=0.01, max_latency_ms=100.0)


def _register(registry):
    adapter = _DummyAdapter()
    return registry.register("ctr_model", adapter, {"auc": 0.75}, {})


def _setup_canary(engine, registry):
    v1 = _register(registry)
    registry.promote("ctr_model", v1, "full")
    time.sleep(0.01)
    v2 = _register(registry)
    engine.promote("ctr_model", v2)
    engine.advance("ctr_model")  # → canary
    return v1, v2


def _make_logs(n=100, prediction=0.7, outcome=1.0, latency_ms=20.0, error=False):
    return [
        PredictionLog(
            user_id=f"u{i}",
            model_version="v1",
            prediction=prediction if outcome == 1.0 else 1 - prediction,
            outcome=outcome if i % 5 == 0 else 0.0,
            latency_ms=latency_ms,
            error=error,
        )
        for i in range(n)
    ]


class TestAUCRollback:
    def test_rollback_triggers_when_auc_drops_below_threshold(self, engine, registry, rollback_cfg):
        v1, v2 = _setup_canary(engine, registry)
        monitor = RollbackMonitor(engine, "ctr_model", rollback_cfg)
        monitor.set_baseline_auc(0.80)

        # Log predictions that yield low AUC (random predictions)
        rng = np.random.default_rng(0)
        for i in range(100):
            monitor.log_prediction(PredictionLog(
                user_id=f"u{i}",
                model_version=v2,
                prediction=rng.random(),  # random = bad AUC
                outcome=float(rng.integers(0, 2)),
                latency_ms=10.0,
            ))
        result = monitor.check_now()
        assert result["rollback_triggered"]

    def test_no_rollback_when_auc_is_within_tolerance(self, engine, registry, rollback_cfg):
        v1, v2 = _setup_canary(engine, registry)
        monitor = RollbackMonitor(engine, "ctr_model", rollback_cfg)
        monitor.set_baseline_auc(0.70)

        # Perfect predictions → high AUC
        for i in range(100):
            monitor.log_prediction(PredictionLog(
                user_id=f"u{i}", model_version=v2,
                prediction=float(i % 2), outcome=float(i % 2), latency_ms=10.0,
            ))
        result = monitor.check_now()
        assert not result["rollback_triggered"]


class TestErrorRateRollback:
    def test_rollback_triggers_when_error_rate_exceeds_max(self, engine, registry, rollback_cfg):
        _setup_canary(engine, registry)
        monitor = RollbackMonitor(engine, "ctr_model", rollback_cfg)

        # 5% error rate → above 1% threshold
        for i in range(100):
            monitor.log_prediction(PredictionLog(
                user_id=f"u{i}", model_version="v2",
                prediction=0.5, outcome=None, latency_ms=10.0,
                error=(i < 5),
            ))
        result = monitor.check_now()
        assert result["rollback_triggered"]
        assert "error_rate" in result["reason"]


class TestLatencyRollback:
    def test_rollback_triggers_when_p99_latency_exceeds_max(self, engine, registry, rollback_cfg):
        _setup_canary(engine, registry)
        monitor = RollbackMonitor(engine, "ctr_model", rollback_cfg)

        latencies = [10.0] * 95 + [200.0] * 5  # p99 → 200ms
        for i, lat in enumerate(latencies):
            monitor.log_prediction(PredictionLog(
                user_id=f"u{i}", model_version="v2",
                prediction=0.5, outcome=None, latency_ms=lat,
            ))
        result = monitor.check_now()
        assert result["rollback_triggered"]
        assert "latency" in result["reason"]

    def test_no_rollback_when_all_metrics_within_tolerance(self, engine, registry, rollback_cfg):
        _setup_canary(engine, registry)
        monitor = RollbackMonitor(engine, "ctr_model", rollback_cfg)
        monitor.set_baseline_auc(0.70)

        for i in range(100):
            monitor.log_prediction(PredictionLog(
                user_id=f"u{i}", model_version="v2",
                prediction=float(i % 2), outcome=float(i % 2), latency_ms=15.0,
            ))
        result = monitor.check_now()
        assert not result["rollback_triggered"]
