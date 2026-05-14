import hashlib
import pickle
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.config import DeploymentConfig
from src.deployment.engine import DeploymentEngine, DeploymentStage
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
def fast_config():
    return DeploymentConfig(
        min_shadow_hours=0.0,
        min_canary_hours=0.0,
        canary_fraction=0.10,
        metric_tolerance=0.1,
    )


@pytest.fixture
def engine(registry, fast_config):
    return DeploymentEngine(registry=registry, config=fast_config)


def _register_version(registry, name="ctr_model"):
    adapter = _DummyAdapter()
    return registry.register(name, adapter, {"auc": 0.75}, {"trigger_reason": "test"})


class TestInitialPromotion:
    def test_new_version_starts_in_shadow(self, engine, registry):
        v1 = _register_version(registry)
        stage = engine.promote("ctr_model", v1)
        assert stage == DeploymentStage.SHADOW
        assert engine.get_current_stage("ctr_model") == DeploymentStage.SHADOW


class TestShadowToCanary:
    def test_advances_to_canary_after_min_shadow_time(self, engine, registry):
        v1 = _register_version(registry)
        engine.promote("ctr_model", v1)
        # min_shadow_hours=0 → should advance immediately
        new_stage = engine.advance("ctr_model")
        assert new_stage == DeploymentStage.CANARY

    def test_does_not_advance_before_min_shadow_time(self, registry):
        cfg = DeploymentConfig(min_shadow_hours=2.0, min_canary_hours=0.0)
        eng = DeploymentEngine(registry=registry, config=cfg)
        v1 = _register_version(registry)
        eng.promote("ctr_model", v1)
        result = eng.advance("ctr_model")
        assert result is None


class TestCanaryRouting:
    def test_canary_routes_10_percent_of_users_to_new_model(self, engine, registry):
        import time
        v1 = _register_version(registry)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register_version(registry)
        engine.promote("ctr_model", v2)
        engine.advance("ctr_model")  # → canary

        users = [f"user_{i}" for i in range(1000)]
        routed_to_v2 = [u for u in users if engine.route("ctr_model", u) == v2]
        fraction = len(routed_to_v2) / len(users)
        assert 0.05 < fraction < 0.15, f"Expected ~10%, got {fraction:.1%}"

    def test_same_user_always_routes_to_same_model(self, engine, registry):
        import time
        v1 = _register_version(registry)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register_version(registry)
        engine.promote("ctr_model", v2)
        engine.advance("ctr_model")

        user = "consistent_user_42"
        results = {engine.route("ctr_model", user) for _ in range(20)}
        assert len(results) == 1, "Same user routed to different models"


class TestFullStage:
    def test_full_stage_routes_all_traffic_to_new_model(self, engine, registry):
        import time
        v1 = _register_version(registry)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register_version(registry)
        engine.promote("ctr_model", v2)
        engine.advance("ctr_model")  # shadow → canary
        engine.advance("ctr_model")  # canary → full

        users = [f"user_{i}" for i in range(100)]
        assert all(engine.route("ctr_model", u) == v2 for u in users)


class TestRollback:
    def test_manual_rollback_sets_stage_to_rolled_back(self, engine, registry):
        import time
        v1 = _register_version(registry)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register_version(registry)
        engine.promote("ctr_model", v2)
        engine.rollback("ctr_model", reason="manual test")
        assert engine.get_current_stage("ctr_model") == DeploymentStage.ROLLED_BACK

    def test_rollback_restores_last_full_version(self, engine, registry):
        import time
        v1 = _register_version(registry)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register_version(registry)
        engine.promote("ctr_model", v2)
        engine.advance("ctr_model")  # → canary
        engine.rollback("ctr_model", reason="test rollback")

        # After rollback, v1 should be serving all traffic
        assert engine.route("ctr_model", "any_user") == v1
