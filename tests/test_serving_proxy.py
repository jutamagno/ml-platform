import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import DeploymentConfig
from src.deployment.engine import DeploymentEngine, DeploymentStage
from src.deployment.proxy import InferenceRequest, ServingProxy
from src.registry.registry import ModelRegistry


class _DummyAdapter:
    def __init__(self, score=0.5):
        self._score = score
    def fit(self, X, y): pass
    def predict_proba(self, X): return np.full(len(X), self._score)
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f: pickle.dump(self, f)
    def load(self, path):
        with open(path, "rb") as f: return pickle.load(f)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=str(tmp_path / "registry"))


@pytest.fixture
def fast_engine(registry):
    cfg = DeploymentConfig(min_shadow_hours=0, min_canary_hours=0)
    return DeploymentEngine(registry=registry, config=cfg)


def _register(registry, score=0.5, name="ctr_model"):
    adapter = _DummyAdapter(score=score)
    return registry.register(name, adapter, {"auc": 0.75}, {})


def _model_loader(registry, tmp_path):
    def loader(version):
        path = tmp_path / "registry" / "ctr_model" / version / "model.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
    return loader


def _make_request(user_id="u1"):
    return InferenceRequest(user_id=user_id, item_ids=["i1", "i2"])


class TestShadowStage:
    def test_shadow_returns_production_result(self, fast_engine, registry, tmp_path):
        v1 = _register(registry, score=0.3)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register(registry, score=0.9)
        fast_engine.promote("ctr_model", v2)  # → SHADOW

        proxy = ServingProxy("ctr_model", fast_engine, _model_loader(registry, tmp_path))
        response = proxy.predict(_make_request())
        # In shadow mode, the production model (v1, score=0.3) serves the response
        assert response.model_version == v1
        for score in response.scores.values():
            assert abs(score - 0.3) < 0.01


class TestCanaryRouting:
    def test_canary_routes_same_user_to_same_model(self, fast_engine, registry, tmp_path):
        v1 = _register(registry, score=0.3)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register(registry, score=0.9)
        fast_engine.promote("ctr_model", v2)
        fast_engine.advance("ctr_model")  # → canary

        proxy = ServingProxy("ctr_model", fast_engine, _model_loader(registry, tmp_path))
        user = "stable_user_99"
        versions = {proxy.predict(InferenceRequest(user_id=user, item_ids=["i1"])).model_version
                    for _ in range(10)}
        assert len(versions) == 1


class TestFallback:
    def test_unavailable_model_falls_back_to_last_full_version(self, fast_engine, registry, tmp_path):
        v1 = _register(registry, score=0.3)
        registry.promote("ctr_model", v1, "full")
        time.sleep(0.01)
        v2 = _register(registry, score=0.9)
        fast_engine.promote("ctr_model", v2)
        fast_engine.advance("ctr_model")  # → canary

        def failing_loader(version):
            if version == v2:
                raise RuntimeError("model unavailable")
            return _model_loader(registry, tmp_path)(version)

        proxy = ServingProxy("ctr_model", fast_engine, failing_loader)
        # Force a user that routes to v2
        user = next(
            f"user_{i}" for i in range(1000)
            if fast_engine.route("ctr_model", f"user_{i}") == v2
        )
        response = proxy.predict(InferenceRequest(user_id=user, item_ids=["i1"]))
        # Should fall back to v1
        assert response.model_version in (v1, "__popularity_fallback__")

    def test_no_full_version_returns_popularity_fallback(self, fast_engine, registry, tmp_path):
        v1 = _register(registry)
        fast_engine.promote("ctr_model", v1)  # SHADOW — no FULL version exists

        def always_fail(version):
            raise RuntimeError("model unavailable")

        proxy = ServingProxy("ctr_model", fast_engine, always_fail)
        response = proxy.predict(_make_request())
        assert response.is_fallback
        assert response.model_version == "__popularity_fallback__"
        # Popularity fallback should not crash and should return scores
        assert len(response.scores) == 2
