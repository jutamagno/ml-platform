import pickle
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.registry.registry import ModelRegistry


class _DummyAdapter:
    def fit(self, X, y): pass
    def predict_proba(self, X): return np.zeros(len(X))
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=str(tmp_path / "registry"))


@pytest.fixture
def adapter():
    return _DummyAdapter()


def _register(registry, adapter, name="ctr_model", metrics=None, meta=None):
    return registry.register(
        model_name=name,
        adapter=adapter,
        metrics=metrics or {"auc": 0.75},
        meta=meta or {"trigger_reason": "test"},
    )


class TestRegister:
    def test_registered_version_is_retrievable(self, registry, adapter):
        version = _register(registry, adapter)
        mv = registry.get_version("ctr_model", version)
        assert mv.version == version
        assert mv.metrics["auc"] == 0.75

    def test_re_registration_of_same_version_raises(self, registry, adapter, tmp_path):
        # Manually create the directory to simulate an existing version
        version = _register(registry, adapter)
        # Patch to force collision by pre-creating the dir
        version_dir = tmp_path / "registry" / "ctr_model" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="already registered"):
            from unittest.mock import patch
            with patch("src.registry.registry.datetime") as mock_dt:
                mock_dt.now.return_value.strftime.return_value = version
                registry.register("ctr_model", adapter, {"auc": 0.8}, {})


class TestGetLatest:
    def test_get_latest_production_returns_most_recently_promoted(self, registry, adapter):
        import time
        v1 = _register(registry, adapter, metrics={"auc": 0.70})
        time.sleep(0.01)
        v2 = _register(registry, adapter, metrics={"auc": 0.72})
        registry.promote("ctr_model", v1, "production")
        registry.promote("ctr_model", v2, "production")
        latest = registry.get_latest("ctr_model", stage="production")
        assert latest.version == v2

    def test_get_latest_raises_when_no_production_version(self, registry, adapter):
        _register(registry, adapter)
        with pytest.raises(KeyError):
            registry.get_latest("ctr_model", stage="production")


class TestCompare:
    def test_compare_returns_metric_differences(self, registry, adapter):
        import time
        v1 = _register(registry, adapter, metrics={"auc": 0.70, "ece": 0.05})
        time.sleep(0.01)
        v2 = _register(registry, adapter, metrics={"auc": 0.75, "ece": 0.03})
        df = registry.compare("ctr_model", v1, v2)
        auc_row = df[df["metric"] == "auc"].iloc[0]
        assert abs(auc_row["diff"] - 0.05) < 1e-9


class TestImmutability:
    def test_artifact_is_immutable_after_registration(self, registry, adapter):
        version = _register(registry, adapter, metrics={"auc": 0.70})
        # Metrics can be updated once (to add post-deployment online metrics)
        registry.update_metrics("ctr_model", version, {"online_auc": 0.71})
        # After the one allowed update, further updates raise
        with pytest.raises(ValueError, match="locked"):
            registry.update_metrics("ctr_model", version, {"online_auc": 0.72})
