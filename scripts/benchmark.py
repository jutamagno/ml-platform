"""
Latency benchmark for ServingProxy.

Measures p50/p95/p99 inference latency across deployment stages.
Run: python scripts/benchmark.py
"""
import pickle
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import fakeredis

from src.config import DeploymentConfig, FeatureStoreConfig
from src.deployment.engine import DeploymentEngine
from src.deployment.proxy import InferenceRequest, ServingProxy
from src.feature_store.store import FeatureStore
from src.registry.registry import ModelRegistry

N_WARMUP = 50
N_REQUESTS = 1_000
N_ITEMS_PER_REQUEST = 10


class _FastAdapter:
    """Minimal adapter: predict_proba using a pre-fitted random model."""

    def fit(self, X, y):
        import numpy as np
        self._weights = np.random.rand(X.shape[1])

    def predict_proba(self, X):
        import numpy as np
        return 1 / (1 + np.exp(-X.values @ self._weights))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._weights, f)

    def load(self, path):
        with open(path, "rb") as f:
            self._weights = pickle.load(f)


def _percentile(data: list[float], p: int) -> float:
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


def _print_row(label: str, latencies: list[float]) -> None:
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    mean = statistics.mean(latencies)
    print(f"  {label:<20}  mean={mean*1000:6.2f}ms  p50={p50*1000:6.2f}ms  p95={p95*1000:6.2f}ms  p99={p99*1000:6.2f}ms")


def _setup_registry(tmp_dir: Path):
    import pandas as pd
    registry = ModelRegistry(root=str(tmp_dir / "registry"))

    for name in ("v1", "v2"):
        adapter = _FastAdapter()
        import numpy as np
        X = pd.DataFrame(np.random.rand(200, 6), columns=[
            "user_clicks_1h", "user_clicks_24h", "user_clicks_7d",
            "item_impressions_1h", "item_impressions_24h", "item_impressions_7d",
        ])
        y = pd.Series((X.iloc[:, 0] + X.iloc[:, 1] > 1.0).astype(int))
        adapter.fit(X, y)
        registry.register("ctr_model", adapter, {"auc": 0.75}, {"trigger_reason": "bench"})

    return registry


def _model_loader(registry: ModelRegistry, tmp_dir: Path):
    def loader(version: str) -> _FastAdapter:
        adapter = _FastAdapter()
        path = tmp_dir / "registry" / "ctr_model" / version / "model.pkl"
        adapter.load(str(path))
        return adapter
    return loader


def _make_requests(n: int) -> list[InferenceRequest]:
    return [
        InferenceRequest(
            user_id=f"user_{i % 500}",
            item_ids=[f"item_{j}" for j in range(N_ITEMS_PER_REQUEST)],
        )
        for i in range(n)
    ]


def run_benchmark(tmp_dir: Path) -> None:
    print("\n" + "=" * 62)
    print("  ServingProxy Latency Benchmark")
    print(f"  {N_REQUESTS} requests × {N_ITEMS_PER_REQUEST} items/request")
    print("=" * 62)

    registry = _setup_registry(tmp_dir)
    versions = registry.list_versions("ctr_model")
    v1, v2 = versions[0].version, versions[1].version

    cfg = DeploymentConfig(min_shadow_hours=0, min_canary_hours=0)
    requests = _make_requests(N_WARMUP + N_REQUESTS)
    loader = _model_loader(registry, tmp_dir)

    stages = [
        ("SHADOW",  lambda e: (e.promote("ctr_model", v2), None)),
        ("CANARY",  lambda e: (e.promote("ctr_model", v2), e.advance("ctr_model"))),
        ("FULL",    lambda e: (e.promote("ctr_model", v2), e.advance("ctr_model"), e.advance("ctr_model"))),
    ]

    print()
    for stage_name, setup_fn in stages:
        engine = DeploymentEngine(registry=registry, config=cfg)
        # For CANARY and FULL we need v1 in FULL first
        registry.promote("ctr_model", v1, "full")
        setup_fn(engine)
        proxy = ServingProxy("ctr_model", engine, loader)

        # Warmup
        for req in requests[:N_WARMUP]:
            proxy.predict(req)

        # Measure
        latencies = []
        for req in requests[N_WARMUP:]:
            t0 = time.perf_counter()
            proxy.predict(req)
            latencies.append(time.perf_counter() - t0)

        _print_row(stage_name, latencies)

    # Baseline: no ML model (popularity fallback)
    engine = DeploymentEngine(registry=registry, config=cfg)

    def always_fail(version):
        raise RuntimeError("unavailable")

    proxy = ServingProxy("ctr_model", engine, always_fail)
    latencies = []
    for req in requests[N_WARMUP:]:
        t0 = time.perf_counter()
        proxy.predict(req)
        latencies.append(time.perf_counter() - t0)
    _print_row("POPULARITY FALLBACK", latencies)

    print()
    print(f"  Rollback budget:  {100.0:.0f}ms p99 threshold")
    print(f"  Shadow overhead:  dual inference (both models run; prod result returned)")
    print()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_benchmark(Path(tmp))
