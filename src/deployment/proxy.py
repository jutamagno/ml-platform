import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

import pandas as pd

from src.deployment.engine import DeploymentEngine, DeploymentStage
from src.deployment.monitor import PredictionLog, RollbackMonitor
from src.training.job import FEATURE_COLS

if TYPE_CHECKING:
    from src.feature_store.store import FeatureStore

logger = logging.getLogger(__name__)

_POPULARITY_FALLBACK = "__popularity_fallback__"
_MODEL_CACHE_MAX = 10


@dataclass
class InferenceRequest:
    user_id: str
    item_ids: list[str]
    context: dict = field(default_factory=dict)


@dataclass
class InferenceResponse:
    model_version: str
    scores: dict[str, float]  # item_id → predicted CTR
    is_fallback: bool = False


class ServingProxy:
    """
    Routes each request to the correct model version based on deployment stage.

    The proxy is the only component that knows about routing — models are
    stateless HTTP endpoints unaware of their deployment stage, making them
    simpler and routing logic independently testable.

    SHADOW: calls production + new model, returns production result, logs both.
    Fallback chain: routed model → last FULL version → popularity scores.
    """

    def __init__(
        self,
        model_name: str,
        engine: DeploymentEngine,
        model_loader: Callable[[str], object],
        monitor: RollbackMonitor | None = None,
        feature_store: "FeatureStore | None" = None,
    ) -> None:
        self._model_name = model_name
        self._engine = engine
        self._load_model = model_loader
        self._monitor = monitor
        self._feature_store = feature_store
        self._model_cache: OrderedDict[str, object] = OrderedDict()

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        stage = self._engine.get_current_stage(self._model_name)
        routed_version = self._engine.route(self._model_name, request.user_id)

        start = time.perf_counter()
        error = False

        try:
            if stage == DeploymentStage.SHADOW:
                response = self._predict_shadow(request, routed_version)
            elif stage in (DeploymentStage.CANARY, DeploymentStage.FULL, DeploymentStage.ROLLED_BACK):
                response = self._predict_version(request, routed_version)
            else:
                response = self._popularity_fallback(request)
        except Exception as exc:
            logger.error("Inference error version=%s: %s", routed_version, exc)
            error = True
            response = self._fallback(request)

        latency_ms = (time.perf_counter() - start) * 1000

        if self._monitor and not response.is_fallback:
            score = next(iter(response.scores.values()), 0.0) if response.scores else 0.0
            self._monitor.log_prediction(
                PredictionLog(
                    user_id=request.user_id,
                    model_version=response.model_version,
                    prediction=score,
                    outcome=None,
                    latency_ms=latency_ms,
                    error=error,
                )
            )

        return response

    def _predict_shadow(self, request: InferenceRequest, prod_version: str) -> InferenceResponse:
        shadow_version = self._engine.get_shadow_version(self._model_name)
        prod_response = self._predict_version(request, prod_version)
        if shadow_version and shadow_version != prod_version:
            try:
                shadow_response = self._predict_version(request, shadow_version)
                logger.debug(
                    "SHADOW user=%s prod_score=%s shadow_score=%s",
                    request.user_id,
                    list(prod_response.scores.values())[:1],
                    list(shadow_response.scores.values())[:1],
                )
            except Exception as exc:
                logger.warning("Shadow inference failed version=%s: %s", shadow_version, exc)
        return prod_response  # always return production result to caller

    def _predict_version(self, request: InferenceRequest, version: str) -> InferenceResponse:
        model = self._get_model(version)
        if model is None:
            return self._fallback(request)

        if self._feature_store is not None:
            X = self._feature_store.get_serving_features(request.user_id, request.item_ids)
        else:
            X = pd.DataFrame(index=range(len(request.item_ids)))

        for col in FEATURE_COLS:
            if col not in X.columns:
                X[col] = 0
        X_feats = X[FEATURE_COLS].fillna(0)
        scores_arr = model.predict_proba(X_feats)
        scores = {iid: float(s) for iid, s in zip(request.item_ids, scores_arr)}
        return InferenceResponse(model_version=version, scores=scores)

    def _fallback(self, request: InferenceRequest) -> InferenceResponse:
        try:
            version = self._engine.get_latest_full_version(self._model_name)
            if version:
                return self._predict_version(request, version)
        except Exception:
            pass
        return self._popularity_fallback(request)

    def _popularity_fallback(self, request: InferenceRequest) -> InferenceResponse:
        scores = {iid: 1.0 / (rank + 1) for rank, iid in enumerate(request.item_ids)}
        return InferenceResponse(model_version=_POPULARITY_FALLBACK, scores=scores, is_fallback=True)

    def _get_model(self, version: str) -> object | None:
        if version in self._model_cache:
            self._model_cache.move_to_end(version)
            return self._model_cache[version]
        try:
            model = self._load_model(version)
        except Exception as exc:
            logger.error("Failed to load model version=%s: %s", version, exc)
            return None
        if len(self._model_cache) >= _MODEL_CACHE_MAX:
            self._model_cache.popitem(last=False)
        self._model_cache[version] = model
        return model
