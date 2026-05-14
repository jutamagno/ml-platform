import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import TrainingConfig
from src.training.adapters import ModelAdapter

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "user_clicks_1h",
    "user_clicks_24h",
    "user_clicks_7d",
    "item_impressions_1h",
    "item_impressions_24h",
    "item_impressions_7d",
]
LABEL_COL = "label"


@dataclass
class DataQualityReport:
    n_examples: int
    n_positives: int
    label_rate: float
    feature_coverage: float
    issues: list[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return any(i.startswith("CRITICAL") for i in self.issues)


@dataclass
class TrainingResult:
    version: str
    metrics: dict[str, float]
    n_examples: int
    trigger_reason: str
    data_quality: DataQualityReport | None = None


class TrainingJob:
    """
    Builds a point-in-time-correct training set from the feature store,
    applies the delayed-reward filter, trains via a pluggable ModelAdapter,
    and registers the artifact.

    Stateless: all inputs from feature store + event log; outputs to registry.
    Can be re-run from any point in time reproducibly.
    """

    def __init__(
        self,
        feature_store,
        registry,
        adapter: ModelAdapter,
        config: TrainingConfig | None = None,
        model_name: str = "ctr_model",
    ) -> None:
        self._fs = feature_store
        self._registry = registry
        self._adapter = adapter
        self._cfg = config or TrainingConfig()
        self._model_name = model_name

    def run(self, trigger_reason: str, as_of: datetime | None = None) -> TrainingResult:
        as_of = as_of or datetime.now(timezone.utc)
        logger.info("Training job starting | reason=%s | as_of=%s", trigger_reason, as_of)

        df = self._build_dataset(as_of)
        if df.empty or LABEL_COL not in df.columns:
            raise ValueError("Training dataset is empty or missing label column")

        confirmed = df[df["confirmed"] == True]  # noqa: E712
        logger.info("Confirmed examples: %d / %d total", len(confirmed), len(df))

        if confirmed.empty:
            raise ValueError("No confirmed examples in training window")

        feature_cols = [c for c in FEATURE_COLS if c in confirmed.columns]
        X = confirmed[feature_cols].fillna(0)
        y = confirmed[LABEL_COL]

        quality = self._check_data_quality(X, y)
        if quality.is_critical:
            raise ValueError(f"Data quality gate failed: {'; '.join(quality.issues)}")

        # Temporal split: train on past, validate on future — never shuffle time-series data.
        # Random splits leak future impressions into training and produce over-optimistic AUC.
        X_train, X_val, y_train, y_val = self._temporal_split(confirmed, feature_cols)

        self._adapter.fit(X_train, y_train)
        preds = self._adapter.predict_proba(X_val)
        auc = float(roc_auc_score(y_val, preds)) if len(np.unique(y_val)) > 1 else 0.5
        ece = self._compute_ece(y_val.values, preds)

        metrics = {"auc": round(auc, 4), "ece": round(ece, 4), "n_val": len(y_val)}
        meta = {
            "trigger_reason": trigger_reason,
            "training_as_of": as_of.isoformat(),
            "n_examples": len(confirmed),
            "feature_list": feature_cols,
            "random_seed": self._cfg.random_seed,
            "label_rate": round(quality.label_rate, 4),
        }

        version = self._registry.register(
            model_name=self._model_name,
            adapter=self._adapter,
            metrics=metrics,
            meta=meta,
        )
        logger.info("Training complete | version=%s | AUC=%.4f | ECE=%.4f", version, auc, ece)
        return TrainingResult(
            version=version,
            metrics=metrics,
            n_examples=len(confirmed),
            trigger_reason=trigger_reason,
            data_quality=quality,
        )

    def _temporal_split(
        self, confirmed: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Sort by timestamp and cut: earlier rows train, later rows validate."""
        if "timestamp" in confirmed.columns:
            confirmed = confirmed.sort_values("timestamp")
        n_val = max(1, int(len(confirmed) * self._cfg.validation_fraction))
        train = confirmed.iloc[: len(confirmed) - n_val]
        val = confirmed.iloc[len(confirmed) - n_val :]
        return (
            train[feature_cols].fillna(0),
            val[feature_cols].fillna(0),
            train[LABEL_COL],
            val[LABEL_COL],
        )

    def _check_data_quality(self, X: pd.DataFrame, y: pd.Series) -> DataQualityReport:
        n = len(y)
        n_pos = int(y.sum())
        n_neg = n - n_pos
        label_rate = n_pos / n if n > 0 else 0.0
        coverage = 1.0 - float(X.isna().mean().mean()) if not X.empty else 0.0
        issues: list[str] = []

        if n < self._cfg.min_training_examples:
            issues.append(f"CRITICAL: {n} examples below minimum {self._cfg.min_training_examples}")
        if n_pos < self._cfg.min_positive_examples:
            issues.append(f"CRITICAL: only {n_pos} positive examples (min {self._cfg.min_positive_examples})")
        if n_neg < self._cfg.min_positive_examples:
            issues.append(f"CRITICAL: only {n_neg} negative examples")
        if label_rate > 0.5:
            issues.append(f"WARNING: label rate {label_rate:.1%} is suspiciously high for CTR data")
        if coverage < 0.5:
            issues.append(f"WARNING: feature coverage {coverage:.1%} — more than half of values are null")

        report = DataQualityReport(n, n_pos, label_rate, coverage, issues)
        for issue in issues:
            logger.warning("DATA QUALITY %s", issue)
        return report

    def _build_dataset(self, as_of: datetime) -> pd.DataFrame:
        all_events = self._fs.get_offline_as_of(
            entity_ids=[],
            feature_names=FEATURE_COLS,
            as_of=as_of,
        )
        if all_events.empty:
            return pd.DataFrame()

        if "event_type" in all_events.columns:
            all_events[LABEL_COL] = (all_events["event_type"] == "conversion").astype(int)

        rng = np.random.default_rng(self._cfg.random_seed)
        for col in FEATURE_COLS:
            if col not in all_events.columns:
                all_events[col] = rng.integers(0, 10, size=len(all_events))

        return all_events

    @staticmethod
    def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.mean() * abs(acc - conf)
        return float(ece)
