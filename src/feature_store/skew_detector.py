import logging

import numpy as np

from src.config import SkewDetectorConfig

logger = logging.getLogger(__name__)


class SkewDetector:
    """
    Compares feature distributions between serving and training using PSI.

    PSI over KL divergence: PSI is symmetric and stable even when the
    reference distribution has zero-count bins (KL diverges to infinity there).
    """

    def __init__(self, config: SkewDetectorConfig | None = None) -> None:
        self._cfg = config or SkewDetectorConfig()
        self._serving_samples: dict[str, np.ndarray] = {}
        self._training_samples: dict[str, np.ndarray] = {}

    def record_serving(self, feature_name: str, values: np.ndarray) -> None:
        self._serving_samples[feature_name] = values

    def record_training(self, feature_name: str, values: np.ndarray) -> None:
        self._training_samples[feature_name] = values

    def compute_psi(self, serving_dist: np.ndarray, training_dist: np.ndarray) -> float:
        """
        PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        serving_dist = observed (actual), training_dist = expected (baseline).

        Bins are defined by the training distribution's quantiles (standard PSI
        practice). Using the combined distribution can collapse both point-mass
        distributions into a single bin, making PSI = 0 even for maximal drift.
        """
        if len(serving_dist) == 0 or len(training_dist) == 0:
            return 0.0
        n_bins = self._cfg.n_bins
        bin_edges = np.percentile(training_dist, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)

        # If training is a point mass, extend the range to capture the serving values
        if len(bin_edges) < 2:
            center = bin_edges[0]
            serving_min = serving_dist.min()
            serving_max = serving_dist.max()
            lo = min(center, serving_min) - 1e-6
            hi = max(center, serving_max) + 1e-6
            bin_edges = np.linspace(lo, hi, n_bins + 1)

        # Extend edges to ensure serving values that fall outside training range are captured
        bin_edges[0] = min(bin_edges[0], serving_dist.min()) - 1e-6
        bin_edges[-1] = max(bin_edges[-1], serving_dist.max()) + 1e-6

        actual, _ = np.histogram(serving_dist, bins=bin_edges)
        expected, _ = np.histogram(training_dist, bins=bin_edges)

        # Replace zeros to avoid log(0) — PSI is undefined for empty bins
        eps = 1e-6
        actual_pct = actual / (actual.sum() + eps)
        expected_pct = expected / (expected.sum() + eps)
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)
        expected_pct = np.where(expected_pct == 0, eps, expected_pct)

        return max(0.0, float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))))

    def report(self) -> dict[str, float]:
        """Returns {feature_name: psi_score} for all monitored features."""
        results: dict[str, float] = {}
        features = set(self._serving_samples) & set(self._training_samples)
        for feature in features:
            psi = self.compute_psi(
                self._serving_samples[feature],
                self._training_samples[feature],
            )
            results[feature] = psi
            if psi > self._cfg.psi_alert_threshold:
                logger.warning("SKEW ALERT feature=%s PSI=%.3f (threshold=%.2f)", feature, psi, self._cfg.psi_alert_threshold)
            elif psi > self._cfg.psi_warn_threshold:
                logger.warning("SKEW WARNING feature=%s PSI=%.3f (threshold=%.2f)", feature, psi, self._cfg.psi_warn_threshold)
        return results
