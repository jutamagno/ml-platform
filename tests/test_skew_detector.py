import numpy as np
import pytest

from src.feature_store.skew_detector import SkewDetector


@pytest.fixture
def detector():
    return SkewDetector()


class TestPSI:
    def test_psi_zero_for_identical_distributions(self, detector):
        dist = np.random.default_rng(0).normal(0, 1, 1000)
        psi = detector.compute_psi(dist, dist)
        assert psi < 0.01

    def test_psi_above_warn_threshold_for_shifted_distribution(self, detector):
        rng = np.random.default_rng(42)
        training = rng.normal(0, 1, 2000)
        serving = rng.normal(5, 1, 2000)  # heavily shifted mean
        psi = detector.compute_psi(serving, training)
        assert psi > 0.2

    def test_psi_symmetric_property(self, detector):
        rng = np.random.default_rng(7)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(2, 1, 1000)
        psi_ab = detector.compute_psi(a, b)
        psi_ba = detector.compute_psi(b, a)
        # PSI is not perfectly symmetric but should be in the same order of magnitude
        assert abs(psi_ab - psi_ba) < max(psi_ab, psi_ba)


class TestReport:
    def test_report_returns_psi_for_each_feature(self, detector):
        rng = np.random.default_rng(0)
        detector.record_serving("feature_a", rng.normal(0, 1, 500))
        detector.record_training("feature_a", rng.normal(0, 1, 500))
        detector.record_serving("feature_b", rng.normal(0, 1, 500))
        detector.record_training("feature_b", rng.normal(5, 1, 500))

        report = detector.report()
        assert "feature_a" in report
        assert "feature_b" in report
        assert report["feature_a"] < 0.05
        assert report["feature_b"] > 0.2

    def test_report_excludes_features_missing_from_one_side(self, detector):
        rng = np.random.default_rng(1)
        detector.record_serving("only_serving", rng.normal(0, 1, 100))
        # no training sample for this feature
        report = detector.report()
        assert "only_serving" not in report

    def test_report_returns_one_score_per_monitored_feature(self, detector):
        rng = np.random.default_rng(2)
        for i in range(5):
            vals = rng.normal(i, 1, 200)
            detector.record_serving(f"f{i}", vals)
            detector.record_training(f"f{i}", vals)
        report = detector.report()
        assert len(report) == 5
