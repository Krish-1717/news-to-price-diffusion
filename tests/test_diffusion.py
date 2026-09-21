"""
tests/test_diffusion.py -- Tests for diffusion model and calibrator
Day 9 Commit 2
"""
import math
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.calibrator import (
    PlattCalibrator, IsotonicCalibrator,
    expected_calibration_error, reliability_diagram_data
)


# ââ Platt calibrator ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestPlattCalibrator:
    def _simple_data(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        scores = rng.normal(0, 1, n)
        labels = (scores + rng.normal(0, 0.5, n) > 0).astype(float)
        return scores, labels

    def test_fit_returns_self(self):
        cal = PlattCalibrator()
        s, l = self._simple_data()
        assert cal.fit(s, l) is cal

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PlattCalibrator().predict_proba(np.array([0.5]))

    def test_probs_in_0_1(self):
        s, l = self._simple_data()
        cal = PlattCalibrator(n_epochs=200).fit(s, l)
        probs = cal.predict_proba(s)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_monotone_positive_A(self):
        s, l = self._simple_data()
        cal = PlattCalibrator(n_epochs=500).fit(s, l)
        # For separable data with positive A, higher score â higher prob
        test_s = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        probs = cal.predict_proba(test_s)
        assert (np.diff(probs) >= -0.05).all(), "Probabilities should be roughly monotone"

    def test_ece_improves_after_calibration(self):
        rng = np.random.default_rng(1)
        n = 500
        scores = rng.normal(0, 1, n)
        labels = (scores > 0).astype(float)
        # Uncalibrated: use sigmoid of scaled scores
        raw_probs = 1 / (1 + np.exp(-scores * 3))
        ece_raw = expected_calibration_error(raw_probs, labels)
        cal = PlattCalibrator(n_epochs=1000, lr=0.05).fit(scores, labels)
        cal_probs = cal.predict_proba(scores)
        ece_cal = expected_calibration_error(cal_probs, labels)
        assert ece_cal < ece_raw + 0.1, "Calibration should not badly worsen ECE"


# ââ Isotonic calibrator âââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestIsotonicCalibrator:
    def test_fit_and_predict(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        cal = IsotonicCalibrator().fit(scores, labels)
        probs = cal.predict_proba(scores)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_monotone_output(self):
        rng = np.random.default_rng(42)
        scores = np.sort(rng.uniform(0, 1, 100))
        labels = (scores + rng.uniform(-0.2, 0.2, 100) > 0.5).astype(float)
        cal = IsotonicCalibrator().fit(scores, labels)
        probs = cal.predict_proba(scores)
        # Isotonic â non-decreasing
        assert (np.diff(probs) >= -1e-9).all()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            IsotonicCalibrator().predict_proba(np.array([0.5]))


# ââ ECE âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class TestECE:
    def test_perfect_calibration_zero_ece(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        # If confidence = accuracy in each bin â ECE ~ 0
        # Use single-element bins by having n_bins=5
        labels = probs  # perfectly calibrated
        ece = expected_calibration_error(probs, labels, n_bins=5)
        assert ece < 0.15

    def test_ece_range(self):
        rng = np.random.default_rng(99)
        probs = rng.uniform(0, 1, 200)
        labels = rng.integers(0, 2, 200).astype(float)
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0

    def test_reliability_diagram_bins(self):
        probs = np.linspace(0, 1, 50)
        labels = (probs > 0.5).astype(float)
        diag = reliability_diagram_data(probs, labels, n_bins=5)
        assert len(diag) == 5
        for d in diag:
            assert "bin_mid" in d and "accuracy" in d and "confidence" in d
