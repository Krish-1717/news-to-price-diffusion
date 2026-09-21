"""
models/calibrator.py -- Probability calibration for diffusion model outputs
Day 9 Commit 1: Platt scaling + isotonic regression calibrators with ECE.
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlattCalibrator:
    """Logistic (Platt) calibration: P(y=1|f) = sigmoid(A*f + B).
    Parameters fit via gradient descent on binary cross-entropy.
    """
    A: float = 1.0
    B: float = 0.0
    lr: float = 0.01
    n_epochs: int = 1000
    _fitted: bool = field(default=False, repr=False)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        """scores: raw model outputs (any range); labels: 0/1."""
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        n = len(scores)
        A, B = self.A, self.B
        for _ in range(self.n_epochs):
            logits = A * scores + B
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            diff = probs - labels
            dA = float(np.dot(diff, scores)) / n
            dB = float(diff.sum()) / n
            A -= self.lr * dA
            B -= self.lr * dB
        self.A, self.B = A, B
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        scores = np.asarray(scores, dtype=float)
        logits = self.A * scores + self.B
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


class IsotonicCalibrator:
    """Isotonic regression calibration via Pool Adjacent Violators (PAV)."""
    _x: Optional[np.ndarray] = None
    _y: Optional[np.ndarray] = None

    def _pav(self, y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators â returns monotone non-decreasing fit."""
        n = len(y)
        target = y.copy().astype(float)
        i = 0
        while i < n:
            j = i + 1
            while j < n and target[j] < target[i]:
                # pool [i..j]
                mean = target[i:j+1].mean()
                target[i:j+1] = mean
                # check left violations
                while i > 0 and target[i] < target[i-1]:
                    i -= 1
                    mean = target[i:j+1].mean()
                    target[i:j+1] = mean
                j += 1
            i = j
        return target

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        order = np.argsort(scores)
        self._x = scores[order]
        self._y = self._pav(labels[order])
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Call fit() first")
        scores = np.asarray(scores, dtype=float)
        return np.interp(scores, self._x, self._y)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE: weighted mean |confidence - accuracy| across equal-width bins."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if lo == bins[-2]:
            mask |= (probs == 1.0)
        cnt = mask.sum()
        if cnt == 0:
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += cnt / n * abs(conf - acc)
    return float(ece)


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 10) -> List[dict]:
    """Return bin-level stats for plotting a reliability diagram."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if lo == bins[-2]:
            mask |= (probs == 1.0)
        cnt = int(mask.sum())
        mid = (lo + hi) / 2
        result.append({
            "bin_mid": round(mid, 4),
            "count": cnt,
            "confidence": float(probs[mask].mean()) if cnt else mid,
            "accuracy": float(labels[mask].mean()) if cnt else 0.0,
        })
    return result
