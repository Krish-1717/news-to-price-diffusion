"""
evaluation/metrics.py -- Probabilistic forecast evaluation for news-to-price-diffusion
Day 13: CRPS, NLL, calibration, energy score â pure Python.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Continuous Ranked Probability Score (CRPS)
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def crps_normal(mu: float, sigma: float, y: float) -> float:
    """
    CRPS for a Gaussian predictive distribution N(mu, sigma^2).
    Closed form:  CRPS = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.
    """
    if sigma <= 0:
        return abs(y - mu)
    z = (y - mu) / sigma
    phi_z = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi_z = _normal_cdf(z)
    return sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / math.sqrt(math.pi))


def crps_ensemble(samples: List[float], y: float) -> float:
    """
    CRPS for an ensemble of samples (empirical distribution).
    CRPS = E|X - y| - 0.5 * E|X - X'|
    Computed in O(n log n) after sorting.
    """
    n = len(samples)
    if n == 0:
        return float("nan")
    s = sorted(samples)
    # E|X - y|
    mae = sum(abs(x - y) for x in s) / n
    # E|X - X'| via sorted trick: sum_i s[i] * (2i - n + 1) / n^2
    energy = sum(s[i] * (2 * i - n + 1) for i in range(n)) / (n * n)
    return mae - 0.5 * energy


# ---------------------------------------------------------------------------
# Negative log-likelihood (Gaussian)
# ---------------------------------------------------------------------------

def nll_gaussian(mu: float, sigma: float, y: float) -> float:
    """NLL under N(mu, sigma^2) for observation y."""
    if sigma <= 0:
        return float("inf")
    z = (y - mu) / sigma
    return 0.5 * z * z + math.log(sigma) + 0.5 * math.log(2 * math.pi)


# ---------------------------------------------------------------------------
# Energy score (multivariate generalisation of CRPS)
# ---------------------------------------------------------------------------

def _l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def energy_score(ensemble: List[List[float]], y: List[float]) -> float:
    """
    Multivariate energy score for ensemble predictions vs observation y.
    ES = E||X - y|| - 0.5 * E||X - X'||
    """
    n = len(ensemble)
    if n == 0:
        return float("nan")
    term1 = sum(_l2(x, y) for x in ensemble) / n
    term2 = sum(_l2(ensemble[i], ensemble[j])
                for i in range(n) for j in range(n) if i != j) / (n * (n - 1)) if n > 1 else 0.0
    return term1 - 0.5 * term2


# ---------------------------------------------------------------------------
# Calibration: reliability diagram bins
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBin:
    """One reliability-diagram bin."""
    confidence: float    # predicted probability of being below threshold
    frequency: float     # empirical frequency
    count: int


def calibration_curve(
    pred_quantiles: List[Tuple[float, float]],  # (predicted_prob, observed_binary)
    n_bins: int = 10,
) -> List[CalibrationBin]:
    """
    Build reliability diagram bins.

    Parameters
    ----------
    pred_quantiles : list of (p_hat, y_bin) where p_hat in [0,1] is the
                     predicted probability and y_bin in {0,1} is whether
                     the outcome fell below the predicted quantile.
    n_bins : number of equal-width bins in [0, 1].
    """
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p_hat, y_bin in pred_quantiles:
        idx = min(int(p_hat * n_bins), n_bins - 1)
        bins[idx].append((p_hat, y_bin))

    result = []
    for i, b in enumerate(bins):
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        freq = sum(y for _, y in b) / len(b)
        result.append(CalibrationBin(conf, freq, len(b)))
    return result


def expected_calibration_error(curve: List[CalibrationBin]) -> float:
    """ECE = weighted mean |confidence - frequency|."""
    total = sum(b.count for b in curve)
    if total == 0:
        return float("nan")
    return sum(b.count * abs(b.confidence - b.frequency) for b in curve) / total


# ---------------------------------------------------------------------------
# Aggregate evaluation report
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Summary of all probabilistic metrics over a test set."""
    mean_crps: float
    mean_nll: float
    mean_energy_score: float
    ece: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"EvalReport(n={self.n_samples})\n"
            f"  CRPS          : {self.mean_crps:.6f}\n"
            f"  NLL           : {self.mean_nll:.6f}\n"
            f"  Energy Score  : {self.mean_energy_score:.6f}\n"
            f"  ECE           : {self.ece:.6f}"
        )


def evaluate(
    observations: List[List[float]],      # true return vectors, shape (N, D)
    ensemble_preds: List[List[List[float]]],  # samples per obs, shape (N, K, D)
    mu_preds: List[float],                # scalar mean pred (1-D shortcut)
    sigma_preds: List[float],             # scalar std pred
    scalar_obs: List[float],              # scalar observation
) -> EvalReport:
    """
    Run full evaluation suite.

    For multivariate energy score: uses ensemble_preds vs observations.
    For univariate CRPS/NLL: uses mu_preds, sigma_preds, scalar_obs.
    """
    n = len(scalar_obs)
    crps_vals = [crps_normal(mu_preds[i], sigma_preds[i], scalar_obs[i]) for i in range(n)]
    nll_vals  = [nll_gaussian(mu_preds[i], sigma_preds[i], scalar_obs[i]) for i in range(n)]
    es_vals   = [energy_score(ensemble_preds[i], observations[i]) for i in range(n)]

    # Calibration: check if obs < mu (should happen ~50% of the time)
    pred_q = [(0.5, float(scalar_obs[i] < mu_preds[i])) for i in range(n)]
    curve = calibration_curve(pred_q, n_bins=5)
    ece = expected_calibration_error(curve)

    return EvalReport(
        mean_crps=sum(crps_vals) / max(n, 1),
        mean_nll=sum(nll_vals) / max(n, 1),
        mean_energy_score=sum(es_vals) / max(n, 1),
        ece=ece,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(0)
    N, K, D = 100, 50, 3

    # Synthetic: true obs from N(0,1), model predicts N(0.1, 1.05)
    obs = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(N)]
    ens = [[[rng.gauss(0.1, 1.05) for _ in range(D)] for _ in range(K)] for _ in range(N)]
    mu  = [0.1] * N
    sig = [1.05] * N
    y1d = [o[0] for o in obs]

    report = evaluate(obs, ens, mu, sig, y1d)
    print("=" * 45)
    print("  EVALUATION METRICS -- synthetic demo")
    print("=" * 45)
    print(report)
    print("=" * 45)

    # Per-function check
    print(f"\n  crps_normal(0,1,0)   = {crps_normal(0,1,0):.6f}  (expected ~0.2338)")
    print(f"  crps_ensemble([0]*5,0) = {crps_ensemble([0.0]*5, 0.0):.6f}  (expected 0)")
    print(f"  nll_gaussian(0,1,0)  = {nll_gaussian(0,1,0):.6f}  (expected ~0.9189)")
