"""
news_day19_coverage_analysis.py
Day 19: Coverage Analysis for diffusion model predictions.
Kupiec POF test, Christoffersen independence test, ECE, tail coverage,
sharpness metric, rolling calibration windows.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Helper: normal distribution
# ---------------------------------------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _norm_ppf(p: float) -> float:
    """Approximate normal percent-point function via bisection."""
    p = max(1e-9, min(1 - 1e-9, p))
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ---------------------------------------------------------------------------
# Prediction interval
# ---------------------------------------------------------------------------
@dataclass
class PredictionInterval:
    lower: float
    upper: float
    point_estimate: float
    confidence: float   # nominal level, e.g. 0.90

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, realized: float) -> bool:
        return self.lower <= realized <= self.upper


# ---------------------------------------------------------------------------
# Empirical coverage
# ---------------------------------------------------------------------------
def empirical_coverage(
    intervals: list[PredictionInterval],
    realizations: list[float],
) -> dict:
    n = len(intervals)
    hits = [1.0 if iv.contains(r) else 0.0 for iv, r in zip(intervals, realizations)]
    coverage = sum(hits) / n
    nominal = intervals[0].confidence if intervals else 0.0
    return {
        'n': n,
        'nominal': nominal,
        'empirical': coverage,
        'coverage_error': coverage - nominal,
        'hits': hits,
    }

# ---------------------------------------------------------------------------
# Kupiec Proportion of Failures (POF) test
# ---------------------------------------------------------------------------
def kupiec_pof_test(
    hits: list[float],  # 1 = covered, 0 = violation
    nominal: float,
) -> dict:
    """
    Likelihood ratio test: H0: P(violation) = 1 - nominal
    LR_POF = -2 * log[L(p0) / L(p_hat)]
    where p0 = expected violation rate, p_hat = observed violation rate.
    Chi-squared with 1 df.
    """
    n = len(hits)
    n_violations = sum(1 for h in hits if h == 0.0)
    p_hat = n_violations / n if n > 0 else 0.0
    p0 = 1 - nominal

    def log_lik(p: float, k: int, n: int) -> float:
        if p <= 0 or p >= 1:
            return float('-inf')
        return k * math.log(p) + (n - k) * math.log(1 - p)

    lr = -2 * (log_lik(p0, n_violations, n) - log_lik(max(p_hat, 1e-10), n_violations, n))
    lr = max(lr, 0.0)

    # Chi-squared p-value (df=1) via regularized incomplete gamma
    p_value = _chi2_pvalue(lr, df=1)

    return {
        'LR_POF': lr,
        'p_value': p_value,
        'observed_violation_rate': p_hat,
        'expected_violation_rate': p0,
        'n_violations': n_violations,
        'n': n,
        'reject_H0': p_value < 0.05,
    }

def _chi2_pvalue(x: float, df: int) -> float:
    """Approximate chi-squared p-value using normal approximation."""
    if x <= 0:
        return 1.0
    # Wilson-Hilferty: chi2(k)/k approx Normal(1, 2/k)
    k = df
    z = (pow(x / k, 1/3) - (1 - 2/(9*k))) / math.sqrt(2/(9*k))
    return max(0.001, 1 - _norm_cdf(z))

# ---------------------------------------------------------------------------
# Christoffersen Independence Test
# ---------------------------------------------------------------------------
def christoffersen_independence_test(hits: list[float]) -> dict:
    """
    Test for independence of interval violations.
    Uses transition matrix and LR test.
    """
    n = len(hits)
    # Count transitions: n_ij = number of times i followed by j
    n00 = n01 = n10 = n11 = 0
    for t in range(n - 1):
        i, j = int(hits[t] == 0.0), int(hits[t+1] == 0.0)
        if i == 0 and j == 0: n00 += 1
        elif i == 0 and j == 1: n01 += 1
        elif i == 1 and j == 0: n10 += 1
        elif i == 1 and j == 1: n11 += 1

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n, 1)

    def ll_restricted():
        return ((n00 + n10) * math.log(max(1 - pi, 1e-10)) +
                (n01 + n11) * math.log(max(pi, 1e-10)))

    def ll_unrestricted():
        return (n00 * math.log(max(1 - pi01, 1e-10)) + n01 * math.log(max(pi01, 1e-10)) +
                n10 * math.log(max(1 - pi11, 1e-10)) + n11 * math.log(max(pi11, 1e-10)))

    lr = -2 * (ll_restricted() - ll_unrestricted())
    lr = max(lr, 0.0)
    p_value = _chi2_pvalue(lr, df=1)

    return {
        'LR_independence': lr,
        'p_value': p_value,
        'pi01': pi01,
        'pi11': pi11,
        'reject_independence': p_value < 0.05,
    }

# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE) for intervals
# ---------------------------------------------------------------------------
def expected_calibration_error(
    intervals: list[PredictionInterval],
    realizations: list[float],
    n_bins: int = 10,
) -> dict:
    """
    ECE: mean absolute difference between confidence and coverage per bin.
    Bins by nominal confidence level.
    """
    bin_width = 1.0 / n_bins
    bins: dict[int, list] = {i: [] for i in range(n_bins)}

    for iv, r in zip(intervals, realizations):
        b = min(int(iv.confidence / bin_width), n_bins - 1)
        bins[b].append((iv.confidence, iv.contains(r)))

    ece = 0.0
    bin_stats = []
    for b_idx, entries in bins.items():
        if not entries:
            continue
        n_b = len(entries)
        nominal_b = sum(e[0] for e in entries) / n_b
        coverage_b = sum(1.0 for e in entries if e[1]) / n_b
        ece += abs(coverage_b - nominal_b) * n_b / len(intervals)
        bin_stats.append({
            'bin': b_idx,
            'n': n_b,
            'nominal': nominal_b,
            'empirical': coverage_b,
        })

    return {'ECE': ece, 'bins': bin_stats}

# ---------------------------------------------------------------------------
# Sharpness metric
# ---------------------------------------------------------------------------
def sharpness(intervals: list[PredictionInterval]) -> dict:
    """Average interval width (sharper = smaller)."""
    widths = [iv.width for iv in intervals]
    n = len(widths)
    avg_width = sum(widths) / n
    std_width = math.sqrt(sum((w - avg_width)**2 for w in widths) / max(n-1, 1))
    return {
        'mean_width': avg_width,
        'std_width': std_width,
        'median_width': sorted(widths)[n // 2],
        'min_width': min(widths),
        'max_width': max(widths),
    }

# ---------------------------------------------------------------------------
# Tail coverage analysis
# ---------------------------------------------------------------------------
def tail_coverage(
    intervals: list[PredictionInterval],
    realizations: list[float],
    tail_pct: float = 0.10,
) -> dict:
    """Coverage rates for tail observations (top and bottom X%)."""
    sorted_pairs = sorted(zip(realizations, intervals), key=lambda x: x[0])
    n = len(sorted_pairs)
    k = max(int(n * tail_pct), 1)

    lower_tail = sorted_pairs[:k]
    upper_tail = sorted_pairs[-k:]

    lower_cov = sum(1.0 for r, iv in lower_tail if iv.contains(r)) / k
    upper_cov = sum(1.0 for r, iv in upper_tail if iv.contains(r)) / k

    return {
        'tail_pct': tail_pct,
        'lower_tail_coverage': lower_cov,
        'upper_tail_coverage': upper_cov,
        'n_tail': k,
    }

# ---------------------------------------------------------------------------
# Rolling calibration windows
# ---------------------------------------------------------------------------
def rolling_coverage(
    intervals: list[PredictionInterval],
    realizations: list[float],
    window: int = 50,
) -> list[dict]:
    """Compute empirical coverage over rolling windows."""
    results = []
    n = len(intervals)
    for i in range(window - 1, n):
        sub_iv = intervals[i - window + 1:i + 1]
        sub_r = realizations[i - window + 1:i + 1]
        cov = empirical_coverage(sub_iv, sub_r)
        results.append({
            'end_idx': i,
            'coverage': cov['empirical'],
            'nominal': cov['nominal'],
            'error': cov['coverage_error'],
        })
    return results

# ---------------------------------------------------------------------------
# Winkler score (interval scoring rule)
# ---------------------------------------------------------------------------
def winkler_score(iv: PredictionInterval, realized: float) -> float:
    """Lower = better. Penalizes wide intervals and misses."""
    alpha = 1 - iv.confidence
    score = iv.width
    if realized < iv.lower:
        score += (2 / alpha) * (iv.lower - realized)
    elif realized > iv.upper:
        score += (2 / alpha) * (realized - iv.upper)
    return score

def mean_winkler_score(
    intervals: list[PredictionInterval],
    realizations: list[float],
) -> float:
    return sum(winkler_score(iv, r) for iv, r in zip(intervals, realizations)) / len(intervals)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("DAY 19: Coverage Analysis for Diffusion Model Predictions")
    print("=" * 60)

    rng = random.Random(42)
    N = 500
    CONFIDENCE = 0.90

    # Simulate a diffusion model producing prediction intervals
    # True process: returns ~ N(0, 0.01)
    true_sigma = 0.01
    intervals, realizations = [], []

    for _ in range(N):
        # Model predicts with slight miscalibration (overconfident)
        model_sigma = true_sigma * 0.85  # model underestimates vol
        z = _norm_ppf(1 - (1 - CONFIDENCE) / 2)
        pt = rng.gauss(0, 0.002)
        lo = pt - z * model_sigma
        hi = pt + z * model_sigma
        realized = rng.gauss(0, true_sigma)

        intervals.append(PredictionInterval(lo, hi, pt, CONFIDENCE))
        realizations.append(realized)

    print(f"\nModel: {CONFIDENCE:.0%} nominal confidence, N={N}")

    # 1. Empirical coverage
    cov_result = empirical_coverage(intervals, realizations)
    print(f"\n1. Empirical Coverage")
    print(f"   Nominal  : {cov_result['nominal']:.2%}")
    print(f"   Empirical: {cov_result['empirical']:.2%}")
    print(f"   Error    : {cov_result['coverage_error']:+.4%}")

    # 2. Kupiec test
    kupiec = kupiec_pof_test(cov_result['hits'], CONFIDENCE)
    print(f"\n2. Kupiec POF Test")
    print(f"   LR stat  : {kupiec['LR_POF']:.4f}")
    print(f"   p-value  : {kupiec['p_value']:.4f}")
    print(f"   Violation: {kupiec['observed_violation_rate']:.4%} (expected {kupiec['expected_violation_rate']:.4%})")
    print(f"   Reject H0: {kupiec['reject_H0']} {'(miscalibrated)' if kupiec['reject_H0'] else ''}")

    # 3. Christoffersen independence test
    christo = christoffersen_independence_test(cov_result['hits'])
    print(f"\n3. Christoffersen Independence Test")
    print(f"   LR stat  : {christo['LR_independence']:.4f}")
    print(f"   p-value  : {christo['p_value']:.4f}")
    print(f"   pi01     : {christo['pi01']:.4f}  (violation after coverage)")
    print(f"   pi11     : {christo['pi11']:.4f}  (violation after violation)")
    print(f"   Clustered: {christo['reject_independence']}")

    # 4. ECE
    ece_result = expected_calibration_error(intervals, realizations)
    print(f"\n4. Expected Calibration Error")
    print(f"   ECE: {ece_result['ECE']:.6f}")

    # 5. Sharpness
    sharp = sharpness(intervals)
    print(f"\n5. Sharpness (Interval Width)")
    print(f"   Mean width  : {sharp['mean_width']:.6f}")
    print(f"   Std width   : {sharp['std_width']:.6f}")
    print(f"   Median width: {sharp['median_width']:.6f}")

    # 6. Tail coverage
    tail = tail_coverage(intervals, realizations, tail_pct=0.10)
    print(f"\n6. Tail Coverage Analysis (10% tails)")
    print(f"   Lower tail coverage: {tail['lower_tail_coverage']:.4%}")
    print(f"   Upper tail coverage: {tail['upper_tail_coverage']:.4%}")
    print(f"   Overall nominal    : {CONFIDENCE:.4%}")

    # 7. Rolling coverage
    rolling = rolling_coverage(intervals, realizations, window=50)
    max_err = max(abs(r['error']) for r in rolling)
    avg_err = sum(abs(r['error']) for r in rolling) / len(rolling)
    print(f"\n7. Rolling Coverage (window=50)")
    print(f"   Max |error|: {max_err:.4%}")
    print(f"   Mean|error|: {avg_err:.4%}")
    print(f"   Sample (every 100 steps):")
    for r in rolling[::100]:
        bar = '#' * int(r['coverage'] * 20)
        print(f"     t={r['end_idx']:>3}: [{bar:<20}] {r['coverage']:.2%} (err {r['error']:+.2%})")

    # 8. Winkler score
    winkler = mean_winkler_score(intervals, realizations)
    print(f"\n8. Mean Winkler Score: {winkler:.6f}")

    print("\n[Done] Day 19: Coverage Analysis complete.")
