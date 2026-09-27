"""
Signal Decay and Half-Life Analysis
Day 15 â news-to-price-diffusion/evaluation/signal_decay.py

Measures how quickly alpha signals decay over time using:
- Rolling IC (Information Coefficient) by forward horizon
- Exponential half-life fitting
- Optimal holding period estimation
- Signal autocorrelation analysis
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecayResult:
    """Signal decay profile across forward horizons."""
    horizons: List[int]              # holding periods (days)
    ic_by_horizon: List[float]       # mean IC at each horizon
    ic_t_stat: List[float]           # t-statistic for each IC
    half_life: Optional[float]       # estimated half-life in days
    optimal_horizon: int             # horizon with highest |IC|
    decay_rate: Optional[float]      # Î» in IC(h) â IC_0 Â· e^{-Î»h}
    ic_0: Optional[float]            # initial IC (extrapolated to h=0)


@dataclass
class SignalAutocorr:
    """Signal autocorrelation structure."""
    lags: List[int]
    autocorr: List[float]
    half_life: Optional[float]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _pearson_ic(scores: List[float], forward_returns: List[float]) -> float:
    """Information coefficient between signal scores and forward returns."""
    n = len(scores)
    if n < 3:
        return 0.0
    ms = sum(scores) / n
    mr = sum(forward_returns) / n
    cov = sum((scores[i] - ms) * (forward_returns[i] - mr) for i in range(n)) / n
    ss = math.sqrt(sum((s - ms) ** 2 for s in scores) / n)
    sr = math.sqrt(sum((r - mr) ** 2 for r in forward_returns) / n)
    if ss < 1e-10 or sr < 1e-10:
        return 0.0
    return cov / (ss * sr)


def _rank_ic(scores: List[float], forward_returns: List[float]) -> float:
    """Spearman rank IC."""
    def _ranks(v: List[float]) -> List[float]:
        n = len(v)
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for rank, idx in enumerate(order):
            r[idx] = float(rank)
        return r

    return _pearson_ic(_ranks(scores), _ranks(forward_returns))


def _ic_t_stat(ic: float, n: int) -> float:
    """t-statistic for IC significance: t = IC * sqrt(n) / sqrt(1 - IC^2)."""
    if n < 3:
        return 0.0
    denom = math.sqrt(max(1 - ic ** 2, 1e-10))
    return ic * math.sqrt(n) / denom


# ---------------------------------------------------------------------------
# Half-life fitting via OLS on log |IC|
# ---------------------------------------------------------------------------

def _fit_exponential_decay(horizons: List[int],
                            ic_values: List[float]
                            ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit IC(h) = IC_0 * exp(-Î» * h) via OLS on log|IC| = log(IC_0) - Î»*h.
    Returns (lambda, IC_0, half_life).
    """
    # Only use horizons where |IC| > 0
    pairs = [(h, abs(ic)) for h, ic in zip(horizons, ic_values) if abs(ic) > 1e-10]
    if len(pairs) < 2:
        return None, None, None

    hs = [p[0] for p in pairs]
    log_ics = [math.log(p[1]) for p in pairs]

    n = len(hs)
    mh = sum(hs) / n
    ml = sum(log_ics) / n

    num = sum((hs[i] - mh) * (log_ics[i] - ml) for i in range(n))
    den = sum((hs[i] - mh) ** 2 for i in range(n))

    if abs(den) < 1e-10:
        return None, None, None

    lam = -num / den            # slope is -Î»
    log_ic0 = ml - lam * (-mh)  # intercept
    ic0 = math.exp(log_ic0)
    half_life = math.log(2) / lam if lam > 1e-10 else None

    return lam, ic0, half_life


# ---------------------------------------------------------------------------
# Main decay analysis
# ---------------------------------------------------------------------------

def compute_signal_decay(
        signals: List[List[float]],           # T Ã N cross-sectional signals
        returns: List[List[float]],            # T Ã N return matrix
        horizons: Optional[List[int]] = None,
        use_rank_ic: bool = True,
) -> DecayResult:
    """
    Compute IC at multiple forward horizons.

    signals:  T observations of N asset scores
    returns:  T observations of N asset returns
    horizons: list of forward horizons (default: [1,2,3,5,10,15,21,42,63])
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 15, 21, 42, 63]

    T = len(signals)
    ic_func = _rank_ic if use_rank_ic else _pearson_ic

    ic_by_horizon: List[float] = []
    t_stats: List[float] = []

    for h in horizons:
        ics: List[float] = []
        for t in range(T - h):
            sig_t = signals[t]
            # Forward h-day cumulative return
            fwd_ret = [
                sum(returns[t + k][n] for k in range(h))
                for n in range(len(sig_t))
            ]
            ic = ic_func(sig_t, fwd_ret)
            ics.append(ic)

        mean_ic = sum(ics) / len(ics) if ics else 0.0
        ic_by_horizon.append(mean_ic)

        n_obs = len(ics)
        t_stat = _ic_t_stat(mean_ic, n_obs)
        t_stats.append(t_stat)

    lam, ic0, half_life = _fit_exponential_decay(horizons, ic_by_horizon)
    decay_rate = lam

    # Optimal horizon = max |IC|
    best_idx = max(range(len(horizons)), key=lambda i: abs(ic_by_horizon[i]))
    optimal_horizon = horizons[best_idx]

    return DecayResult(
        horizons=horizons,
        ic_by_horizon=ic_by_horizon,
        ic_t_stat=t_stats,
        half_life=half_life,
        optimal_horizon=optimal_horizon,
        decay_rate=decay_rate,
        ic_0=ic0,
    )


# ---------------------------------------------------------------------------
# Signal autocorrelation
# ---------------------------------------------------------------------------

def signal_autocorrelation(signals: List[List[float]],
                            max_lag: int = 20) -> SignalAutocorr:
    """
    Compute autocorrelation of cross-sectional signal (mean over assets).
    Useful for understanding signal persistence.
    """
    # Flatten to single time series (mean signal each period)
    T = len(signals)
    mean_sig = [sum(signals[t]) / len(signals[t]) for t in range(T)]

    mu = sum(mean_sig) / T
    var = sum((s - mu) ** 2 for s in mean_sig) / T
    if var < 1e-12:
        return SignalAutocorr(lags=list(range(1, max_lag + 1)),
                              autocorr=[0.0] * max_lag, half_life=None)

    lags = list(range(1, max_lag + 1))
    acf: List[float] = []
    for lag in lags:
        cov = sum((mean_sig[t] - mu) * (mean_sig[t - lag] - mu)
                  for t in range(lag, T)) / (T - lag)
        acf.append(cov / var)

    # Fit half-life to autocorrelation decay
    _, _, hl = _fit_exponential_decay(lags, acf)

    return SignalAutocorr(lags=lags, autocorr=acf, half_life=hl)


# ---------------------------------------------------------------------------
# Rolling IC analysis
# ---------------------------------------------------------------------------

def rolling_ic(signals: List[List[float]],
               returns: List[List[float]],
               horizon: int = 1,
               window: int = 60,
               use_rank_ic: bool = True) -> List[float]:
    """Rolling window IC for a single holding horizon."""
    T = len(signals)
    ic_func = _rank_ic if use_rank_ic else _pearson_ic
    result: List[float] = []

    for t in range(T - horizon):
        win_start = max(0, t - window)
        win_signals = signals[win_start: t + 1]
        win_fwd = [
            [sum(returns[s + k][n] for k in range(horizon))
             for n in range(len(signals[0]))]
            for s in range(win_start, t + 1)
        ]

        ics = [ic_func(win_signals[i], win_fwd[i])
               for i in range(len(win_signals))]
        result.append(sum(ics) / len(ics) if ics else 0.0)

    return result


def ic_statistics(ics: List[float]) -> Dict[str, float]:
    """Compute IC summary statistics."""
    n = len(ics)
    if n == 0:
        return {}
    mu = sum(ics) / n
    sigma = math.sqrt(sum((ic - mu) ** 2 for ic in ics) / max(n - 1, 1))
    positive_frac = sum(1 for ic in ics if ic > 0) / n
    ir = mu / sigma if sigma > 1e-10 else 0.0
    t_stat = _ic_t_stat(mu, n)

    return {
        "mean_ic": mu,
        "ic_std": sigma,
        "ic_ir": ir,
        "t_stat": t_stat,
        "positive_fraction": positive_frac,
        "max_ic": max(ics),
        "min_ic": min(ics),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    rng = random.Random(42)
    T, N = 252, 30

    # Simulate a momentum-like signal with known half-life ~10 days
    def gen_signal_with_decay(t: int) -> List[float]:
        return [rng.gauss(0, 1) for _ in range(N)]

    # Returns: slightly correlated with lagged signal (mean-reverting)
    signals = [gen_signal_with_decay(t) for t in range(T)]

    # Synthetic returns: weak correlation with signal at horizon 1-5
    returns = []
    for t in range(T):
        ret = [0.2 * signals[t][n] * rng.gauss(0, 0.03) + rng.gauss(0, 0.01)
               for n in range(N)]
        returns.append(ret)

    # Signal decay analysis
    decay = compute_signal_decay(signals, returns,
                                  horizons=[1, 2, 3, 5, 10, 21])

    print("Signal decay profile:")
    print(f"{'Horizon':>8} {'IC':>8} {'t-stat':>8}")
    print("-" * 28)
    for h, ic, t in zip(decay.horizons, decay.ic_by_horizon, decay.ic_t_stat):
        sig = "*" if abs(t) > 1.96 else ""
        print(f"{h:>8}  {ic:>7.4f}  {t:>7.3f}{sig}")

    if decay.half_life:
        print(f"\nEstimated half-life: {decay.half_life:.1f} days")
    print(f"Optimal horizon:     {decay.optimal_horizon} days")

    # Autocorrelation
    acf = signal_autocorrelation(signals, max_lag=10)
    print(f"\nSignal autocorrelation (first 5 lags): "
          f"{[round(a, 3) for a in acf.autocorr[:5]]}")
    if acf.half_life:
        print(f"Signal autocorr half-life: {acf.half_life:.1f} days")

    # Rolling IC stats
    roll_ic = rolling_ic(signals, returns, horizon=1, window=60)
    stats = ic_statistics(roll_ic)
    print(f"\nRolling IC statistics (horizon=1):")
    for k, v in stats.items():
        print(f"  {k:<22}: {v:.4f}")
