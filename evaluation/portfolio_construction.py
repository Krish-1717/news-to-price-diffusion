"""
News-Driven Portfolio Construction
Day 16 â news-to-price-diffusion/evaluation/portfolio_construction.py

Builds and backtests long-short equity portfolios using news sentiment signals,
topic exposures, and diffusion model outputs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NewsSignal:
    """News-derived signal for one asset on one date."""
    asset: str
    date: int           # integer date index
    sentiment: float    # aggregate sentiment score
    topic_vec: List[float] = field(default_factory=list)
    confidence: float = 1.0     # signal confidence / reliability


@dataclass
class PortfolioSnapshot:
    """Portfolio weights at a point in time."""
    date: int
    weights: Dict[str, float]     # asset â weight
    total_exposure: float = 0.0   # sum of |weights|

    def __post_init__(self):
        self.total_exposure = sum(abs(w) for w in self.weights.values())


@dataclass
class BacktestResult:
    """Backtest performance summary."""
    dates: List[int]
    portfolio_returns: List[float]
    cumulative_returns: List[float]
    sharpe_ratio: float
    annualised_return: float
    annualised_vol: float
    max_drawdown: float
    calmar_ratio: float
    hit_rate: float
    turnover: float


# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_signals(
        signals: List[NewsSignal],
        decay: float = 0.7,          # EWM decay per day
        max_age: int = 5,            # discard signals older than this
) -> Dict[str, Dict[int, float]]:
    """
    Aggregate raw news signals into smoothed daily scores per asset.
    Returns {asset: {date: score}}.
    """
    # Group by asset
    by_asset: Dict[str, List[Tuple[int, float, float]]] = {}
    for sig in signals:
        by_asset.setdefault(sig.asset, []).append(
            (sig.date, sig.sentiment, sig.confidence)
        )

    result: Dict[str, Dict[int, float]] = {}
    for asset, entries in by_asset.items():
        all_dates = sorted(set(d for d, _, _ in entries))
        score_by_date: Dict[int, float] = {}

        for date in all_dates:
            # EWM weighted aggregate over recent signals
            total_w = 0.0
            weighted_s = 0.0
            for d, sentiment, conf in entries:
                age = date - d
                if 0 <= age <= max_age:
                    w = conf * (decay ** age)
                    weighted_s += w * sentiment
                    total_w += w
            score_by_date[date] = weighted_s / total_w if total_w > 1e-10 else 0.0

        result[asset] = score_by_date
    return result


# ---------------------------------------------------------------------------
# Portfolio construction rules
# ---------------------------------------------------------------------------

def long_short_from_scores(
        scores: Dict[str, float],
        n_long: int = 5,
        n_short: int = 5,
        gross_exposure: float = 1.0,
        min_score_diff: float = 0.0,
) -> Dict[str, float]:
    """
    Rank assets by score â long top n, short bottom n, equal weight.
    gross_exposure controls total portfolio leverage.
    """
    items = sorted(scores.items(), key=lambda x: -x[1])
    long_assets = [a for a, s in items[:n_long] if s > min_score_diff]
    short_assets = [a for a, s in items[-n_short:] if s < -min_score_diff]

    weights: Dict[str, float] = {}
    leg_size = gross_exposure / 2

    if long_assets:
        w_long = leg_size / len(long_assets)
        for a in long_assets:
            weights[a] = w_long

    if short_assets:
        w_short = leg_size / len(short_assets)
        for a in short_assets:
            weights[a] = weights.get(a, 0.0) - w_short

    return weights


def score_weighted_portfolio(
        scores: Dict[str, float],
        gross_exposure: float = 1.0,
        clip_weight: float = 0.25,
) -> Dict[str, float]:
    """
    Score-proportional weights, clipped and normalised.
    """
    assets = list(scores.keys())
    raw = {a: scores[a] for a in assets}

    # Normalise to zero-sum
    mean_s = sum(raw.values()) / len(raw)
    demeaned = {a: raw[a] - mean_s for a in raw}

    # Normalise by total absolute score
    total_abs = sum(abs(v) for v in demeaned.values())
    if total_abs < 1e-10:
        return {a: 0.0 for a in assets}

    raw_w = {a: demeaned[a] / total_abs * gross_exposure for a in demeaned}

    # Clip
    clipped = {a: max(-clip_weight, min(clip_weight, w)) for a, w in raw_w.items()}

    # Re-normalise
    clip_abs = sum(abs(w) for w in clipped.values())
    if clip_abs < 1e-10:
        return {a: 0.0 for a in assets}
    return {a: w / clip_abs * gross_exposure for a, w in clipped.items()}


def risk_parity_sentiment(
        scores: Dict[str, float],
        vols: Dict[str, float],
        gross_exposure: float = 1.0,
) -> Dict[str, float]:
    """
    Risk-parity weighted by sentiment direction, inversely proportional to vol.
    """
    assets = list(scores.keys())
    vol_weights = {
        a: (1.0 / vols.get(a, 0.20)) * (1 if scores[a] >= 0 else -1)
        for a in assets
    }
    total = sum(abs(w) for w in vol_weights.values())
    if total < 1e-10:
        return {a: 0.0 for a in assets}
    return {a: w / total * gross_exposure for a, w in vol_weights.items()}


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_news_portfolio(
        signals: List[NewsSignal],
        returns: Dict[str, List[float]],      # asset â list of daily returns
        dates: List[int],
        strategy: str = "long_short",          # "long_short" | "score_weight"
        n_long: int = 5, n_short: int = 5,
        signal_decay: float = 0.7,
        rebalance_freq: int = 5,               # rebalance every N days
        transaction_cost: float = 0.001,       # round-trip
) -> BacktestResult:
    """
    Event-driven backtest of news-driven portfolio.
    """
    asset_scores = aggregate_daily_signals(signals, decay=signal_decay)
    all_assets = list(returns.keys())
    T = len(dates)

    portfolio_returns: List[float] = []
    prev_weights: Dict[str, float] = {}
    cumulative = 1.0
    cum_list: List[float] = []
    turnover_total = 0.0

    for t_idx, date in enumerate(dates):
        # Get current scores for all assets
        scores: Dict[str, float] = {}
        for asset in all_assets:
            asset_hist = asset_scores.get(asset, {})
            # Use most recent available score on or before this date
            past_dates = [d for d in asset_hist if d <= date]
            if past_dates:
                scores[asset] = asset_hist[max(past_dates)]

        # Rebalance
        if t_idx % rebalance_freq == 0 and scores:
            if strategy == "long_short":
                new_weights = long_short_from_scores(scores, n_long, n_short)
            else:
                new_weights = score_weighted_portfolio(scores)

            # Turnover
            all_keys = set(list(prev_weights.keys()) + list(new_weights.keys()))
            turn = sum(abs(new_weights.get(a, 0.0) - prev_weights.get(a, 0.0))
                       for a in all_keys)
            turnover_total += turn
            prev_weights = new_weights

        # Apply current weights to returns
        port_ret = 0.0
        for asset, w in prev_weights.items():
            asset_rets = returns.get(asset, [0.0] * T)
            if t_idx < len(asset_rets):
                port_ret += w * asset_rets[t_idx]

        # Transaction costs on rebalance
        if t_idx % rebalance_freq == 0 and scores:
            port_ret -= transaction_cost * turnover_total / max(t_idx // rebalance_freq, 1)

        portfolio_returns.append(port_ret)
        cumulative *= (1 + port_ret)
        cum_list.append(cumulative)

    # Performance statistics
    n = len(portfolio_returns)
    mu = sum(portfolio_returns) / n if n else 0.0
    var = sum((r - mu) ** 2 for r in portfolio_returns) / max(n - 1, 1)
    sigma = math.sqrt(var)

    ann_ret = mu * 252
    ann_vol = sigma * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    # Max drawdown
    peak = 1.0
    max_dd = 0.0
    for c in cum_list:
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    calmar = ann_ret / max_dd if max_dd > 1e-10 else 0.0
    hit_rate = sum(1 for r in portfolio_returns if r > 0) / max(n, 1)
    n_rebal = max(t_idx // rebalance_freq, 1)
    avg_turnover = turnover_total / n_rebal

    return BacktestResult(
        dates=dates,
        portfolio_returns=portfolio_returns,
        cumulative_returns=cum_list,
        sharpe_ratio=sharpe,
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        hit_rate=hit_rate,
        turnover=avg_turnover,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    assets = ["AAPL", "MSFT", "NVDA", "AMZN", "JPM",
              "XOM", "GS", "TSLA", "META", "JNJ"]
    T = 252
    dates = list(range(T))

    # Simulate returns
    returns = {a: [rng.gauss(0.0003, 0.015) for _ in range(T)] for a in assets}

    # Simulate news signals (random sentiment)
    signals: List[NewsSignal] = []
    for t in range(0, T, 3):  # news every 3 days
        for asset in assets:
            if rng.random() > 0.5:  # 50% chance of news
                signals.append(NewsSignal(
                    asset=asset, date=t,
                    sentiment=rng.gauss(0, 0.5),
                    confidence=rng.uniform(0.3, 1.0),
                ))

    print(f"Generated {len(signals)} news signals for {T} trading days")

    result = backtest_news_portfolio(
        signals, returns, dates,
        strategy="long_short",
        n_long=3, n_short=3,
        rebalance_freq=5,
    )

    print(f"\nBacktest Results:")
    print(f"  Annualised Return: {result.annualised_return:.2%}")
    print(f"  Annualised Vol:    {result.annualised_vol:.2%}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:      {result.max_drawdown:.2%}")
    print(f"  Calmar Ratio:      {result.calmar_ratio:.3f}")
    print(f"  Hit Rate:          {result.hit_rate:.2%}")
    print(f"  Avg Turnover:      {result.turnover:.2%}")
    final_cum = result.cumulative_returns[-1] if result.cumulative_returns else 1.0
    print(f"  Total Return:      {(final_cum - 1):.2%}")
