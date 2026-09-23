"""
data/market_features.py -- Market feature engineering for news-to-price-diffusion
Day 12: returns, volatility, momentum, microstructure features, normalization pipeline
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Basic stats helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))

def _rolling(xs: List[float], window: int) -> List[Optional[float]]:
    """Sliding-window mean; None for positions with insufficient history."""
    result: List[Optional[float]] = []
    for i, _ in enumerate(xs):
        if i < window - 1:
            result.append(None)
        else:
            result.append(_mean(xs[i - window + 1: i + 1]))
    return result

def _rolling_std(xs: List[float], window: int) -> List[Optional[float]]:
    result: List[Optional[float]] = []
    for i, _ in enumerate(xs):
        if i < window - 1:
            result.append(None)
        else:
            result.append(_std(xs[i - window + 1: i + 1]))
    return result


# ---------------------------------------------------------------------------
# Raw bar data
# ---------------------------------------------------------------------------

@dataclass
class OHLCV:
    """Single daily bar."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def return_(self) -> float:
        """Log return: ln(close/open)."""
        return math.log(self.close / self.open) if self.open > 0 else 0.0

    @property
    def range_pct(self) -> float:
        """(High - Low) / Open â intraday range normalised by open."""
        return (self.high - self.low) / self.open if self.open > 0 else 0.0

    @property
    def close_position(self) -> float:
        """Where close sits in the high-low range [0=low, 1=high]."""
        rng = self.high - self.low
        return (self.close - self.low) / rng if rng > 0 else 0.5


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

@dataclass
class MarketFeatureVector:
    """Normalised feature vector for one bar, ready for model input."""
    date: str

    # Price / return features
    log_return: float          # log(close_t / close_{t-1})
    range_pct: float           # (H-L)/Open
    close_position: float      # close in H-L range [0..1]

    # Momentum features
    mom_5d: Optional[float]    # 5-day cumulative return
    mom_20d: Optional[float]   # 20-day cumulative return
    mom_60d: Optional[float]   # 60-day cumulative return

    # Volatility features
    realized_vol_5d: Optional[float]   # 5d rolling std of log returns
    realized_vol_20d: Optional[float]  # 20d rolling std

    # Volume features
    volume_change: Optional[float]     # log(vol_t / vol_{t-1})
    volume_zscore: Optional[float]     # (vol_t - mean_20d) / std_20d

    # Microstructure
    amihud: Optional[float]    # |return| / volume (illiquidity proxy)

    def to_list(self) -> List[float]:
        """Return feature vector as a list; None filled with 0."""
        return [
            self.log_return,
            self.range_pct,
            self.close_position,
            self.mom_5d or 0.0,
            self.mom_20d or 0.0,
            self.mom_60d or 0.0,
            self.realized_vol_5d or 0.0,
            self.realized_vol_20d or 0.0,
            self.volume_change or 0.0,
            self.volume_zscore or 0.0,
            self.amihud or 0.0,
        ]

    @classmethod
    def dim(cls) -> int:
        return 11


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

class MarketFeaturePipeline:
    """
    Converts a list of OHLCV bars into normalised MarketFeatureVectors.

    Steps
    -----
    1. Compute raw features (returns, momentum, vol, volume stats)
    2. Optionally z-score normalise each feature over the training window
    """

    def __init__(self, normalise: bool = True, lookback: int = 252):
        self.normalise = normalise
        self.lookback = lookback
        self._means: Dict[str, float] = {}
        self._stds: Dict[str, float] = {}

    def _compute_raw(self, bars: List[OHLCV]) -> List[MarketFeatureVector]:
        n = len(bars)
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        log_returns = [0.0] + [
            math.log(closes[i] / closes[i - 1]) if closes[i - 1] > 0 else 0.0
            for i in range(1, n)
        ]

        vol_5 = _rolling_std(log_returns, 5)
        vol_20 = _rolling_std(log_returns, 20)
        vol_mean_20 = _rolling(volumes, 20)
        vol_std_20 = _rolling_std(volumes, 20)

        features = []
        for i, bar in enumerate(bars):
            # Momentum
            mom_5 = (
                sum(log_returns[max(0, i - 4): i + 1])
                if i >= 4 else None
            )
            mom_20 = (
                sum(log_returns[max(0, i - 19): i + 1])
                if i >= 19 else None
            )
            mom_60 = (
                sum(log_returns[max(0, i - 59): i + 1])
                if i >= 59 else None
            )

            # Volume
            vol_change = (
                math.log(volumes[i] / volumes[i - 1])
                if i > 0 and volumes[i - 1] > 0 else None
            )
            vol_mu = vol_mean_20[i]
            vol_sg = vol_std_20[i]
            vol_z = (
                (volumes[i] - vol_mu) / vol_sg
                if vol_mu is not None and vol_sg and vol_sg > 0 else None
            )

            # Amihud illiquidity
            amihud = (
                abs(log_returns[i]) / volumes[i]
                if volumes[i] > 0 else None
            )

            features.append(MarketFeatureVector(
                date=bar.date,
                log_return=log_returns[i],
                range_pct=bar.range_pct,
                close_position=bar.close_position,
                mom_5d=mom_5,
                mom_20d=mom_20,
                mom_60d=mom_60,
                realized_vol_5d=vol_5[i],
                realized_vol_20d=vol_20[i],
                volume_change=vol_change,
                volume_zscore=vol_z,
                amihud=amihud,
            ))
        return features

    def fit_transform(self, bars: List[OHLCV]) -> List[MarketFeatureVector]:
        """Compute features and fit normalisation statistics on the full series."""
        features = self._compute_raw(bars)
        if not self.normalise:
            return features
        # Fit per-column stats
        col_names = [
            "log_return","range_pct","close_position",
            "mom_5d","mom_20d","mom_60d",
            "realized_vol_5d","realized_vol_20d",
            "volume_change","volume_zscore","amihud",
        ]
        matrix = [f.to_list() for f in features]
        for j, name in enumerate(col_names):
            vals = [row[j] for row in matrix if row[j] != 0.0]
            mu = _mean(vals) if vals else 0.0
            sg = _std(vals) if vals else 1.0
            self._means[name] = mu
            self._stds[name] = sg if sg > 0 else 1.0
        return self._normalise_features(features)

    def transform(self, bars: List[OHLCV]) -> List[MarketFeatureVector]:
        """Apply pre-fitted normalisation to new bars."""
        return self._normalise_features(self._compute_raw(bars))

    def _normalise_features(self, features: List[MarketFeatureVector]) -> List[MarketFeatureVector]:
        if not self._means:
            return features
        for f in features:
            row = f.to_list()
            col_names = [
                "log_return","range_pct","close_position",
                "mom_5d","mom_20d","mom_60d",
                "realized_vol_5d","realized_vol_20d",
                "volume_change","volume_zscore","amihud",
            ]
            for j, name in enumerate(col_names):
                mu = self._means.get(name, 0.0)
                sg = self._stds.get(name, 1.0)
                row[j] = (row[j] - mu) / sg
            (f.log_return, f.range_pct, f.close_position,
             f.mom_5d, f.mom_20d, f.mom_60d,
             f.realized_vol_5d, f.realized_vol_20d,
             f.volume_change, f.volume_zscore, f.amihud) = row
        return features


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(99)
    price = 100.0

    bars = []
    for i in range(120):
        ret = rng.gauss(0.0004, 0.012)
        price *= (1 + ret)
        high  = price * (1 + abs(rng.gauss(0, 0.005)))
        low   = price * (1 - abs(rng.gauss(0, 0.005)))
        vol   = abs(rng.gauss(1_000_000, 200_000))
        bars.append(OHLCV(
            date=f"2024-{i//22+1:02d}-{i%22+1:02d}",
            open=price / (1 + ret),
            high=high,
            low=low,
            close=price,
            volume=vol,
        ))

    pipeline = MarketFeaturePipeline(normalise=True)
    features = pipeline.fit_transform(bars)

    print(f"Computed {len(features)} feature vectors, dim={MarketFeatureVector.dim()}")
    print()
    print(f"{'Date':<14} {'logRet':>8} {'RangePct':>9} {'Mom5d':>8} {'Vol20d':>8} {'VolZ':>8}")
    print("-" * 55)
    for fv in features[-5:]:
        print(
            f"{fv.date:<14} "
            f"{fv.log_return:>88.3f} "
            f"{fv.range_pct:>9!{fv.mom_5d or 0):>8.3f} "
            f"{fv.realized_vol_20d or 0):>8.3f} "
            f"{(fv.volume_zscore or 0):>8.3f}"
        )
