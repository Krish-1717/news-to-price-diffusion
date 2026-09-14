"""
Synthetic data generator for smoke-testing the pipeline before connecting real data.

Design principles:
  - NO lookahead: all features computed from data available at t-1 (yesterday close)
  - Return labels are next-day log-returns (available at t, after close)
  - Generates realistic fat-tailed return distributions via Student-t noise
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SyntheticConfig:
    n_days: int = 2000
    feature_dim: int = 16
    sentiment_range: Tuple[float, float] = (-1.0, 1.0)
    return_vol: float = 0.012
    df_t: float = 4.0
    seed: int = 42


class SyntheticDataset:
    """
    Generates (features, sentiment, next_return) triples with no lookahead.

    Timeline:
        day t-1: OHLCV known -> features computed
        day t:   close known -> next_return = log(close_t / close_{t-1})
    """

    def __init__(self, cfg: SyntheticConfig = SyntheticConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        log_returns = self._simulate_returns(rng)
        prices = np.exp(np.cumsum(log_returns))
        features = self._build_features(prices, log_returns, rng)
        noise = rng.standard_normal(cfg.n_days) * 0.4
        raw_sentiment = log_returns / cfg.return_vol * 0.3 + noise
        sentiment = np.clip(raw_sentiment, *cfg.sentiment_range)
        feat_cols = [f"feat_{i}" for i in range(cfg.feature_dim)]
        self.df = pd.DataFrame(features, columns=feat_cols)
        self.df["sentiment"] = sentiment
        self.df["next_return"] = log_returns
        self.df.index.name = "day"

    def _simulate_returns(self, rng):
        cfg = self.cfg
        t_innov = rng.standard_t(cfg.df_t, size=cfg.n_days) / np.sqrt(
            cfg.df_t / (cfg.df_t - 2))
        jumps = rng.binomial(1, 0.02, cfg.n_days) * rng.normal(0, 0.03, cfg.n_days)
        return t_innov * cfg.return_vol + jumps - 0.5 * cfg.return_vol**2

    def _build_features(self, prices, returns, rng):
        cfg = self.cfg
        n = cfg.n_days
        feat = np.zeros((n, cfg.feature_dim))
        windows = [5, 10, 20, 60]
        for i, w in enumerate(windows):
            for j in range(n):
                start = max(0, j - w)
                feat[j, i] = np.mean(returns[start:j]) if j > 0 else 0.0
                feat[j, i + 4] = np.std(returns[start:j]) if j > 1 else cfg.return_vol
        for j in range(n):
            feat[j, 8] = prices[j-1] / np.max(prices[max(0,j-52):j]) if j > 0 else 1.0
            feat[j, 9] = prices[j-1] / np.mean(prices[max(0,j-20):j]) if j > 0 else 1.0
            feat[j, 10] = returns[j-1] if j > 0 else 0.0
        feat[:, 11:] = rng.standard_normal((n, cfg.feature_dim - 11)) * 0.1
        return feat.astype(np.float32)

    def train_val_split(self, val_frac=0.15, test_frac=0.10):
        """Walk-forward split — no shuffling, strictly ordered."""
        n = len(self.df)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test
        splits = []
        for start, end in [(0, n_train), (n_train, n_train+n_val), (n_train+n_val, n)]:
            subset = SyntheticDataset.__new__(SyntheticDataset)
            subset.cfg = self.cfg
            subset.df = self.df.iloc[start:end].copy()
            splits.append(subset)
        return tuple(splits)

    def as_tensors(self):
        feat_cols = [c for c in self.df.columns if c.startswith("feat_")]
        X_market = self.df[feat_cols].values.astype(np.float32)
        X_sentiment = self.df["sentiment"].values.astype(np.float32).reshape(-1, 1)
        y = self.df["next_return"].values.astype(np.float32)
        return X_market, X_sentiment, y

    def __len__(self):
        return len(self.df)
