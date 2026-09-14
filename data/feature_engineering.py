"""
feature_engineering.py — Combine price + sentiment into model-ready tensors.

Day 2: Merges market context (from real_loader) and news sentiment
(from finnhub_sentiment) into a single context tensor for the DDPM.

Context tensor shape: (T, CONTEXT_DIM=17)
  - market features : 14 rolling return/vol/momentum/RSI/skew features
  - sentiment       : bullish, bearish, article_count_norm (t-1 shifted)

All features z-score normalized using training-set statistics only.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict

from data.real_loader import PriceLoader
from data.finnhub_sentiment import SentimentLoader

MARKET_DIM = 4 * 3 + 2    # 4 windows x (ret, vol, mom) + rsi + skew = 14
SENTIMENT_DIM = 3          # bullish, bearish, article_count_norm
CONTEXT_DIM = MARKET_DIM + SENTIMENT_DIM  # 17


@dataclass
class NormStats:
    """Mean and std computed on training data only (no leakage to val/test)."""
    mean: np.ndarray   # shape (CONTEXT_DIM,)
    std: np.ndarray    # shape (CONTEXT_DIM,)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (self.std + 1e-8)

    def denormalize(self, X: np.ndarray) -> np.ndarray:
        return X * (self.std + 1e-8) + self.mean


class FeaturePipeline:
    """
    End-to-end pipeline: ticker → (X_context, y_returns, splits).

    Parameters
    ----------
    ticker : str
    start, end : str  ISO date range
    finnhub_api_key : str or None
    use_finbert : bool
    cache : bool
    """

    def __init__(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end: str = "2023-12-31",
        finnhub_api_key: Optional[str] = None,
        use_finbert: bool = False,
        cache: bool = True,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.price_loader = PriceLoader([ticker], cache=cache)
        self.sent_loader = SentimentLoader(
            api_key=finnhub_api_key,
            use_finbert=use_finbert,
            cache=cache,
        )

    def run(self, val_frac=0.15, test_frac=0.10) -> Dict[str, np.ndarray]:
        """
        Build and split context tensors.

        Returns dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test (np.ndarray)
          norm_stats  (NormStats)
          dates  (pd.DatetimeIndex)
        """
        # 1. Price features
        prices = self.price_loader.fetch(start=self.start, end=self.end)
        X_market, y, dates = self.price_loader.build_features(prices, self.ticker)

        # 2. Sentiment (lagged by 1 day for no-lookahead)
        sent_df = self.sent_loader.fetch(self.ticker, start=self.start, end=self.end)
        X_sent = self.sent_loader.align(sent_df, dates, shift=1)

        # 3. Concatenate context tensor
        X = np.concatenate([X_market, X_sent], axis=1).astype(np.float32)
        assert X.shape[1] == CONTEXT_DIM, \
            f"Context dim mismatch: got {X.shape[1]}, expected {CONTEXT_DIM}"

        # 4. Walk-forward split (strict temporal order, no shuffling)
        n = len(X)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val - n_test

        X_train, y_train = X[:n_train], y[:n_train]
        X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

        # 5. Normalize using training stats only (prevents lookahead)
        norm_stats = NormStats(
            mean=X_train.mean(axis=0),
            std=X_train.std(axis=0),
        )
        X_train = norm_stats.normalize(X_train)
        X_val   = norm_stats.normalize(X_val)
        X_test  = norm_stats.normalize(X_test)

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "norm_stats": norm_stats,
            "dates": dates,
        }


def describe_splits(data: dict) -> None:
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        y = data[f"y_{split}"]
        print(
            f"  {split:5s}: X={X.shape}  "
            f"y mean={y.mean():.5f}  std={y.std():.5f}  "
            f"skew={float(pd.Series(y).skew()):.3f}"
        )


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(f"Building feature pipeline for {ticker} ...")
    pipeline = FeaturePipeline(ticker=ticker, start="2020-01-01", end="2023-12-31")
    try:
        data = pipeline.run()
        print(f"Context dim: {data['X_train'].shape[1]}  (expected {CONTEXT_DIM})")
        describe_splits(data)
        print("\nfeature_engineering.py OK")
    except ImportError as e:
        print(f"Skipping (missing dependency): {e}")
