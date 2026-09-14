"""
real_loader.py — Fetch and cache real OHLCV price data via yfinance.

Day 2: Replaces the synthetic generator with real market data.
All features are constructed from t-1 data only (no lookahead).

Usage:
    from data.real_loader import PriceLoader
    loader = PriceLoader(["AAPL", "MSFT", "GOOGL"])
    df = loader.fetch(start="2018-01-01", end="2023-12-31")
    X, y, dates = loader.build_features(df, ticker="AAPL")
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

CACHE_DIR = Path(__file__).parent.parent / "data" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [5, 10, 21, 63]  # trading days: 1wk, 2wk, 1mo, 3mo


def _compute_features(returns: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame(index=returns.index)
    for w in WINDOWS:
        feat[f"ret_{w}d_lag1"] = returns.rolling(w).mean().shift(1)
        feat[f"vol_{w}d_lag1"] = returns.rolling(w).std().shift(1)
        feat[f"mom_{w}d_lag1"] = (1 + returns).rolling(w).apply(
            lambda x: x.prod() - 1, raw=True
        ).shift(1)
    delta = returns.copy()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    feat["rsi_14_lag1"] = (100 - 100 / (1 + rs)).shift(1) / 100.0
    feat["skew_21d_lag1"] = returns.rolling(21).skew().shift(1)
    return feat


class PriceLoader:
    def __init__(self, tickers: List[str], cache: bool = True):
        self.tickers = tickers
        self.cache = cache

    def fetch(self, start="2018-01-01", end="2023-12-31") -> pd.DataFrame:
        if not HAS_YFINANCE:
            raise ImportError("yfinance not installed.")
        all_prices = {}
        for ticker in self.tickers:
            cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
            if self.cache and cache_path.exists():
                df = pd.read_parquet(cache_path)
            else:
                raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
                if raw.empty:
                    continue
                df = raw[["Close"]].rename(columns={"Close": "close"})
                if self.cache:
                    df.to_parquet(cache_path)
            all_prices[ticker] = df["close"]
        return pd.DataFrame(all_prices)

    def build_features(self, prices, ticker, min_history=63):
        if ticker not in prices.columns:
            raise ValueError(f"Ticker {ticker} not in prices DataFrame")
        close = prices[ticker].dropna()
        log_ret = np.log(close / close.shift(1)).dropna()
        feat_df = _compute_features(log_ret)
        valid_mask = feat_df.notna().all(axis=1)
        log_ret = log_ret[valid_mask]
        feat_df = feat_df[valid_mask]
        if len(log_ret) <= min_history:
            raise ValueError(f"Not enough data for {ticker}")
        X = feat_df.values.astype(np.float32)
        y = log_ret.values.astype(np.float32)
        return X, y, log_ret.index

    @staticmethod
    def walk_forward_split(X, y, val_frac=0.15, test_frac=0.10):
        n = len(X)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        n_train = n - n_val - n_test
        return {
            "X_train": X[:n_train], "y_train": y[:n_train],
            "X_val": X[n_train:n_train+n_val], "y_val": y[n_train:n_train+n_val],
            "X_test": X[n_train+n_val:], "y_test": y[n_train+n_val:],
        }
