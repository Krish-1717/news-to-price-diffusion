"""
test_real_loader.py — Unit tests for real_loader.py and finnhub_sentiment.py

Day 2: Validates feature engineering, no-lookahead guarantee,
and walk-forward split logic using mocked APIs (no network calls).
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.real_loader import _compute_features, PriceLoader, WINDOWS
from data.finnhub_sentiment import SentimentLoader
from data.feature_engineering import (
    FeaturePipeline, NormStats, CONTEXT_DIM, MARKET_DIM, SENTIMENT_DIM
)


def _fake_returns(n=500, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-02", periods=n)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=dates, name="SPY")


def _fake_close(n=500, seed=42):
    returns = _fake_returns(n=n, seed=seed)
    prices = (1 + returns).cumprod() * 100.0
    prices.name = "SPY"
    return prices


class TestComputeFeatures(unittest.TestCase):

    def setUp(self):
        self.returns = _fake_returns(n=200)

    def test_output_shape(self):
        feat = _compute_features(self.returns)
        expected_cols = len(WINDOWS) * 3 + 2
        self.assertEqual(feat.shape[1], expected_cols)

    def test_lag_naming(self):
        feat = _compute_features(self.returns)
        for col in feat.columns:
            self.assertIn("lag1", col, f"Column {col} should have lag1 suffix")

    def test_rsi_normalized(self):
        feat = _compute_features(self.returns).dropna()
        rsi = feat["rsi_14_lag1"]
        self.assertTrue((rsi >= 0.0).all())
        self.assertTrue((rsi <= 1.0).all())

    def test_no_future_bleed(self):
        feat_original = _compute_features(self.returns)
        modified = self.returns.copy()
        modified.iloc[-1] = 999.0
        feat_modified = _compute_features(modified)
        pd.testing.assert_frame_equal(
            feat_original.iloc[:-1], feat_modified.iloc[:-1],
            check_exact=False, rtol=1e-5,
        )

    def test_skewness_present(self):
        feat = _compute_features(self.returns)
        self.assertIn("skew_21d_lag1", feat.columns)


class TestPriceLoader(unittest.TestCase):

    @patch("data.real_loader.HAS_YFINANCE", True)
    @patch("data.real_loader.yf")
    def test_fetch_returns_dataframe(self, mock_yf):
        prices = _fake_close(n=300)
        mock_yf.download.return_value = pd.DataFrame({"Close": prices})
        loader = PriceLoader(["SPY"], cache=False)
        result = loader.fetch(start="2020-01-01", end="2021-12-31")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("SPY", result.columns)

    def test_build_features_shapes(self):
        prices_df = pd.DataFrame({"SPY": _fake_close(n=300)})
        loader = PriceLoader(["SPY"], cache=False)
        X, y, dates = loader.build_features(prices_df, "SPY")
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], MARKET_DIM)
        self.assertEqual(len(dates), len(y))

    def test_build_features_dtype(self):
        prices_df = pd.DataFrame({"SPY": _fake_close(n=200)})
        loader = PriceLoader(["SPY"], cache=False)
        X, y, _ = loader.build_features(prices_df, "SPY")
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_walk_forward_no_overlap(self):
        n = 400
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        splits = PriceLoader.walk_forward_split(X, y, val_frac=0.15, test_frac=0.10)
        total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        self.assertEqual(total, n)

    def test_build_features_unknown_ticker(self):
        prices_df = pd.DataFrame({"SPY": _fake_close(n=200)})
        loader = PriceLoader(["AAPL"], cache=False)
        with self.assertRaises(ValueError):
            loader.build_features(prices_df, "AAPL")


class TestSentimentLoader(unittest.TestCase):

    def test_synthetic_shape(self):
        start, end = "2021-01-01", "2021-12-31"
        df = SentimentLoader._synthetic("SPY", start, end)
        expected = pd.bdate_range(start=start, end=end)
        self.assertEqual(len(df), len(expected))
        self.assertListEqual(list(df.columns), ["bullish", "bearish", "article_count_norm"])

    def test_synthetic_bounds(self):
        df = SentimentLoader._synthetic("AAPL", "2020-01-01", "2022-12-31")
        self.assertTrue((df["bullish"] >= 0.0).all())
        self.assertTrue((df["bullish"] <= 1.0).all())

    def test_synthetic_deterministic(self):
        df1 = SentimentLoader._synthetic("SPY", "2021-01-01", "2021-06-30")
        df2 = SentimentLoader._synthetic("SPY", "2021-01-01", "2021-06-30")
        pd.testing.assert_frame_equal(df1, df2)

    def test_align_shape(self):
        start, end = "2021-01-01", "2021-06-30"
        loader = SentimentLoader(cache=False)
        df = loader.fetch("SPY", start=start, end=end)
        price_dates = pd.bdate_range(start=start, end=end)
        arr = loader.align(df, price_dates, shift=1)
        self.assertEqual(arr.shape, (len(price_dates), SENTIMENT_DIM))
        self.assertEqual(arr.dtype, np.float32)

    def test_align_neutral_pad(self):
        start, end = "2021-01-01", "2021-06-30"
        loader = SentimentLoader(cache=False)
        df = loader.fetch("SPY", start=start, end=end)
        price_dates = pd.bdate_range(start=start, end=end)
        arr = loader.align(df, price_dates, shift=1)
        self.assertAlmostEqual(arr[0, 0], 0.5, places=5)


class TestNormStats(unittest.TestCase):

    def test_normalize_zero_mean(self):
        X = np.random.randn(1000, CONTEXT_DIM).astype(np.float32)
        ns = NormStats(mean=X.mean(axis=0), std=X.std(axis=0))
        X_norm = ns.normalize(X)
        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-5)

    def test_roundtrip(self):
        X = np.random.randn(500, CONTEXT_DIM).astype(np.float32) * 3 + 1
        ns = NormStats(mean=X.mean(axis=0), std=X.std(axis=0))
        X_rt = ns.denormalize(ns.normalize(X))
        np.testing.assert_allclose(X_rt, X, rtol=1e-4)


class TestFeaturePipeline(unittest.TestCase):

    @patch("data.real_loader.HAS_YFINANCE", True)
    @patch("data.real_loader.yf")
    def test_context_dim(self, mock_yf):
        prices = _fake_close(n=400)
        mock_yf.download.return_value = pd.DataFrame({"Close": prices})
        pipeline = FeaturePipeline("SPY", start="2020-01-01", end="2021-06-30", cache=False)
        data = pipeline.run()
        self.assertEqual(data["X_train"].shape[1], CONTEXT_DIM)

    @patch("data.real_loader.HAS_YFINANCE", True)
    @patch("data.real_loader.yf")
    def test_no_nan(self, mock_yf):
        prices = _fake_close(n=400)
        mock_yf.download.return_value = pd.DataFrame({"Close": prices})
        pipeline = FeaturePipeline("SPY", start="2020-01-01", end="2021-06-30", cache=False)
        data = pipeline.run()
        for split in ("train", "val", "test"):
            self.assertFalse(np.any(np.isnan(data[f"X_{split}"])))
            self.assertFalse(np.any(np.isnan(data[f"y_{split}"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
