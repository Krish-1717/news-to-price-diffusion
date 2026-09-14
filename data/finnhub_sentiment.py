"""
finnhub_sentiment.py — Fetch news sentiment scores via Finnhub API.

Day 2: Provides the "news context" vector that conditions the DDPM.

Finnhub's /company-news endpoint returns articles with a sentiment
score. We aggregate by trading day into a (T, 3) tensor:
  [bullish_score, bearish_score, article_count_normalized]
"""

import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    import finnhub
    HAS_FINNHUB = True
except ImportError:
    HAS_FINNHUB = False

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

CACHE_DIR = Path(__file__).parent.parent / "data" / ".cache" / "sentiment"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class SentimentLoader:
    """
    Fetch and aggregate Finnhub news sentiment for a ticker.

    If Finnhub is unavailable, falls back to a synthetic sentiment
    signal (useful for offline testing and CI).
    """

    SENTIMENT_DIM = 3  # [bullish, bearish, article_count_norm]

    def __init__(self, api_key=None, use_finbert=False, cache=True):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.use_finbert = use_finbert and HAS_TRANSFORMERS
        self.cache = cache
        self._client = None
        self._finbert = None
        if self.use_finbert:
            self._finbert = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True,
            )

    @property
    def client(self):
        if not HAS_FINNHUB:
            raise ImportError("finnhub-python not installed.")
        if self._client is None:
            if not self.api_key:
                raise ValueError("No Finnhub API key. Set FINNHUB_API_KEY env var.")
            self._client = finnhub.Client(api_key=self.api_key)
        return self._client

    def fetch(self, ticker, start, end):
        cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
        if self.cache and cache_path.exists():
            return pd.read_parquet(cache_path)
        if not self.api_key or not HAS_FINNHUB:
            df = self._synthetic(ticker, start, end)
        else:
            df = self._fetch_live(ticker, start, end)
        if self.cache:
            df.to_parquet(cache_path)
        return df

    def _fetch_live(self, ticker, start, end):
        start_dt = datetime.date.fromisoformat(start)
        end_dt = datetime.date.fromisoformat(end)
        all_records = []
        cur = start_dt
        while cur < end_dt:
            chunk_end = min(cur + datetime.timedelta(days=30), end_dt)
            try:
                articles = self.client.company_news(
                    ticker, _from=cur.isoformat(), to=chunk_end.isoformat()
                )
                all_records.extend(articles)
            except Exception as e:
                print(f"[warn] Finnhub error for {ticker} {cur}: {e}")
            cur = chunk_end + datetime.timedelta(days=1)
            time.sleep(0.12)  # rate limit
        return self._aggregate(all_records, start, end)

    def _aggregate(self, articles, start, end):
        date_range = pd.bdate_range(start=start, end=end)
        records = {d: {"bullish": 0.0, "bearish": 0.0, "count": 0} for d in date_range}
        for art in articles:
            ts = art.get("datetime", 0)
            date = pd.Timestamp(ts, unit="s").normalize()
            if date not in records:
                continue
            if self.use_finbert and self._finbert and art.get("headline"):
                scores = self._finbert_score(art["headline"])
            else:
                sent = art.get("sentiment", {})
                scores = {
                    "bullish": float(sent.get("bullishPercent", 0.5)),
                    "bearish": float(sent.get("bearishPercent", 0.5)),
                }
            records[date]["bullish"] += scores.get("bullish", 0.5)
            records[date]["bearish"] += scores.get("bearish", 0.5)
            records[date]["count"] += 1
        rows = []
        max_count = max((v["count"] for v in records.values()), default=1) or 1
        for date in date_range:
            r = records[date]
            n = r["count"]
            if n > 0:
                rows.append({
                    "bullish": r["bullish"] / n,
                    "bearish": r["bearish"] / n,
                    "article_count_norm": n / max_count,
                })
            else:
                rows.append({"bullish": 0.5, "bearish": 0.5, "article_count_norm": 0.0})
        return pd.DataFrame(rows, index=date_range)

    def _finbert_score(self, headline):
        result = self._finbert(headline[:512])[0]
        scores = {r["label"].lower(): r["score"] for r in result}
        return {
            "bullish": scores.get("positive", 0.5),
            "bearish": scores.get("negative", 0.5),
        }

    @staticmethod
    def _synthetic(ticker, start, end):
        """AR(1) synthetic sentiment: phi=0.7, mean-reverting around 0.5."""
        rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        phi = 0.7
        bull = np.zeros(n)
        bull[0] = 0.5
        for t in range(1, n):
            bull[t] = phi * bull[t - 1] + (1 - phi) * 0.5 + rng.normal(0, 0.05)
        bull = np.clip(bull, 0.1, 0.9)
        bear = 1.0 - bull + rng.normal(0, 0.02, n)
        bear = np.clip(bear, 0.1, 0.9)
        count = rng.integers(0, 10, n) / 10.0
        return pd.DataFrame(
            {"bullish": bull, "bearish": bear, "article_count_norm": count},
            index=dates,
        )

    def align(self, sentiment_df, price_dates, shift=1):
        """
        Align sentiment to price dates with shift for no-lookahead.
        shift=1 means arr[t] uses sentiment from t-1.
        """
        aligned = sentiment_df.reindex(price_dates, method="ffill").fillna(0.5)
        arr = aligned[["bullish", "bearish", "article_count_norm"]].values.astype(np.float32)
        if shift > 0:
            arr = np.roll(arr, shift, axis=0)
            arr[:shift] = 0.5
        return arr


if __name__ == "__main__":
    ticker = "AAPL"
    loader = SentimentLoader(cache=True)
    df = loader.fetch(ticker, start="2022-01-01", end="2022-12-31")
    print(f"Sentiment for {ticker}: {df.shape}")
    print(df.head())
    print(f"\nMean bullish: {df['bullish'].mean():.3f}")
    print("finnhub_sentiment.py OK")
