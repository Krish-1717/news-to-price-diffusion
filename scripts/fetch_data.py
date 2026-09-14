"""
fetch_data.py — CLI to download and cache all data for a list of tickers.

Day 2: Pre-fetches price + sentiment data so training can run offline.

Usage:
    python scripts/fetch_data.py --tickers AAPL MSFT GOOGL SPY QQQ
        --start 2018-01-01 --end 2023-12-31
        --finnhub_key YOUR_KEY

Without --finnhub_key, sentiment falls back to the synthetic AR(1) signal.
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.real_loader import PriceLoader
from data.finnhub_sentiment import SentimentLoader
from data.feature_engineering import FeaturePipeline, describe_splits


def parse_args():
    p = argparse.ArgumentParser(description="Pre-fetch price + sentiment data")
    p.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "MSFT"])
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end",   default="2023-12-31")
    p.add_argument("--finnhub_key", default=None)
    p.add_argument("--use_finbert", action="store_true")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--summary", action="store_true",
                   help="Print train/val/test split stats for each ticker")
    return p.parse_args()


def main():
    args = parse_args()
    cache = not args.no_cache

    print("=" * 60)
    print("  news-to-price-diffusion -- Data Fetch (Day 2)")
    print("=" * 60)
    print(f"  Tickers : {args.tickers}")
    print(f"  Period  : {args.start} to {args.end}")
    print(f"  Finnhub : {'API key set' if args.finnhub_key else 'synthetic fallback'}")
    print(f"  Cache   : {cache}")
    print()

    # Step 1: Download prices
    print("[1/3] Downloading prices via yfinance ...")
    t0 = time.time()
    try:
        price_loader = PriceLoader(args.tickers, cache=cache)
        prices = price_loader.fetch(start=args.start, end=args.end)
        print(f"  OK Prices: {prices.shape}  ({time.time()-t0:.1f}s)")
        available = [t for t in args.tickers if t in prices.columns]
    except ImportError:
        print("  FAIL yfinance not installed. Run: pip install yfinance")
        available = []

    # Step 2: Sentiment per ticker
    print(f"[2/3] Fetching sentiment ...")
    sent_loader = SentimentLoader(
        api_key=args.finnhub_key, use_finbert=args.use_finbert, cache=cache
    )
    for ticker in available:
        t0 = time.time()
        df = sent_loader.fetch(ticker, start=args.start, end=args.end)
        print(f"  OK {ticker:6s} sentiment: {df.shape}  bull={df['bullish'].mean():.3f}  ({time.time()-t0:.1f}s)")

    # Step 3: Build feature tensors
    if args.summary and available:
        print(f"[3/3] Building feature tensors + split stats ...")
        for ticker in available:
            print(f"  {ticker}:")
            try:
                pipeline = FeaturePipeline(
                    ticker=ticker, start=args.start, end=args.end,
                    finnhub_api_key=args.finnhub_key,
                    use_finbert=args.use_finbert, cache=cache,
                )
                data = pipeline.run()
                describe_splits(data)
            except Exception as e:
                print(f"  FAIL {e}")
    else:
        print(f"[3/3] Skipping split stats (pass --summary to enable)")

    print("All data cached. Ready to train.")
    print("  Next: python scripts/train.py --ticker SPY")


if __name__ == "__main__":
    main()
