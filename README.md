# News-to-Price Diffusion

A generative model that outputs a **full distribution of next-day returns** conditioned on market context and news sentiment, using a Denoising Diffusion Probabilistic Model (DDPM).

## Why a distribution?

Point-forecast models hide uncertainty. A return distribution lets you:
- Set position size by the width of the predicted interval
- Detect bimodal outlooks (earnings/macro events)
- Backtest tail-risk strategies without simulation tricks

## Architecture

```
[Market features]  [Sentiment score]
        ↓                ↓
     Feature concat (128-d)
              ↓
         Condition MLP
              ↓
     DDPM Reverse Process (T=200 steps)
         UNet 1-D denoiser
              ↓
    N samples → return distribution
```

## Day-by-Day Build Plan

| Day | Milestone |
|-----|-----------|
| 1 | Scaffold, data layer, DDPM math, training loop, inference smoke-test |
| 2 | Real data (yfinance + Finnhub news), sentiment via FinBERT |
| 3 | Walk-forward validation (no lookahead) |
| 4 | Calibration — reliability diagrams, CRPS |
| 5 | Backtest harness with distribution-aware position sizing |
| 6 | Live inference endpoint (FastAPI) |
| 7–10 | Tuning, ablations, model card |

## Quick Start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py          # synthetic data smoke-test
python scripts/train.py --epochs 50   # full training run
python scripts/infer.py --ticker SPY
```

## Repo Structure

```
data/          synthetic + real data loaders
models/        DDPM architecture (UNet1D + diffusion math)
scripts/       train, infer, smoke_test
tests/         unit tests
```# News-to-Price Diffusion

A generative model that outputs a **full distribution of next-day returns** conditioned on market context and news sentiment, using a Denoising Diffusion Probabilistic Model (DDPM).

## Why a distribution?

Point-forecast models hide uncertainty. A return distribution lets you:
- Set position size by the width of the predicted interval
- Detect bimodal outlooks (earnings/macro events)
- Backtest tail-risk strategies without simulation tricks

## Architecture

```
[Market features]  [Sentiment score]
        |                |
     Feature concat (128-d)
              |
         Condition MLP
              |
     DDPM Reverse Process (T=200 steps)
              |
    N samples -> return distribution
```

## Day-by-Day Build Plan

| Day | Milestone |
|-----|-----------|
| 1 | Scaffold, data layer, DDPM math, training loop, inference smoke-test |
| 2 | Real data (yfinance + Finnhub), sentiment via FinBERT |
| 3 | Walk-forward validation (no lookahead) |
| 4 | Calibration: reliability diagrams, CRPS |
| 5 | Backtest harness with distribution-aware position sizing |
| 6 | Live inference endpoint (FastAPI) |

## Quick Start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/train.py --epochs 50
python scripts/infer.py --ticker SPY
```

## Repo Structure

data/ - synthetic + real data loaders
models/ - DDPM architecture and diffusion math
scripts/ - train, infer, smoke_test
tests/ - unit tests
