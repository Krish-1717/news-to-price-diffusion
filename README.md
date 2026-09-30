# News-to-Price Diffusion

A 30-day build of a production NLP pipeline that converts financial news into alpha signals — covering sentiment analysis, event studies, multimodal fusion, and a high-throughput processing FSM. Pure Python stdlib only.

## What's Inside

From raw news text to trading signals: Loughran-McDonald lexicon scoring, multimodal price+sentiment fusion, event-study abnormal returns, and a 7,200 article/sec production pipeline.

### Days 1–18 — Core NLP & Signal Infrastructure

Early days cover text preprocessing, named entity recognition, basic sentiment scoring, and initial price-news correlation models. See commit history for the full progression.

### Days 19–30 — Production Pipeline (`quant_code/`)

| File | Topic |
|------|-------|
| `news_day19_coverage_analysis.py` | News coverage bias detection, source weighting, cross-coverage consistency |
| `news_day20_consistency_model.py` | Cross-source signal consistency, ensemble agreement scoring |
| `news_day21_online_learning.py` | Passive-Aggressive & Perceptron online learners, concept drift detection |
| `news_day22_model_card.py` | Model card generation: fairness audit, calibration, coverage gaps |
| `news_day23_latent_factor_diffusion.py` | Latent factor extraction, news-to-price diffusion kernels |
| `news_day24_causal_analysis.py` | Granger causality, transfer entropy, news-return causal graphs |
| `news_day25_sentiment_analysis.py` | Loughran-McDonald lexicon, negation-aware scoring (window=3), entity sentiment |
| `news_day26_multimodal_fusion.py` | ModalityGate, LateFusionEnsemble, price momentum + vol regime + sentiment |
| `news_day27_event_study.py` | OLS market model, abnormal returns, CAR, CAAR t-test, bootstrap p-value |
| `news_day28_signal_backtest.py` | Composite signal, sigmoid position sizing, vol-targeting, IC decay analysis |
| `news_day29_topic_modeling.py` | TF-IDF, soft k-means TopicCluster, MinHash deduplication, co-occurrence graph |
| `news_day30_production_pipeline.py` | FSM pipeline (IDLE→INGESTING→PREPROCESSING→SCORING→SIGNAL_GEN→ALERTING), 7,200 art/sec |

## Key Concepts

- **Sentiment scoring** — Loughran-McDonald financial lexicon, negation window handling, entity-level attribution
- **Multimodal fusion** — Late-fusion ensemble combining price momentum, volatility regime, and NLP sentiment signals
- **Event study** — OLS market model residuals, CAR/CAAR, bootstrap significance testing
- **Online learning** — Passive-Aggressive and Perceptron classifiers with concept-drift detection
- **Topic modeling** — TF-IDF matrix, soft k-means clustering, MinHash Jaccard deduplication
- **Causal analysis** — Granger causality F-test, transfer entropy, directed information graphs
- **Production FSM** — Finite state machine with 6 states, async ingestion, monitoring dashboard
- **Signal backtesting** — IC decay curves, vol-targeted position sizing, multi-period signal attribution

## System Throughput

The Day 30 production pipeline achieves **7,200 articles/second** throughput with:
- Parallel preprocessing workers
- Streaming sentiment scoring
- Rolling signal aggregation
- Real-time alert dispatch

## Running the Code

```bash
# Run any module directly
python quant_code/news_day25_sentiment_analysis.py
python quant_code/news_day30_production_pipeline.py

# Run all Day 19-30 demos
python run_demos.py
```

## Requirements

Pure Python 3.8+ standard library only — no external packages required.

```
python >= 3.8
# No pip install needed
```

## Architecture

```
news-to-price-diffusion/
├── quant_code/                        # Days 19-30 production modules
│   ├── news_day19_coverage_analysis.py
│   ├── news_day20_consistency_model.py
│   ├── ...
│   ├── news_day29_topic_modeling.py
│   └── news_day30_production_pipeline.py
└── run_demos.py                       # Demo runner for all modules
```
