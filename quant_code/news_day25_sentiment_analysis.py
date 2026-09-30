"""
news_day25_sentiment_analysis.py
Day 25: Financial Sentiment Analysis — Loughran-McDonald lexicon scoring,
sentence-level polarity, entity extraction, news impact scoring,
event-driven signal construction.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
import re
from dataclasses import dataclass, field
from collections import Counter

# ---------------------------------------------------------------------------
# 1. Finance-domain sentiment lexicon (Loughran-McDonald style)
# ---------------------------------------------------------------------------
POSITIVE_WORDS = {
    'growth', 'profit', 'gain', 'increase', 'strong', 'beat', 'exceed',
    'record', 'high', 'surge', 'rally', 'outperform', 'upgrade', 'positive',
    'improve', 'recovery', 'expansion', 'revenue', 'earnings', 'dividend',
    'opportunity', 'robust', 'solid', 'momentum', 'advantage', 'efficient',
    'innovative', 'leading', 'upside', 'accelerate', 'confident', 'optimistic',
    'buy', 'overweight', 'attractive', 'strong_buy', 'raised', 'lift',
    'breakthrough', 'partnership', 'acquisition', 'launch', 'approval',
}

NEGATIVE_WORDS = {
    'loss', 'decline', 'fall', 'decrease', 'weak', 'miss', 'disappoint',
    'low', 'plunge', 'selloff', 'underperform', 'downgrade', 'negative',
    'worsen', 'recession', 'contraction', 'deficit', 'write-off', 'impairment',
    'risk', 'concern', 'uncertainty', 'headwind', 'challenge', 'pressure',
    'inefficient', 'lagging', 'downside', 'decelerate', 'pessimistic',
    'sell', 'underweight', 'unattractive', 'lowered', 'reduce',
    'lawsuit', 'recall', 'investigation', 'default', 'bankruptcy', 'fraud',
    'cut', 'layoff', 'restructure', 'warning', 'guidance_cut', 'miss',
}

UNCERTAINTY_WORDS = {
    'may', 'might', 'could', 'uncertain', 'unclear', 'potential', 'possible',
    'estimate', 'approximately', 'pending', 'contingent', 'subject_to',
    'risk', 'volatility', 'fluctuate', 'depend', 'expect', 'anticipate',
}

NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nor', "n't", 'without',
                   'lack', 'fail', 'unable', 'impossible'}

# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    # lowercase, remove punctuation (except hyphens for compound words)
    text = text.lower()
    text = re.sub(r"[^\w\s\-']", ' ', text)
    return [w.strip("'") for w in text.split() if w.strip("'")]

def remove_stopwords(tokens: list[str]) -> list[str]:
    STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'shall', 'should', 'can', 'could', 'may', 'might',
                  'must', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by',
                  'from', 'as', 'into', 'through', 'during', 'its', 'it',
                  'this', 'that', 'these', 'those', 'and', 'or', 'but', 'if',
                  'than', 'so', 'yet', 'both', 'also', 'such', 'up', 'out',
                  'about', 'above', 'after', 'before', 'between', 'each'}
    return [t for t in tokens if t not in STOPWORDS]

# ---------------------------------------------------------------------------
# 3. Sentiment scoring
# ---------------------------------------------------------------------------
@dataclass
class SentimentScore:
    positive: float
    negative: float
    uncertainty: float
    net: float        # positive - negative
    polarity: float   # in [-1, 1]
    subjectivity: float  # ratio of sentiment words
    word_count: int

def score_text(text: str, window: int = 3) -> SentimentScore:
    """
    Score text using lexicon + negation handling within a window.
    Negation in the preceding `window` tokens flips sentiment.
    """
    tokens = tokenize(text)
    n = len(tokens)
    pos_count = neg_count = unc_count = 0

    negation_active = [False] * n
    for i, tok in enumerate(tokens):
        if tok in NEGATION_WORDS:
            for j in range(i+1, min(i+1+window, n)):
                negation_active[j] = True

    for i, tok in enumerate(tokens):
        if tok in POSITIVE_WORDS:
            if negation_active[i]:
                neg_count += 1
            else:
                pos_count += 1
        elif tok in NEGATIVE_WORDS:
            if negation_active[i]:
                pos_count += 1
            else:
                neg_count += 1
        if tok in UNCERTAINTY_WORDS:
            unc_count += 1

    total_sentiment = pos_count + neg_count
    total = max(n, 1)

    net = (pos_count - neg_count) / max(total_sentiment, 1)
    polarity = (pos_count - neg_count) / max(n, 1)
    subjectivity = total_sentiment / total

    return SentimentScore(
        positive=pos_count / total,
        negative=neg_count / total,
        uncertainty=unc_count / total,
        net=net,
        polarity=max(-1.0, min(1.0, polarity * 10)),  # scale to [-1,1]
        subjectivity=subjectivity,
        word_count=n,
    )

def score_sentences(text: str) -> list[tuple[str, SentimentScore]]:
    """Score each sentence separately."""
    sentences = re.split(r'[.!?]+', text)
    results = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:
            results.append((sent, score_text(sent)))
    return results

# ---------------------------------------------------------------------------
# 4. Entity extraction (rule-based)
# ---------------------------------------------------------------------------
TICKER_PATTERN = re.compile(r'\b[A-Z]{2,5}\b')
MONEY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d+)?(?:[BMK])?')
PCT_PATTERN = re.compile(r'[-+]?\d+(?:\.\d+)?%')
QUARTER_PATTERN = re.compile(r'Q[1-4]\s*\d{4}|FY\d{4}')

@dataclass
class NewsEntities:
    tickers: list[str]
    money_mentions: list[str]
    percentages: list[str]
    quarters: list[str]

def extract_entities(text: str) -> NewsEntities:
    """Extract financial entities from text."""
    # Filter likely non-ticker all-caps (common English words)
    COMMON_CAPS = {'THE', 'AND', 'FOR', 'CEO', 'CFO', 'COO', 'IPO', 'EPS',
                    'GDP', 'US', 'EU', 'UK', 'YOY', 'QOQ', 'MOM', 'YTD',
                    'ATH', 'ATL', 'ETF', 'SEC', 'FED', 'IMF', 'WHO', 'AI',
                    'ML', 'IT', 'OR', 'IN', 'AT', 'TO', 'BY', 'OF'}
    tickers = [t for t in TICKER_PATTERN.findall(text) if t not in COMMON_CAPS]
    return NewsEntities(
        tickers=list(dict.fromkeys(tickers)),  # deduplicate, preserve order
        money_mentions=MONEY_PATTERN.findall(text),
        percentages=PCT_PATTERN.findall(text),
        quarters=QUARTER_PATTERN.findall(text),
    )

# ---------------------------------------------------------------------------
# 5. News impact scoring
# ---------------------------------------------------------------------------
HEADLINE_MULTIPLIERS = {
    # High impact events
    'earnings': 2.0, 'guidance': 1.8, 'acquisition': 2.5, 'merger': 2.5,
    'lawsuit': 1.8, 'fda': 2.0, 'approval': 2.0, 'recall': 2.0,
    'bankruptcy': 3.0, 'default': 3.0, 'fraud': 2.5, 'investigation': 2.0,
    'dividend': 1.5, 'buyback': 1.5, 'layoff': 1.5, 'ipo': 2.0,
    # Medium impact
    'partnership': 1.3, 'contract': 1.2, 'product': 1.1, 'launch': 1.3,
    'upgrade': 1.2, 'downgrade': 1.2, 'analyst': 1.0, 'rating': 1.0,
}

def news_impact_score(text: str, headline: bool = False) -> dict:
    """
    Compute overall news impact score combining:
    - Sentiment polarity
    - Uncertainty discount
    - Topic multiplier (high-impact events)
    - Headline premium
    """
    sentiment = score_text(text)
    tokens = tokenize(text)

    # Topic multiplier: max multiplier from detected high-impact words
    topic_mult = 1.0
    for tok in tokens:
        if tok in HEADLINE_MULTIPLIERS:
            topic_mult = max(topic_mult, HEADLINE_MULTIPLIERS[tok])

    # Uncertainty discount: high uncertainty → reduced signal confidence
    uncertainty_discount = 1.0 - 0.5 * sentiment.uncertainty

    # Headline premium: headlines are more informative
    headline_mult = 1.5 if headline else 1.0

    impact = (sentiment.polarity * topic_mult *
               uncertainty_discount * headline_mult)

    # Confidence: based on subjectivity and length
    confidence = min(1.0, sentiment.subjectivity * math.log(max(sentiment.word_count, 10)) / 5)

    return {
        'sentiment': sentiment.polarity,
        'uncertainty': sentiment.uncertainty,
        'topic_multiplier': topic_mult,
        'uncertainty_discount': uncertainty_discount,
        'impact_score': impact,
        'confidence': confidence,
        'net_sentiment': sentiment.net,
    }

# ---------------------------------------------------------------------------
# 6. Event-driven signal construction
# ---------------------------------------------------------------------------
@dataclass
class NewsEvent:
    timestamp: int    # day index
    ticker: str
    headline: str
    body: str = ''

def build_sentiment_signal(events: list[NewsEvent],
                             horizon: int = 5,
                             decay: float = 0.7) -> dict[str, list[tuple[int, float]]]:
    """
    Build per-ticker sentiment time series.
    Signal at day t = sum over past events (exponential decay by age).
    Returns {ticker: [(day, signal_value), ...]}.
    """
    from collections import defaultdict
    ticker_events: dict[str, list] = defaultdict(list)
    for ev in events:
        impact = news_impact_score(ev.headline + ' ' + ev.body, headline=True)
        ticker_events[ev.ticker].append((ev.timestamp, impact['impact_score']))

    max_day = max((ev.timestamp for ev in events), default=0) + horizon
    signals: dict[str, list] = {}

    for ticker, ev_list in ticker_events.items():
        signal_series = []
        for t in range(max_day):
            # Sum decayed signals from past events
            sig = sum(
                score * (decay ** (t - t_ev))
                for t_ev, score in ev_list
                if 0 <= t - t_ev <= horizon
            )
            signal_series.append((t, sig))
        signals[ticker] = signal_series

    return signals

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 25: Financial Sentiment Analysis")
    print("=" * 65)

    # Sample financial news headlines and articles
    texts = [
        ("AAPL Q3 Earnings Beat: Revenue Surges 12%, Raises FY2026 Guidance",
         "Apple Inc. reported record Q3 2026 earnings, with revenue exceeding analyst "
         "estimates by $2.3B. EPS of $1.45 beat consensus of $1.32. The company raised "
         "its full-year revenue guidance by 5%, citing strong iPhone demand and services growth.",
         True),

        ("TSLA Misses Q2 Estimates, Cuts Delivery Guidance Amid Demand Concerns",
         "Tesla reported Q2 deliveries of 385K units, missing the 420K consensus estimate. "
         "Management cut full-year delivery guidance citing macro headwinds and pricing pressure. "
         "The stock fell 8% in after-hours trading. CFO warns of continued margin compression.",
         True),

        ("FED Signals Uncertainty on Rate Path, Markets Could See Volatility",
         "The Federal Reserve indicated it may not cut rates as aggressively as markets "
         "anticipated, citing uncertain inflation trajectory. Investors could face risk "
         "if economic conditions do not improve. The Fed will carefully monitor data.",
         False),
    ]

    print("\n1. Headline & Article Sentiment Scoring")
    for headline, body, is_headline in texts:
        print(f"\n   Headline: {headline[:60]}...")
        h_score = score_text(headline)
        b_score = score_text(body)
        impact = news_impact_score(headline + ' ' + body, headline=True)
        print(f"   Headline sentiment  : polarity={h_score.polarity:+.3f}  uncertainty={h_score.uncertainty:.3f}")
        print(f"   Article sentiment   : polarity={b_score.polarity:+.3f}  uncertainty={b_score.uncertainty:.3f}")
        print(f"   Topic multiplier    : {impact['topic_multiplier']:.1f}x")
        print(f"   Impact score        : {impact['impact_score']:+.4f}")
        print(f"   Confidence          : {impact['confidence']:.3f}")

    print("\n2. Entity Extraction")
    sample = "AAPL beat Q3 2026 estimates; NVDA and MSFT also reported gains. Revenue up 12%. EPS $1.45."
    entities = extract_entities(sample)
    print(f"   Text: {sample}")
    print(f"   Tickers  : {entities.tickers}")
    print(f"   Money    : {entities.money_mentions}")
    print(f"   Percents : {entities.percentages}")
    print(f"   Quarters : {entities.quarters}")

    print("\n3. Sentence-Level Polarity")
    article = ("Revenue surged to record highs in Q3. However, management warned that "
                "macro headwinds could pressure margins. The company did not provide guidance "
                "for Q4, citing uncertainty. Analysts remain bullish on the long-term opportunity.")
    sentence_scores = score_sentences(article)
    for sent, score in sentence_scores:
        bar = '▲' if score.polarity > 0.05 else ('▼' if score.polarity < -0.05 else '–')
        print(f"   {bar} {score.polarity:+.3f}  {sent[:60]}...")

    print("\n4. Negation Handling")
    neg_tests = [
        "Revenue grew strongly",
        "Revenue did not grow",
        "Revenue failed to grow",
        "No decline in earnings",
    ]
    for t in neg_tests:
        s = score_text(t)
        print(f"   '{t}': polarity={s.polarity:+.4f}  pos={s.positive:.3f}  neg={s.negative:.3f}")

    print("\n5. Event-Driven Sentiment Signal")
    rng = random.Random(42)
    events = [
        NewsEvent(0, 'AAPL', 'Apple beats earnings, raises guidance'),
        NewsEvent(1, 'AAPL', 'Analyst upgrades AAPL to Buy'),
        NewsEvent(2, 'TSLA', 'Tesla misses deliveries guidance cut'),
        NewsEvent(3, 'AAPL', 'Apple dividend increase announced'),
        NewsEvent(4, 'TSLA', 'Tesla faces investigation by SEC'),
        NewsEvent(5, 'AAPL', 'New product launch boosts AAPL outlook'),
    ]
    signals = build_sentiment_signal(events, horizon=5, decay=0.7)
    print(f"   {'Day':>4} | {'AAPL signal':>12} | {'TSLA signal':>12}")
    print("   " + "-" * 35)
    aapl_sig = dict(signals.get('AAPL', []))
    tsla_sig = dict(signals.get('TSLA', []))
    for day in range(10):
        a = aapl_sig.get(day, 0.0)
        t = tsla_sig.get(day, 0.0)
        print(f"   {day:>4} | {a:>+12.4f} | {t:>+12.4f}")

    print("\n[Done] Day 25: Sentiment Analysis complete.")
