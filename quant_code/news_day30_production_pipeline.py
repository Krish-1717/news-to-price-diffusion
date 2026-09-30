"""
news_day30_production_pipeline.py
Day 30: Production NLP Pipeline — end-to-end news-to-signal pipeline:
ingest → preprocess → sentiment score → entity extraction → topic tag →
signal generation → trade signal → monitoring dashboard.
Finite state machine orchestration, health metrics, alert thresholds.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import re
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

# ---------------------------------------------------------------------------
# 1. Domain objects
# ---------------------------------------------------------------------------
@dataclass
class RawArticle:
    article_id: str
    timestamp: float       # Unix epoch
    source: str
    headline: str
    body: str = ''

@dataclass
class ProcessedArticle:
    article_id: str
    timestamp: float
    source: str
    tokens: list[str]
    entities: list[str]
    sentiment_score: float     # [-1, 1]
    topic: str
    is_duplicate: bool

@dataclass
class NewsSignal:
    timestamp: float
    composite_score: float     # [-1, 1]  positive = bullish
    entity_scores: dict[str, float]
    n_articles: int
    confidence: float

@dataclass
class TradeSignal:
    timestamp: float
    direction: str             # 'BUY' | 'SELL' | 'FLAT'
    strength: float            # [0, 1]
    composite_news_score: float
    trigger: str

# ---------------------------------------------------------------------------
# 2. Pipeline state machine
# ---------------------------------------------------------------------------
class PipelineState(Enum):
    IDLE = 'IDLE'
    INGESTING = 'INGESTING'
    PREPROCESSING = 'PREPROCESSING'
    SCORING = 'SCORING'
    SIGNAL_GEN = 'SIGNAL_GEN'
    ALERTING = 'ALERTING'
    ERROR = 'ERROR'

# ---------------------------------------------------------------------------
# 3. Preprocessing module
# ---------------------------------------------------------------------------
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'to',
    'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'this',
    'that', 'it', 'and', 'or', 'but', 'not', 'no', 'so', 'if', 'said',
    'says', 'report', 'new', 'also', 'has', 'have', 'had', 'will', 'can',
}

FINANCIAL_ENTITIES = {
    'fed', 'federal reserve', 'fomc', 'ecb', 'bank of england',
    'apple', 'microsoft', 'google', 'amazon', 'tesla', 'nvidia', 'meta',
    'sp500', 's&p', 'nasdaq', 'dow jones', 'ftse', 'dax',
    'oil', 'gold', 'dollar', 'euro', 'bitcoin',
    'opec', 'china', 'europe', 'uk', 'usa',
    'inflation', 'gdp', 'unemployment', 'interest rate',
}

def tokenize(text: str) -> list[str]:
    words = re.sub(r'[^a-zA-Z\s]', ' ', text.lower()).split()
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]

def extract_entities(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for entity in FINANCIAL_ENTITIES:
        if entity in lower:
            found.append(entity.title().replace(' ', '_'))
    return list(set(found))

# ---------------------------------------------------------------------------
# 4. Sentiment module (Loughran-McDonald lite)
# ---------------------------------------------------------------------------
LM_POSITIVE = {
    'surge', 'rally', 'beat', 'record', 'growth', 'gain', 'profit', 'rise',
    'boost', 'strong', 'positive', 'recover', 'advance', 'expand', 'improve',
    'outperform', 'exceed', 'exceed', 'better', 'higher', 'jump', 'soar',
    'robust', 'optimistic', 'bullish', 'upgrade', 'opportunity', 'momentum'
}
LM_NEGATIVE = {
    'fall', 'drop', 'slump', 'decline', 'loss', 'miss', 'weak', 'risk',
    'fear', 'crisis', 'recession', 'default', 'downgrade', 'cut', 'layoff',
    'deficit', 'bear', 'volatile', 'uncertain', 'concern', 'worse', 'lower',
    'plunge', 'crash', 'sell', 'disappointing', 'warning', 'shortage', 'drag'
}
NEGATORS = {'not', 'no', 'never', 'neither', 'nor', 'barely', 'hardly', 'fails'}

def score_sentiment(tokens: list[str]) -> float:
    """Negation-aware sentiment scoring in [-1, 1]."""
    pos, neg = 0, 0
    n = len(tokens)
    for i, tok in enumerate(tokens):
        # Check negation window = 3
        negated = any(tokens[max(0,i-3):i][j] in NEGATORS for j in range(min(3, i)))
        if tok in LM_POSITIVE:
            if negated: neg += 1
            else: pos += 1
        elif tok in LM_NEGATIVE:
            if negated: pos += 0.5
            else: neg += 1
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

# ---------------------------------------------------------------------------
# 5. Simple topic tagger (keyword-based)
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    'MONETARY_POLICY':  ['rate', 'fed', 'central_bank', 'inflation', 'fomc', 'hike', 'cut'],
    'EARNINGS':         ['earnings', 'revenue', 'profit', 'quarterly', 'eps', 'beat', 'miss'],
    'COMMODITIES':      ['oil', 'gold', 'commodity', 'opec', 'crude', 'energy'],
    'TECH':             ['tech', 'technology', 'software', 'ai', 'chip', 'semiconductor'],
    'MACRO':            ['gdp', 'unemployment', 'payroll', 'growth', 'recession', 'economy'],
    'FX_RATES':         ['dollar', 'euro', 'currency', 'forex', 'exchange'],
}

def tag_topic(tokens: list[str]) -> str:
    tok_set = set(tokens)
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in tok_set)
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else 'GENERAL'

# ---------------------------------------------------------------------------
# 6. Deduplication (character shingling)
# ---------------------------------------------------------------------------
def _shingles(text: str, k: int = 4) -> set[str]:
    t = re.sub(r'\s+', ' ', text.lower())
    return {t[i:i+k] for i in range(len(t) - k + 1)}

def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)

# ---------------------------------------------------------------------------
# 7. Signal generation
# ---------------------------------------------------------------------------
def generate_news_signal(processed: list[ProcessedArticle],
                          window_seconds: float = 3600.0,
                          current_time: float = None) -> NewsSignal:
    """Aggregate processed articles within time window into a news signal."""
    if current_time is None:
        current_time = max((a.timestamp for a in processed), default=0.0)
    in_window = [a for a in processed
                 if not a.is_duplicate and current_time - a.timestamp <= window_seconds]
    if not in_window:
        return NewsSignal(current_time, 0.0, {}, 0, 0.0)

    # Recency weighting: w = exp(-age / half_life)
    half_life = window_seconds / 3
    def w(a):
        age = current_time - a.timestamp
        return math.exp(-age / max(half_life, 1.0))

    total_w = sum(w(a) for a in in_window)
    composite = sum(w(a) * a.sentiment_score for a in in_window) / max(total_w, 1e-10)

    # Per-entity sentiment
    entity_scores: dict[str, list[float]] = defaultdict(list)
    for a in in_window:
        for e in a.entities:
            entity_scores[e].append(a.sentiment_score)
    entity_avg = {e: sum(v)/len(v) for e, v in entity_scores.items()}

    confidence = min(1.0, len(in_window) / 10.0) * min(1.0, abs(composite) * 3)
    return NewsSignal(current_time, composite, entity_avg, len(in_window), confidence)

# ---------------------------------------------------------------------------
# 8. Trade signal
# ---------------------------------------------------------------------------
def signal_to_trade(news_signal: NewsSignal,
                     bullish_threshold: float = 0.15,
                     bearish_threshold: float = -0.15,
                     min_confidence: float = 0.2) -> TradeSignal:
    s = news_signal.composite_score
    c = news_signal.confidence
    if c < min_confidence:
        return TradeSignal(news_signal.timestamp, 'FLAT', 0.0, s, 'LOW_CONFIDENCE')
    if s > bullish_threshold:
        strength = min(1.0, (s - bullish_threshold) / (1 - bullish_threshold)) * c
        return TradeSignal(news_signal.timestamp, 'BUY', strength, s, f'BULLISH(c={c:.2f})')
    if s < bearish_threshold:
        strength = min(1.0, (bearish_threshold - s) / (1 + bearish_threshold)) * c
        return TradeSignal(news_signal.timestamp, 'SELL', strength, s, f'BEARISH(c={c:.2f})')
    return TradeSignal(news_signal.timestamp, 'FLAT', 0.0, s, 'NEUTRAL')

# ---------------------------------------------------------------------------
# 9. Production pipeline orchestrator
# ---------------------------------------------------------------------------
@dataclass
class PipelineMetrics:
    articles_ingested: int = 0
    articles_processed: int = 0
    duplicates_removed: int = 0
    signals_generated: int = 0
    errors: int = 0
    processing_times_ms: list[float] = field(default_factory=list)
    sentiment_distribution: list[float] = field(default_factory=list)

    def avg_latency_ms(self) -> float:
        return sum(self.processing_times_ms) / max(len(self.processing_times_ms), 1)

    def throughput_per_sec(self, elapsed: float) -> float:
        return self.articles_processed / max(elapsed, 1e-6)

class NewsPipeline:
    def __init__(self, dedup_threshold: float = 0.45, signal_window: float = 3600.0):
        self.state = PipelineState.IDLE
        self.dedup_threshold = dedup_threshold
        self.signal_window = signal_window
        self.metrics = PipelineMetrics()
        self._shingled_cache: deque = deque(maxlen=500)
        self._processed: list[ProcessedArticle] = []
        self._signals: list[NewsSignal] = []
        self._trades: list[TradeSignal] = []
        self._alerts: list[str] = []

    def _transition(self, new_state: PipelineState) -> None:
        self.state = new_state

    def _is_duplicate(self, headline: str) -> bool:
        s = _shingles(headline)
        for prev_s in self._shingled_cache:
            if _jaccard(s, prev_s) >= self.dedup_threshold:
                return True
        self._shingled_cache.append(s)
        return False

    def ingest(self, articles: list[RawArticle]) -> list[RawArticle]:
        self._transition(PipelineState.INGESTING)
        self.metrics.articles_ingested += len(articles)
        return articles

    def preprocess(self, articles: list[RawArticle]) -> list[ProcessedArticle]:
        self._transition(PipelineState.PREPROCESSING)
        processed = []
        for a in articles:
            t0 = time.monotonic()
            full_text = a.headline + ' ' + a.body
            tokens = tokenize(full_text)
            entities = extract_entities(full_text)
            sentiment = score_sentiment(tokens)
            topic = tag_topic(tokens)
            is_dup = self._is_duplicate(a.headline)
            if is_dup:
                self.metrics.duplicates_removed += 1
            self.metrics.processing_times_ms.append((time.monotonic() - t0) * 1000)
            self.metrics.sentiment_distribution.append(sentiment)
            pa = ProcessedArticle(
                a.article_id, a.timestamp, a.source,
                tokens, entities, sentiment, topic, is_dup
            )
            processed.append(pa)
            self._processed.append(pa)
        self.metrics.articles_processed += len(processed)
        return processed

    def score(self, processed: list[ProcessedArticle],
               current_time: float) -> NewsSignal:
        self._transition(PipelineState.SCORING)
        signal = generate_news_signal(self._processed, self.signal_window, current_time)
        self._signals.append(signal)
        self.metrics.signals_generated += 1
        return signal

    def generate_trade(self, signal: NewsSignal) -> TradeSignal:
        self._transition(PipelineState.SIGNAL_GEN)
        trade = signal_to_trade(signal)
        self._trades.append(trade)
        return trade

    def check_alerts(self, signal: NewsSignal) -> list[str]:
        self._transition(PipelineState.ALERTING)
        alerts = []
        if abs(signal.composite_score) > 0.6:
            alerts.append(f"EXTREME_SENTIMENT: score={signal.composite_score:+.3f}")
        if signal.n_articles > 15:
            alerts.append(f"NEWS_SPIKE: {signal.n_articles} articles in window")
        if signal.confidence > 0.7 and abs(signal.composite_score) > 0.3:
            alerts.append(f"HIGH_CONVICTION: score={signal.composite_score:+.3f} conf={signal.confidence:.2f}")
        for entity, score in signal.entity_scores.items():
            if abs(score) > 0.7:
                alerts.append(f"ENTITY_ALERT: {entity} sentiment={score:+.3f}")
        self._alerts.extend(alerts)
        self._transition(PipelineState.IDLE)
        return alerts

    def run_batch(self, articles: list[RawArticle],
                   current_time: float) -> tuple[NewsSignal, TradeSignal, list[str]]:
        """Full pipeline: ingest → preprocess → score → trade → alert."""
        raw = self.ingest(articles)
        processed = self.preprocess(raw)
        signal = self.score(processed, current_time)
        trade = self.generate_trade(signal)
        alerts = self.check_alerts(signal)
        return signal, trade, alerts

    def dashboard(self, elapsed: float) -> str:
        lines = []
        w = 60
        lines.append('┌' + '─' * (w - 2) + '┐')
        lines.append(f'│{"  NEWS PIPELINE MONITORING DASHBOARD":^{w-2}}│')
        lines.append('├' + '─' * (w - 2) + '┤')
        m = self.metrics

        def row(label, value):
            pad = max(0, w - 3 - len(label) - len(str(value)))
            return f'│ {label}{" " * pad}{value} │'

        lines.append(row('State:', self.state.value))
        lines.append(row('Articles ingested:', m.articles_ingested))
        lines.append(row('Articles processed:', m.articles_processed))
        lines.append(row('Duplicates removed:', m.duplicates_removed))
        lines.append(row('Signals generated:', m.signals_generated))
        lines.append(row('Avg latency (ms):', f'{m.avg_latency_ms():.2f}'))
        lines.append(row('Throughput (art/s):', f'{m.throughput_per_sec(elapsed):.1f}'))
        lines.append(row('Errors:', m.errors))
        lines.append(row('Total alerts:', len(self._alerts)))

        if self._signals:
            last_sig = self._signals[-1]
            lines.append('├' + '─' * (w - 2) + '┤')
            lines.append(row('Latest signal score:', f'{last_sig.composite_score:+.4f}'))
            lines.append(row('Latest confidence:', f'{last_sig.confidence:.4f}'))
            lines.append(row('Articles in window:', last_sig.n_articles))

        if self._trades:
            last_trade = self._trades[-1]
            lines.append('├' + '─' * (w - 2) + '┤')
            lines.append(row('Latest trade signal:', last_trade.direction))
            lines.append(row('Signal strength:', f'{last_trade.strength:.4f}'))
            lines.append(row('Trigger:', last_trade.trigger))

        if self._alerts:
            lines.append('├' + '─' * (w - 2) + '┤')
            lines.append(f'│ Recent Alerts:{" " * (w - 17)}│')
            for alert in self._alerts[-3:]:
                a_str = alert[:w-4]
                pad = max(0, w - 3 - len(a_str))
                lines.append(f'│ ! {a_str}{" " * pad}│')

        lines.append('└' + '─' * (w - 2) + '┘')
        return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 30: Production News-to-Signal Pipeline")
    print("=" * 65)

    rng = random.Random(42)

    raw_feed = [
        RawArticle('a001', 1000.0, 'Reuters',
            'Federal Reserve signals more rate hikes ahead to fight inflation', ''),
        RawArticle('a002', 1100.0, 'Bloomberg',
            'Fed chair signals continued rate increases as inflation remains high', ''),  # near-dup
        RawArticle('a003', 1200.0, 'WSJ',
            'Markets rally strongly after positive jobs report beats expectations', ''),
        RawArticle('a004', 1300.0, 'FT',
            'Apple reports record revenue growth driven by strong iPhone sales', ''),
        RawArticle('a005', 1400.0, 'CNBC',
            'Oil prices surge as OPEC announces surprise production cuts', ''),
        RawArticle('a006', 1500.0, 'Reuters',
            'Tech stocks slump on fears of tightening financial conditions', ''),
        RawArticle('a007', 1600.0, 'Bloomberg',
            'Tesla beats delivery estimates electric vehicle demand remains robust', ''),
        RawArticle('a008', 1700.0, 'WSJ',
            'Dollar strengthens after hawkish Federal Reserve commentary', ''),
        RawArticle('a009', 1800.0, 'Reuters',
            'China GDP growth slows amid property crisis and weak consumer demand', ''),
        RawArticle('a010', 1900.0, 'FT',
            'Gold prices rise as recession fears increase safe haven demand', ''),
        RawArticle('a011', 2000.0, 'Bloomberg',
            'Nvidia earnings surge on AI chip demand exceeding analyst expectations', ''),
        RawArticle('a012', 2100.0, 'CNBC',
            'Markets crash sharply on unexpected poor economic data warning signs', ''),
        RawArticle('a013', 2200.0, 'Reuters',
            'Strong corporate earnings beat estimates across tech and energy sectors', ''),
        RawArticle('a014', 2300.0, 'WSJ',
            'Federal Reserve rate hike signals dampen market optimism significantly', ''),
        RawArticle('a015', 2400.0, 'Bloomberg',
            'Inflation drops to target level raising hopes for rate pause', ''),
    ]

    pipeline = NewsPipeline(dedup_threshold=0.40, signal_window=2400.0)
    t_start = time.monotonic()

    print("\n--- Batch 1: First 8 articles (t=0-1700) ---")
    batch1 = raw_feed[:8]
    signal1, trade1, alerts1 = pipeline.run_batch(batch1, current_time=1800.0)

    print(f"  Processed: {pipeline.metrics.articles_processed} articles "
          f"({pipeline.metrics.duplicates_removed} duplicates removed)")
    print(f"  Signal: score={signal1.composite_score:+.4f}  "
          f"confidence={signal1.confidence:.4f}  n={signal1.n_articles}")
    print(f"  Trade: {trade1.direction}  strength={trade1.strength:.4f}  [{trade1.trigger}]")
    if alerts1:
        print(f"  Alerts: {alerts1}")

    print("\n  Per-article breakdown:")
    print(f"  {'ID':>6} | {'Sentiment':>10} | {'Topic':>20} | {'Dup':>4} | {'Entities'}")
    print("  " + "-" * 70)
    for pa in pipeline._processed:
        ents = ', '.join(pa.entities[:3])
        print(f"  {pa.article_id:>6} | {pa.sentiment_score:>+10.4f} | {pa.topic:>20} | "
              f"{'Y' if pa.is_duplicate else 'N':>4} | {ents}")

    print(f"\n--- Batch 2: Remaining 7 articles (t=1800-2400) ---")
    batch2 = raw_feed[8:]
    signal2, trade2, alerts2 = pipeline.run_batch(batch2, current_time=2400.0)

    print(f"  Processed total: {pipeline.metrics.articles_processed}  "
          f"Duplicates: {pipeline.metrics.duplicates_removed}")
    print(f"  Signal: score={signal2.composite_score:+.4f}  "
          f"confidence={signal2.confidence:.4f}  n={signal2.n_articles}")
    print(f"  Trade: {trade2.direction}  strength={trade2.strength:.4f}  [{trade2.trigger}]")
    if alerts2:
        print(f"  Alerts: {alerts2}")

    print("\n  Entity sentiment heatmap:")
    if signal2.entity_scores:
        for entity, score in sorted(signal2.entity_scores.items(), key=lambda x: -abs(x[1]))[:8]:
            bar_len = int(abs(score) * 15)
            bar = ('█' if score > 0 else '░') * bar_len
            print(f"  {entity:20} {bar:15} {score:+.4f}")

    print("\n--- Signal History ---")
    print(f"  {'Batch':>6} | {'Score':>8} | {'Conf':>6} | {'N':>4} | {'Trade'}")
    print("  " + "-" * 45)
    for i, (sig, tr) in enumerate(zip(pipeline._signals, pipeline._trades)):
        print(f"  {i+1:>6} | {sig.composite_score:>+8.4f} | "
              f"{sig.confidence:>6.4f} | {sig.n_articles:>4} | {tr.direction}")

    elapsed = time.monotonic() - t_start

    print("\n--- Pipeline Monitoring Dashboard ---")
    print(pipeline.dashboard(elapsed))

    print("\n--- Backtest Signal Quality ---")
    # Compute signal vs next-period direction correlation
    true_directions = [1, 0, 1, -1, 1, 0, 1, -1, -1, 0,
                       1, -1, 1, -1, 1][:len(pipeline._processed)]
    pred_directions = [1 if pa.sentiment_score > 0.05 else (-1 if pa.sentiment_score < -0.05 else 0)
                       for pa in pipeline._processed]
    n = min(len(true_directions), len(pred_directions))
    correct = sum(1 for t, p in zip(true_directions[:n], pred_directions[:n]) if t == p)
    print(f"  Directional accuracy (in-sample):  {correct}/{n} = {correct/max(n,1):.1%}")
    pos_only = [(t, p) for t, p in zip(true_directions[:n], pred_directions[:n]) if p != 0]
    if pos_only:
        c2 = sum(1 for t, p in pos_only if t == p)
        print(f"  Signal precision (when non-zero): {c2}/{len(pos_only)} = {c2/len(pos_only):.1%}")

    print(f"\n  Avg processing latency: {pipeline.metrics.avg_latency_ms():.3f} ms/article")
    print(f"  Total throughput:       {pipeline.metrics.throughput_per_sec(elapsed):.0f} articles/sec")

    print("\n[Done] Day 30: Production Pipeline complete.")
    print("=" * 65)
    print("30-DAY QUANTITATIVE FINANCE CHALLENGE COMPLETE!")
    print("=" * 65)
    print("  Options Track:    10 days, 10 modules — BS → Heston → SVI → IR")
    print("  Portfolio Track:  10 days, 10 modules — MVO → BL → Kelly → Factor")
    print("  News/NLP Track:   10 days, 10 modules — Sentiment → Events → Prod")
    print(f"  All pure Python stdlib — 30 × 3 = 90 modules")
    print("=" * 65)
