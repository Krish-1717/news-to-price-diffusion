"""
models/news_encoder.py -- News text encoder (TF-IDF + MLP) for news-to-price-diffusion
Day 13: Converts raw news headlines/bodies into conditioning vectors for the score network.
Pure Python, no external dependencies.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","has","have","had",
    "do","does","did","will","would","could","should","may","might","this",
    "that","these","those","it","its","he","she","they","we","you","i","me",
    "him","her","them","us","our","your","their","my","his","not","no","its",
])

def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords, return tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in _STOPWORDS]
    return tokens


def _ngrams(tokens: List[str], n: int) -> List[str]:
    """Generate character n-grams from each token (for OOV robustness)."""
    result = list(tokens)
    for tok in tokens:
        for i in range(len(tok) - n + 1):
            result.append(tok[i:i + n])
    return result


# ---------------------------------------------------------------------------
# TF-IDF vectoriser
# ---------------------------------------------------------------------------

@dataclass
class TFIDFVectoriser:
    """
    Minimal TF-IDF implementation.
    Fit on a corpus of documents, then transform new documents to sparse vectors.
    """
    max_vocab: int = 5000
    ngram_n: int = 3          # character n-gram size (set 0 to disable)
    smooth_idf: bool = True

    _vocab: Dict[str, int] = field(default_factory=dict)
    _idf: List[float] = field(default_factory=list)
    _n_docs: int = 0

    def fit(self, corpus: List[str]) -> "TFIDFVectoriser":
        """Build vocabulary and IDF from a list of documents."""
        self._n_docs = len(corpus)
        doc_freq: Dict[str, int] = {}
        for doc in corpus:
            tokens = _tokenise(doc)
            if self.ngram_n > 0:
                tokens = _ngrams(tokens, self.ngram_n)
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # Sort by document frequency descending, take top max_vocab
        sorted_terms = sorted(doc_freq.items(), key=lambda x: -x[1])[:self.max_vocab]
        self._vocab = {term: i for i, (term, _) in enumerate(sorted_terms)}

        # IDF = log((1+N)/(1+df)) + 1  (smoothed)
        N = self._n_docs
        self._idf = []
        for term, _ in sorted_terms:
            df = doc_freq[term]
            if self.smooth_idf:
                idf = math.log((1 + N) / (1 + df)) + 1.0
            else:
                idf = math.log(N / df) + 1.0
            self._idf.append(idf)

        return self

    def transform(self, doc: str) -> Dict[int, float]:
        """
        Convert document to TF-IDF sparse vector {term_idx: tfidf_weight}.
        Returns normalised (L2) vector.
        """
        tokens = _tokenise(doc)
        if self.ngram_n > 0:
            tokens = _ngrams(tokens, self.ngram_n)

        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n_tokens = max(len(tokens), 1)

        vec: Dict[int, float] = {}
        for term, count in tf.items():
            if term in self._vocab:
                idx = self._vocab[term]
                tf_weight = count / n_tokens
                vec[idx] = tf_weight * self._idf[idx]

        # L2 normalise
        norm = math.sqrt(sum(v * v for v in vec.values())) + 1e-12
        return {k: v / norm for k, v in vec.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


# ---------------------------------------------------------------------------
# News encoder (TF-IDF -> dense projection)
# ---------------------------------------------------------------------------

def _relu(x: float) -> float:
    return max(0.0, x)


def _dense(x: List[float], W: List[List[float]], b: List[float]) -> List[float]:
    return [sum(W[i][j] * x[j] for j in range(len(x))) + b[i] for i in range(len(W))]


@dataclass
class NewsEncoderConfig:
    """Configuration for the news encoder."""
    vocab_size: int = 5000
    hidden_dim: int = 128
    output_dim: int = 64       # matches ScoreNetworkConfig.cond_dim
    dropout_p: float = 0.1     # not applied in pure-Python inference


class NewsEncoder:
    """
    Two-layer MLP that projects sparse TF-IDF vectors to dense conditioning vectors.

    Architecture:
      sparse_tfidf (vocab_size) -> [dense, ReLU] -> (hidden_dim,) -> [dense] -> (output_dim,)
    """

    def __init__(self, config: NewsEncoderConfig, rng=None):
        import random
        self.config = config
        rng = rng or random.Random(1)
        lim1 = math.sqrt(6 / (config.vocab_size + config.hidden_dim))
        lim2 = math.sqrt(6 / (config.hidden_dim + config.output_dim))
        self.W1 = [[rng.uniform(-lim1, lim1) for _ in range(config.vocab_size)]
                   for _ in range(config.hidden_dim)]
        self.b1 = [0.0] * config.hidden_dim
        self.W2 = [[rng.uniform(-lim2, lim2) for _ in range(config.hidden_dim)]
                   for _ in range(config.output_dim)]
        self.b2 = [0.0] * config.output_dim

    def _sparse_to_dense(self, sparse: Dict[int, float]) -> List[float]:
        """Materialise sparse TF-IDF dict into a dense vector."""
        v = [0.0] * self.config.vocab_size
        for idx, val in sparse.items():
            if 0 <= idx < self.config.vocab_size:
                v[idx] = val
        return v

    def encode(self, tfidf_sparse: Dict[int, float]) -> List[float]:
        """
        Encode a sparse TF-IDF vector to a dense conditioning vector.

        Parameters
        ----------
        tfidf_sparse : output of TFIDFVectoriser.transform()

        Returns
        -------
        cond : dense vector of shape (output_dim,)
        """
        dense_in = self._sparse_to_dense(tfidf_sparse)
        h = _dense(dense_in, self.W1, self.b1)
        h = [_relu(v) for v in h]
        out = _dense(h, self.W2, self.b2)
        # L2-normalise output
        norm = math.sqrt(sum(v * v for v in out)) + 1e-12
        return [v / norm for v in out]

    def encode_text(self, text: str, vectoriser: TFIDFVectoriser) -> List[float]:
        """Convenience: encode raw text string directly."""
        sparse = vectoriser.transform(text)
        return self.encode(sparse)


# ---------------------------------------------------------------------------
# Sentiment features (rule-based, no dependencies)
# ---------------------------------------------------------------------------

_POS_WORDS = frozenset([
    "beat", "beats", "surpass", "surge", "soar", "record", "growth", "profit",
    "gain", "rally", "rise", "strong", "upgrade", "bullish", "positive", "exceed",
    "outperform", "revenue", "earnings", "dividend", "buyback", "approved", "win",
])
_NEG_WORDS = frozenset([
    "miss", "misses", "fall", "drop", "decline", "loss", "cut", "lower", "weak",
    "downgrade", "bearish", "negative", "fail", "disappoint", "recall", "lawsuit",
    "fine", "penalty", "fraud", "investigation", "layoff", "bankruptcy", "default",
])

@dataclass
class SentimentFeatures:
    """Rule-based sentiment features for a news document."""
    pos_count: int
    neg_count: int
    total_words: int
    ticker_mentioned: bool

    @property
    def sentiment_score(self) -> float:
        """Net sentiment in [-1, +1]."""
        if self.total_words == 0:
            return 0.0
        return (self.pos_count - self.neg_count) / math.sqrt(self.total_words + 1)

    def to_vector(self) -> List[float]:
        """4-dim feature vector."""
        return [
            self.sentiment_score,
            self.pos_count / (self.total_words + 1),
            self.neg_count / (self.total_words + 1),
            float(self.ticker_mentioned),
        ]


def extract_sentiment(text: str, ticker: Optional[str] = None) -> SentimentFeatures:
    """Extract sentiment features from raw text."""
    tokens = _tokenise(text)
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    mentioned = ticker is not None and ticker.lower() in text.lower()
    return SentimentFeatures(pos, neg, len(tokens), mentioned)


# ---------------------------------------------------------------------------
# Combined news feature (TF-IDF encoding + sentiment)
# ---------------------------------------------------------------------------

def build_news_conditioning(
    text: str,
    vectoriser: TFIDFVectoriser,
    encoder: NewsEncoder,
    ticker: Optional[str] = None,
) -> List[float]:
    """
    Build full news conditioning vector by combining:
      - Dense TF-IDF encoding (output_dim,)
      - Sentiment features (4,)
    Returns vector of length output_dim + 4.
    """
    dense = encoder.encode_text(text, vectoriser)
    sentiment = extract_sentiment(text, ticker).to_vector()
    return dense + sentiment


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus = [
        "Apple beats earnings expectations revenue surges record high",
        "Microsoft misses quarterly earnings guidance cut stock drops",
        "Fed raises interest rates inflation concerns grow",
        "Tesla reports record deliveries beats analyst estimates",
        "Amazon layoffs continue cost cutting measures weakness",
        "Google search revenue growth strong advertising market",
        "Oil prices surge geopolitical tensions supply concerns",
        "Bank earnings disappoint loan losses rise recession fears",
    ]

    vectoriser = TFIDFVectoriser(max_vocab=500, ngram_n=3)
    vectoriser.fit(corpus)

    cfg = NewsEncoderConfig(vocab_size=vectoriser.vocab_size, hidden_dim=64, output_dim=64)
    encoder = NewsEncoder(cfg)

    print("=" * 55)
    print("  NEWS ENCODER -- TF-IDF + sentiment demo")
    print("=" * 55)
    print(f"  Vocab size: {vectoriser.vocab_size}")

    test_headlines = [
        ("AAPL", "Apple beats Q3 earnings expectations revenue surges 12%"),
        ("MSFT", "Microsoft misses quarterly revenue forecast cuts guidance"),
        ("TSLA", "Tesla record deliveries beat Wall Street estimates strong demand"),
    ]

    for ticker, headline in test_headlines:
        sent = extract_sentiment(headline, ticker)
        cond = build_news_conditioning(headline, vectoriser, encoder, ticker)
        print(f"\n  [{ticker}] {headline[:50]}")
        print(f"    Sentiment: {sent.sentiment_score:+.3f}  (pos={sent.pos_count}, neg={sent.neg_count})")
        print(f"    Cond vector: dim={len(cond)}  norm={math.sqrt(sum(v**2 for v in cond)):.4f}")
    print("=" * 55)
