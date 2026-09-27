"""
Financial News Topic Modeling â LDA
Day 16 â news-to-price-diffusion/nlp/topic_modeling.py

Pure-Python Latent Dirichlet Allocation (LDA) via collapsed Gibbs sampling,
with financial domain topics and topic-to-price signal mapping.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "this", "that",
    "it", "its", "as", "up", "after", "before", "than", "more", "also",
    "into", "about", "over", "under", "between", "through", "during", "per",
    "which", "who", "what", "when", "where", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "so", "then", "too", "very", "s", "t", "just", "their", "they",
    "them", "we", "our", "us", "i", "my", "he", "she", "his", "her",
}


def tokenize(text: str, min_len: int = 3) -> List[str]:
    """Lowercase, split on non-alpha, remove stop words and short tokens."""
    tokens = []
    for tok in text.lower().split():
        # Remove punctuation
        clean = "".join(c for c in tok if c.isalpha())
        if len(clean) >= min_len and clean not in STOP_WORDS:
            tokens.append(clean)
    return tokens


def build_vocabulary(docs: List[List[str]],
                      min_df: int = 2,
                      max_df_frac: float = 0.95) -> Dict[str, int]:
    """Build vocabulary with document-frequency filtering."""
    df: Dict[str, int] = {}
    n_docs = len(docs)
    for doc in docs:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1

    max_df = int(max_df_frac * n_docs)
    vocab = {w: i for i, w in enumerate(
        sorted(w for w, cnt in df.items() if min_df <= cnt <= max_df)
    )}
    return vocab


# ---------------------------------------------------------------------------
# LDA Gibbs sampler
# ---------------------------------------------------------------------------

@dataclass
class LDAModel:
    n_topics: int
    alpha: float           # document-topic Dirichlet prior
    beta: float            # topic-word Dirichlet prior
    vocab: Dict[str, int]
    n_words: int = 0

    # Count matrices (updated during training)
    n_dk: List[List[int]] = field(default_factory=list)   # doc Ã topic
    n_kw: List[List[int]] = field(default_factory=list)   # topic Ã word
    n_k:  List[int]        = field(default_factory=list)  # topic total counts

    topic_word_: List[List[float]] = field(default_factory=list)   # Ï
    doc_topic_:  List[List[float]] = field(default_factory=list)   # Î¸

    def __post_init__(self):
        self.n_words = len(self.vocab)


def fit_lda(docs_text: List[str],
            n_topics: int = 10,
            alpha: float = 0.1,
            beta: float = 0.01,
            n_iter: int = 100,
            seed: int = 42) -> LDAModel:
    """
    Fit LDA via collapsed Gibbs sampling.
    """
    rng = random.Random(seed)

    # Tokenize
    docs_tok = [tokenize(d) for d in docs_text]
    vocab = build_vocabulary(docs_tok)
    V = len(vocab)
    D = len(docs_tok)

    # Convert to word indices
    docs_idx: List[List[int]] = []
    for doc in docs_tok:
        docs_idx.append([vocab[w] for w in doc if w in vocab])

    # Initialise counts
    n_dk = [[0] * n_topics for _ in range(D)]
    n_kw = [[0] * V for _ in range(n_topics)]
    n_k  = [0] * n_topics

    # Random initial topic assignments
    assignments: List[List[int]] = []
    for d, doc in enumerate(docs_idx):
        z_d = []
        for w in doc:
            k = rng.randint(0, n_topics - 1)
            z_d.append(k)
            n_dk[d][k] += 1
            n_kw[k][w] += 1
            n_k[k] += 1
        assignments.append(z_d)

    # Gibbs iterations
    for it in range(n_iter):
        for d, doc in enumerate(docs_idx):
            n_d = len(doc)
            for i, w in enumerate(doc):
                k_old = assignments[d][i]

                # Remove current assignment
                n_dk[d][k_old] -= 1
                n_kw[k_old][w] -= 1
                n_k[k_old] -= 1

                # Compute conditional distribution
                probs = [
                    (n_dk[d][k] + alpha) *
                    (n_kw[k][w] + beta) /
                    (n_k[k] + V * beta)
                    for k in range(n_topics)
                ]

                # Sample new topic
                total = sum(probs)
                r = rng.random() * total
                k_new = 0
                cumsum = 0.0
                for k, p in enumerate(probs):
                    cumsum += p
                    if cumsum >= r:
                        k_new = k
                        break

                assignments[d][i] = k_new
                n_dk[d][k_new] += 1
                n_kw[k_new][w] += 1
                n_k[k_new] += 1

    # Compute normalised distributions
    phi = [
        [(n_kw[k][w] + beta) / (n_k[k] + V * beta) for w in range(V)]
        for k in range(n_topics)
    ]
    theta = [
        [(n_dk[d][k] + alpha) / (len(docs_idx[d]) + n_topics * alpha)
         for k in range(n_topics)]
        for d in range(D)
    ]

    model = LDAModel(n_topics=n_topics, alpha=alpha, beta=beta,
                     vocab=vocab, n_words=V)
    model.n_dk = n_dk
    model.n_kw = n_kw
    model.n_k = n_k
    model.topic_word_ = phi
    model.doc_topic_ = theta
    return model


def top_words(model: LDAModel, topic: int, n: int = 10) -> List[Tuple[str, float]]:
    """Top n words for a given topic."""
    id2word = {v: k for k, v in model.vocab.items()}
    phi_k = model.topic_word_[topic]
    ranked = sorted(range(model.n_words), key=lambda w: -phi_k[w])
    return [(id2word[w], phi_k[w]) for w in ranked[:n]]


def infer_topic(model: LDAModel, text: str,
                n_iter: int = 20, seed: int = 0) -> List[float]:
    """
    Infer topic distribution for a new document (online Gibbs).
    Returns Î¸ vector of length n_topics.
    """
    rng = random.Random(seed)
    tokens = tokenize(text)
    doc = [model.vocab[w] for w in tokens if w in model.vocab]
    if not doc:
        return [1.0 / model.n_topics] * model.n_topics

    V = model.n_words
    n_topics = model.n_topics
    n_dk_doc = [0] * n_topics
    z_doc = [rng.randint(0, n_topics - 1) for _ in doc]
    for k in z_doc:
        n_dk_doc[k] += 1

    for _ in range(n_iter):
        for i, w in enumerate(doc):
            k_old = z_doc[i]
            n_dk_doc[k_old] -= 1
            probs = [
                (n_dk_doc[k] + model.alpha) *
                (model.n_kw[k][w] + model.beta) /
                (model.n_k[k] + V * model.beta)
                for k in range(n_topics)
            ]
            total = sum(probs)
            r = rng.random() * total
            k_new = 0
            cs = 0.0
            for k, p in enumerate(probs):
                cs += p
                if cs >= r:
                    k_new = k
                    break
            z_doc[i] = k_new
            n_dk_doc[k_new] += 1

    n_doc = len(doc)
    return [(n_dk_doc[k] + model.alpha) / (n_doc + n_topics * model.alpha)
            for k in range(n_topics)]


# ---------------------------------------------------------------------------
# Topic-to-signal mapping
# ---------------------------------------------------------------------------

def topic_sentiment_signal(theta: List[float],
                            topic_sentiments: List[float]) -> float:
    """
    Map document topic distribution to a sentiment signal.
    topic_sentiments[k] = bullish (+) or bearish (-) association of topic k.
    Returns weighted sum: Î£ Î¸_k * sentiment_k.
    """
    return sum(t * s for t, s in zip(theta, topic_sentiments))


def corpus_topic_signals(model: LDAModel,
                          topic_sentiments: List[float]) -> List[float]:
    """Compute sentiment signal for every document in the training corpus."""
    return [topic_sentiment_signal(theta, topic_sentiments)
            for theta in model.doc_topic_]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    headlines = [
        "Federal Reserve raises interest rates by 25 basis points amid inflation concerns",
        "Stock market rallies as inflation data comes in lower than expected",
        "Tech stocks surge on strong earnings from major semiconductor companies",
        "Oil prices fall sharply as OPEC increases production quota",
        "Federal Reserve signals pause in rate hikes, dollar weakens",
        "Nvidia reports record revenue driven by AI chip demand",
        "Recession fears grow as GDP contracts for second consecutive quarter",
        "Gold reaches all-time high as investors seek safe haven assets",
        "Apple unveils new product lineup, shares jump in after-hours trading",
        "Inflation remains sticky, bond yields hit multi-year highs",
        "Banks report strong loan growth but rising credit losses",
        "Commodity markets rally on China demand recovery expectations",
        "Tech sector leads gains as artificial intelligence investments surge",
        "Fed Chair hints at future rate cuts if inflation cools further",
        "Energy stocks rise on supply disruption fears in Middle East",
        "Market volatility spikes as geopolitical tensions escalate",
        "Amazon and Microsoft cloud revenue beats analyst estimates",
        "Treasury yields invert, signaling potential economic slowdown",
        "Consumer spending data shows resilience despite high interest rates",
        "Biotech stocks fall after clinical trial failure announcement",
    ] * 3  # repeat for more data

    print(f"Fitting LDA on {len(headlines)} documents...")
    model = fit_lda(headlines, n_topics=5, n_iter=50, seed=42)

    print("\nTop words per topic:")
    for k in range(model.n_topics):
        words = top_words(model, k, n=6)
        word_str = ", ".join(f"{w}({p:.3f})" for w, p in words)
        print(f"  Topic {k}: {word_str}")

    # Assign manual sentiments to topics
    topic_sentiments = [0.5, -0.3, 0.8, -0.6, 0.2]

    # Infer new document
    new_doc = "Fed rate cut boosts stock market rally, tech leads gains"
    theta_new = infer_topic(model, new_doc)
    signal = topic_sentiment_signal(theta_new, topic_sentiments)

    print(f"\nNew document: '{new_doc}'")
    print(f"Topic distribution: {[round(t, 3) for t in theta_new]}")
    print(f"Sentiment signal:   {signal:.4f}")
