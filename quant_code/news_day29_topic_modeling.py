"""
news_day29_topic_modeling.py
Day 29: Topic Modeling & News Intelligence — TF-IDF, soft topic clustering,
news deduplication (shingling + Jaccard), entity co-occurrence graph,
topic trend tracking over time.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import re
import random
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Text preprocessing
# ---------------------------------------------------------------------------
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'this', 'that', 'it', 'its',
    'and', 'or', 'but', 'not', 'no', 'so', 'if', 'then', 'than', 'also',
    'into', 'over', 'after', 'before', 'between', 'through', 'during',
    'said', 'says', 'according', 'report', 'new', 'year', 'share'
}

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, keep words ≥3 chars."""
    words = re.sub(r'[^a-zA-Z\s]', ' ', text.lower()).split()
    return [w for w in words if len(w) >= 3 and w not in STOPWORDS]

def simple_stem(word: str) -> str:
    """Very lightweight suffix-stripping stemmer."""
    suffixes = ['ing', 'tion', 'tions', 'ment', 'ments', 'ed', 'er', 'ers',
                'est', 'ly', 'ness', 'ful', 'less', 'able', 'ible', 'al',
                'ive', 'ous', 'ary', 'ory']
    for suf in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)]
    return word

def preprocess(text: str, stem: bool = True) -> list[str]:
    tokens = tokenize(text)
    if stem:
        tokens = [simple_stem(t) for t in tokens]
    return tokens

# ---------------------------------------------------------------------------
# 2. TF-IDF
# ---------------------------------------------------------------------------
class TFIDF:
    def __init__(self, stem: bool = True):
        self.stem = stem
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.n_docs = 0
        self.doc_tokens: list[list[str]] = []

    def fit(self, documents: list[str]) -> 'TFIDF':
        self.n_docs = len(documents)
        self.doc_tokens = [preprocess(d, self.stem) for d in documents]
        self.df = defaultdict(int)
        for toks in self.doc_tokens:
            for w in set(toks):
                self.df[w] += 1
        self.idf = {w: math.log((self.n_docs + 1) / (df + 1)) + 1
                    for w, df in self.df.items()}
        return self

    def transform(self, doc_idx: int) -> dict[str, float]:
        toks = self.doc_tokens[doc_idx]
        if not toks:
            return {}
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        max_tf = max(tf.values())
        return {w: (cnt / max_tf) * self.idf.get(w, 1.0) for w, cnt in tf.items()}

    def top_terms(self, doc_idx: int, k: int = 10) -> list[tuple[str, float]]:
        scores = self.transform(doc_idx)
        return sorted(scores.items(), key=lambda x: -x[1])[:k]

    def global_top_terms(self, k: int = 20) -> list[tuple[str, float]]:
        """Most frequent terms weighted by IDF (corpus-level importance)."""
        term_score: dict[str, float] = defaultdict(float)
        for idx in range(self.n_docs):
            for w, s in self.transform(idx).items():
                term_score[w] += s
        return sorted(term_score.items(), key=lambda x: -x[1])[:k]

    def cosine_similarity(self, idx_a: int, idx_b: int) -> float:
        va, vb = self.transform(idx_a), self.transform(idx_b)
        num = sum(va.get(w, 0) * vb.get(w, 0) for w in va)
        da = math.sqrt(sum(v**2 for v in va.values()))
        db = math.sqrt(sum(v**2 for v in vb.values()))
        return num / max(da * db, 1e-10)

# ---------------------------------------------------------------------------
# 3. Soft topic clustering (centroid-based NMF-lite)
# ---------------------------------------------------------------------------
class TopicCluster:
    """
    Soft k-means clustering in TF-IDF space.
    Each topic is a centroid dict; documents get soft assignments.
    """
    def __init__(self, n_topics: int = 5, n_iter: int = 30, seed: int = 42):
        self.K = n_topics
        self.n_iter = n_iter
        self.seed = seed
        self.centroids: list[dict[str, float]] = []
        self.topic_labels: list[str] = []

    def _vec_add(self, a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0.0) + v
        return out

    def _vec_scale(self, a: dict, s: float) -> dict:
        return {k: v * s for k, v in a.items()}

    def _cosine(self, a: dict, b: dict) -> float:
        num = sum(a.get(w, 0) * b.get(w, 0) for w in a)
        da = math.sqrt(sum(v**2 for v in a.values()))
        db = math.sqrt(sum(v**2 for v in b.values()))
        return num / max(da * db, 1e-10)

    def fit(self, tfidf: TFIDF) -> 'TopicCluster':
        n = tfidf.n_docs
        vecs = [tfidf.transform(i) for i in range(n)]

        # Initialize centroids from random documents
        rng = random.Random(self.seed)
        idxs = rng.sample(range(n), min(self.K, n))
        self.centroids = [dict(vecs[i]) for i in idxs]

        for _ in range(self.n_iter):
            # Assignment
            assignments = []
            for v in vecs:
                sims = [self._cosine(v, c) for c in self.centroids]
                assignments.append(sims.index(max(sims)))

            # Update centroids
            new_centroids = [{} for _ in range(self.K)]
            counts = [0] * self.K
            for i, a in enumerate(assignments):
                new_centroids[a] = self._vec_add(new_centroids[a], vecs[i])
                counts[a] += 1
            for k in range(self.K):
                if counts[k] > 0:
                    self.centroids[k] = self._vec_scale(new_centroids[k], 1 / counts[k])

        # Label topics by top centroid terms
        self.topic_labels = []
        for k in range(self.K):
            top = sorted(self.centroids[k].items(), key=lambda x: -x[1])[:4]
            self.topic_labels.append('+'.join(w for w, _ in top))

        return self

    def assign(self, vec: dict[str, float]) -> list[float]:
        """Soft assignment: normalized cosine similarities to each centroid."""
        sims = [self._cosine(vec, c) for c in self.centroids]
        total = sum(sims) + 1e-10
        return [s / total for s in sims]

    def top_topic_terms(self, k: int, n: int = 8) -> list[tuple[str, float]]:
        return sorted(self.centroids[k].items(), key=lambda x: -x[1])[:n]

# ---------------------------------------------------------------------------
# 4. News deduplication (shingling + Jaccard / MinHash sketch)
# ---------------------------------------------------------------------------
def shingles(text: str, k: int = 3) -> set[str]:
    """Character k-shingles for near-duplicate detection."""
    t = text.lower()
    return {t[i:i+k] for i in range(len(t) - k + 1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)

def minhash_signature(shingle_set: set[str], n_hashes: int = 64,
                       seed: int = 0) -> list[int]:
    """MinHash sketch for fast Jaccard estimation."""
    rng = random.Random(seed)
    params = [(rng.randint(1, (1 << 31) - 1),
               rng.randint(0, (1 << 31) - 1)) for _ in range(n_hashes)]
    MOD = (1 << 31) - 1
    sig = []
    for a, b in params:
        mn = min((a * hash(s) + b) % MOD for s in shingle_set) if shingle_set else MOD
        sig.append(mn)
    return sig

def minhash_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)

def deduplicate(headlines: list[str], threshold: float = 0.5,
                use_minhash: bool = True) -> list[int]:
    """Return indices of unique headlines (remove near-duplicates)."""
    n = len(headlines)
    if use_minhash:
        sigs = [minhash_signature(shingles(h)) for h in headlines]
        def sim(i, j): return minhash_jaccard(sigs[i], sigs[j])
    else:
        shin = [shingles(h) for h in headlines]
        def sim(i, j): return jaccard(shin[i], shin[j])

    kept = []
    removed = set()
    for i in range(n):
        if i in removed:
            continue
        kept.append(i)
        for j in range(i + 1, n):
            if j not in removed and sim(i, j) >= threshold:
                removed.add(j)
    return kept

# ---------------------------------------------------------------------------
# 5. Entity co-occurrence graph
# ---------------------------------------------------------------------------
@dataclass
class CooccurrenceGraph:
    """Simple undirected co-occurrence graph for entity pairs."""
    edges: dict[tuple[str, str], int] = field(default_factory=dict)
    node_count: dict[str, int] = field(default_factory=dict)

    def add_document(self, entities: list[str]) -> None:
        for e in entities:
            self.node_count[e] = self.node_count.get(e, 0) + 1
        unique = sorted(set(entities))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                key = (unique[i], unique[j])
                self.edges[key] = self.edges.get(key, 0) + 1

    def top_pairs(self, k: int = 10) -> list[tuple[tuple[str, str], int]]:
        return sorted(self.edges.items(), key=lambda x: -x[1])[:k]

    def entity_centrality(self) -> list[tuple[str, float]]:
        """Degree centrality: sum of co-occurrence counts per entity."""
        centrality: dict[str, float] = defaultdict(float)
        for (a, b), w in self.edges.items():
            centrality[a] += w
            centrality[b] += w
        return sorted(centrality.items(), key=lambda x: -x[1])

# ---------------------------------------------------------------------------
# 6. Topic trend over time
# ---------------------------------------------------------------------------
def topic_trend(tfidf: TFIDF, clusters: TopicCluster,
                timestamps: list[int], n_bins: int = 4) -> list[list[float]]:
    """
    Track topic share over time buckets.
    Returns list of [topic_0_share, ..., topic_K_share] per time bin.
    """
    if not timestamps:
        return []
    mn, mx = min(timestamps), max(timestamps)
    span = max(mx - mn, 1)
    bins: list[list[list[float]]] = [[] for _ in range(n_bins)]
    for i in range(tfidf.n_docs):
        b = min(int((timestamps[i] - mn) / span * n_bins), n_bins - 1)
        vec = tfidf.transform(i)
        bins[b].append(clusters.assign(vec))
    results = []
    for b in bins:
        if not b:
            results.append([0.0] * clusters.K)
        else:
            avg = [sum(row[k] for row in b) / len(b) for k in range(clusters.K)]
            results.append(avg)
    return results

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 29: Topic Modeling & News Intelligence")
    print("=" * 65)

    articles = [
        "Federal Reserve raises interest rates to combat inflation. Markets react sharply to monetary policy tightening.",
        "Fed raises rates again as inflation remains elevated above target. Bond yields surge.",
        "Interest rates hiked by Federal Reserve to fight persistent inflation. Treasury yields climb.",  # near-dup of #1
        "Apple reports record quarterly earnings driven by iPhone sales and services revenue growth.",
        "Technology stocks surge after strong earnings from Apple Microsoft and Google parent Alphabet.",
        "Apple iPhone sales beat expectations in latest quarterly results boosting tech sector.",
        "Oil prices climb as OPEC production cuts reduce global crude supply. Energy sector rallies.",
        "OPEC agrees to extend production cuts, supporting crude oil prices. Energy stocks rise.",
        "Tesla delivers record number of electric vehicles in third quarter, beating analyst estimates.",
        "Electric vehicle sales accelerate globally as Tesla and rivals expand production capacity.",
        "China GDP growth slows as property sector crisis deepens, raising recession fears.",
        "Bank of England raises interest rates to 15-year high as UK inflation remains stubbornly high.",
        "Unemployment rate falls to 50-year low as labor market remains resilient despite rate hikes.",
        "Semiconductor shortage eases as chip makers boost production capacity to meet demand.",
        "Artificial intelligence investments surge as companies race to deploy large language models.",
        "AI chip demand drives Nvidia revenue to record quarterly highs, stock jumps.",
        "Job market adds stronger than expected payrolls, wages rise, Fed likely to hold rates.",
        "Gold prices rise as investors seek safe haven assets amid global economic uncertainty.",
        "Dollar strengthens against major currencies as Fed maintains hawkish monetary policy stance.",
        "Corporate earnings season beats expectations with technology and energy sectors leading gains.",
    ]
    timestamps = list(range(0, len(articles) * 3, 3))  # 3-day spacing

    print("\n1. TF-IDF Fitting")
    tfidf = TFIDF(stem=True)
    tfidf.fit(articles)
    print(f"   Corpus: {tfidf.n_docs} docs, {len(tfidf.df)} unique terms")
    top_corpus = tfidf.global_top_terms(k=10)
    print(f"   Top corpus terms: {', '.join(f'{w}({s:.2f})' for w, s in top_corpus[:8])}")

    print("\n2. Per-Document Top Terms (first 3 docs)")
    for i in range(3):
        top = tfidf.top_terms(i, k=5)
        print(f"   Doc {i}: {', '.join(f'{w}({s:.2f})' for w, s in top)}")

    print("\n3. Near-Duplicate Detection (MinHash)")
    unique_ids = deduplicate(articles, threshold=0.35)
    print(f"   Original: {len(articles)} articles → Unique: {len(unique_ids)} after dedup")
    removed = [i for i in range(len(articles)) if i not in unique_ids]
    print(f"   Removed as near-duplicates: {removed}")
    for i in removed:
        print(f"   [{i}] {articles[i][:60]}...")

    print("\n4. Topic Clustering (K=5 topics)")
    clusters = TopicCluster(n_topics=5, n_iter=40, seed=42)
    clusters.fit(tfidf)
    print(f"\n   {'Topic':>6} | Label")
    print("   " + "-" * 50)
    for k in range(clusters.K):
        top_terms = clusters.top_topic_terms(k, n=6)
        label = ', '.join(f"{w}({s:.2f})" for w, s in top_terms[:4])
        print(f"   {k:>6} | {label}")

    print("\n   Document → Topic assignments (top 2):")
    for i in range(min(8, tfidf.n_docs)):
        vec = tfidf.transform(i)
        probs = clusters.assign(vec)
        top2 = sorted(enumerate(probs), key=lambda x: -x[1])[:2]
        t1, p1 = top2[0]
        t2, p2 = top2[1]
        snippet = articles[i][:45]
        print(f"   {i:2}. [{snippet}...]  T{t1}:{p1:.2f}  T{t2}:{p2:.2f}")

    print("\n5. Cosine Similarity (potential duplicates)")
    print(f"   {'Doc i':>5} | {'Doc j':>5} | {'Cosine Sim':>12}")
    print("   " + "-" * 28)
    pairs = [(0,1),(0,2),(1,2),(3,4),(3,5),(6,7),(8,9),(0,3)]
    for i, j in pairs:
        sim = tfidf.cosine_similarity(i, j)
        print(f"   {i:>5} | {j:>5} | {sim:>12.4f}")

    print("\n6. Entity Co-occurrence Graph")
    entity_lists = [
        ['FederalReserve', 'Markets', 'Inflation'],
        ['FederalReserve', 'Inflation', 'BondYields'],
        ['FederalReserve', 'Inflation', 'TreasuryYields'],
        ['Apple', 'iPhone', 'TechSector'],
        ['Apple', 'Microsoft', 'Google', 'TechSector'],
        ['Apple', 'iPhone', 'TechSector'],
        ['OPEC', 'CrudeOil', 'EnergySector'],
        ['OPEC', 'CrudeOil', 'EnergySector'],
        ['Tesla', 'ElectricVehicles'],
        ['Tesla', 'ElectricVehicles', 'AutoSector'],
        ['China', 'PropertySector', 'GDP'],
        ['BankOfEngland', 'Inflation', 'InterestRates'],
        ['FederalReserve', 'LaborMarket', 'Unemployment'],
        ['Semiconductors', 'ChipMakers'],
        ['AI', 'LargeLanguageModels', 'TechSector'],
        ['Nvidia', 'AI', 'TechSector'],
        ['FederalReserve', 'LaborMarket', 'Wages'],
        ['Gold', 'SafeHaven'],
        ['Dollar', 'FederalReserve', 'CurrencyMarkets'],
        ['TechSector', 'EnergySector', 'EarningsSeason'],
    ]
    graph = CooccurrenceGraph()
    for el in entity_lists:
        graph.add_document(el)

    print(f"   Top co-occurring pairs:")
    for (a, b), cnt in graph.top_pairs(k=8):
        print(f"   {a} ↔ {b}: {cnt}")
    print(f"\n   Entity centrality:")
    for entity, score in graph.entity_centrality()[:8]:
        print(f"   {entity:20}: {score:.0f}")

    print("\n7. Topic Trend Over Time (4 time buckets)")
    trends = topic_trend(tfidf, clusters, timestamps, n_bins=4)
    print(f"   {'Bin':>4} | " + " | ".join(f"T{k:>4}" for k in range(clusters.K)))
    print("   " + "-" * (8 + clusters.K * 9))
    for b, shares in enumerate(trends):
        row = " | ".join(f"{s:>5.3f}" for s in shares)
        print(f"   {b:>4} | {row}")
    dominant = [f"T{max(range(clusters.K), key=lambda k: t[k])}" for t in trends]
    print(f"\n   Dominant topic per bin: {dominant}")

    print("\n[Done] Day 29: Topic Modeling complete.")
