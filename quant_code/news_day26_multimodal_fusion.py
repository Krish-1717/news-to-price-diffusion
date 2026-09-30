"""
news_day26_multimodal_fusion.py
Day 26: Multi-Modal Fusion — Combine news sentiment + price momentum +
vol regime into a unified return prediction. Attention-like weighting,
cross-modal alignment, late fusion ensemble, signal quality gating.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def std(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1))

def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1 + math.exp(-x))
    return math.exp(x) / (1 + math.exp(x))

def relu(x: float) -> float:
    return max(0.0, x)

def dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))

# ---------------------------------------------------------------------------
# 1. Signal modalities
# ---------------------------------------------------------------------------
@dataclass
class PriceMomentumSignal:
    """Short/medium/long momentum + mean-reversion + vol-adjusted."""
    returns: list[float]   # historical daily returns
    current_t: int

    def short_momentum(self, window: int = 5) -> float:
        start = max(0, self.current_t - window)
        return sum(self.returns[start:self.current_t])

    def medium_momentum(self, window: int = 21) -> float:
        start = max(0, self.current_t - window)
        return sum(self.returns[start:self.current_t])

    def long_momentum(self, window: int = 63, skip: int = 5) -> float:
        start = max(0, self.current_t - window)
        end = max(0, self.current_t - skip)
        return sum(self.returns[start:end])

    def mean_reversion(self, window: int = 5) -> float:
        return -self.short_momentum(window)  # contrarian

    def realized_vol(self, window: int = 21) -> float:
        start = max(0, self.current_t - window)
        r_w = self.returns[start:self.current_t]
        return std(r_w) * math.sqrt(252) if len(r_w) > 1 else 0.15

    def feature_vector(self) -> list[float]:
        rv = max(self.realized_vol(), 1e-4)
        return [
            self.short_momentum() / rv,
            self.medium_momentum() / rv,
            self.long_momentum() / rv,
            self.mean_reversion() / rv,
            rv,  # vol regime proxy
        ]

@dataclass
class VolRegimeSignal:
    """Volatility regime features: GARCH-like rolling stats."""
    returns: list[float]
    current_t: int
    lambda_ewma: float = 0.94

    def ewma_vol(self) -> float:
        if self.current_t < 2:
            return 0.015
        var = self.returns[0]**2
        for r in self.returns[1:self.current_t]:
            var = self.lambda_ewma * var + (1 - self.lambda_ewma) * r**2
        return math.sqrt(max(var, 1e-8) * 252)

    def vol_trend(self, window: int = 10) -> float:
        """Recent change in vol (vol acceleration)."""
        if self.current_t < window * 2:
            return 0.0
        vol_now = self.ewma_vol()
        # Approximate past vol
        old_t = max(0, self.current_t - window)
        VolRegimeSignal(self.returns, old_t, self.lambda_ewma).ewma_vol()
        vol_then = VolRegimeSignal(self.returns, old_t).ewma_vol()
        return (vol_now - vol_then) / max(vol_then, 1e-8)

    def vol_zscore(self, window: int = 63) -> float:
        """Current vol relative to recent history."""
        start = max(0, self.current_t - window)
        vols = []
        for t in range(start, self.current_t):
            v = VolRegimeSignal(self.returns, t, self.lambda_ewma).ewma_vol()
            vols.append(v)
        if len(vols) < 5:
            return 0.0
        cur_vol = self.ewma_vol()
        return (cur_vol - mean(vols)) / max(std(vols), 1e-8)

    def feature_vector(self) -> list[float]:
        vol = self.ewma_vol()
        return [
            vol,
            self.vol_trend(),
            self.vol_zscore(),
            sigmoid(self.vol_zscore()),  # regime probability (high vol = stress)
        ]

@dataclass
class SentimentSignalFeatures:
    """News sentiment features over multiple lookback windows."""
    sentiment_series: list[float]   # daily aggregated sentiment scores
    current_t: int

    def recent_sentiment(self, window: int = 3) -> float:
        start = max(0, self.current_t - window)
        s = self.sentiment_series[start:self.current_t]
        return mean(s) if s else 0.0

    def sentiment_momentum(self, window: int = 5) -> float:
        """Change in sentiment."""
        if self.current_t < 2:
            return 0.0
        return self.recent_sentiment(3) - self.recent_sentiment(window)

    def sentiment_vol(self, window: int = 10) -> float:
        """Disagreement / consistency of sentiment."""
        start = max(0, self.current_t - window)
        s = self.sentiment_series[start:self.current_t]
        return std(s) if len(s) > 1 else 0.0

    def feature_vector(self) -> list[float]:
        return [
            self.recent_sentiment(1),
            self.recent_sentiment(3),
            self.recent_sentiment(5),
            self.sentiment_momentum(5),
            self.sentiment_vol(10),
        ]

# ---------------------------------------------------------------------------
# 2. Attention-like cross-modal weighting
# ---------------------------------------------------------------------------
class ModalityGate:
    """
    Learns per-modality quality gates via sigmoid gating.
    Gate_i = sigmoid(w_gate_i . features_i + b_i)
    Final prediction = sum_i (gate_i * modal_pred_i)
    """
    def __init__(self, modal_dims: list[int], seed: int = 0):
        rng = random.Random(seed)
        self.n_modalities = len(modal_dims)
        self.gate_W = [[rng.gauss(0, 0.1) for _ in range(d)] for d in modal_dims]
        self.gate_b = [0.0] * self.n_modalities
        self.pred_W = [[rng.gauss(0, 0.1) for _ in range(d)] for d in modal_dims]
        self.pred_b = [0.0] * self.n_modalities

    def forward(self, modal_features: list[list[float]]) -> tuple[float, list[float]]:
        """
        Returns (final_pred, per-modality gates).
        """
        gates = [sigmoid(dot(self.gate_W[i], modal_features[i]) + self.gate_b[i])
                  for i in range(self.n_modalities)]
        preds = [math.tanh(dot(self.pred_W[i], modal_features[i]) + self.pred_b[i])
                  for i in range(self.n_modalities)]
        total_gate = sum(gates)
        if total_gate < 1e-8:
            return 0.0, gates
        final = sum(gates[i] * preds[i] for i in range(self.n_modalities)) / total_gate
        return final, gates

    def sgd_update(self, modal_features: list[list[float]],
                    target: float, lr: float = 0.01) -> float:
        """Single SGD step."""
        final, gates = self.forward(modal_features)
        loss = (final - target)**2
        d_final = 2 * (final - target)

        total_gate = sum(gates)
        if total_gate < 1e-8:
            return loss

        for i in range(self.n_modalities):
            # Gradient through gate and prediction
            g_i = gates[i]
            p_i = math.tanh(dot(self.pred_W[i], modal_features[i]) + self.pred_b[i])

            # dLoss/d_pred_i = d_final * g_i / total_gate
            d_pred = d_final * g_i / total_gate
            d_tanh = 1 - p_i**2
            for j, f in enumerate(modal_features[i]):
                self.pred_W[i][j] -= lr * d_pred * d_tanh * f
            self.pred_b[i] -= lr * d_pred * d_tanh

            # dLoss/d_gate_i (simplified)
            d_gate = d_final * p_i / max(total_gate, 1e-8)
            d_sigmoid = g_i * (1 - g_i)
            for j, f in enumerate(modal_features[i]):
                self.gate_W[i][j] -= lr * d_gate * d_sigmoid * f
            self.gate_b[i] -= lr * d_gate * d_sigmoid

        return loss

# ---------------------------------------------------------------------------
# 3. Late fusion ensemble
# ---------------------------------------------------------------------------
class LateFusionEnsemble:
    """
    Trains separate predictors per modality, then learns optimal blend weights.
    Uses equal initialization, updates blend via softmax gradient.
    """
    def __init__(self, n_modalities: int, modal_dims: list[int], hidden: int = 8, seed: int = 0):
        rng = random.Random(seed)
        self.n = n_modalities
        self.modal_dims = modal_dims
        self.hidden = hidden

        # Per-modality: simple linear + relu + linear
        self.W1 = [[[rng.gauss(0, 0.1) for _ in range(d)] for _ in range(hidden)] for d in modal_dims]
        self.b1 = [[0.0]*hidden for _ in range(n_modalities)]
        self.W2 = [[rng.gauss(0, 0.1) for _ in range(hidden)] for _ in range(n_modalities)]
        self.b2 = [0.0] * n_modalities

        # Blend weights (log-space for softmax)
        self.log_w = [0.0] * n_modalities

    def modal_predict(self, features: list[float], i: int) -> float:
        h = [relu(dot(self.W1[i][j], features) + self.b1[i][j]) for j in range(self.hidden)]
        return math.tanh(dot(self.W2[i], h) + self.b2[i])

    def predict(self, modal_features: list[list[float]]) -> tuple[float, list[float]]:
        preds = [self.modal_predict(modal_features[i], i) for i in range(self.n)]
        weights = softmax(self.log_w)
        final = sum(weights[i] * preds[i] for i in range(self.n))
        return final, weights

    def train_step(self, modal_features: list[list[float]], target: float,
                    lr: float = 0.01) -> float:
        final, weights = self.predict(modal_features)
        loss = (final - target)**2
        d_final = 2 * (final - target)

        preds = [self.modal_predict(modal_features[i], i) for i in range(self.n)]

        # Update blend weights
        for i in range(self.n):
            d_wi = d_final * preds[i] * weights[i] * (1 - weights[i])
            self.log_w[i] -= lr * d_wi

        # Update per-modality networks (simplified: MSE on each modal pred)
        for i in range(self.n):
            err = preds[i] - target
            h_pre = [dot(self.W1[i][j], modal_features[i]) + self.b1[i][j] for j in range(self.hidden)]
            h = [relu(v) for v in h_pre]
            d_out = 2 * err * (1 - preds[i]**2) * weights[i]
            for j in range(self.hidden):
                self.W2[i][j] -= lr * d_out * h[j]
                d_h = d_out * self.W2[i][j] * (1.0 if h_pre[j] > 0 else 0.0)
                for k, f in enumerate(modal_features[i]):
                    self.W1[i][j][k] -= lr * d_h * f
                self.b1[i][j] -= lr * d_h
            self.b2[i] -= lr * d_out

        return loss

# ---------------------------------------------------------------------------
# 4. Signal quality gating
# ---------------------------------------------------------------------------
def signal_quality_gate(momentum_features: list[float],
                          vol_features: list[float],
                          sentiment_features: list[float]) -> dict:
    """
    Assess quality/reliability of each modality's signal.
    High uncertainty → gate down; high confidence → gate up.
    """
    # Momentum quality: lower when vol is high (momentum unreliable in crashes)
    vol_level = vol_features[0]  # ewma_vol
    mom_quality = max(0.0, 1.0 - (vol_level - 0.15) / 0.30)  # degrades above 15% vol

    # Vol quality: always useful (structural feature)
    vol_quality = 1.0

    # Sentiment quality: lower when sentiment vol is high (noisy news)
    sent_vol = abs(sentiment_features[4])  # sentiment_vol feature
    sent_quality = max(0.1, 1.0 - sent_vol * 5)

    return {
        'momentum_gate': min(1.0, max(0.0, mom_quality)),
        'vol_gate': vol_quality,
        'sentiment_gate': min(1.0, max(0.0, sent_quality)),
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 26: Multi-Modal Fusion")
    print("=" * 65)

    rng = random.Random(42)
    T = 400

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Simulate price returns and sentiment
    price_returns = [0.0005 + 0.012 * randn() for _ in range(T)]
    sentiment_raw = [0.01 * randn() + 0.001 for _ in range(T)]  # slightly positive bias

    # Regime: high-vol crash at t=200..230
    for t in range(200, 230):
        price_returns[t] = -0.003 + 0.025 * randn()
        sentiment_raw[t] = -0.05 + 0.02 * randn()

    # True target: 1-day forward return
    targets = price_returns[1:] + [0.0]

    print("\n1. Per-Modality Feature Vectors (t=250)")
    t = 250
    mom = PriceMomentumSignal(price_returns, t)
    vol = VolRegimeSignal(price_returns, t)
    sent = SentimentSignalFeatures(sentiment_raw, t)

    mom_f = mom.feature_vector()
    vol_f = vol.feature_vector()
    sent_f = sent.feature_vector()

    print(f"   Momentum features  : {[round(f, 4) for f in mom_f]}")
    print(f"   Vol regime features: {[round(f, 4) for f in vol_f]}")
    print(f"   Sentiment features : {[round(f, 4) for f in sent_f]}")

    print("\n2. Signal Quality Gates")
    gates = signal_quality_gate(mom_f, vol_f, sent_f)
    for name, g in gates.items():
        bar = '█' * int(g * 20)
        print(f"   {name:20s}: {g:.3f}  [{bar:<20}]")

    print("\n3. Attention-Like Modality Gating Model")
    modal_dims = [len(mom_f), len(vol_f), len(sent_f)]
    gate_model = ModalityGate(modal_dims, seed=0)

    losses = []
    n_train = 300
    for t_i in range(50, n_train):
        mom_t = PriceMomentumSignal(price_returns, t_i).feature_vector()
        vol_t = VolRegimeSignal(price_returns, t_i).feature_vector()
        sent_t = SentimentSignalFeatures(sentiment_raw, t_i).feature_vector()
        loss = gate_model.sgd_update([mom_t, vol_t, sent_t], targets[t_i], lr=0.005)
        losses.append(loss)

    # Rolling loss
    def rolling_mean(xs, w):
        return [mean(xs[max(0,i-w):i+1]) for i in range(len(xs))]
    roll_loss = rolling_mean(losses, 30)
    print(f"   Initial loss (avg first 30): {mean(losses[:30]):.6f}")
    print(f"   Final   loss (avg last 30) : {mean(losses[-30:]):.6f}")

    # Test
    n_correct = 0
    n_test = 0
    for t_i in range(n_train, min(T-1, n_train+80)):
        mom_t = PriceMomentumSignal(price_returns, t_i).feature_vector()
        vol_t = VolRegimeSignal(price_returns, t_i).feature_vector()
        sent_t = SentimentSignalFeatures(sentiment_raw, t_i).feature_vector()
        pred, modal_gates = gate_model.forward([mom_t, vol_t, sent_t])
        if (pred > 0) == (targets[t_i] > 0):
            n_correct += 1
        n_test += 1
    print(f"   Test directional accuracy: {n_correct/max(n_test,1):.2%} (n={n_test})")

    print("\n4. Late Fusion Ensemble")
    ensemble = LateFusionEnsemble(3, modal_dims, hidden=8, seed=0)
    e_losses = []
    for t_i in range(50, n_train):
        mom_t = PriceMomentumSignal(price_returns, t_i).feature_vector()
        vol_t = VolRegimeSignal(price_returns, t_i).feature_vector()
        sent_t = SentimentSignalFeatures(sentiment_raw, t_i).feature_vector()
        l = ensemble.train_step([mom_t, vol_t, sent_t], targets[t_i], lr=0.005)
        e_losses.append(l)

    final_pred, final_weights = ensemble.predict([mom_f, vol_f, sent_f])
    print(f"   Blend weights: momentum={final_weights[0]:.3f}  vol={final_weights[1]:.3f}  sentiment={final_weights[2]:.3f}")
    print(f"   Ensemble loss improvement: {mean(e_losses[:30]):.6f} → {mean(e_losses[-30:]):.6f}")

    print("\n5. Cross-Modal Alignment (correlation of modal predictions)")
    all_mom_preds, all_vol_preds, all_sent_preds = [], [], []
    for t_i in range(100, 300):
        mom_t = PriceMomentumSignal(price_returns, t_i).feature_vector()
        vol_t = VolRegimeSignal(price_returns, t_i).feature_vector()
        sent_t = SentimentSignalFeatures(sentiment_raw, t_i).feature_vector()
        pm = gate_model.pred_W[0]
        pv = gate_model.pred_W[1]
        ps = gate_model.pred_W[2]
        all_mom_preds.append(math.tanh(dot(pm, mom_t) + gate_model.pred_b[0]))
        all_vol_preds.append(math.tanh(dot(pv, vol_t) + gate_model.pred_b[1]))
        all_sent_preds.append(math.tanh(dot(ps, sent_t) + gate_model.pred_b[2]))

    def pearson(a, b):
        n = len(a)
        ma, mb = mean(a), mean(b)
        num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
        den = math.sqrt(sum((a[i]-ma)**2 for i in range(n)) * sum((b[i]-mb)**2 for i in range(n)))
        return num / max(den, 1e-10)

    print(f"   Momentum vs Vol sentiment correlation  : {pearson(all_mom_preds, all_vol_preds):+.4f}")
    print(f"   Momentum vs News sentiment correlation : {pearson(all_mom_preds, all_sent_preds):+.4f}")
    print(f"   Vol vs News sentiment correlation      : {pearson(all_vol_preds, all_sent_preds):+.4f}")
    print(f"   (Low cross-modal correlation = diverse information sources = better fusion)")

    print("\n[Done] Day 26: Multi-Modal Fusion complete.")
