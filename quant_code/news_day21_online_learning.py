"""
news_day21_online_learning.py
Day 21: Online Learning for News Diffusion — AdaGrad SGD, EMA weight smoothing,
CUSUM drift detector, experience replay buffer, incremental calibration.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

# ---------------------------------------------------------------------------
# Activations and helpers
# ---------------------------------------------------------------------------
def relu(x: float) -> float:
    return max(0.0, x)

def relu_deriv(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1 + math.exp(-x))
    return math.exp(x) / (1 + math.exp(x))

def dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))

def mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    return [dot(row, v) for row in M]

# ---------------------------------------------------------------------------
# Two-layer MLP with AdaGrad
# ---------------------------------------------------------------------------
class AdaGradMLP:
    """
    Two-layer MLP trained with AdaGrad for adaptive learning rates.
    Architecture: input_dim -> hidden -> output_dim (linear output)
    """
    def __init__(self, input_dim: int, hidden: int, output_dim: int, lr: float = 0.01,
                 eps: float = 1e-7, seed: int = 0):
        rng = random.Random(seed)
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden)

        self.W1 = [[rng.gauss(0, scale1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[rng.gauss(0, scale2) for _ in range(hidden)] for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

        # AdaGrad accumulators
        self.G_W1 = [[eps] * input_dim for _ in range(hidden)]
        self.G_b1 = [eps] * hidden
        self.G_W2 = [[eps] * hidden for _ in range(output_dim)]
        self.G_b2 = [eps] * output_dim

        self.lr = lr
        self.eps = eps
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim

    def forward(self, x: list[float]) -> tuple[list[float], list[float], list[float]]:
        """Returns (output, hidden_pre_act, hidden_post_act)."""
        h_pre = [dot(self.W1[j], x) + self.b1[j] for j in range(self.hidden)]
        h = [relu(v) for v in h_pre]
        out = [dot(self.W2[k], h) + self.b2[k] for k in range(self.output_dim)]
        return out, h_pre, h

    def backward(self, x: list[float], h_pre: list[float], h: list[float],
                 output: list[float], target: list[float]) -> float:
        """MSE loss backward + AdaGrad update. Returns scalar loss."""
        loss = sum((output[k] - target[k])**2 for k in range(self.output_dim)) / self.output_dim

        # Output layer gradients
        dout = [(output[k] - target[k]) * 2.0 / self.output_dim for k in range(self.output_dim)]

        dW2 = [[dout[k] * h[j] for j in range(self.hidden)] for k in range(self.output_dim)]
        db2 = list(dout)

        # Hidden layer gradients
        dh = [sum(self.W2[k][j] * dout[k] for k in range(self.output_dim))
              for j in range(self.hidden)]
        dh_pre = [dh[j] * relu_deriv(h_pre[j]) for j in range(self.hidden)]

        dW1 = [[dh_pre[j] * x[i] for i in range(self.input_dim)] for j in range(self.hidden)]
        db1 = list(dh_pre)

        # AdaGrad update
        for j in range(self.hidden):
            for i in range(self.input_dim):
                self.G_W1[j][i] += dW1[j][i]**2
                self.W1[j][i] -= self.lr / math.sqrt(self.G_W1[j][i]) * dW1[j][i]
            self.G_b1[j] += db1[j]**2
            self.b1[j] -= self.lr / math.sqrt(self.G_b1[j]) * db1[j]

        for k in range(self.output_dim):
            for j in range(self.hidden):
                self.G_W2[k][j] += dW2[k][j]**2
                self.W2[k][j] -= self.lr / math.sqrt(self.G_W2[k][j]) * dW2[k][j]
            self.G_b2[k] += db2[k]**2
            self.b2[k] -= self.lr / math.sqrt(self.G_b2[k]) * db2[k]

        return loss

    def train_step(self, x: list[float], y: list[float]) -> float:
        output, h_pre, h = self.forward(x)
        return self.backward(x, h_pre, h, output, y)

    def predict(self, x: list[float]) -> list[float]:
        out, _, _ = self.forward(x)
        return out

# ---------------------------------------------------------------------------
# EMA weight smoothing (exponential moving average of predictions)
# ---------------------------------------------------------------------------
class EMAPredictor:
    """
    Maintains an EMA of recent predictions for smoothing.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.ema: Optional[list[float]] = None

    def update(self, new_pred: list[float]) -> list[float]:
        if self.ema is None:
            self.ema = list(new_pred)
        else:
            self.ema = [self.alpha * n + (1 - self.alpha) * e
                        for n, e in zip(new_pred, self.ema)]
        return list(self.ema)

# ---------------------------------------------------------------------------
# CUSUM drift detector
# ---------------------------------------------------------------------------
class CUSUMDriftDetector:
    """
    Page-Hinkley CUSUM for detecting distribution drift in a stream.
    Monitors both positive and negative cumulative deviations.
    """
    def __init__(self, threshold: float = 5.0, k: float = 0.5):
        self.threshold = threshold
        self.k = k
        self.C_plus = 0.0
        self.C_minus = 0.0
        self.mean_est = 0.0
        self.std_est = 1.0
        self.n = 0
        self.drift_detected = False
        self.last_drift_idx = 0

    def update(self, value: float, idx: int) -> bool:
        """Update CUSUM with new observation. Returns True if drift detected."""
        self.n += 1
        # Update mean/std online
        delta = value - self.mean_est
        self.mean_est += delta / self.n
        if self.n > 1:
            self.std_est = math.sqrt(max(
                self.std_est**2 * (self.n - 2) / (self.n - 1) + delta**2 / self.n,
                1e-8
            ))

        standardized = (value - self.mean_est) / max(self.std_est, 1e-8)
        self.C_plus = max(0.0, self.C_plus + standardized - self.k)
        self.C_minus = max(0.0, self.C_minus - standardized - self.k)

        if self.C_plus > self.threshold or self.C_minus > self.threshold:
            self.drift_detected = True
            self.last_drift_idx = idx
            self.C_plus = self.C_minus = 0.0  # reset after alarm
            self.n = 0  # restart statistics
            return True
        return False

# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.
    Enables mini-batch learning from past experiences.
    """
    def __init__(self, capacity: int = 500, seed: int = 42):
        self.buffer: deque = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, x: list[float], y: list[float]):
        self.buffer.append((list(x), list(y)))

    def sample(self, batch_size: int) -> list[tuple[list[float], list[float]]]:
        n = min(batch_size, len(self.buffer))
        indices = self.rng.sample(range(len(self.buffer)), n)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

# ---------------------------------------------------------------------------
# Online learning trainer (stream + replay)
# ---------------------------------------------------------------------------
class OnlineLearner:
    """
    Online learning loop that combines:
    - Immediate learning from new data point
    - Replay from buffer for stability
    - EMA smoothing of predictions
    - CUSUM drift monitoring
    """
    def __init__(self, model: AdaGradMLP, replay_capacity: int = 500,
                 replay_batch: int = 16, ema_alpha: float = 0.1,
                 cusum_threshold: float = 5.0, seed: int = 0):
        self.model = model
        self.buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)
        self.replay_batch = replay_batch
        self.ema = EMAPredictor(alpha=ema_alpha)
        self.drift = CUSUMDriftDetector(threshold=cusum_threshold)
        self.losses: list[float] = []
        self.drift_events: list[int] = []
        self.step_count = 0

    def step(self, x: list[float], y: list[float]) -> dict:
        """Process one new (x, y) pair in the stream."""
        self.step_count += 1

        # 1. Predict on new data
        pred = self.model.predict(x)
        ema_pred = self.ema.update(pred)

        # 2. Compute prediction error for drift detection
        error = math.sqrt(sum((pred[k] - y[k])**2 for k in range(len(y))))
        drifted = self.drift.update(error, self.step_count)
        if drifted:
            self.drift_events.append(self.step_count)

        # 3. Learn from new data point
        loss = self.model.train_step(x, y)
        self.losses.append(loss)

        # 4. Store in replay buffer
        self.buffer.push(x, y)

        # 5. Experience replay
        replay_loss = 0.0
        if len(self.buffer) >= self.replay_batch:
            batch = self.buffer.sample(self.replay_batch)
            replay_losses = [self.model.train_step(bx, by) for bx, by in batch]
            replay_loss = sum(replay_losses) / len(replay_losses)

        return {
            'step': self.step_count,
            'loss': loss,
            'replay_loss': replay_loss,
            'drift_detected': drifted,
            'pred': pred,
            'ema_pred': ema_pred,
            'error': error,
        }

    def rolling_loss(self, window: int = 50) -> float:
        tail = self.losses[-window:]
        return sum(tail) / max(len(tail), 1)

# ---------------------------------------------------------------------------
# Incremental calibration check
# ---------------------------------------------------------------------------
def incremental_calibration(predictions: list[float], actuals: list[float],
                              window: int = 50) -> list[dict]:
    """
    Track rolling calibration: mean error, std error, and coverage of
    ±1sigma prediction intervals.
    """
    results = []
    n = len(predictions)
    for i in range(window - 1, n):
        sub_p = predictions[i - window + 1:i + 1]
        sub_a = actuals[i - window + 1:i + 1]
        errors = [sub_p[j] - sub_a[j] for j in range(len(sub_p))]
        bias = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        std_e = math.sqrt(sum((e - bias)**2 for e in errors) / max(len(errors)-1, 1))
        results.append({'idx': i, 'bias': bias, 'rmse': rmse, 'std_err': std_e})
    return results

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 21: Online Learning for News Diffusion")
    print("=" * 65)

    rng = random.Random(42)
    D_in, D_hidden, D_out = 8, 16, 4
    N_STREAM = 600

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Data stream: regime change at t=300
    def gen_sample(t: int) -> tuple[list[float], list[float]]:
        x = [randn() for _ in range(D_in)]
        if t < 300:
            y = [0.5 * x[0] + 0.3 * x[1] + 0.1 * randn() for _ in range(D_out)]
        else:  # regime change: different mapping
            y = [-0.5 * x[2] + 0.4 * x[3] + 0.1 * randn() for _ in range(D_out)]
        return x, y

    model = AdaGradMLP(D_in, D_hidden, D_out, lr=0.05, seed=0)
    learner = OnlineLearner(model, replay_capacity=300, replay_batch=16,
                             ema_alpha=0.1, cusum_threshold=4.0, seed=0)

    print(f"\nOnline learning stream ({N_STREAM} steps, regime change at t=300)")
    all_preds, all_actuals = [], []

    for t in range(N_STREAM):
        x, y = gen_sample(t)
        result = learner.step(x, y)
        all_preds.append(result['pred'][0])
        all_actuals.append(y[0])

        if (t + 1) % 100 == 0 or result['drift_detected']:
            rolling = learner.rolling_loss(50)
            drift_marker = " << DRIFT DETECTED" if result['drift_detected'] else ""
            print(f"  t={t+1:4d}: loss={result['loss']:.5f}, "
                  f"rolling50={rolling:.5f}, "
                  f"error={result['error']:.5f}{drift_marker}")

    print(f"\n  Drift events detected at: {learner.drift_events}")
    print(f"  Final rolling-50 loss: {learner.rolling_loss(50):.5f}")
    print(f"  Buffer size: {len(learner.buffer)}")

    # Incremental calibration
    print("\nIncremental Calibration (every 100 steps):")
    cal = incremental_calibration(all_preds, all_actuals, window=50)
    for r in cal[::100]:
        print(f"  t={r['idx']:4d}: bias={r['bias']:+.5f}, rmse={r['rmse']:.5f}, std_err={r['std_err']:.5f}")

    # EMA smoothing test
    print("\nEMA Smoothing (final 5 predictions):")
    ema_pred = EMAPredictor(alpha=0.1)
    for i, p in enumerate(all_preds[-5:]):
        smoothed = ema_pred.update([p])
        print(f"  raw={p:.5f}  ema={smoothed[0]:.5f}")

    print("\n[Done] Day 21: Online Learning complete.")
