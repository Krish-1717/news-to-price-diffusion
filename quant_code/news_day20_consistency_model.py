"""
news_day20_consistency_model.py
Day 20: Consistency Model (Song et al. 2023) for news-price diffusion.
Single-step generation, consistency training loss, EMA parameters,
multi-step sampling, ODE trajectory approximation.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x**2 for x in v))

def _dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))

# ---------------------------------------------------------------------------
# Noise schedule (EDM-style, continuous)
# ---------------------------------------------------------------------------
def sigma_schedule(t: float, sigma_min: float = 0.002, sigma_max: float = 80.0) -> float:
    """EDM sigma schedule: sigma(t) = t (noise level = time)."""
    return t

def sigma_data_const() -> float:
    return 0.5  # typical data sigma for normalization

# ---------------------------------------------------------------------------
# Consistency function target: f_theta(x_t, t) -> x_0
# Architecture: simple linear network + skip connection
# ---------------------------------------------------------------------------
class ConsistencyNet:
    """
    Consistency model network.
    Parametrize F as:
    F(x, t) = c_skip(t) * x + c_out(t) * F_theta(x / c_in(t), t)
    where F_theta is a learned function.
    """
    def __init__(self, d: int = 8, hidden: int = 32, seed: int = 0):
        rng = random.Random(seed)
        self.d = d
        self.hidden = hidden
        self.sigma_data = sigma_data_const()

        # F_theta: [x_scaled(d), t_embed(4)] -> d
        d_in = d + 4
        self.W1 = [[rng.gauss(0, (2/d_in)**0.5) for _ in range(d_in)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[rng.gauss(0, (2/hidden)**0.5) for _ in range(hidden)] for _ in range(d)]
        self.b2 = [0.0] * d

        # EMA copy
        self.W1_ema = [row[:] for row in self.W1]
        self.b1_ema = self.b1[:]
        self.W2_ema = [row[:] for row in self.W2]
        self.b2_ema = self.b2[:]

    def _t_embed(self, t: float) -> list[float]:
        """Fourier time embedding."""
        freqs = [1.0, 2.0, 4.0, 8.0]
        return [math.sin(t * f) if i % 2 == 0 else math.cos(t * f)
                for i, f in enumerate(freqs)]

    def _c_skip(self, sigma: float) -> float:
        s2 = self.sigma_data ** 2
        return s2 / (sigma**2 + s2)

    def _c_out(self, sigma: float) -> float:
        s2 = self.sigma_data ** 2
        return sigma * self.sigma_data / math.sqrt(sigma**2 + s2)

    def _c_in(self, sigma: float) -> float:
        s2 = self.sigma_data ** 2
        return 1.0 / math.sqrt(sigma**2 + s2)

    def _forward_raw(self, x: list[float], t: float, use_ema: bool = False) -> list[float]:
        W1 = self.W1_ema if use_ema else self.W1
        b1 = self.b1_ema if use_ema else self.b1
        W2 = self.W2_ema if use_ema else self.W2
        b2 = self.b2_ema if use_ema else self.b2

        t_emb = self._t_embed(t)
        inp = x + t_emb

        h = [max(sum(W1[j][i] * inp[i] for i in range(len(inp))) + b1[j], 0.0)
             for j in range(self.hidden)]
        out = [sum(W2[j][i] * h[i] for i in range(self.hidden)) + b2[j]
               for j in range(self.d)]
        return out

    def forward(self, x_t: list[float], sigma: float, use_ema: bool = False) -> list[float]:
        """Consistency function: maps (x_t, sigma) -> x_0_prediction."""
        c_skip = self._c_skip(sigma)
        c_out = self._c_out(sigma)
        c_in = self._c_in(sigma)

        x_scaled = [xi * c_in for xi in x_t]
        F_raw = self._forward_raw(x_scaled, sigma, use_ema)

        return [c_skip * x_t[i] + c_out * F_raw[i] for i in range(self.d)]

    def update_ema(self, mu: float = 0.9999):
        """EMA weight update."""
        for j in range(len(self.W1)):
            for i in range(len(self.W1[0])):
                self.W1_ema[j][i] = mu * self.W1_ema[j][i] + (1 - mu) * self.W1[j][i]
            self.b1_ema[j] = mu * self.b1_ema[j] + (1 - mu) * self.b1[j]
        for j in range(len(self.W2)):
            for i in range(len(self.W2[0])):
                self.W2_ema[j][i] = mu * self.W2_ema[j][i] + (1 - mu) * self.W2[j][i]
            self.b2_ema[j] = mu * self.b2_ema[j] + (1 - mu) * self.b2[j]

    def sgd_update(self, grad_W1, grad_W2, grad_b1, grad_b2, lr: float):
        for j in range(len(self.W1)):
            for i in range(len(self.W1[0])):
                self.W1[j][i] -= lr * grad_W1[j][i]
            self.b1[j] -= lr * grad_b1[j]
        for j in range(len(self.W2)):
            for i in range(len(self.W2[0])):
                self.W2[j][i] -= lr * grad_W2[j][i]
            self.b2[j] -= lr * grad_b2[j]


# ---------------------------------------------------------------------------
# Consistency Training Loss
# ---------------------------------------------------------------------------
def consistency_loss(
    net: ConsistencyNet,
    x0: list[float],
    sigma1: float,
    sigma2: float,
    rng: random.Random,
) -> tuple[float, list[float]]:
    """
    Consistency training loss:
    L = ||f(x + sigma2*eps, sigma2) - f_ema(x + sigma1*eps, sigma1)||^2
    where sigma1 < sigma2.
    """
    d = len(x0)
    eps = [rng.gauss(0, 1) for _ in range(d)]

    x_t2 = [x0[i] + sigma2 * eps[i] for i in range(d)]
    x_t1 = [x0[i] + sigma1 * eps[i] for i in range(d)]

    # Student prediction (online network)
    f2 = net.forward(x_t2, sigma2, use_ema=False)
    # Teacher prediction (EMA network)
    f1 = net.forward(x_t1, sigma1, use_ema=True)

    diff = [f2[i] - f1[i] for i in range(d)]
    loss = sum(di**2 for di in diff) / d

    return loss, diff


def consistency_training_step(
    net: ConsistencyNet,
    x0: list[float],
    sigma_min: float = 0.002,
    sigma_max: float = 5.0,
    n_discretize: int = 40,
    lr: float = 1e-3,
    rng: random.Random = None,
    ema_mu: float = 0.999,
) -> float:
    """One training step."""
    if rng is None:
        rng = random.Random()

    # Sample a pair (sigma_n, sigma_{n+1}) from schedule
    k = rng.randint(0, n_discretize - 2)
    sigma_n = sigma_min + (sigma_max - sigma_min) * k / n_discretize
    sigma_n1 = sigma_min + (sigma_max - sigma_min) * (k + 1) / n_discretize

    loss, diff = consistency_loss(net, x0, sigma_n, sigma_n1, rng)

    # Approximate backprop (finite differences on output)
    d = len(x0)
    eps = [rng.gauss(0, 1) for _ in range(d)]
    x_t2 = [x0[i] + sigma_n1 * eps[i] for i in range(d)]

    # Compute gradient through F_theta(x_t2, sigma_n1)
    c_in = net._c_in(sigma_n1)
    c_out = net._c_out(sigma_n1)

    x_scaled = [xi * c_in for xi in x_t2]
    t_emb = net._t_embed(sigma_n1)
    inp = x_scaled + t_emb

    # Forward pass manually for gradient
    h_pre = [sum(net.W1[j][i] * inp[i] for i in range(len(inp))) + net.b1[j]
             for j in range(net.hidden)]
    h = [max(hp, 0.0) for hp in h_pre]
    F_raw = [sum(net.W2[j][i] * h[i] for i in range(net.hidden)) + net.b2[j]
             for j in range(d)]

    # Loss gradient w.r.t. F_raw: dL/dF_raw = (2/d) * diff * c_out
    grad_F_raw = [2 / d * diff[j] * c_out for j in range(d)]

    # Backward through W2
    grad_W2 = [[grad_F_raw[j] * h[i] for i in range(net.hidden)] for j in range(d)]
    grad_b2 = grad_F_raw[:]
    grad_h = [sum(net.W2[j][i] * grad_F_raw[j] for j in range(d)) for i in range(net.hidden)]

    # Through ReLU
    grad_h_pre = [grad_h[i] * (1.0 if h_pre[i] > 0 else 0.0) for i in range(net.hidden)]

    # Through W1
    grad_W1 = [[grad_h_pre[j] * inp[i] for i in range(len(inp))] for j in range(net.hidden)]
    grad_b1 = grad_h_pre[:]

    net.sgd_update(grad_W1, grad_W2, grad_b1, grad_b2, lr)
    net.update_ema(ema_mu)

    return loss


# ---------------------------------------------------------------------------
# Sampling: single-step generation
# ---------------------------------------------------------------------------
def single_step_sample(
    net: ConsistencyNet,
    sigma_max: float = 5.0,
    seed: int = 0,
) -> list[float]:
    """Generate from noise in ONE network evaluation."""
    rng = random.Random(seed)
    x_T = [rng.gauss(0, sigma_max) for _ in range(net.d)]
    return net.forward(x_T, sigma_max, use_ema=True)


# ---------------------------------------------------------------------------
# Multi-step sampling
# ---------------------------------------------------------------------------
def multistep_sample(
    net: ConsistencyNet,
    sigma_min: float = 0.002,
    sigma_max: float = 5.0,
    n_steps: int = 5,
    seed: int = 0,
) -> list[float]:
    """
    Multi-step consistency sampling:
    Start from sigma_max, iteratively denoise via consistency function + noise injection.
    """
    rng = random.Random(seed)
    d = net.d

    sigmas = [sigma_max - (sigma_max - sigma_min) * i / (n_steps - 1) for i in range(n_steps)]

    x = [rng.gauss(0, sigma_max) for _ in range(d)]

    for step_idx in range(n_steps - 1):
        sigma = sigmas[step_idx]
        sigma_next = sigmas[step_idx + 1]

        # Map to clean data
        x0_hat = net.forward(x, sigma, use_ema=True)

        # Re-corrupt to sigma_next
        eps = [rng.gauss(0, 1) for _ in range(d)]
        x = [x0_hat[i] + sigma_next * eps[i] for i in range(d)]

    # Final denoising step
    x0_final = net.forward(x, sigmas[-1], use_ema=True)
    return x0_final


# ---------------------------------------------------------------------------
# Consistency Distillation from pre-trained score model (mock)
# ---------------------------------------------------------------------------
def mock_score_model(x_t: list[float], sigma: float, x0_true: list[float]) -> list[float]:
    """Mock oracle score = (x_t - x0_true) / sigma^2."""
    return [(x_t[i] - x0_true[i]) / max(sigma**2, 1e-6) for i in range(len(x_t))]


def distillation_step(
    net: ConsistencyNet,
    x0: list[float],
    sigma_n: float,
    sigma_n1: float,
    rng: random.Random,
    n_ode_steps: int = 1,
    lr: float = 1e-3,
) -> float:
    """
    Consistency distillation: use score model to compute ODE trajectory step.
    Target = f_ema(x + sigma_n1*eps - step * score * (sigma_n1 - sigma_n), sigma_n)
    """
    d = len(x0)
    eps = [rng.gauss(0, 1) for _ in range(d)]
    x_t_n1 = [x0[i] + sigma_n1 * eps[i] for i in range(d)]

    # ODE step (Euler) using mock score
    score = mock_score_model(x_t_n1, sigma_n1, x0)
    step_size = sigma_n1 - sigma_n
    x_t_n = [x_t_n1[i] - step_size * score[i] for i in range(d)]

    # Teacher: EMA model at sigma_n
    f1 = net.forward(x_t_n, sigma_n, use_ema=True)
    # Student: online model at sigma_n1
    f2 = net.forward(x_t_n1, sigma_n1, use_ema=False)

    diff = [f2[i] - f1[i] for i in range(d)]
    loss = sum(di**2 for di in diff) / d

    # Simplified gradient update
    c_in = net._c_in(sigma_n1)
    c_out = net._c_out(sigma_n1)
    x_scaled = [xi * c_in for xi in x_t_n1]
    t_emb = net._t_embed(sigma_n1)
    inp = x_scaled + t_emb

    h_pre = [sum(net.W1[j][i] * inp[i] for i in range(len(inp))) + net.b1[j]
             for j in range(net.hidden)]
    h = [max(hp, 0.0) for hp in h_pre]

    grad_F = [2 / d * diff[j] * c_out for j in range(d)]
    grad_W2 = [[grad_F[j] * h[i] for i in range(net.hidden)] for j in range(d)]
    grad_b2 = grad_F[:]
    grad_h = [sum(net.W2[j][i] * grad_F[j] for j in range(d)) for i in range(net.hidden)]
    grad_h_pre = [grad_h[i] * (1.0 if h_pre[i] > 0 else 0.0) for i in range(net.hidden)]
    grad_W1 = [[grad_h_pre[j] * inp[i] for i in range(len(inp))] for j in range(net.hidden)]
    grad_b1 = grad_h_pre[:]

    net.sgd_update(grad_W1, grad_W2, grad_b1, grad_b2, lr)
    net.update_ema(0.999)
    return loss


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("DAY 20: Consistency Model for News-Price Diffusion")
    print("=" * 60)

    rng = random.Random(42)
    D = 8
    net = ConsistencyNet(d=D, hidden=24, seed=0)

    # Synthetic data: bimodal distribution
    def sample_data(rng: random.Random) -> list[float]:
        if rng.random() < 0.5:
            return [rng.gauss(2.0, 0.3) for _ in range(D)]
        else:
            return [rng.gauss(-2.0, 0.3) for _ in range(D)]

    print("\n1. Consistency Training")
    train_rng = random.Random(1)
    losses = []
    for step in range(500):
        x0 = sample_data(train_rng)
        loss = consistency_training_step(net, x0, sigma_min=0.01, sigma_max=3.0,
                                          lr=3e-4, rng=train_rng, ema_mu=0.995)
        losses.append(loss)
        if step % 100 == 0:
            avg_loss = sum(losses[-50:]) / min(len(losses), 50)
            print(f"   Step {step:>4}: avg loss = {avg_loss:.6f}")

    print("\n2. Single-Step Generation")
    single_samples = [single_step_sample(net, sigma_max=3.0, seed=i) for i in range(5)]
    for i, s in enumerate(single_samples):
        mean_s = sum(s) / D
        print(f"   Sample {i+1}: mean={mean_s:.4f}, norm={_norm(s):.4f}")

    print("\n3. Multi-Step Generation (5 steps)")
    multi_samples = [multistep_sample(net, sigma_min=0.01, sigma_max=3.0, n_steps=5, seed=i)
                     for i in range(5)]
    for i, s in enumerate(multi_samples):
        mean_s = sum(s) / D
        print(f"   Sample {i+1}: mean={mean_s:.4f}, norm={_norm(s):.4f}")

    print("\n4. Consistency Distillation (from mock score model)")
    net2 = ConsistencyNet(d=D, hidden=24, seed=1)
    dist_losses = []
    for step in range(200):
        x0 = sample_data(train_rng)
        k = train_rng.randint(0, 10)
        sigma_n = 0.01 + 2.99 * k / 10
        sigma_n1 = 0.01 + 2.99 * (k + 1) / 10
        loss = distillation_step(net2, x0, sigma_n, sigma_n1, train_rng, lr=3e-4)
        dist_losses.append(loss)
    avg = sum(dist_losses[-50:]) / 50
    print(f"   Distillation final loss: {avg:.6f}")

    print("\n5. EMA Parameter Convergence")
    w1_diffs = [abs(net.W1[j][0] - net.W1_ema[j][0]) for j in range(min(5, net.hidden))]
    print(f"   Max |W1 - W1_ema|: {max(w1_diffs):.6f}")

    print("\n6. Generation Quality Check")
    n_gen = 100
    gen_means = [sum(single_step_sample(net, sigma_max=3.0, seed=s)) / D for s in range(n_gen)]
    pos_pct = sum(1 for m in gen_means if m > 0) / n_gen
    neg_pct = sum(1 for m in gen_means if m < 0) / n_gen
    print(f"   Generated {n_gen} samples")
    print(f"   Positive mean: {pos_pct:.2%}  Negative mean: {neg_pct:.2%}")

    print("\n[Done] Day 20: Consistency Model complete.")
