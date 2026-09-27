"""
Latent Diffusion Model for Financial Time Series
Day 15 â news-to-price-diffusion/diffusion/latent_diffusion.py

Implements a simplified latent diffusion process:
  1. Encoder: compresses return windows into a latent vector
  2. Forward diffusion: progressively adds Gaussian noise
  3. Reverse diffusion (score network): denoises step-by-step
  4. Decoder: reconstructs return sequences from latent

All pure Python / pure math â no ML frameworks required.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility math
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _matvec(M: List[List[float]], v: List[float]) -> List[float]:
    return [_dot(row, v) for row in M]


def _add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _scale(s: float, v: List[float]) -> List[float]:
    return [s * x for x in v]


def _tanh(x: float) -> float:
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e = math.exp(2 * x)
    return (e - 1) / (e + 1)


def _relu(x: float) -> float:
    return max(0.0, x)


def _softplus(x: float) -> float:
    return math.log(1 + math.exp(min(x, 30)))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in v))


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

@dataclass
class NoiseSchedule:
    T: int              # total diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self):
        # Linear beta schedule
        self.betas = [
            self.beta_start + (self.beta_end - self.beta_start) * t / (self.T - 1)
            for t in range(self.T)
        ]
        self.alphas = [1.0 - b for b in self.betas]
        # Cumulative product Î±Ì_t = â_{sâ¤t} Î±_s
        self.alpha_bar = []
        prod = 1.0
        for a in self.alphas:
            prod *= a
            self.alpha_bar.append(prod)

    def q_sample(self, x0: List[float], t: int,
                 rng: random.Random) -> Tuple[List[float], List[float]]:
        """
        Sample x_t ~ q(x_t | x_0) = N(âÎ±Ì_t x_0, (1-Î±Ì_t) I).
        Returns (x_t, noise).
        """
        ab = self.alpha_bar[t]
        sqrt_ab = math.sqrt(ab)
        sqrt_one_minus_ab = math.sqrt(1.0 - ab)
        noise = [rnd.gauss(0, 1) for _ in x0]
        x_t = [sqrt_ab * x0[i] + sqrt_one_minus_ab * noise[i]
               for i in range(len(x0))]
        return x_t, noise

    def denoise_step(self, x_t: List[float], eps_pred: List[float],
                     t: int, rng: random.Random) -> List[float]:
        """
        DDPM reverse step: x_{t-1} ~ p_Î¸(x_{t-1} | x_t).
        """
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        ab_t = self.alpha_bar[t]
        ab_prev = self.alpha_bar[t - 1] if t > 0 else 1.0

        # Predicted x_0
        sqrt_ab = math.sqrt(ab_t)
        sqrt_one_minus_ab = math.sqrt(1.0 - ab_t)
        x0_pred = [
            (x_t[i] - sqrt_one_minus_ab * eps_pred[i]) / (sqrt_ab + 1e-8)
            for i in range(len(x_t))
        ]

        # Posterior mean
        coef1 = math.sqrt(ab_prev) * beta_t / (1.0 - ab_t + 1e-8)
        coef2 = math.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t + 1e-8)
        mu = [coef1 * x0_pred[i] + coef2 * x_t[i] for i in range(len(x_t))]

        # Posterior variance
        var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t + 1e-8)
        std = math.sqrt(max(var, 1e-8))
        if t == 0:
            return mu
        return [mu[i] + std * rng.gauss(0, 1) for i in range(len(mu))]


# ---------------------------------------------------------------------------
# Simple MLP layer
# ---------------------------------------------------------------------------

@dataclass
class LinearLayer:
    W: List[List[float]]
    b: List[float]
    activation: str = "relu"   # "relu", "tanh", "none"

    @classmethod
    def random_init(cls, in_dim: int, out_dim: int,
                    activation: str = "relu",
                    rng: Optional[random.Random] = None) -> "LinearLayer":
        rng = rng or random.Random(0)
        scale = math.sqrt(2.0 / in_dim)
        W = [[rng.gauss(0, scale) for _ in range(in_dim)] for _ in range(out_dim)]
        b = [0.0] * out_dim
        return cls(W=W, b=b, activation=activation)

    def forward(self, x: List[float]) -> List[float]:
        out = _add(_matvec(self.W, x), self.b)
        if self.activation == "relu":
            return [_relu(v) for v in out]
        elif self.activation == "tanh":
            return [_tanh(v) for v in out]
        return out


@dataclass
class MLP:
    layers: List[LinearLayer]

    def forward(self, x: List[float]) -> List[float]:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    @classmethod
    def build(cls, dims: List[int], activations: Optional[List[str]] = None,
              rng: Optional[random.Random] = None) -> "MLP":
        rng = rng or random.Random(42)
        acts = activations or (["relu"] * (len(dims) - 2) + ["none"])
        layers = [
            LinearLayer.random_init(dims[i], dims[i + 1], acts[i], rng)
            for i in range(len(dims) - 1)
        ]
        return cls(layers=layers)


# ---------------------------------------------------------------------------
# Latent Diffusion Model
# ---------------------------------------------------------------------------

@dataclass
class LDMConfig:
    seq_len: int = 20         # input window length
    latent_dim: int = 8       # latent space dimension
    hidden_dim: int = 32      # MLP hidden dimension
    T: int = 50               # diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 0.02


class LatentDiffusionModel:
    """
    Latent diffusion model for financial return sequences.

    Architecture:
      encoder  : seq_len â latent_dim
      score_net: latent_dim + 1 (time embedding) â latent_dim  (noise predictor)
      decoder  : latent_dim â seq_len
    """

    def __init__(self, config: LDMConfig, seed: int = 0):
        self.config = config
        rng = random.Random(seed)
        self.schedule = NoiseSchedule(config.T, config.beta_start, config.beta_end)

        # Encoder
        self.encoder = MLP.build(
            [config.seq_len, config.hidden_dim, config.latent_dim],
            activations=["relu", "none"], rng=rng
        )

        # Score network (noise predictor) â takes [z_t || t_embed]
        t_embed_dim = 4
        self.score_net = MLP.build(
            [config.latent_dim + t_embed_dim,
             config.hidden_dim, config.hidden_dim,
             config.latent_dim],
            activations=["relu", "relu", "none"], rng=rng
        )

        # Decoder
        self.decoder = MLP.build(
            [config.latent_dim, config.hidden_dim, config.seq_len],
            activations=["relu", "none"], rng=rng
        )

    def _time_embedding(self, t: int) -> List[float]:
        """Sinusoidal time embedding (4-dim)."""
        freq = [1.0, 10.0, 100.0, 1000.0]
        return [math.sin(t / f) if i % 2 == 0 else math.cos(t / f)
                for i, f in enumerate(freq)]

    def encode(self, x: List[float]) -> List[float]:
        return self.encoder.forward(x)

    def decode(self, z: List[float]) -> List[float]:
        return self.decoder.forward(z)

    def predict_noise(self, z_t: List[float], t: int) -> List[float]:
        t_emb = self._time_embedding(t)
        inp = z_t + t_emb
        return self.score_net.forward(inp)

    def forward_process(self, x: List[float],
                        t: int,
                        rng: random.Random) -> Tuple[List[float], List[float]]:
        """Encode then add noise: returns (z_t, noise)."""
        z0 = self.encode(x)
        z_t, noise = self.schedule.q_sample(z0, t, rng)
        return z_t, noise

    @staticmethod
    def _mse_loss(pred: List[float], target: List[float]) -> float:
        return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(pred)

    def compute_loss(self, x: List[float], rng: random.Random) -> float:
        """Compute diffusion training loss for one sample."""
        t = rng.randint(0, self.config.T - 1)
        z_t, noise = self.forward_process(x, t, rng)
        eps_pred = self.predict_noise(z_t, t)
        return self._mse_loss(eps_pred, noise)

    @staticmethod
    def _sgd_update(layer: LinearLayer, x_inp: List[float],
                    grad_out: List[float], lr: float) -> List[float]:
        """Single SGD step on one layer; returns upstream gradient."""
        n_in = len(layer.W[0])
        n_out = len(layer.W)
        grad_in = [0.0] * n_in
        for j in range(n_out):
            for k in range(n_in):
                layer.W[j][k] -= lr * grad_out[j] * x_inp[k]
            layer.b[j] -= lr * grad_out[j]
            for k in range(n_in):
                grad_in[k] += layer.W[j][k] * grad_out[j]
        return grad_in

    def sample(self, n_steps: Optional[int] = None,
               rng: Optional[random.Random] = None) -> List[float]:
        """Generate a new return sequence via reverse diffusion."""
        rng = rng or random.Random(random.randint(0, 2**31))
        T = n_steps or self.config.T
        d = self.config.latent_dim

        # Start from pure noise
        z_t = [rng.gauss(0, 1) for _ in range(d)]

        for t in range(T - 1, -1, -1):
            eps = self.predict_noise(z_t, t)
            z_t = self.schedule.denoise_step(z_t, eps, t, rng)

        return self.decode(z_t)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _rng_mod
    rng = random.Random(42)

    config = LDMConfig(seq_len=20, latent_dim=8, hidden_dim=32, T=20)
    model = LatentDiffusionModel(config, seed=42)

    # Synthetic training data: AR(1) return sequences
    def generate_ar1(n: int, phi: float = 0.3) -> List[float]:
        seq = [rng.gauss(0, 0.01)]
        for _ in range(n - 1):
            seq.append(phi * seq[-1] + rng.gauss(0, 0.01))
        return seq

    # Quick training demo (few steps, no backprop â just loss monitoring)
    print("Training loss over 10 samples:")
    for i in range(10):
        x = generate_ar1(config.seq_len)
        loss = model.compute_loss(x, rng)
        print(f"  Sample {i+1}: loss = {loss:.6f}")

    # Generate synthetic sequences
    print("\nGenerated return sequences (5 samples):")
    for i in range(5):
        seq = model.sample(rng=random.Random(i))
        mean = sum(seq) / len(seq)
        vol = math.sqrt(sum((r - mean) ** 2 for r in seq) / len(seq))
        print(f"  Sample {i+1}: mean={mean:.5f}, vol={vol:.5f}, "
              f"seq={[round(r, 4) for r in seq[:5]]}...")

    # Encode-decode roundtrip
    x_orig = generate_ar1(config.seq_len, phi=0.5)
    z = model.encode(x_orig)
    x_recon = model.decode(z)
    recon_err = sum((a - b) ** 2 for a, b in zip(x_orig, x_recon)) / len(x_orig)
    print(f"\nEncoder-decoder reconstruction MSE: {recon_err:.8f}")
    print(f"Latent norm: {_norm(z):.4f}")
