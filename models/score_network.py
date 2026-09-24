"""
models/score_network.py -- Score function / noise-prediction network for news-to-price-diffusion
Day 13: Diffusion model core -- noise schedule, forward process, score network architecture,
        denoising loss, and reverse-process sampler. Pure Python + math only (no PyTorch/numpy).

Design: DDPM-style score network
  - Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
  - Score network: epsilon_theta(x_t, t, c)  [c = news conditioning vector]
  - Training loss: L = E[||epsilon - epsilon_theta(x_t, t, c)||^2]
  - Reverse: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta) + sigma_t * z
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

@dataclass
class NoiseSchedule:
    """
    Linear or cosine noise schedule for DDPM.

    Attributes
    ----------
    T : total diffusion steps
    beta : list of beta_t values (variance of each forward step)
    alpha : 1 - beta_t
    alpha_bar : cumulative product of alpha_t
    """
    T: int
    beta: List[float] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)
    alpha_bar: List[float] = field(default_factory=list)

    @classmethod
    def linear(cls, T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> "NoiseSchedule":
        """Linear schedule from Ho et al. (2020) DDPM."""
        betas = [beta_start + (beta_end - beta_start) * t / (T - 1) for t in range(T)]
        alphas = [1.0 - b for b in betas]
        alpha_bars = []
        cumulative = 1.0
        for a in alphas:
            cumulative *= a
            alpha_bars.append(cumulative)
        return cls(T=T, beta=betas, alpha=alphas, alpha_bar=alpha_bars)

    @classmethod
    def cosine(cls, T: int, s: float = 0.008) -> "NoiseSchedule":
        """
        Cosine schedule from Nichol & Dhariwal (2021).
        Produces smoother alpha_bar trajectory that avoids near-zero
        values at small t (better for high-resolution data).
        """
        def f(t: int) -> float:
            return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2

        alpha_bars = [f(t) / f(0) for t in range(T + 1)]
        betas = [min(1.0 - alpha_bars[t] / alpha_bars[t - 1], 0.999)
                 for t in range(1, T + 1)]
        alphas = [1.0 - b for b in betas]
        alpha_bars_t = alpha_bars[1:]
        return cls(T=T, beta=betas, alpha=alphas, alpha_bar=alpha_bars_t)

    def sigma(self, t: int) -> float:
        """Posterior std at step t (for reverse process sampling)."""
        if t == 0:
            return 0.0
        ab_prev = self.alpha_bar[t - 1] if t > 0 else 1.0
        return math.sqrt(self.beta[t] * (1.0 - ab_prev) / (1.0 - self.alpha_bar[t]))

    def snr(self, t: int) -> float:
        """Signal-to-noise ratio: alpha_bar_t / (1 - alpha_bar_t)."""
        ab = self.alpha_bar[t]
        return ab / (1.0 - ab + 1e-12)


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def _randn(size: int, rng: random.Random) -> List[float]:
    """Sample from N(0,1) using Box-Muller."""
    result = []
    for _ in range((size + 1) // 2):
        u1 = rng.random() + 1e-12
        u2 = rng.random()
        z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        result.extend([z1, z2])
    return result[:size]


def q_sample(
    x0: List[float],
    t: int,
    schedule: NoiseSchedule,
    rng: random.Random,
) -> Tuple[List[float], List[float]]:
    """
    Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I).

    Returns
    -------
    x_t : noisy sample at step t
    epsilon : the noise that was added (used as training target)
    """
    ab = schedule.alpha_bar[t]
    sqrt_ab = math.sqrt(ab)
    sqrt_one_minus_ab = math.sqrt(1.0 - ab)
    epsilon = _randn(len(x0), rng)
    x_t = [sqrt_ab * x + sqrt_one_minus_ab * e for x, e in zip(x0, epsilon)]
    return x_t, epsilon


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: int, dim: int) -> List[float]:
    """
    Sinusoidal positional embedding for diffusion timestep t.
    Returns a vector of length `dim`.
    """
    half = dim // 2
    freqs = [math.exp(-math.log(10000) * i / (half - 1)) for i in range(half)]
    emb = []
    for f in freqs:
        emb.append(math.sin(t * f))
        emb.append(math.cos(t * f))
    return emb[:dim]


# ---------------------------------------------------------------------------
# Minimal MLP score network (no autograd -- forward pass only)
# ---------------------------------------------------------------------------

def _relu(x: float) -> float:
    return max(0.0, x)


def _linear(x: List[float], W: List[List[float]], b: List[float]) -> List[float]:
    """Dense layer: y = Wx + b."""
    return [sum(W[i][j] * x[j] for j in range(len(x))) + b[i]
            for i in range(len(W))]


@dataclass
class ScoreNetworkConfig:
    """Configuration for the score network MLP."""
    data_dim: int = 11
    cond_dim: int = 64
    time_emb_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    T: int = 1000


class ScoreNetwork:
    """
    Epsilon-prediction network for DDPM.

    Architecture:
      input = [x_t (data_dim), time_emb (time_emb_dim), cond (cond_dim)]
      -> MLP with ReLU activations
      -> output: epsilon_hat (data_dim)
    """

    def __init__(self, config: ScoreNetworkConfig, rng: Optional[random.Random] = None):
        self.config = config
        self._rng = rng or random.Random(0)
        self._init_weights()

    def _glorot(self, fan_in: int, fan_out: int) -> List[List[float]]:
        """Xavier uniform initialisation."""
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return [[self._rng.uniform(-limit, limit) for _ in range(fan_in)]
                for _ in range(fan_out)]

    def _init_weights(self):
        cfg = self.config
        in_dim = cfg.data_dim + cfg.time_emb_dim + cfg.cond_dim
        dims = [in_dim] + cfg.hidden_dims + [cfg.data_dim]
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []
        for i in range(len(dims) - 1):
            self.weights.append(self._glorot(dims[i], dims[i + 1]))
            self.biases.append([0.0] * dims[i + 1])

    def forward(
        self,
        x_t: List[float],
        t: int,
        cond: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Predict epsilon (noise) given noisy input x_t, timestep t, and
        optional conditioning vector cond (e.g. news embedding).
        """
        cfg = self.config
        t_emb = sinusoidal_embedding(t, cfg.time_emb_dim)
        if cond is None:
            cond = [0.0] * cfg.cond_dim
        h = x_t + t_emb + cond
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = _linear(h, W, b)
            if i < len(self.weights) - 1:
                h = [_relu(v) for v in h]
        return h

    def ddpm_loss(
        self,
        x0: List[float],
        t: int,
        schedule: NoiseSchedule,
        cond: Optional[List[float]] = None,
        rng: Optional[random.Random] = None,
    ) -> float:
        """
        Compute DDPM MSE loss for one sample.
        L_t = ||epsilon - epsilon_theta(x_t, t, c)||^2
        """
        rng = rng or self._rng
        x_t, epsilon = q_sample(x0, t, schedule, rng)
        epsilon_hat = self.forward(x_t, t, cond)
        return sum((e - eh) ** 2 for e, eh in zip(epsilon, epsilon_hat)) / len(epsilon)


# ---------------------------------------------------------------------------
# Reverse process sampler (DDPM ancestral sampling)
# ---------------------------------------------------------------------------

class DDPMSampler:
    """DDPM reverse-process sampler."""

    def __init__(self, schedule: NoiseSchedule, network: ScoreNetwork):
        self.schedule = schedule
        self.network = network

    def sample(
        self,
        cond: Optional[List[float]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[float]:
        """Generate one sample by running the full reverse chain x_T -> x_0."""
        rng = rng or random.Random()
        cfg = self.network.config
        sch = self.schedule
        x = _randn(cfg.data_dim, rng)
        for t in range(sch.T - 1, -1, -1):
            eps_hat = self.network.forward(x, t, cond)
            ab = sch.alpha_bar[t]
            sqrt_1mab = math.sqrt(1.0 - ab)
            coef = sch.beta[t] / sqrt_1mab
            mean = [(1.0 / math.sqrt(sch.alpha[t])) * (xi - coef * ei)
                    for xi, ei in zip(x, eps_hat)]
            if t > 0:
                sigma = sch.sigma(t)
                z = _randn(cfg.data_dim, rng)
                x = [m + sigma * zi for m, zi in zip(mean, z)]
            else:
                x = mean
        return x

    def ddim_sample(
        self,
        steps: int = 50,
        cond: Optional[List[float]] = None,
        rng: Optional[random.Random] = None,
        eta: float = 0.0,
    ) -> List[float]:
        """
        DDIM deterministic sampler (Song et al. 2020).
        eta=0 -> fully deterministic; eta=1 -> matches DDPM variance.
        """
        rng = rng or random.Random()
        cfg = self.network.config
        sch = self.schedule
        stride = max(1, sch.T // steps)
        ts = list(range(sch.T - 1, -1, -stride))[:steps]
        x = _randn(cfg.data_dim, rng)
        for i, t in enumerate(ts):
            ab_t = sch.alpha_bar[t]
            ab_prev = sch.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else 1.0
            eps_hat = self.network.forward(x, t, cond)
            x0_hat = [(xi - math.sqrt(1 - ab_t) * ei) / math.sqrt(ab_t)
                      for xi, ei in zip(x, eps_hat)]
            sigma = eta * math.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            dir_xt = [math.sqrt(max(0, 1 - ab_prev - sigma ** 2)) * ei for ei in eps_hat]
            noise = [sigma * zi for zi in _randn(cfg.data_dim, rng)] if eta > 0 else [0.0] * cfg.data_dim
            x = [math.sqrt(ab_prev) * x0i + dxi + ni
                 for x0i, dxi, ni in zip(x0_hat, dir_xt, noise)]
        return x


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    cfg = ScoreNetworkConfig(data_dim=11, cond_dim=64, time_emb_dim=32,
                              hidden_dims=[128, 128, 64], T=100)
    schedule = NoiseSchedule.cosine(T=cfg.T)
    net = ScoreNetwork(cfg, rng=rng)

    print("=" * 55)
    print("  SCORE NETWORK -- forward pass & loss demo")
    print("=" * 55)

    x0 = [rng.gauss(0, 1) for _ in range(cfg.data_dim)]
    cond = [rng.gauss(0, 0.1) for _ in range(cfg.cond_dim)]

    print("  Forward process noise levels:")
    for t in [0, 10, 25, 50, 75, 99]:
        ab = schedule.alpha_bar[]
        snr_db = 10 * math.log10(schedule.snr(t) + 1e-12)
        print(f"    t={t:>3}  alpha_bar={ab:.4f}  SNR={snr_db:+1f}dB")

    print()
    print("  DDPM training loss at random t:")
    for _ in range(5):
        t = rng.randint(0, cfg.T - 1)
        loss = net.ddpm_loss(x0, t, schedule, cond, rng)
        print(f"    t={t:>3}  loss={loss:.4f}")

    print()
    sampler = DDPMSampler(schedule, net)
    sample = sampler.ddim_sample(steps=20, cond=cond, rng=rng)
    print(f"  DDIM sample (20 steps): mean={sum(sample)/len(sample):.4f}")
    print("=" * 55)
