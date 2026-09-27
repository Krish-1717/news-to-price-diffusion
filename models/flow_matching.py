"""
Flow Matching for Financial Return Generation
Day 16 â news-to-price-diffusion/models/flow_matching.py

Implements Conditional Flow Matching (Lipman et al. 2022):
  - Builds a vector field that transforms Gaussian noise â target distribution
  - Conditions on news sentiment / macro features
  - Enables fast straight-line ODE sampling (fewer steps than diffusion)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _scale(s: float, v: List[float]) -> List[float]:
    return [s * x for x in v]


def _matvec(M: List[List[float]], v: List[float]) -> List[float]:
    return [_dot(row, v) for row in M]


def _relu(x: float) -> float:
    return max(0.0, x)


def _tanh(x: float) -> float:
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e = math.exp(2 * x)
    return (e - 1) / (e + 1)


def _sigmoid(x: float) -> float:
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Conditional optimal transport flow matching
# ---------------------------------------------------------------------------

@dataclass
class FlowMatchingConfig:
    """Configuration for the flow matching model."""
    x_dim: int = 10             # return vector dimension
    cond_dim: int = 8           # conditioning feature dimension
    hidden_dim: int = 64        # MLP hidden width
    n_hidden: int = 3           # number of hidden layers
    sigma_min: float = 1e-4     # minimum noise level
    ode_steps: int = 20         # ODE solver steps for generation
    seed: int = 42


# ---------------------------------------------------------------------------
# Simple MLP (vector field network)
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    W: List[List[float]]
    b: List[float]
    activation: str  # "relu", "tanh", "none"

    @classmethod
    def init(cls, in_d: int, out_d: int, act: str, rng: random.Random) -> "Layer":
        scale = math.sqrt(2.0 / in_d)
        W = [[rng.gauss(0, scale) for _ in range(in_d)] for _ in range(out_d)]
        b = [0.0] * out_d
        return cls(W=W, b=b, activation=act)

    def forward(self, x: List[float]) -> List[float]:
        out = _add(_matvec(self.W, x), self.b)
        if self.activation == "relu":
            return [_relu(v) for v in out]
        if self.activation == "tanh":
            return [_tanh(v) for v in out]
        return out


class VectorFieldNet:
    """
    MLP that approximates the flow matching vector field.
    Input: [x (x_dim), t (time, 1), cond (cond_dim)]
    Output: vector field u_Î¸ (x_dim)
    """

    def __init__(self, config: FlowMatchingConfig):
        rng = random.Random(config.seed)
        in_d = config.x_dim + 1 + config.cond_dim   # x + t + cond
        dims = [in_d] + [config.hidden_dim] * config.n_hidden + [config.x_dim]
        acts = ["relu"] * config.n_hidden + ["none"]
        self.layers = [
            Layer.init(dims[i], dims[i + 1], acts[i], rng)
            for i in range(len(dims) - 1)
        ]

    def forward(self, x: List[float], t: float,
                cond: List[float]) -> List[float]:
        inp = x + [t] + cond
        h = inp
        for layer in self.layers:
            h = layer.forward(h)
        return h


# ---------------------------------------------------------------------------
# Conditional flow matching target
# ---------------------------------------------------------------------------

def cfm_target_field(x0: List[float], x1: List[float], t: float) -> List[float]:
    """
    Conditional flow matching target: u(x|x0, x1, t) = x1 - x0.
    The optimal-transport path is: x_t = (1-t)*x0 + t*x1.
    The constant vector field equals x1 - x0 regardless of t.
    """
    return _sub(x1, x0)


def interpolate(x0: List[float], x1: List[float],
                t: float, sigma_min: float = 1e-4) -> List[float]:
    """
    Stochastic interpolant: x_t = (1-t)*x0 + t*x1 + sigma_min*eps.
    For CFM we use the deterministic version (sigma_min â 0).
    """
    d = len(x0)
    x_t = _add(_scale(1 - t, x0), _scale(t, x1))
    # Add tiny noise for numerical stability
    return x_t


# ---------------------------------------------------------------------------
# Training step (manual gradient approximation via finite difference)
# ---------------------------------------------------------------------------

def _mse(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def _grad_output_mse(pred: List[float], target: List[float]) -> List[float]:
    n = len(pred)
    return [2 * (pred[i] - target[i]) / n for i in range(n)]


def train_step_fd(net: VectorFieldNet,
                   x0: List[float], x1: List[float],
                   cond: List[float], t: float,
                   lr: float = 1e-3, eps: float = 1e-3) -> float:
    """
    Finite-difference gradient step for one (x0, x1, cond, t) sample.
    Returns loss before update.
    """
    x_t = interpolate(x0, x1, t)
    u_target = cfm_target_field(x0, x1, t)
    u_pred = net.forward(x_t, t, cond)
    loss = _mse(u_pred, u_target)

    # Finite-difference gradient w.r.t. each weight
    for layer in net.layers:
        for i in range(len(layer.W)):
            for j in range(len(layer.W[i])):
                layer.W[i][j] += eps
                pred_plus = net.forward(x_t, t, cond)
                loss_plus = _mse(pred_plus, u_target)
                layer.W[i][j] -= 2 * eps
                pred_minus = net.forward(x_t, t, cond)
                loss_minus = _mse(pred_minus, u_target)
                layer.W[i][j] += eps  # restore
                grad = (loss_plus - loss_minus) / (2 * eps)
                layer.W[i][j] -= lr * grad

            layer.b[i] += eps
            loss_plus = _mse(net.forward(x_t, t, cond), u_target)
            layer.b[i] -= 2 * eps
            loss_minus = _mse(net.forward(x_t, t, cond), u_target)
            layer.b[i] += eps
            grad = (loss_plus - loss_minus) / (2 * eps)
            layer.b[i] -= lr * grad

    return loss


# ---------------------------------------------------------------------------
# ODE solver (Euler / RK4) for generation
# ---------------------------------------------------------------------------

def euler_sample(net: VectorFieldNet,
                  cond: List[float],
                  x0: Optional[List[float]] = None,
                  n_steps: int = 20,
                  rng: Optional[random.Random] = None) -> List[float]:
    """
    Sample from model by integrating ODE from t=0 to t=1.
    x_{t+dt} = x_t + dt * u_Î¸(x_t, t, cond)
    """
    rng = rng or random.Random(0)
    d = net.layers[-1].b.__len__()  # output dim = x_dim

    if x0 is None:
        x = [rng.gauss(0, 1) for _ in range(d)]
    else:
        x = x0[:]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        v = net.forward(x, t, cond)
        x = _add(x, _scale(dt, v))

    return x


def rk4_sample(net: VectorFieldNet,
                cond: List[float],
                x0: Optional[List[float]] = None,
                n_steps: int = 10,
                rng: Optional[random.Random] = None) -> List[float]:
    """RK4 ODE solver â more accurate with fewer steps."""
    rng = rng or random.Random(0)
    d = len(net.layers[-1].b)

    if x0 is None:
        x = [rng.gauss(0, 1) for _ in range(d)]
    else:
        x = x0[:]

    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        k1 = net.forward(x, t, cond)
        k2 = net.forward(_add(x, _scale(dt / 2, k1)), t + dt / 2, cond)
        k3 = net.forward(_add(x, _scale(dt / 2, k2)), t + dt / 2, cond)
        k4 = net.forward(_add(x, _scale(dt, k3)), t + dt, cond)
        x = _add(x, _scale(dt / 6, _add(_add(_add(k1, _scale(2, k2)),
                                              _scale(2, k3)), k4)))
    return x


# ---------------------------------------------------------------------------
# Conditional generation utilities
# ---------------------------------------------------------------------------

def encode_news_condition(sentiment: float, topic_probs: List[float],
                           macro_features: Optional[List[float]] = None) -> List[float]:
    """
    Encode news/macro features into a conditioning vector.
    Returns a fixed-length vector.
    """
    cond = [sentiment] + topic_probs[:5]  # first 5 topics
    if macro_features:
        cond += macro_features[:2]
    # Pad or truncate to 8 dims
    while len(cond) < 8:
        cond.append(0.0)
    return cond[:8]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = random.Random(42)
    config = FlowMatchingConfig(x_dim=5, cond_dim=8, hidden_dim=16, n_hidden=2,
                                 ode_steps=10)
    net = VectorFieldNet(config)

    # Synthetic training data: returns from two regimes
    def bull_returns() -> List[float]:
        return [rng.gauss(0.002, 0.01) for _ in range(config.x_dim)]

    def bear_returns() -> List[float]:
        return [rng.gauss(-0.002, 0.02) for _ in range(config.x_dim)]

    def bull_cond() -> List[float]:
        return encode_news_condition(0.7, [0.4, 0.1, 0.2, 0.1, 0.2])

    def bear_cond() -> List[float]:
        return encode_news_condition(-0.6, [0.1, 0.5, 0.1, 0.2, 0.1])

    print("Training flow matching model (10 steps)...")
    losses = []
    for i in range(10):
        is_bull = rng.random() > 0.4
        x1 = bull_returns() if is_bull else bear_returns()
        cond = bull_cond() if is_bull else bear_cond()
        x0 = [rng.gauss(0, 1) for _ in range(config.x_dim)]
        t = rng.random()
        loss = train_step_fd(net, x0, x1, cond, t, lr=0.005)
        losses.append(loss)

    print(f"  Initial loss: {losses[0]:.4f}  Final loss: {losses[-1]:.4f}")

    print("\nGenerating samples:")
    for label, cond_fn in [("Bull", bull_cond), ("Bear", bear_cond)]:
        cond = cond_fn()
        sample = rk4_sample(net, cond, n_steps=10, rng=random.Random(0))
        mean_ret = sum(sample) / len(sample)
        print(f"  {label}: mean_return={mean_ret:.5f}  "
              f"sample={[round(r, 4) for r in sample]}")
