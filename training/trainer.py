"""
training/trainer.py -- DDPM training loop for news-to-price-diffusion
Day 13: Mini-batch SGD, gradient clipping, learning-rate schedule, checkpoint saving.
Pure Python + stdlib; no frameworks required.
"""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyper-parameters for the DDPM training loop."""
    learning_rate: float = 1e-3
    lr_warmup_steps: int = 500
    lr_decay_steps: int = 10_000
    gradient_clip: float = 1.0
    batch_size: int = 32
    max_steps: int = 20_000
    eval_every: int = 500
    save_every: int = 2_000
    seed: int = 42
    weight_decay: float = 1e-4


# ---------------------------------------------------------------------------
# Parameter store (flat dict of param tensors + moment estimates for Adam)
# ---------------------------------------------------------------------------

class AdamOptimiser:
    """
    Adam with optional weight decay (AdamW-style).
    Works on a flat list of (param, grad) pairs where each is a list[float].
    """
    def __init__(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self._m: List[List[float]] = []
        self._v: List[List[float]] = []
        self._step = 0
        self._initialised = False

    def _init(self, params: List[List[float]]) -> None:
        self._m = [[0.0] * len(p) for p in params]
        self._v = [[0.0] * len(p) for p in params]
        self._initialised = True

    def step(self, params: List[List[float]], grads: List[List[float]]) -> None:
        """Update params in-place given gradients."""
        if not self._initialised:
            self._init(params)
        self._step += 1
        t = self._step
        bc1 = 1 - self.b1 ** t
        bc2 = 1 - self.b2 ** t
        for i, (p, g) in enumerate(zip(params, grads)):
            for j in range(len(p)):
                gj = g[j] + self.wd * p[j]   # weight decay
                self._m[i][j] = self.b1 * self._m[i][j] + (1 - self.b1) * gj
                self._v[i][j] = self.b2 * self._v[i][j] + (1 - self.b2) * gj * gj
                m_hat = self._m[i][j] / bc1
                v_hat = self._v[i][j] / bc2
                p[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


def clip_grad_norm(grads: List[List[float]], max_norm: float) -> float:
    """Clip gradient lists to max L2 norm. Returns pre-clip norm."""
    total_sq = sum(g_i ** 2 for g in grads for g_i in g)
    norm = math.sqrt(total_sq) + 1e-12
    if norm > max_norm:
        scale = max_norm / norm
        for g in grads:
            for j in range(len(g)):
                g[j] *= scale
    return norm


# ---------------------------------------------------------------------------
# Learning-rate schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------

def lr_schedule(step: int, base_lr: float, warmup_steps: int, decay_steps: int) -> float:
    """
    Linear warm-up for `warmup_steps`, then cosine decay to 0 over `decay_steps`.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(decay_steps - warmup_steps, 1), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Minimal dataset abstraction
# ---------------------------------------------------------------------------

@dataclass
class NewsReturnSample:
    """One training example: (news_cond_vector, next_day_return_vector)."""
    cond: List[float]       # news conditioning from NewsEncoder, shape (cond_dim,)
    x0: List[float]         # normalised return features, shape (data_dim,)


class DataLoader:
    """Shuffle + batch a list of NewsReturnSample."""
    def __init__(self, dataset: List[NewsReturnSample], batch_size: int, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self._rng = random.Random(seed)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        self._rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield [self.dataset[i] for i in batch_idx]

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------

@dataclass
class TrainMetrics:
    """Accumulated metrics for one logging interval."""
    loss_sum: float = 0.0
    grad_norm_sum: float = 0.0
    n_steps: int = 0
    elapsed_sec: float = 0.0

    def update(self, loss: float, grad_norm: float, dt: float) -> None:
        self.loss_sum += loss
        self.grad_norm_sum += grad_norm
        self.n_steps += 1
        self.elapsed_sec += dt

    def summarise(self) -> Dict[str, float]:
        n = max(self.n_steps, 1)
        return {
            "loss": self.loss_sum / n,
            "grad_norm": self.grad_norm_sum / n,
            "steps_per_sec": n / max(self.elapsed_sec, 1e-9),
        }

    def reset(self) -> None:
        self.loss_sum = self.grad_norm_sum = self.elapsed_sec = 0.0
        self.n_steps = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    DDPM training loop.

    The network and schedule objects must expose:
      - network.ddpm_loss(x0_batch, t_batch, schedule, cond_batch, rng) -> float
      - network._params() -> List[List[float]]          (all trainable params)
      - network._grads()  -> List[List[float]]          (corresponding gradients)
      - schedule.T        -> int                        (number of diffusion steps)

    In practice these are filled by the ScoreNetwork from models/score_network.py.
    """

    def __init__(
        self,
        network,
        schedule,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        log_fn: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ):
        self.net = network
        self.sched = schedule
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config or TrainerConfig()
        self.log_fn = log_fn or (lambda step, metrics: print(f"  step {step:6d} | " +
                                  " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())))
        self._rng = random.Random(self.cfg.seed)
        self._optimiser = AdamOptimiser(
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.global_step = 0
        self.best_val_loss = float("inf")
        self._checkpoints: List[Dict] = []

    def _sample_t(self, batch_size: int) -> List[int]:
        """Sample random diffusion timesteps uniformly in [1, T]."""
        return [self._rng.randint(1, self.sched.T) for _ in range(batch_size)]

    def _train_step(self, batch: List[NewsReturnSample]) -> Tuple[float, float]:
        """Run one forward+backward step. Returns (loss, grad_norm)."""
        t_batch = self._sample_t(len(batch))
        x0_batch = [s.x0 for s in batch]
        cond_batch = [s.cond for s in batch]

        # Forward: compute DDPM loss (calls network's own backward internally
        # in a real framework; here we rely on the network exposing _grads())
        loss = self.net.ddpm_loss(x0_batch, t_batch, self.sched, cond_batch, self._rng)

        params = self.net._params()
        grads = self.net._grads()
        grad_norm = clip_grad_norm(grads, self.cfg.gradient_clip)

        # LR schedule
        self._optimiser.lr = lr_schedule(
            self.global_step,
            self.cfg.learning_rate,
            self.cfg.lr_warmup_steps,
            self.cfg.lr_decay_steps,
        )
        self._optimiser.step(params, grads)
        return loss, grad_norm

    def _eval(self) -> float:
        """Compute mean DDPM loss on val set (no gradient)."""
        if self.val_loader is None:
            return float("nan")
        total, n = 0.0, 0
        for batch in self.val_loader:
            t_batch = self._sample_t(len(batch))
            x0_batch = [s.x0 for s in batch]
            cond_batch = [s.cond for s in batch]
            total += self.net.ddpm_loss(x0_batch, t_batch, self.sched, cond_batch, self._rng)
            n += 1
        return total / max(n, 1)

    def _save_checkpoint(self, val_loss: float) -> None:
        """Save a minimal in-memory checkpoint."""
        ckpt = {
            "step": self.global_step,
            "val_loss": val_loss,
            "lr": self._optimiser.lr,
        }
        self._checkpoints.append(ckpt)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

    def train(self) -> None:
        """Main training loop."""
        metrics = TrainMetrics()
        step = 0

        while step < self.cfg.max_steps:
            for batch in self.train_loader:
                if step >= self.cfg.max_steps:
                    break
                t0 = time.time()
                loss, gnorm = self._train_step(batch)
                dt = time.time() - t0
                metrics.update(loss, gnorm, dt)
                self.global_step = step = step + 1

                if step % self.cfg.eval_every == 0:
                    summary = metrics.summarise()
                    summary["val_loss"] = self._eval()
                    summary["lr"] = self._optimiser.lr
                    self.log_fn(step, summary)
                    metrics.reset()

                if step % self.cfg.save_every == 0:
                    val_loss = self._eval()
                    self._save_checkpoint(val_loss)


# ---------------------------------------------------------------------------
# CLI demo (synthetic data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    print("=" * 55)
    print("  TRAINER -- synthetic data smoke-test")
    print("=" * 55)

    DATA_DIM, COND_DIM = 4, 8
    N_TRAIN, N_VAL = 128, 32
    rng = random.Random(0)

    def rand_vec(d):
        return [rng.gauss(0, 1) for _ in range(d)]

    train_data = [NewsReturnSample(rand_vec(COND_DIM), rand_vec(DATA_DIM)) for _ in range(N_TRAIN)]
    val_data   = [NewsReturnSample(rand_vec(COND_DIM), rand_vec(DATA_DIM)) for _ in range(N_VAL)]

    train_loader = DataLoader(train_data, batch_size=16, seed=1)
    val_loader   = DataLoader(val_data,   batch_size=16, seed=2)

    cfg = TrainerConfig(max_steps=100, eval_every=50, save_every=100,
                        learning_rate=1e-3, batch_size=16)

    # Stub network and schedule for smoke-test
    class _StubNet:
        def ddpm_loss(self, x0, t, sched, cond, rng):
            return rng.uniform(0.5, 1.5)
        def _params(self): return [[0.0]]
        def _grads(self): return [[0.0]]

    class _StubSched:
        T = 100

    trainer = Trainer(_StubNet(), _StubSched(), train_loader, val_loader, cfg)
    trainer.train()
    print(f"  Done. Best val loss: {trainer.best_val_loss:.4f}")
    print("=" * 55)
