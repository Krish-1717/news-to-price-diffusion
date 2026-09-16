"""
ddpm.py — 1-D Denoising Diffusion Probabilistic Model (DDPM) for return distributions.

Day 3: Model architecture.

The DDPM learns the distribution of one-day log-returns conditioned on a
CONTEXT_DIM-dimensional context vector of market and sentiment features.

All context features use .shift(1) (prior-day values) so there is zero
look-ahead. The model outputs a return distribution from which we can
sample or compute moments (mean, VaR, CVaR).

Architecture:
  Forward process q(x_t | x_{t-1}) = N(sqrt(1-β_t) x_{t-1}, β_t I)
  Reverse process p_θ(x_{t-1} | x_t, c) with learned denoiser ε_θ
  Denoiser:  TimeConditionedMLP — a residual MLP that takes
               (x_t, sinusoidal_timestep_embedding, context_projection)
             and predicts the noise ε.

Noise schedule: cosine schedule (Nichol & Dhariwal 2021) — better
  variance preservation than the linear schedule for small T.

References:
  Ho et al. (2020). Denoising Diffusion Probabilistic Models.
  Nichol & Dhariwal (2021). Improved Denoising Diffusion Probabilistic Models.

Usage:
  from models.ddpm import DDPM, DDPMConfig
  cfg = DDPMConfig()
  model = DDPM(cfg)
  loss = model.training_loss(x0, context)   # x0: (B,), context: (B, CONTEXT_DIM)
  samples = model.sample(context, n_samples=500)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DDPMConfig:
    # Diffusion
    T: int = 1000               # number of diffusion steps
    beta_schedule: str = "cosine"  # "cosine" or "linear"
    beta_start: float = 1e-4   # (used only for linear schedule)
    beta_end:   float = 0.02   # (used only for linear schedule)

    # Data / conditioning
    context_dim: int = 17       # CONTEXT_DIM: market + sentiment features
    x_dim: int = 1              # 1-D return

    # Denoiser MLP
    hidden_dim: int = 256
    n_layers: int = 6
    time_emb_dim: int = 64
    context_proj_dim: int = 64
    dropout: float = 0.1

    # Training
    clip_denoised: bool = True
    clip_range: float = 0.20    # clip return predictions to ±20% (daily)


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encode scalar timestep t ∈ {1,...,T} as a fixed sinusoidal embedding,
    then project through a small MLP.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer timesteps in [1, T]
        returns: (B, dim) embedding
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float()[:, None] * freqs[None]   # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


# ── Residual MLP block ────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """One residual layer: Linear → LayerNorm → SiLU → Dropout → Linear + skip."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ── Conditioned denoiser ──────────────────────────────────────────────────────

class TimeConditionedMLP(nn.Module):
    """
    Noise-prediction network ε_θ(x_t, t, context).

    Input:
      x_t      : (B, 1) noisy return at diffusion step t
      t        : (B,)   integer timestep
      context  : (B, context_dim) conditioning features

    Output:
      eps_pred : (B, 1) predicted noise
    """
    def __init__(self, cfg: DDPMConfig):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(cfg.time_emb_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.context_proj_dim),
            nn.SiLU(),
            nn.Linear(cfg.context_proj_dim, cfg.context_proj_dim),
        )

        # Project all inputs into hidden_dim
        in_dim = cfg.x_dim + cfg.time_emb_dim + cfg.context_proj_dim
        self.input_proj = nn.Linear(in_dim, cfg.hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.x_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,       # (B, 1)
        t:   torch.Tensor,        # (B,)
        ctx: torch.Tensor,        # (B, context_dim)
    ) -> torch.Tensor:
        t_emb   = self.time_emb(t)        # (B, time_emb_dim)
        c_emb   = self.context_proj(ctx)  # (B, context_proj_dim)
        h = torch.cat([x_t, t_emb, c_emb], dim=-1)  # (B, in_dim)
        h = self.input_proj(h)            # (B, hidden_dim)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)        # (B, 1)


# ── Noise schedule ────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021, eq. 17).
    Returns β_t for t = 1 ... T, shape (T,).
    Clips β to [1e-5, 0.9999].
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)
    alpha_bar = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return beta.clamp(1e-5, 0.9999)


def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear schedule from Ho et al. (2020)."""
    return torch.linspace(beta_start, beta_end, T)


# ── DDPM ─────────────────────────────────────────────────────────────────────

class DDPM(nn.Module):
    """
    1-D DDPM for return distributions conditioned on context.

    Key methods:
      training_loss(x0, context)     → scalar loss (simple MSE on noise)
      sample(context, n_samples)     → return samples from p_θ(x_0 | c)
      q_sample(x0, t, eps)           → forward diffusion x_t
      p_sample_one_step(x_t, t, ctx) → one reverse step
    """

    def __init__(self, cfg: DDPMConfig):
        super().__init__()
        self.cfg = cfg
        self.denoiser = TimeConditionedMLP(cfg)

        # Build schedule buffers (not parameters)
        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.T)
        else:
            betas = linear_beta_schedule(cfg.T, cfg.beta_start, cfg.beta_end)

        alphas      = 1.0 - betas
        alpha_bar   = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("betas",          betas)
        self.register_buffer("alphas",         alphas)
        self.register_buffer("alpha_bar",      alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_var",  posterior_var)
        self.register_buffer("sqrt_alpha_bar",       alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer("sqrt_recip_alpha",     (1.0 / alphas).sqrt())
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar)
        )

    # ── Forward (noising) process ─────────────────────────────────────────────

    def q_sample(
        self,
        x0:  torch.Tensor,   # (B, 1)
        t:   torch.Tensor,   # (B,) integer in [0, T-1]
        eps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I).
        Returns (x_t, eps).
        """
        if eps is None:
            eps = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_alpha_bar[t][:, None]           # (B, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None]  # (B, 1)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps

    # ── Reverse (denoising) process ───────────────────────────────────────────

    def p_sample_one_step(
        self,
        x_t: torch.Tensor,   # (B, 1)
        t:   torch.Tensor,   # (B,) integer in [0, T-1]
        ctx: torch.Tensor,   # (B, context_dim)
    ) -> torch.Tensor:
        """
        One reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t, context).
        Returns x_{t-1} of shape (B, 1).
        """
        # Predict noise
        eps_pred = self.denoiser(x_t, t, ctx)

        # Predict x_0 from x_t and eps_pred
        coef1 = self.posterior_mean_coef1[t][:, None]  # (B, 1)
        coef2 = self.posterior_mean_coef2[t][:, None]

        x0_pred = (x_t - self.sqrt_one_minus_alpha_bar[t][:, None] * eps_pred)                   / self.sqrt_alpha_bar[t][:, None]

        if self.cfg.clip_denoised:
            x0_pred = x0_pred.clamp(-self.cfg.clip_range, self.cfg.clip_range)

        posterior_mean = coef1 * x0_pred + coef2 * x_t

        # Add noise for t > 0
        noise = torch.randn_like(x_t)
        var   = self.posterior_var[t][:, None].clamp(min=1e-20)
        x_prev = posterior_mean + (t > 0).float()[:, None] * var.sqrt() * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,   # (B, context_dim) or (context_dim,)
        n_samples: int = 500,
    ) -> torch.Tensor:
        """
        Full reverse chain: sample n_samples returns from p_θ(x_0 | context).

        If context is 1-D, it is broadcast to (n_samples, context_dim).
        Returns samples of shape (n_samples, 1).
        """
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(n_samples, -1)

        device = next(self.parameters()).device
        context = context.to(device)
        x = torch.randn(n_samples, self.cfg.x_dim, device=device)

        for step in reversed(range(self.cfg.T)):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            x = self.p_sample_one_step(x, t, context)

        return x  # (n_samples, 1)

    # ── Training loss ─────────────────────────────────────────────────────────

    def training_loss(
        self,
        x0:      torch.Tensor,   # (B, 1)  observed returns
        context: torch.Tensor,   # (B, context_dim)
    ) -> torch.Tensor:
        """
        Simple DDPM objective: E[||ε - ε_θ(x_t, t, context)||²].
        t is sampled uniformly from {0, ..., T-1}.
        Returns a scalar loss.
        """
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.cfg.T, (B,), device=device)  # (B,)
        x_t, eps = self.q_sample(x0, t)
        eps_pred = self.denoiser(x_t, t, context)
        return F.mse_loss(eps_pred, eps)

    # ── Convenience: moments from samples ────────────────────────────────────

    def predictive_moments(
        self,
        context: torch.Tensor,
        n_samples: int = 1000,
        var_level: float = 0.05,  # VaR / CVaR level
    ) -> dict:
        """
        Sample the return distribution and compute moments.
        Returns dict with keys: mean, std, var (VaR), cvar, skew, kurt.
        """
        samples = self.sample(context, n_samples).squeeze(-1)  # (n_samples,)
        q = torch.quantile(samples, var_level)
        cvar_mask = samples <= q
        cvar = samples[cvar_mask].mean() if cvar_mask.any() else q

        n = samples.numel()
        mu   = samples.mean()
        sig  = samples.std(unbiased=True)
        z    = (samples - mu) / (sig + 1e-8)
        skew = (z ** 3).mean()
        kurt = (z ** 4).mean() - 3.0  # excess kurtosis

        return {
            "mean": mu.item(),
            "std":  sig.item(),
            "var":  q.item(),       # Value-at-Risk (lower tail)
            "cvar": cvar.item(),    # Conditional VaR
            "skew": skew.item(),
            "kurt": kurt.item(),
        }
