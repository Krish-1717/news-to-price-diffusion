"""
DDPM for return distributions, conditioned on market features + sentiment.

References: Ho et al. NeurIPS 2020 (https://arxiv.org/abs/2006.11239)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule — keeps noise low near t=0 and t=T."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1.0 - ac[1:] / ac[:-1]
    return torch.clamp(betas, min=1e-5, max=0.999)


class SinusoidalPosEmb(nn.Module):
    """Encodes timestep t as a sinusoidal embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalDenoiser(nn.Module):
    """
    MLP denoiser: (x_t, t, context) -> epsilon_pred

    context_dim = n_market_features + 1 (sentiment)
    """
    def __init__(self, context_dim: int, time_emb_dim: int = 64,
                 hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        in_dim = 1 + time_emb_dim + context_dim
        layers = []
        for i in range(n_layers):
            out = hidden_dim if i < n_layers - 1 else hidden_dim // 2
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, out), nn.GELU()]
        layers.append(nn.Linear(hidden_dim // 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.float())
        x_in = torch.cat([x_t.unsqueeze(-1), t_emb, context], dim=-1)
        return self.net(x_in).squeeze(-1)


class DDPM(nn.Module):
    """
    Full DDPM with cosine schedule, conditional on a context vector.

    Usage:
        ddpm = DDPM(context_dim=17)
        loss = ddpm.loss(x0, context)         # training
        samples = ddpm.sample(context, n=500) # inference -> return distribution
    """

    def __init__(self, context_dim: int, T: int = 200, schedule: str = "cosine",
                 hidden_dim: int = 256, n_layers: int = 4, time_emb_dim: int = 64):
        super().__init__()
        self.T = T
        betas = cosine_beta_schedule(T) if schedule == "cosine" else                 torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", ac)
        self.register_buffer("alphas_cumprod_prev", ac_prev)
        self.register_buffer("sqrt_alphas_cumprod", ac.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - ac).sqrt())
        self.register_buffer("posterior_variance", betas * (1.0 - ac_prev) / (1.0 - ac))
        self.denoiser = ConditionalDenoiser(context_dim, hidden_dim=hidden_dim,
                                            n_layers=n_layers, time_emb_dim=time_emb_dim)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return self.sqrt_alphas_cumprod[t] * x0 + self.sqrt_one_minus_alphas_cumprod[t] * noise

    def loss(self, x0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        return F.mse_loss(self.denoiser(x_t, t, context), noise)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, n_samples: int = 500) -> torch.Tensor:
        """Returns (B, n_samples) return distribution."""
        device = next(self.parameters()).device
        B = context.shape[0]
        ctx = context.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)
        x = torch.randn(B * n_samples, device=device)
        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((B * n_samples,), t_idx, device=device, dtype=torch.long)
            eps = self.denoiser(x, t_batch, ctx)
            mean = (1.0 / self.alphas[t_idx].sqrt()) * (
                x - self.betas[t_idx] / self.sqrt_one_minus_alphas_cumprod[t_idx] * eps)
            if t_idx > 0:
                x = mean + self.posterior_variance[t_idx].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.reshape(B, n_samples)
