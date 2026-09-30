"""
news_day23_latent_factor_diffusion.py
Day 23: Latent Factor Diffusion — SVD-based factor extraction from news embeddings,
factor-conditioned score network, factor loading matrix, orthogonalization,
and conditional generation from news factor states.
Pure Python stdlib only.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers: vector and matrix ops (no numpy)
# ---------------------------------------------------------------------------

def dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))

def mat_vec(M: list[list[float]], v: list[float]) -> list[float]:
    return [dot(row, v) for row in M]

def vec_scale(v: list[float], s: float) -> list[float]:
    return [vi * s for vi in v]

def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]

def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [ai - bi for ai, bi in zip(a, b)]

def norm(v: list[float]) -> float:
    return math.sqrt(sum(vi**2 for vi in v))

def normalize(v: list[float]) -> list[float]:
    n = norm(v)
    return [vi / n for vi in v] if n > 1e-10 else v

def outer(a: list[float], b: list[float]) -> list[list[float]]:
    return [[ai * bj for bj in b] for ai in a]

def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


# ---------------------------------------------------------------------------
# Power iteration SVD (thin SVD for embedding matrix)
# ---------------------------------------------------------------------------

def power_iteration_svd(
    X: list[list[float]],   # T x D matrix
    k: int = 5,
    n_iter: int = 50,
    seed: int = 42,
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """
    Thin SVD via randomized power iteration.
    Returns (U, S, V^T) where X ≈ U * diag(S) * V^T.
    U: T x k, S: k, V^T: k x D.
    """
    rng = random.Random(seed)
    T = len(X)
    D = len(X[0]) if T > 0 else 0

    # Compute X^T X (D x D) via outer products
    # Too slow for large D; use randomized approach
    # Initialize random k-dimensional subspace
    U = []  # orthogonal vectors in R^T
    S = []
    VT = []  # vectors in R^D

    # Deflation approach: find top-k singular vectors
    X_residual = [row[:] for row in X]  # copy

    for comp in range(k):
        # Random init vector in R^D
        v = normalize([rng.gauss(0, 1) for _ in range(D)])

        for _ in range(n_iter):
            # u = X v / ||X v||
            u_raw = [dot(X_residual[t], v) for t in range(T)]
            u = normalize(u_raw)

            # v = X^T u / ||X^T u||
            v_raw = [sum(X_residual[t][d] * u[t] for t in range(T)) for d in range(D)]
            v = normalize(v_raw)

        # Singular value: sigma = u^T X v
        sigma = sum(u[t] * dot(X_residual[t], v) for t in range(T))
        sigma = abs(sigma)

        U.append(u)
        S.append(sigma)
        VT.append(v)

        # Deflate: X <- X - sigma * u * v^T
        for t in range(T):
            for d in range(D):
                X_residual[t][d] -= sigma * u[t] * v[d]

    # Return U as T x k (transpose of list), VT as k x D
    return U, S, VT


# ---------------------------------------------------------------------------
# Factor extraction from news embeddings
# ---------------------------------------------------------------------------

@dataclass
class NewsEmbedding:
    """Simulated news embedding (D-dimensional)."""
    text: str
    embedding: list[float]
    timestamp: float
    asset: str


@dataclass
class FactorModel:
    """Learned factor model from news embeddings."""
    factor_vecs: list[list[float]]    # k x D (V^T rows)
    singular_values: list[float]       # k
    explained_variance: list[float]    # k
    total_variance: float
    n_factors: int


def extract_news_factors(
    embeddings: list[list[float]],  # T x D
    n_factors: int = 5,
) -> FactorModel:
    """Extract latent factors from news embedding matrix via SVD."""
    T = len(embeddings)
    D = len(embeddings[0]) if T > 0 else 0

    # Center embeddings
    means = [sum(embeddings[t][d] for t in range(T)) / T for d in range(D)]
    centered = [[embeddings[t][d] - means[d] for d in range(D)] for t in range(T)]

    U, S, VT = power_iteration_svd(centered, k=n_factors)

    # Explained variance = S_i^2 / sum(S^2)
    total_var = sum(s**2 for s in S)
    explained = [s**2 / max(total_var, 1e-10) for s in S]

    return FactorModel(
        factor_vecs=VT,
        singular_values=S,
        explained_variance=explained,
        total_variance=total_var,
        n_factors=n_factors,
    )


def project_to_factors(
    embedding: list[float],
    factor_model: FactorModel,
) -> list[float]:
    """Project a news embedding onto the learned factor space. Returns factor scores."""
    return [dot(factor_model.factor_vecs[k], embedding) for k in range(factor_model.n_factors)]


def orthogonalize_factors(factor_scores: list[float], cov_factors: list[list[float]]) -> list[float]:
    """
    Gram-Schmidt orthogonalization of factor scores w.r.t. estimated factor covariance.
    Useful for making factors uncorrelated in factor-conditioned generation.
    """
    # Simplified: use Cholesky-like sequential orthogonalization
    n = len(factor_scores)
    ortho = factor_scores.copy()

    for i in range(1, n):
        for j in range(i):
            proj = sum(cov_factors[i][k] * ortho[k] for k in range(n))
            norm_j_sq = cov_factors[j][j]
            if norm_j_sq > 1e-10:
                ortho[i] -= (proj / norm_j_sq) * ortho[j]

    return ortho


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> list[float]:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    def f(t):
        return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    f0 = f(0)
    for t in range(1, T + 1):
        alpha_bar_t = f(t) / f0
        alpha_bar_prev = f(t - 1) / f0
        beta_t = 1 - alpha_bar_t / max(alpha_bar_prev, 1e-10)
        betas.append(min(max(beta_t, 1e-5), 0.999))
    return betas


def alpha_bar(betas: list[float]) -> list[float]:
    """Cumulative product of (1 - beta_t)."""
    result = []
    prod = 1.0
    for b in betas:
        prod *= (1 - b)
        result.append(prod)
    return result


# ---------------------------------------------------------------------------
# Factor-conditioned score network
# ---------------------------------------------------------------------------

class FactorConditionedScoreNet:
    """
    A simple MLP-based score network conditioned on factor state.
    Architecture: [x_t, t_embed, factor_embed] -> score prediction.
    Implemented as a linear model for pure-Python compatibility.
    """

    def __init__(self, d_data: int = 8, n_factors: int = 5, seed: int = 0):
        rng = random.Random(seed)
        self.d_data = d_data
        self.n_factors = n_factors

        # Linear weights: W: (d_data + n_factors + 1) -> d_data
        d_in = d_data + n_factors + 1  # +1 for time embedding
        self.W1 = [[rng.gauss(0, 0.1) for _ in range(d_in)] for _ in range(d_data * 2)]
        self.b1 = [0.0] * (d_data * 2)
        self.W2 = [[rng.gauss(0, 0.1) for _ in range(d_data * 2)] for _ in range(d_data)]
        self.b2 = [0.0] * d_data

    def _silu(self, x: list[float]) -> list[float]:
        return [xi / (1 + math.exp(-xi)) for xi in x]

    def forward(self, x_t: list[float], t: float, factors: list[float]) -> list[float]:
        """Predict score (noise) from noisy data, time, and factor conditioning."""
        # Build input
        inp = x_t + factors + [t]

        # Layer 1
        h = [sum(self.W1[j][i] * inp[i] for i in range(len(inp))) + self.b1[j]
             for j in range(len(self.b1))]
        h = self._silu(h)

        # Layer 2
        out = [sum(self.W2[j][i] * h[i] for i in range(len(h))) + self.b2[j]
               for j in range(self.d_data)]
        return out

    def score(self, x_t: list[float], t: float, factors: list[float],
              alpha_bar_t: float) -> list[float]:
        """Score function: score = -eps_theta / sqrt(1 - alpha_bar)."""
        eps = self.forward(x_t, t, factors)
        scale = -1.0 / max(math.sqrt(1 - alpha_bar_t), 1e-8)
        return [scale * e for e in eps]

    def update_weights(self, grad_W1: list[list[float]], grad_W2: list[list[float]],
                       grad_b1: list[float], grad_b2: list[float], lr: float = 1e-3):
        """SGD weight update."""
        for j in range(len(self.W1)):
            for i in range(len(self.W1[0])):
                self.W1[j][i] -= lr * grad_W1[j][i]
            self.b1[j] -= lr * grad_b1[j]
        for j in range(len(self.W2)):
            for i in range(len(self.W2[0])):
                self.W2[j][i] -= lr * grad_W2[j][i]
            self.b2[j] -= lr * grad_b2[j]


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    net: FactorConditionedScoreNet,
    x0: list[float],
    factors: list[float],
    betas: list[float],
    alpha_bars: list[float],
    lr: float = 1e-3,
    rng: random.Random = None,
) -> float:
    """Single training step: diffuse x0 and predict noise."""
    if rng is None:
        rng = random.Random()

    T = len(betas)
    t_idx = rng.randint(0, T - 1)
    ab = alpha_bars[t_idx]

    # Forward diffusion: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps
    eps_true = [rng.gauss(0, 1) for _ in range(len(x0))]
    x_t = [math.sqrt(ab) * x0[i] + math.sqrt(1 - ab) * eps_true[i] for i in range(len(x0))]

    # Predict noise
    t_norm = t_idx / T
    eps_pred = net.forward(x_t, t_norm, factors)

    # MSE loss gradient
    loss = sum((eps_pred[i] - eps_true[i])**2 for i in range(len(x0))) / len(x0)

    # Backprop through linear net (simplified)
    grad_out = [(eps_pred[i] - eps_true[i]) * 2 / len(x0) for i in range(len(x0))]

    # Layer 2 gradients
    inp_l2 = [sum(net.W1[j][i] * (x_t + factors + [t_norm])[i] for i in range(len(x_t) + len(factors) + 1)) + net.b1[j]
              for j in range(len(net.b1))]
    h = [xi / (1 + math.exp(-xi)) for xi in inp_l2]

    grad_W2 = [[grad_out[j] * h[i] for i in range(len(h))] for j in range(len(grad_out))]
    grad_b2 = grad_out[:]

    # Layer 1 gradients (approximate)
    grad_h = [sum(net.W2[j][i] * grad_out[j] for j in range(len(grad_out))) for i in range(len(h))]
    silu_grad = [math.sigmoid(xi) * (1 + xi * (1 - math.sigmoid(xi))) if hasattr(math, 'sigmoid') else
                 (1 / (1 + math.exp(-xi))) * (1 + xi * (1 - 1 / (1 + math.exp(-xi))))
                 for xi in inp_l2]
    grad_h_pre = [grad_h[i] * silu_grad[i] for i in range(len(h))]

    inp = x_t + factors + [t_norm]
    grad_W1 = [[grad_h_pre[j] * inp[i] for i in range(len(inp))] for j in range(len(grad_h_pre))]
    grad_b1 = grad_h_pre[:]

    net.update_weights(grad_W1, grad_W2, grad_b1, grad_b2, lr)

    return loss


# ---------------------------------------------------------------------------
# Factor-conditioned generation (DDPM sampling)
# ---------------------------------------------------------------------------

def factor_conditioned_sample(
    net: FactorConditionedScoreNet,
    factors: list[float],
    betas: list[float],
    alpha_bars: list[float],
    n_steps: int = None,
    seed: int = 42,
) -> list[float]:
    """
    Generate a sample x0 conditioned on news factors using DDPM reverse process.
    """
    if n_steps is None:
        n_steps = len(betas)

    rng = random.Random(seed)
    T = len(betas)

    # Start from pure noise
    x = [rng.gauss(0, 1) for _ in range(net.d_data)]

    for t_idx in range(T - 1, -1, -1):
        t_norm = t_idx / T
        ab = alpha_bars[t_idx]
        ab_prev = alpha_bars[t_idx - 1] if t_idx > 0 else 1.0
        b = betas[t_idx]

        # Predict noise
        eps = net.forward(x, t_norm, factors)

        # Compute x0 prediction
        x0_pred = [(x[i] - math.sqrt(1 - ab) * eps[i]) / max(math.sqrt(ab), 1e-8)
                   for i in range(net.d_data)]

        # Reverse diffusion
        coef1 = math.sqrt(ab_prev) * b / max(1 - ab, 1e-8)
        coef2 = math.sqrt(1 - b) * (1 - ab_prev) / max(1 - ab, 1e-8)

        x_mean = [coef1 * x0_pred[i] + coef2 * x[i] for i in range(net.d_data)]

        if t_idx > 0:
            noise_scale = math.sqrt(b * (1 - ab_prev) / max(1 - ab, 1e-8))
            noise = [rng.gauss(0, 1) for _ in range(net.d_data)]
            x = [x_mean[i] + noise_scale * noise[i] for i in range(net.d_data)]
        else:
            x = x_mean

    return x


# ---------------------------------------------------------------------------
# Factor loading matrix
# ---------------------------------------------------------------------------

def estimate_factor_loadings(
    returns: list[list[float]],    # T x N (asset returns)
    factor_scores: list[list[float]],  # T x k (news factor scores at each time)
) -> list[list[float]]:
    """
    Estimate factor loading matrix B (N x k) via OLS:
    r_t = B * f_t + epsilon_t
    """
    T = len(returns)
    N = len(returns[0]) if T > 0 else 0
    k = len(factor_scores[0]) if T > 0 else 0

    # OLS: B = (F^T F)^{-1} F^T R   (k x k solve for each asset)
    # F: T x k matrix of factor scores
    F = factor_scores  # T x k

    # F^T F: k x k
    FtF = [[sum(F[t][i] * F[t][j] for t in range(T)) for j in range(k)] for i in range(k)]

    # F^T R: k x N
    FtR = [[sum(F[t][i] * returns[t][n] for t in range(T)) for n in range(N)] for i in range(k)]

    # Solve FtF * B = FtR for B (simplified: use diagonal approximation)
    # For exact solution we'd need LU or Cholesky, approximate here
    diag = [FtF[i][i] for i in range(k)]

    # B: k x N (then transpose to N x k)
    B_T = [[FtR[i][n] / max(diag[i], 1e-10) for n in range(N)] for i in range(k)]
    # Transpose to N x k
    B = [[B_T[j][n] for j in range(k)] for n in range(N)]

    return B


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("DAY 23: Latent Factor Diffusion")
    print("=" * 65)

    rng = random.Random(42)
    D_EMBED = 16   # embedding dimension
    T_NEWS = 100   # number of news items
    K_FACTORS = 4  # latent factors
    N_ASSETS = 5

    # Simulate news embeddings with 3 true latent factors
    true_topics = [[rng.gauss(0, 1) for _ in range(D_EMBED)] for _ in range(K_FACTORS)]
    embeddings = []
    for _ in range(T_NEWS):
        # Mix of topics
        weights = [rng.gauss(0, 1) for _ in range(K_FACTORS)]
        emb = [sum(weights[k] * true_topics[k][d] for k in range(K_FACTORS)) + rng.gauss(0, 0.1)
               for d in range(D_EMBED)]
        embeddings.append(emb)

    print(f"\n1. Factor Extraction from News Embeddings")
    print(f"   Input: {T_NEWS} news embeddings × {D_EMBED} dimensions")

    factor_model = extract_news_factors(embeddings, n_factors=K_FACTORS)

    print(f"\n   Singular values: {[f'{s:.4f}' for s in factor_model.singular_values]}")
    print(f"   Explained variance per factor:")
    cumulative = 0.0
    for i, ev in enumerate(factor_model.explained_variance):
        cumulative += ev
        print(f"     Factor {i+1}: {ev:.4%}  (cumulative: {cumulative:.4%})")

    # --- Project some embeddings ---
    print(f"\n2. News Embedding → Factor Scores")
    for i in range(3):
        scores = project_to_factors(embeddings[i], factor_model)
        print(f"   News {i+1} factor scores: {[f'{s:.4f}' for s in scores]}")

    # --- Factor loading matrix ---
    print(f"\n3. Factor Loading Matrix (Assets × Factors)")
    all_scores = [project_to_factors(emb, factor_model) for emb in embeddings]
    asset_returns = [[rng.gauss(0.0002, 0.01) + 0.3 * all_scores[t][i % K_FACTORS]
                      for i in range(N_ASSETS)] for t in range(T_NEWS)]

    B = estimate_factor_loadings(asset_returns, all_scores)
    print(f"   {'':>10} | " + " | ".join(f"F{k+1:>8}" for k in range(K_FACTORS)))
    print("   " + "-" * (15 + K_FACTORS * 12))
    for n in range(N_ASSETS):
        row = f"   Asset {n+1:>3}  | " + " | ".join(f"{B[n][k]:>8.4f}" for k in range(K_FACTORS))
        print(row)

    # --- Train factor-conditioned score network ---
    print(f"\n4. Training Factor-Conditioned Score Network")
    T_DIFF = 50
    betas = cosine_beta_schedule(T_DIFF)
    alpha_bars_ = alpha_bar(betas)

    net = FactorConditionedScoreNet(d_data=D_EMBED // 2, n_factors=K_FACTORS, seed=0)

    # Train on first 50 embeddings with their factor scores
    train_losses = []
    train_rng = random.Random(0)
    for step in range(200):
        t_idx = step % T_NEWS
        x0 = embeddings[t_idx][:net.d_data]  # use first d_data dims
        factors = all_scores[t_idx]
        loss = training_step(net, x0, factors, betas, alpha_bars_, lr=1e-3, rng=train_rng)
        if step % 50 == 0:
            train_losses.append(loss)
            print(f"   Step {step:>4}: loss = {loss:.6f}")

    # --- Factor-conditioned generation ---
    print(f"\n5. Factor-Conditioned Generation")

    # Generate with different factor states
    bullish_factors = [2.0, -1.0, 0.5, -0.3][:K_FACTORS]
    bearish_factors = [-2.0, 1.0, -0.5, 0.3][:K_FACTORS]

    bull_sample = factor_conditioned_sample(net, bullish_factors, betas, alpha_bars_, seed=1)
    bear_sample = factor_conditioned_sample(net, bearish_factors, betas, alpha_bars_, seed=2)
    neutral_sample = factor_conditioned_sample(net, [0.0] * K_FACTORS, betas, alpha_bars_, seed=3)

    print(f"   Bullish factors  → L2 norm: {norm(bull_sample):.4f}, mean: {sum(bull_sample)/len(bull_sample):.4f}")
    print(f"   Bearish factors  → L2 norm: {norm(bear_sample):.4f}, mean: {sum(bear_sample)/len(bear_sample):.4f}")
    print(f"   Neutral factors  → L2 norm: {norm(neutral_sample):.4f}, mean: {sum(neutral_sample)/len(neutral_sample):.4f}")

    # Cosine similarity between bull and bear samples
    cos_sim = dot(bull_sample, bear_sample) / max(norm(bull_sample) * norm(bear_sample), 1e-8)
    print(f"\n   Bull-Bear cosine similarity: {cos_sim:.4f}  (lower = more different)")

    print("\n[Done] Day 23: Latent Factor Diffusion complete.")
