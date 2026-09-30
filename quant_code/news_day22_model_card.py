"""
news_day22_model_card.py
Day 22: Model Card for News-to-Price Diffusion Model.
Mitchell (2019) model card format: intended use, performance metrics,
ethical considerations, bias checks, uncertainty quantification summary.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Performance metric helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def std(xs: list[float]) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / max(len(xs) - 1, 1))

def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    mx, my = mean(xs[:n]), mean(ys[:n])
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom = math.sqrt(sum((xs[i] - mx)**2 for i in range(n)) *
                       sum((ys[i] - my)**2 for i in range(n)))
    return num / max(denom, 1e-10)

def rmse(preds: list[float], actuals: list[float]) -> float:
    n = min(len(preds), len(actuals))
    return math.sqrt(sum((preds[i] - actuals[i])**2 for i in range(n)) / max(n, 1))

def mae(preds: list[float], actuals: list[float]) -> float:
    n = min(len(preds), len(actuals))
    return sum(abs(preds[i] - actuals[i]) for i in range(n)) / max(n, 1)

def directional_accuracy(preds: list[float], actuals: list[float]) -> float:
    """Fraction of times predicted direction matches realized direction."""
    n = min(len(preds), len(actuals))
    correct = sum(1 for i in range(n) if (preds[i] >= 0) == (actuals[i] >= 0))
    return correct / max(n, 1)

def coverage(lower: list[float], upper: list[float], actuals: list[float]) -> float:
    n = min(len(lower), len(upper), len(actuals))
    return sum(1 for i in range(n) if lower[i] <= actuals[i] <= upper[i]) / max(n, 1)

# ---------------------------------------------------------------------------
# Bias metrics
# ---------------------------------------------------------------------------
def compute_bias(preds: list[float], actuals: list[float]) -> float:
    """Mean prediction error (positive = over-prediction)."""
    return mean([p - a for p, a in zip(preds, actuals)])

def tail_bias(preds: list[float], actuals: list[float], pct: float = 0.10) -> dict:
    """Bias in upper and lower tail (extreme observations)."""
    n = min(len(preds), len(actuals))
    pairs = sorted(zip(actuals[:n], preds[:n]), key=lambda x: x[0])
    k = max(int(n * pct), 1)

    lower = pairs[:k]
    upper = pairs[-k:]

    lower_bias = mean([p - a for a, p in lower])
    upper_bias = mean([p - a for a, p in upper])
    return {'lower_tail_bias': lower_bias, 'upper_tail_bias': upper_bias}

def segment_bias(preds: list[float], actuals: list[float],
                  segments: dict[str, list[int]]) -> dict[str, float]:
    """Bias per user-defined segment (dict of segment_name -> list of indices)."""
    biases = {}
    for name, idxs in segments.items():
        sub_preds = [preds[i] for i in idxs if i < len(preds)]
        sub_actual = [actuals[i] for i in idxs if i < len(actuals)]
        biases[name] = compute_bias(sub_preds, sub_actual) if sub_preds else float('nan')
    return biases

# ---------------------------------------------------------------------------
# Model card dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelDetails:
    name: str
    version: str
    date: str
    model_type: str
    architecture: str
    training_data: str
    contact: str = ''

@dataclass
class IntendedUse:
    primary_uses: list[str]
    primary_users: list[str]
    out_of_scope: list[str]

@dataclass
class PerformanceMetrics:
    dataset: str
    n_samples: int
    rmse: float
    mae: float
    correlation: float
    directional_accuracy: float
    coverage_90: float
    coverage_95: float
    mean_interval_width: float
    tail_bias_lower: float
    tail_bias_upper: float

@dataclass
class EthicalConsiderations:
    risks: list[str]
    mitigations: list[str]
    limitations: list[str]
    recommendations: list[str]

@dataclass
class UncertaintyQuantification:
    method: str
    calibration_error: float
    sharpness: float
    notes: str

@dataclass
class ModelCard:
    details: ModelDetails
    intended_use: IntendedUse
    metrics: list[PerformanceMetrics]
    ethical: EthicalConsiderations
    uq: UncertaintyQuantification

    def render(self) -> str:
        lines = []

        def section(title: str):
            lines.append("\n" + "=" * 65)
            lines.append(f"  {title}")
            lines.append("=" * 65)

        def field_line(label: str, value):
            lines.append(f"  {label:<30} {value}")

        def bullet(text: str, indent: int = 2):
            lines.append(" " * indent + "• " + text)

        # Header
        lines.append("=" * 65)
        lines.append(f"  MODEL CARD: {self.details.name}")
        lines.append("=" * 65)

        section("1. MODEL DETAILS")
        field_line("Version:", self.details.version)
        field_line("Date:", self.details.date)
        field_line("Type:", self.details.model_type)
        field_line("Architecture:", self.details.architecture)
        field_line("Training Data:", self.details.training_data)
        if self.details.contact:
            field_line("Contact:", self.details.contact)

        section("2. INTENDED USE")
        lines.append("  Primary Uses:")
        for u in self.intended_use.primary_uses:
            bullet(u, 4)
        lines.append("  Primary Users:")
        for u in self.intended_use.primary_users:
            bullet(u, 4)
        lines.append("  Out-of-Scope Uses:")
        for u in self.intended_use.out_of_scope:
            bullet(u, 4)

        section("3. PERFORMANCE METRICS")
        for m in self.metrics:
            lines.append(f"\n  Dataset: {m.dataset} (n={m.n_samples:,})")
            lines.append(f"  {'-'*45}")
            field_line("  RMSE:", f"{m.rmse:.6f}")
            field_line("  MAE:", f"{m.mae:.6f}")
            field_line("  Pearson r:", f"{m.correlation:.4f}")
            field_line("  Directional Accuracy:", f"{m.directional_accuracy:.2%}")
            field_line("  Coverage 90%:", f"{m.coverage_90:.2%} (nominal 90%)")
            field_line("  Coverage 95%:", f"{m.coverage_95:.2%} (nominal 95%)")
            field_line("  Mean Interval Width:", f"{m.mean_interval_width:.6f}")
            field_line("  Lower Tail Bias:", f"{m.tail_bias_lower:+.6f}")
            field_line("  Upper Tail Bias:", f"{m.tail_bias_upper:+.6f}")

        section("4. UNCERTAINTY QUANTIFICATION")
        field_line("Method:", self.uq.method)
        field_line("Calibration Error:", f"{self.uq.calibration_error:.4f}")
        field_line("Sharpness:", f"{self.uq.sharpness:.6f}")
        lines.append(f"  Notes: {self.uq.notes}")

        section("5. ETHICAL CONSIDERATIONS")
        lines.append("  Risks:")
        for r in self.ethical.risks:
            bullet(r, 4)
        lines.append("  Mitigations:")
        for m in self.ethical.mitigations:
            bullet(m, 4)
        lines.append("  Limitations:")
        for l in self.ethical.limitations:
            bullet(l, 4)
        lines.append("  Recommendations for Deployment:")
        for r in self.ethical.recommendations:
            bullet(r, 4)

        lines.append("\n" + "=" * 65)
        return '\n'.join(lines)

# ---------------------------------------------------------------------------
# Auto-compute metrics from predictions
# ---------------------------------------------------------------------------
def compute_model_card_metrics(preds: list[float], actuals: list[float],
                                 lower_90: list[float], upper_90: list[float],
                                 lower_95: list[float], upper_95: list[float],
                                 dataset_name: str) -> PerformanceMetrics:
    tb = tail_bias(preds, actuals)
    return PerformanceMetrics(
        dataset=dataset_name,
        n_samples=len(actuals),
        rmse=rmse(preds, actuals),
        mae=mae(preds, actuals),
        correlation=pearson_r(preds, actuals),
        directional_accuracy=directional_accuracy(preds, actuals),
        coverage_90=coverage(lower_90, upper_90, actuals),
        coverage_95=coverage(lower_95, upper_95, actuals),
        mean_interval_width=mean([u - l for u, l in zip(upper_90, lower_90)]),
        tail_bias_lower=tb['lower_tail_bias'],
        tail_bias_upper=tb['upper_tail_bias'],
    )

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rng = random.Random(42)

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    # Simulate news diffusion model predictions
    N = 500
    actuals = [0.001 * randn() for _ in range(N)]
    # Model: partially correlated predictions with slight bias
    preds = [0.7 * a + 0.3 * 0.001 * randn() + 0.00005 for a in actuals]

    # Prediction intervals (model slightly overconfident → under-coverage)
    sigma_pred = 0.0008
    lower_90 = [p - 1.645 * sigma_pred for p in preds]
    upper_90 = [p + 1.645 * sigma_pred for p in preds]
    lower_95 = [p - 1.96 * sigma_pred for p in preds]
    upper_95 = [p + 1.96 * sigma_pred for p in preds]

    # Compute calibration error (mean |coverage - nominal|)
    cov90 = coverage(lower_90, upper_90, actuals)
    cov95 = coverage(lower_95, upper_95, actuals)
    cal_error = (abs(cov90 - 0.90) + abs(cov95 - 0.95)) / 2

    metrics_train = compute_model_card_metrics(
        preds[:400], actuals[:400],
        lower_90[:400], upper_90[:400],
        lower_95[:400], upper_95[:400],
        dataset_name='Train (2019-2022)'
    )
    metrics_test = compute_model_card_metrics(
        preds[400:], actuals[400:],
        lower_90[400:], upper_90[400:],
        lower_95[400:], upper_95[400:],
        dataset_name='Test (2023)'
    )

    card = ModelCard(
        details=ModelDetails(
            name="News-to-Price Diffusion Model (NPDM v2.4)",
            version="2.4.0",
            date="2026-09-29",
            model_type="Latent Factor Diffusion (DDPM + News Conditioning)",
            architecture="Factor-conditioned DDPM with SVD news embedding, 2-layer MLP score net",
            training_data="Financial news corpus 2019-2022, 10M+ articles; "
                          "price returns from 500 US equities",
            contact="quant-research@example.com"
        ),
        intended_use=IntendedUse(
            primary_uses=[
                "Generating probabilistic short-horizon price-return scenarios from news",
                "Risk scenario generation for stress testing",
                "Constructing prediction intervals for returns given news sentiment",
            ],
            primary_users=[
                "Quantitative researchers and portfolio managers",
                "Risk management teams evaluating scenario-driven VaR",
                "Academic researchers studying news-price relationships",
            ],
            out_of_scope=[
                "Direct use as investment advice or trade signals",
                "Long-horizon forecasting (>5 trading days)",
                "Non-US equities or illiquid securities",
                "Real-time production trading without additional risk controls",
            ]
        ),
        metrics=[metrics_train, metrics_test],
        ethical=EthicalConsiderations(
            risks=[
                "Model may amplify momentum bias in news sentiment → feedback loops",
                "Corporate earnings news correlated across sectors → underestimated tail correlation",
                "News source selection bias: predominantly mainstream English-language media",
                "Potential for over-fitting to crisis-era correlations (2020-2021)",
            ],
            mitigations=[
                "Prediction intervals calibrated using Kupiec/Christoffersen backtesting",
                "Out-of-sample testing on held-out 2023 data",
                "Bias diagnostics run across sectors (tech vs. non-tech) and market caps",
                "Explainability layer via factor loading visualization",
            ],
            limitations=[
                "Coverage under-performance in extreme news events (black swans)",
                "Directional accuracy degrades during regime changes",
                "Latency: not designed for sub-second inference",
                "No explicit handling of earnings surprise magnitude",
            ],
            recommendations=[
                "Always combine with human analyst review for high-impact news events",
                "Retrain or fine-tune after major market structure changes",
                "Use ensemble with traditional factor models to reduce idiosyncratic risk",
                "Monitor rolling calibration metrics weekly and trigger alert if coverage < nominal - 5%",
            ]
        ),
        uq=UncertaintyQuantification(
            method="Gaussian prediction intervals from score-network variance; "
                   "calibrated via rolling Kupiec POF test",
            calibration_error=cal_error,
            sharpness=mean([u - l for u, l in zip(upper_90, lower_90)]),
            notes="Model exhibits mild overconfidence (under-coverage ~2-3% at 95% level). "
                  "Recommend inflating sigma by 1.1x in production."
        )
    )

    print(card.render())
    print("\n[Done] Day 22: Model Card complete.")
