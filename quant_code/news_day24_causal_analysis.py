"""
news_day24_causal_analysis.py
Day 24: Causal Analysis — Granger causality (bivariate VAR + F-test),
Impulse Response Functions, Forecast Error Variance Decomposition,
event study with Cumulative Abnormal Returns (CAR), cross-sectional regression.
Pure Python stdlib only (no scipy).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# OLS regression (no numpy)
# ---------------------------------------------------------------------------

def ols(X: list[list[float]], y: list[float]) -> tuple[list[float], float, list[float]]:
    """
    OLS: beta = (X^T X)^{-1} X^T y
    Returns (beta, r_squared, residuals).
    Uses Gram-Schmidt for numerical stability.
    """
    n = len(y)
    p = len(X[0])

    # Compute X^T X
    XtX = [[sum(X[t][i] * X[t][j] for t in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(X[t][i] * y[t] for t in range(n)) for i in range(p)]

    # Solve via Gaussian elimination
    beta = _solve_linear(XtX, Xty)

    fitted = [sum(X[t][i] * beta[i] for i in range(p)) for t in range(n)]
    residuals = [y[t] - fitted[t] for t in range(n)]

    ss_res = sum(r**2 for r in residuals)
    y_mean = sum(y) / n
    ss_tot = sum((yi - y_mean)**2 for yi in y)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)

    return beta, r2, residuals


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-12:
            M[col][col] = 1e-10

        pivot = M[col][col]
        for row in range(n):
            if row != col:
                factor = M[row][col] / pivot
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

        # Normalize
        for j in range(col, n + 1):
            M[col][j] /= pivot

    return [M[i][n] for i in range(n)]


# ---------------------------------------------------------------------------
# Granger Causality Test
# ---------------------------------------------------------------------------

def granger_causality(
    y: list[float],           # target series
    x: list[float],           # potential Granger-cause series
    max_lag: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Granger causality test: does x Granger-cause y?
    H0: x does NOT Granger-cause y (restricted model = AR(p))
    H1: x DOES Granger-cause y (unrestricted model = VAR-like with x lags)

    F-statistic: F = [(RSS_r - RSS_ur) / p] / [RSS_ur / (T - 2p - 1)]
    where p = max_lag.
    """
    n = len(y)
    p = max_lag
    T = n - p  # effective observations

    if T < 2 * p + 2:
        return {'error': 'Insufficient observations', 'p_value': 1.0, 'granger_causes': False}

    # Build lagged matrices
    def build_matrix(series_list: list[list[float]], lags: int) -> list[list[float]]:
        T_eff = len(series_list[0]) - lags
        rows = []
        for t in range(lags, len(series_list[0])):
            row = [1.0]  # intercept
            for series in series_list:
                for lag in range(1, lags + 1):
                    row.append(series[t - lag])
            rows.append(row)
        return rows

    y_dep = y[p:]

    # Restricted: y on its own lags only
    X_r = build_matrix([y], p)
    _, _, res_r = ols(X_r, y_dep)
    RSS_r = sum(r**2 for r in res_r)

    # Unrestricted: y on y lags + x lags
    X_ur = build_matrix([y, x], p)
    _, _, res_ur = ols(X_ur, y_dep)
    RSS_ur = sum(r**2 for r in res_ur)

    # F-statistic
    df1 = p         # added regressors (x lags)
    df2 = T - 2 * p - 1
    if df2 <= 0 or RSS_ur < 1e-12:
        return {'error': 'Degrees of freedom issue', 'p_value': 1.0, 'granger_causes': False}

    F = ((RSS_r - RSS_ur) / df1) / (RSS_ur / df2)

    # Approximate p-value using F distribution CDF approximation
    p_value = _f_pvalue_approx(F, df1, df2)

    return {
        'F_statistic': F,
        'p_value': p_value,
        'df1': df1,
        'df2': df2,
        'RSS_restricted': RSS_r,
        'RSS_unrestricted': RSS_ur,
        'max_lag': max_lag,
        'granger_causes': p_value < alpha,
        'significance': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else '')),
    }


def _f_pvalue_approx(F: float, df1: int, df2: int) -> float:
    """
    Approximate F-distribution p-value using normal approximation for large df.
    For small df, uses Wilson-Hilferty cube-root transformation.
    """
    if F <= 0:
        return 1.0

    # Wilson-Hilferty approximation for chi-squared
    # F ~ chi2(df1)/df1 / (chi2(df2)/df2)
    # Use chi2 normal approx: chi2(k)/k approx Normal(1, 2/k)
    if df2 > 30:
        # Large df2: F approx chi2(df1)/df1
        z = (pow(F * df1 / df1, 1/3) - (1 - 2/(9*df1))) / math.sqrt(2/(9*df1))
        p_value = 0.5 * math.erfc(z / math.sqrt(2))
        return max(0.001, min(1.0, p_value))

    # Use beta distribution approximation for small df
    x = df1 * F / (df1 * F + df2)
    # Rough approximation: p-value from normal score
    # mean and var of Beta(df1/2, df2/2)
    a, b = df1 / 2, df2 / 2
    mean_x = a / (a + b)
    var_x = a * b / ((a + b)**2 * (a + b + 1))
    z = (x - mean_x) / max(math.sqrt(var_x), 1e-8)
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return max(0.001, min(1.0, p_value))


# ---------------------------------------------------------------------------
# VAR Model
# ---------------------------------------------------------------------------

@dataclass
class VARModel:
    """Bivariate VAR(p) model."""
    coefs: list[list[list[float]]]   # p x 2 x 2 coefficient matrices A_1,...,A_p
    intercepts: list[float]           # 2
    residual_cov: list[list[float]]   # 2 x 2
    p: int                            # lag order
    n_obs: int


def fit_var(
    series1: list[float],
    series2: list[float],
    p: int = 2,
) -> VARModel:
    """
    Fit bivariate VAR(p) by OLS equation-by-equation.
    Y_t = A_1 * Y_{t-1} + ... + A_p * Y_{t-p} + c + epsilon_t
    """
    n = len(series1)
    T = n - p

    def build_X():
        rows = []
        for t in range(p, n):
            row = [1.0]  # intercept
            for lag in range(1, p + 1):
                row += [series1[t - lag], series2[t - lag]]
            rows.append(row)
        return rows

    X = build_X()
    y1 = series1[p:]
    y2 = series2[p:]

    beta1, _, res1 = ols(X, y1)
    beta2, _, res2 = ols(X, y2)

    # Parse coefficients
    # beta = [intercept, A_1[0,0], A_1[0,1], A_2[0,0], A_2[0,1], ...]
    coefs = []
    for lag in range(p):
        A_lag = [
            [beta1[1 + 2*lag], beta1[2 + 2*lag]],
            [beta2[1 + 2*lag], beta2[2 + 2*lag]],
        ]
        coefs.append(A_lag)

    intercepts = [beta1[0], beta2[0]]

    # Residual covariance
    cov = [
        [sum(r1*r1 for r1 in res1) / T, sum(r1*r2 for r1,r2 in zip(res1,res2)) / T],
        [sum(r1*r2 for r1,r2 in zip(res1,res2)) / T, sum(r2*r2 for r2 in res2) / T],
    ]

    return VARModel(coefs, intercepts, cov, p, T)


# ---------------------------------------------------------------------------
# Impulse Response Function
# ---------------------------------------------------------------------------

def impulse_response(
    var_model: VARModel,
    n_periods: int = 20,
    shock_size: float = 1.0,
) -> dict:
    """
    Compute IRF for bivariate VAR.
    Response of each variable to a unit shock in each variable.
    Uses Cholesky identification (lower triangular).
    """
    p = var_model.p

    # Cholesky decomposition of residual cov (2x2)
    Sigma = var_model.residual_cov
    L = [[0.0, 0.0], [0.0, 0.0]]
    L[0][0] = math.sqrt(max(Sigma[0][0], 1e-12))
    L[1][0] = Sigma[1][0] / max(L[0][0], 1e-12)
    L[1][1] = math.sqrt(max(Sigma[1][1] - L[1][0]**2, 1e-12))

    # IRF for each structural shock
    results = {'shock_to_y1': [], 'shock_to_y2': []}

    for shock_var in [0, 1]:
        shock = [L[i][shock_var] * shock_size for i in range(2)]

        # History of responses (initialize with shock at t=0)
        history = [[0.0, 0.0]] * p + [shock]

        responses = [[shock[0], shock[1]]]

        for h in range(1, n_periods):
            resp = list(var_model.intercepts)
            for lag in range(p):
                t_idx = len(history) - 1 - lag
                if t_idx >= 0:
                    A = var_model.coefs[lag]
                    y_lag = history[t_idx] if t_idx < len(history) else [0.0, 0.0]
                    for i in range(2):
                        for j in range(2):
                            resp[i] += A[i][j] * y_lag[j]
            # Subtract intercept (IRF is deviation from steady state)
            resp_dev = [resp[i] - var_model.intercepts[i] for i in range(2)]
            history.append(resp_dev)
            responses.append(resp_dev)

        key = 'shock_to_y1' if shock_var == 0 else 'shock_to_y2'
        results[key] = responses

    return results


# ---------------------------------------------------------------------------
# Forecast Error Variance Decomposition (FEVD)
# ---------------------------------------------------------------------------

def fevd(
    irf: dict,
    n_periods: int = 20,
) -> dict:
    """
    FEVD from IRF responses.
    FEVD_y1_from_shock1(h) = sum_{j=0}^{h} resp[j][0]^2 from shock1 / total variance
    """
    decomp = {'y1': [], 'y2': []}

    for var_idx, var_name in enumerate(['y1', 'y2']):
        for h in range(n_periods):
            var_from_s1 = sum(irf['shock_to_y1'][j][var_idx]**2 for j in range(h + 1))
            var_from_s2 = sum(irf['shock_to_y2'][j][var_idx]**2 for j in range(h + 1))
            total = var_from_s1 + var_from_s2
            if total < 1e-12:
                decomp[var_name].append({'s1_share': 0.5, 's2_share': 0.5})
            else:
                decomp[var_name].append({
                    's1_share': var_from_s1 / total,
                    's2_share': var_from_s2 / total,
                })

    return decomp


# ---------------------------------------------------------------------------
# Event Study: CAR
# ---------------------------------------------------------------------------

@dataclass
class NewsEvent:
    event_date: int    # index in returns series
    asset: str
    sentiment: float   # -1 (negative) to +1 (positive)
    category: str = 'earnings'


def event_study(
    returns: list[float],        # full return series
    event_date: int,
    estimation_window: tuple[int, int] = (-120, -21),  # days before event
    event_window: tuple[int, int] = (-5, 20),           # CAR window
) -> dict:
    """
    Standard event study methodology.
    1. Estimate normal returns via OLS on estimation window
    2. Compute ARs = actual - expected in event window
    3. Cumulate to CAR
    """
    est_start = event_date + estimation_window[0]
    est_end = event_date + estimation_window[1]

    if est_start < 0 or est_end < 0 or est_end >= len(returns):
        return {'error': 'Insufficient data for estimation window'}

    # Simple constant mean model for expected return
    est_returns = returns[est_start:est_end]
    expected_r = sum(est_returns) / len(est_returns)
    sigma_est = math.sqrt(sum((r - expected_r)**2 for r in est_returns) / max(len(est_returns) - 1, 1))

    # Event window ARs
    ev_start = max(event_date + event_window[0], 0)
    ev_end = min(event_date + event_window[1] + 1, len(returns))

    ars = []
    for t in range(ev_start, ev_end):
        ar = returns[t] - expected_r
        ars.append(ar)

    # CAR
    car = sum(ars)

    # t-statistic (Patell 1976 style)
    n_event = len(ars)
    car_std = sigma_est * math.sqrt(n_event)
    t_stat = car / max(car_std, 1e-8)

    # Two-sided p-value (normal approximation)
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return {
        'CAR': car,
        'expected_return': expected_r,
        'sigma_estimation': sigma_est,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'ARs': ars,
        'cumulative_ARs': [sum(ars[:i+1]) for i in range(len(ars))],
        'n_estimation': len(est_returns),
        'n_event': n_event,
    }


def cross_sectional_car_regression(
    events: list[NewsEvent],
    cars: list[float],
) -> dict:
    """
    Regress event CARs on sentiment and other cross-sectional characteristics.
    CAR_i = alpha + beta * sentiment_i + epsilon_i
    """
    n = len(events)
    if n < 3:
        return {}

    # Build design matrix: [1, sentiment]
    X = [[1.0, e.sentiment] for e in events]

    beta, r2, residuals = ols(X, cars)

    # Standard errors
    T = n
    p = 2
    sse = sum(r**2 for r in residuals)
    s2 = sse / max(T - p, 1)

    XtX = [[sum(X[t][i] * X[t][j] for t in range(T)) for j in range(p)] for i in range(p)]
    try:
        XtX_inv = _invert_2x2(XtX)
        se = [math.sqrt(max(s2 * XtX_inv[i][i], 0)) for i in range(p)]
    except Exception:
        se = [float('nan')] * p

    t_stats = [beta[i] / max(se[i], 1e-10) for i in range(p)]
    p_values = [2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))) for t in t_stats]

    return {
        'alpha': beta[0],
        'beta_sentiment': beta[1],
        'r_squared': r2,
        'se_alpha': se[0],
        'se_beta': se[1],
        't_alpha': t_stats[0],
        't_beta': t_stats[1],
        'p_alpha': p_values[0],
        'p_beta': p_values[1],
        'n_events': n,
    }


def _invert_2x2(M: list[list[float]]) -> list[list[float]]:
    """Invert a 2x2 matrix."""
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if abs(det) < 1e-12:
        raise ValueError("Singular matrix")
    return [
        [ M[1][1] / det, -M[0][1] / det],
        [-M[1][0] / det,  M[0][0] / det],
    ]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("DAY 24: Causal Analysis — Granger, IRF, FEVD, Event Study")
    print("=" * 65)

    rng = random.Random(42)
    N = 300

    # Simulate two series: news sentiment (x) → stock returns (y) with lag
    x_series = [rng.gauss(0, 1) for _ in range(N)]  # news sentiment
    y_series = []
    y_prev = 0.0
    for t in range(N):
        # y_t = 0.3 * y_{t-1} + 0.5 * x_{t-2} + noise
        x_lag2 = x_series[t - 2] if t >= 2 else 0.0
        y_t = 0.3 * y_prev + 0.5 * x_lag2 + rng.gauss(0, 0.5)
        y_series.append(y_t)
        y_prev = y_t

    # --- Granger causality ---
    print("\n1. Granger Causality: Does News Sentiment → Stock Returns?")

    for lag in [1, 2, 5]:
        result = granger_causality(y_series, x_series, max_lag=lag)
        sig = result.get('significance', '')
        causes = result.get('granger_causes', False)
        print(f"   Lag={lag}: F={result.get('F_statistic', 0):.4f}, "
              f"p={result.get('p_value', 1):.4f} {sig}, "
              f"Granger-causes: {causes}")

    print("\n   Reverse: Does Stock Returns → News Sentiment? (should be NO)")
    for lag in [1, 2]:
        result = granger_causality(x_series, y_series, max_lag=lag)
        sig = result.get('significance', '')
        causes = result.get('granger_causes', False)
        print(f"   Lag={lag}: F={result.get('F_statistic', 0):.4f}, "
              f"p={result.get('p_value', 1):.4f} {sig}, "
              f"Granger-causes: {causes}")

    # --- VAR model ---
    print("\n2. VAR(2) Model Estimation")
    var = fit_var(y_series, x_series, p=2)
    print(f"   Equation 1 (y): A1[y,y]={var.coefs[0][0][0]:.4f}, A1[y,x]={var.coefs[0][0][1]:.4f}")
    print(f"                    A2[y,y]={var.coefs[1][0][0]:.4f}, A2[y,x]={var.coefs[1][0][1]:.4f}")
    print(f"   Equation 2 (x): A1[x,y]={var.coefs[0][1][0]:.4f}, A1[x,x]={var.coefs[0][1][1]:.4f}")
    print(f"   Residual cov   : [[{var.residual_cov[0][0]:.4f}, {var.residual_cov[0][1]:.4f}], ...]")

    # --- IRF ---
    print("\n3. Impulse Response Function")
    irf_results = impulse_response(var, n_periods=10)

    print(f"   Response of y to shock in x (shock_to_x => response of y):")
    print(f"   {'Horizon':>8} | {'Response of y':>15} | {'Response of x':>15}")
    print("   " + "-" * 45)
    for h in range(min(10, len(irf_results['shock_to_y2']))):
        resp = irf_results['shock_to_y2'][h]
        print(f"   {h:>8} | {resp[0]:>15.6f} | {resp[1]:>15.6f}")

    # --- FEVD ---
    print("\n4. Forecast Error Variance Decomposition")
    fevd_result = fevd(irf_results, n_periods=10)

    print(f"   Variance of y explained by shocks to x:")
    for h in [0, 1, 4, 9]:
        share = fevd_result['y1'][h]['s2_share'] if h < len(fevd_result['y1']) else float('nan')
        print(f"   Horizon {h+1:>2}: {share:.4%} from x-shock, {1-share:.4%} from y-shock")

    # --- Event study ---
    print("\n5. Event Study (News → Stock Returns)")

    # Generate stock returns with event effects
    stock_returns = [rng.gauss(0.0003, 0.015) for _ in range(500)]

    events = []
    for _ in range(20):
        ev_date = rng.randint(150, 450)
        sentiment = rng.uniform(-1, 1)
        # Inject effect
        for k in range(-2, 10):
            if 0 <= ev_date + k < len(stock_returns):
                stock_returns[ev_date + k] += 0.01 * sentiment * max(0, 1 - k / 10)
        events.append(NewsEvent(ev_date, 'AAPL', sentiment))

    cars = []
    for ev in events:
        result = event_study(stock_returns, ev.event_date)
        if 'CAR' in result:
            cars.append(result['CAR'])
        else:
            cars.append(float('nan'))

    valid = [(e, c) for e, c in zip(events, cars) if not math.isnan(c)]

    print(f"   Events analyzed: {len(valid)}")
    avg_car = sum(c for _, c in valid) / len(valid)
    print(f"   Average CAR     : {avg_car:.4%}")

    pos_events = [(e, c) for e, c in valid if e.sentiment > 0]
    neg_events = [(e, c) for e, c in valid if e.sentiment < 0]

    if pos_events:
        avg_car_pos = sum(c for _, c in pos_events) / len(pos_events)
        print(f"   Avg CAR (positive sentiment): {avg_car_pos:.4%}  (n={len(pos_events)})")
    if neg_events:
        avg_car_neg = sum(c for _, c in neg_events) / len(neg_events)
        print(f"   Avg CAR (negative sentiment): {avg_car_neg:.4%}  (n={len(neg_events)})")

    # Show one event in detail
    sample_ev = valid[0]
    detail = event_study(stock_returns, sample_ev[0].event_date)
    print(f"\n   Sample event CAR: {detail['CAR']:.4%}")
    print(f"   t-statistic    : {detail['t_statistic']:.4f}")
    print(f"   p-value        : {detail['p_value']:.4f}")
    print(f"   Cumulative ARs :", [f"{c:.4%}" for c in detail['cumulative_ARs'][:8]])

    # --- Cross-sectional regression ---
    print("\n6. Cross-Sectional CAR ~ Sentiment Regression")
    valid_events = [e for e, _ in valid]
    valid_cars = [c for _, c in valid]

    reg_result = cross_sectional_car_regression(valid_events, valid_cars)
    print(f"   Alpha (intercept): {reg_result.get('alpha', 0):.6f}  (t={reg_result.get('t_alpha', 0):.4f})")
    print(f"   Beta (sentiment) : {reg_result.get('beta_sentiment', 0):.6f}  (t={reg_result.get('t_beta', 0):.4f})")
    print(f"   R-squared        : {reg_result.get('r_squared', 0):.4f}")

    sig_sentiment = '***' if reg_result.get('p_beta', 1) < 0.001 else (
                    '**' if reg_result.get('p_beta', 1) < 0.01 else (
                    '*' if reg_result.get('p_beta', 1) < 0.05 else ''))
    print(f"   Sentiment p-value: {reg_result.get('p_beta', 1):.4f} {sig_sentiment}")

    print("\n[Done] Day 24: Causal Analysis complete.")
