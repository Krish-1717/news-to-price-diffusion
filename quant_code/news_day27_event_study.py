"""
news_day27_event_study.py
Day 27: Event Study Framework — abnormal returns, cumulative AR (CAR),
cross-sectional CAAR with t-test, bootstrap significance testing.
Pure Python stdlib only.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)

def std(xs: list[float], ddof: int = 1) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / max(len(xs) - ddof, 1))

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def t_pvalue_two_sided(t_stat: float, df: int) -> float:
    """
    Approximate two-sided p-value for t-distribution via normal approximation
    (accurate for df > 20). Uses Wilson-Hilferty for df <= 20.
    """
    if df <= 0:
        return 1.0
    # For large df: t → N(0,1)
    if df >= 30:
        return 2 * (1 - _norm_cdf(abs(t_stat)))
    # Cornish-Fisher approximation: t(df) ≈ N(0,1) + correction
    x = abs(t_stat)
    # Abramowitz-Stegun approximation
    g = x * (1 - (x**2 - 1) / (4 * df))
    p_approx = 2 * (1 - _norm_cdf(g))
    return max(0.0, min(1.0, p_approx))

# ---------------------------------------------------------------------------
# 1. Market model (OLS estimation in estimation window)
# ---------------------------------------------------------------------------
@dataclass
class MarketModel:
    """Simple OLS: R_i = alpha + beta * R_m + epsilon."""
    alpha: float
    beta: float
    sigma_e: float  # residual std dev
    r_squared: float

def fit_market_model(stock_returns: list[float],
                      market_returns: list[float]) -> MarketModel:
    """OLS regression of stock returns on market returns."""
    n = len(stock_returns)
    x_bar = mean(market_returns)
    y_bar = mean(stock_returns)

    sxx = sum((x - x_bar)**2 for x in market_returns)
    sxy = sum((x - x_bar) * (y - y_bar)
              for x, y in zip(market_returns, stock_returns))

    if abs(sxx) < 1e-12:
        return MarketModel(y_bar, 0.0, std(stock_returns), 0.0)

    beta = sxy / sxx
    alpha = y_bar - beta * x_bar

    resids = [y - (alpha + beta * x) for x, y in zip(market_returns, stock_returns)]
    sigma_e = std(resids)
    ss_res = sum(r**2 for r in resids)
    ss_tot = sum((y - y_bar)**2 for y in stock_returns)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)

    return MarketModel(alpha, beta, sigma_e, r2)

def abnormal_return(stock_ret: float, market_ret: float, model: MarketModel) -> float:
    """AR_t = R_t - (alpha + beta * R_m_t)."""
    return stock_ret - (model.alpha + model.beta * market_ret)

# ---------------------------------------------------------------------------
# 2. Event study: single security
# ---------------------------------------------------------------------------
@dataclass
class EventStudyResult:
    ticker: str
    event_day: int
    ar: list[float]          # abnormal returns [event-window]
    car: list[float]         # cumulative ARs
    t_stats: list[float]     # t-stat at each day (AR / sigma_e)
    pre_window: int
    post_window: int

def single_event_study(ticker: str,
                        stock_returns: list[float],
                        market_returns: list[float],
                        event_day: int,
                        estimation_window: int = 120,
                        pre_window: int = 10,
                        post_window: int = 20) -> EventStudyResult:
    """
    Run event study for a single stock.
    - Estimation window: [event_day - estimation_window - pre_window, event_day - pre_window)
    - Event window: [event_day - pre_window, event_day + post_window]
    """
    est_start = event_day - estimation_window - pre_window
    est_end   = event_day - pre_window

    if est_start < 0:
        raise ValueError(f"Not enough data before event: est_start={est_start}")

    # Fit market model on estimation window
    est_stock  = stock_returns[est_start:est_end]
    est_market = market_returns[est_start:est_end]
    model = fit_market_model(est_stock, est_market)

    # Compute ARs in event window
    ar_list = []
    car_list = []
    t_list = []
    cum = 0.0
    for offset in range(-pre_window, post_window + 1):
        idx = event_day + offset
        if 0 <= idx < len(stock_returns):
            ar = abnormal_return(stock_returns[idx], market_returns[idx], model)
        else:
            ar = 0.0
        cum += ar
        t = ar / max(model.sigma_e, 1e-8)
        ar_list.append(ar)
        car_list.append(cum)
        t_list.append(t)

    return EventStudyResult(
        ticker=ticker,
        event_day=event_day,
        ar=ar_list,
        car=car_list,
        t_stats=t_list,
        pre_window=pre_window,
        post_window=post_window,
    )

# ---------------------------------------------------------------------------
# 3. Cross-sectional CAAR (Cumulative Average Abnormal Return)
# ---------------------------------------------------------------------------
@dataclass
class CAARResult:
    window_days: list[int]      # offset from event: e.g. [-10, -9, ..., 20]
    caar: list[float]           # cross-sectional average CAR
    caar_t_stat: list[float]    # t-stat (CAAR / cross-sectional std)
    caar_pvalue: list[float]    # approximate p-value
    n_events: int

def cross_sectional_caar(event_results: list[EventStudyResult]) -> CAARResult:
    """
    Pool event-study results: CAAR[τ] = mean(CAR_i[τ]) over all events i.
    t-stat = CAAR[τ] / (std(CAR_i[τ]) / sqrt(N)).
    """
    n = len(event_results)
    if n == 0:
        return CAARResult([], [], [], [], 0)

    # Align on offset indices (use first result's structure as template)
    pre  = event_results[0].pre_window
    post = event_results[0].post_window
    offsets = list(range(-pre, post + 1))
    n_days = len(offsets)

    caar = []
    t_stats = []
    pvalues = []

    for d in range(n_days):
        cars_d = [r.car[d] for r in event_results if d < len(r.car)]
        if not cars_d:
            caar.append(0.0)
            t_stats.append(0.0)
            pvalues.append(1.0)
            continue
        mu_car = mean(cars_d)
        std_car = std(cars_d)
        t = mu_car / max(std_car / math.sqrt(len(cars_d)), 1e-10)
        p = t_pvalue_two_sided(t, len(cars_d) - 1)
        caar.append(mu_car)
        t_stats.append(t)
        pvalues.append(p)

    return CAARResult(offsets, caar, t_stats, pvalues, n)

# ---------------------------------------------------------------------------
# 4. Bootstrap significance test
# ---------------------------------------------------------------------------
def bootstrap_caar_pvalue(event_results: list[EventStudyResult],
                            test_window: tuple[int, int] = (0, 5),
                            n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Bootstrap test for CAAR over test_window [t1, t2] (offsets from event).
    Null: randomly shuffle event dates to break signal.
    """
    rng = random.Random(seed)
    pre = event_results[0].pre_window

    def caar_window(results, t1, t2):
        """CAAR from day t1 to t2 (inclusive, as offsets)."""
        d1 = t1 + pre  # index in car list
        d2 = t2 + pre
        cars = [r.car[d2] - (r.car[d1 - 1] if d1 > 0 else 0.0)
                for r in results if d2 < len(r.car)]
        return mean(cars)

    # Observed CAAR
    t1, t2 = test_window
    obs_caar = caar_window(event_results, t1, t2)

    # Bootstrap null distribution: shuffle AR within each series
    boot_caars = []
    for _ in range(n_boot):
        shuffled_results = []
        for r in event_results:
            ar_shuffled = r.ar[:]
            rng.shuffle(ar_shuffled)
            # Reconstruct CAR
            car_boot = []
            cum = 0.0
            for ar in ar_shuffled:
                cum += ar
                car_boot.append(cum)
            shuffled_results.append(EventStudyResult(
                r.ticker, r.event_day, ar_shuffled, car_boot,
                [0.0] * len(ar_shuffled), r.pre_window, r.post_window
            ))
        boot_caars.append(caar_window(shuffled_results, t1, t2))

    p_val = sum(1 for b in boot_caars if abs(b) >= abs(obs_caar)) / n_boot

    return {
        'observed_caar': obs_caar,
        'bootstrap_mean': mean(boot_caars),
        'bootstrap_std': std(boot_caars),
        'p_value': p_val,
        'significant_5pct': p_val < 0.05,
        'n_bootstrap': n_boot,
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 27: Event Study Framework")
    print("=" * 65)

    rng = random.Random(42)
    T_total = 500

    def randn_rng():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rng.random())

    # Simulate market returns
    market_returns = [randn_rng() * 0.01 for _ in range(T_total)]

    # Simulate 10 earnings events with positive surprise
    n_events = 10
    event_days = sorted(rng.sample(range(150, 450), n_events))
    tickers = [f'STOCK_{i:02d}' for i in range(n_events)]

    print(f"\nSimulating {n_events} earnings events with +1.5% abnormal return at event day")

    event_results = []
    for i, (ticker, ev_day) in enumerate(zip(tickers, event_days)):
        # Stock: beta=1.0 + unique alpha + event impact
        beta = 0.8 + rng.random() * 0.5
        alpha_daily = (rng.random() - 0.3) * 0.001  # slight drift
        stock_rets = []
        for t in range(T_total):
            eps = randn_rng() * 0.015
            r = alpha_daily + beta * market_returns[t] + eps
            if t == ev_day:
                r += 0.015 + randn_rng() * 0.005  # +1.5% event shock
            stock_rets.append(r)

        result = single_event_study(ticker, stock_rets, market_returns, ev_day,
                                     estimation_window=120, pre_window=10, post_window=20)
        event_results.append(result)

    print("\n1. Individual Event Study: STOCK_00")
    r0 = event_results[0]
    model_check = fit_market_model(
        [0.0] * 10,  # placeholder
        [0.0] * 10
    )
    print(f"   {'Day':>5} | {'AR':>8} | {'CAR':>8} | {'t-stat':>8} | {'sig':>5}")
    print("   " + "-" * 45)
    offsets = list(range(-r0.pre_window, r0.post_window + 1))
    for off, ar, car, t in zip(offsets, r0.ar, r0.car, r0.t_stats):
        sig = '**' if abs(t) > 2.58 else ('*' if abs(t) > 1.96 else '')
        marker = ' <-- event' if off == 0 else ''
        print(f"   {off:>5} | {ar:>+8.4f} | {car:>+8.4f} | {t:>+8.3f} | {sig:>5}{marker}")

    print("\n2. Cross-Sectional CAAR (10 events)")
    caar_result = cross_sectional_caar(event_results)

    print(f"   {'Day':>5} | {'CAAR':>8} | {'t-stat':>8} | {'p-value':>9} | {'sig':>5}")
    print("   " + "-" * 50)
    for off, caar, t, p in zip(caar_result.window_days, caar_result.caar,
                                 caar_result.caar_t_stat, caar_result.caar_pvalue):
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
        marker = ' <-- event' if off == 0 else ''
        print(f"   {off:>5} | {caar:>+8.4f} | {t:>+8.3f} | {p:>9.4f} | {sig:>5}{marker}")

    print("\n3. Post-Event Drift Window [0, +5]")
    # Find CAAR at day 0 and day +5
    idx0 = caar_result.window_days.index(0)
    idx5 = caar_result.window_days.index(5)
    drift_05 = caar_result.caar[idx5]
    print(f"   CAAR[0,+5]  = {drift_05:+.4f}  (t={caar_result.caar_t_stat[idx5]:+.3f}  p={caar_result.caar_pvalue[idx5]:.4f})")
    drift_110 = caar_result.caar[caar_result.window_days.index(10)]
    print(f"   CAAR[0,+10] = {drift_110:+.4f}  (t={caar_result.caar_t_stat[caar_result.window_days.index(10)]:+.3f}  p={caar_result.caar_pvalue[caar_result.window_days.index(10)]:.4f})")
    drift_120 = caar_result.caar[caar_result.window_days.index(20)]
    print(f"   CAAR[0,+20] = {drift_120:+.4f}  (t={caar_result.caar_t_stat[caar_result.window_days.index(20)]:+.3f}  p={caar_result.caar_pvalue[caar_result.window_days.index(20)]:.4f})")

    print("\n4. Bootstrap Significance Test (window [0, +5])")
    boot = bootstrap_caar_pvalue(event_results, test_window=(0, 5), n_boot=1000)
    print(f"   Observed CAAR     : {boot['observed_caar']:+.4f}")
    print(f"   Bootstrap mean    : {boot['bootstrap_mean']:+.4f}")
    print(f"   Bootstrap std     : {boot['bootstrap_std']:.4f}")
    print(f"   Bootstrap p-value : {boot['p_value']:.4f}")
    print(f"   Significant (5%) : {boot['significant_5pct']}")

    print("\n5. Negative-Surprise Control (event with -2% shock)")
    neg_event_results = []
    neg_event_day = 250
    for i in range(10):
        beta = 0.8 + rng.random() * 0.5
        stock_rets = []
        for t in range(T_total):
            eps = randn_rng() * 0.015
            r = 0.0 + beta * market_returns[t] + eps
            if t == neg_event_day:
                r -= 0.020  # -2% shock
            stock_rets.append(r)
        result = single_event_study(f'NEG_{i:02d}', stock_rets, market_returns,
                                     neg_event_day, estimation_window=120)
        neg_event_results.append(result)

    neg_caar = cross_sectional_caar(neg_event_results)
    neg_boot = bootstrap_caar_pvalue(neg_event_results, test_window=(0, 5), n_boot=1000)
    idx0n = neg_caar.window_days.index(0)
    print(f"   CAAR at event day : {neg_caar.caar[idx0n]:+.4f}  (t={neg_caar.caar_t_stat[idx0n]:+.3f})")
    print(f"   Bootstrap p-value : {neg_boot['p_value']:.4f}")
    print(f"   Significant (5%): {neg_boot['significant_5pct']}")

    print("\n[Done] Day 27: Event Study complete.")
