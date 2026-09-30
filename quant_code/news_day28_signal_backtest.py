"""
news_day28_signal_backtest.py
Day 28: Trading Signal Backtesting Pipeline — combine news sentiment + price
momentum + vol regime, position sizing, backtest engine with transaction costs,
performance metrics, signal decay analysis.
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

# ---------------------------------------------------------------------------
# 1. Signal generation
# ---------------------------------------------------------------------------
def price_momentum_signal(returns: list[float], t: int,
                           lookback: int = 21, skip: int = 1) -> float:
    """
    Price momentum: cumulative return from t-lookback to t-skip.
    Normalized to z-score over past 63 days.
    """
    if t < lookback + skip + 63:
        return 0.0
    raw_signals = []
    for s in range(63):
        tt = t - s
        if tt < lookback + skip:
            break
        end = tt - skip
        start = end - lookback
        if start < 0:
            break
        cum = sum(returns[start:end])
        raw_signals.append(cum)
    if len(raw_signals) < 5:
        return 0.0
    sig = raw_signals[0]  # current
    m = mean(raw_signals)
    s = std(raw_signals)
    return (sig - m) / max(s, 1e-8)

def ewma_vol_regime(returns: list[float], t: int,
                     lambda_: float = 0.94, window: int = 63) -> float:
    """
    Vol regime signal: current EWMA vol / rolling mean of EWMA vol.
    >1 = high vol regime, <1 = low vol regime.
    """
    if t < window + 1:
        return 1.0
    # Compute EWMA vol series
    ewma_vols = []
    var = returns[0]**2
    for i in range(1, t + 1):
        var = lambda_ * var + (1 - lambda_) * returns[i - 1]**2
        ewma_vols.append(math.sqrt(var * 252))  # annualized
    if len(ewma_vols) < window:
        return 1.0
    current_vol = ewma_vols[-1]
    avg_vol = mean(ewma_vols[-window:])
    return current_vol / max(avg_vol, 1e-8)

def sentiment_signal(sentiment_series: list[float], t: int,
                      window: int = 5) -> float:
    """
    Moving average of recent sentiment scores.
    Returns z-score of sentiment vs past 63-day baseline.
    """
    if t < window + 63:
        return 0.0
    recent = mean(sentiment_series[t - window: t])
    baseline = mean(sentiment_series[t - 63: t])
    baseline_std = std(sentiment_series[t - 63: t])
    return (recent - baseline) / max(baseline_std, 1e-8)

def composite_signal(mom: float, vol_regime: float, sent: float,
                      w_mom: float = 0.5, w_sent: float = 0.3, w_vol: float = 0.2,
                      vol_threshold: float = 1.5) -> float:
    """
    Combine signals with vol-regime scaling.
    In high-vol regime: reduce signal strength (defensive).
    """
    raw = w_mom * mom + w_sent * sent
    # Vol regime penalty: scale down in high-vol environments
    vol_penalty = 1.0 / max(1.0, (vol_regime / vol_threshold)**0.5)
    return raw * vol_penalty

# ---------------------------------------------------------------------------
# 2. Position sizing
# ---------------------------------------------------------------------------
def signal_to_weight(signal: float,
                      max_position: float = 0.20,
                      signal_scale: float = 2.0) -> float:
    """
    Sigmoid-based position sizing: w = max_pos * tanh(signal / scale).
    Smooth saturation at max position.
    """
    return max_position * math.tanh(signal / signal_scale)

def vol_target_weight(signal: float, realized_vol: float,
                       target_vol: float = 0.10,
                       max_position: float = 0.30) -> float:
    """
    Volatility-targeting: scale position so that position * vol = target_vol.
    w = (target_vol / realized_vol) * sign(signal) * min(|signal|, 1)
    """
    if realized_vol < 1e-6:
        return 0.0
    scale = min(target_vol / realized_vol, max_position)
    return scale * math.tanh(signal)

# ---------------------------------------------------------------------------
# 3. Backtest engine
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    dates: list[int]
    portfolio_value: list[float]
    daily_returns: list[float]
    weights: list[float]
    signals: list[float]
    turnover: list[float]
    n_trades: int
    total_costs: float

def backtest(returns: list[float],
              signals: list[float],
              max_position: float = 0.20,
              signal_scale: float = 2.0,
              transaction_cost: float = 0.0005,
              bid_ask_spread: float = 0.0002,
              slippage_bps: float = 0.5,
              rebal_freq: int = 1,
              initial_value: float = 1.0) -> BacktestResult:
    """
    Single-asset backtest engine.
    Position = signal_to_weight(signal_t) applied to one risky asset.
    Cash = 1 - |position|.
    """
    T = len(returns)
    if len(signals) != T:
        raise ValueError("returns and signals must be same length")

    slippage = slippage_bps / 10000

    port_value = initial_value
    current_weight = 0.0
    portfolio_values = [port_value]
    daily_returns_out = []
    weights_out = [0.0]
    signals_out = [signals[0] if signals else 0.0]
    turnover_out = [0.0]
    n_trades = 0
    total_costs = 0.0

    for t in range(1, T):
        # Target weight from signal
        if t % rebal_freq == 0:
            target_weight = signal_to_weight(signals[t], max_position, signal_scale)
        else:
            target_weight = current_weight

        # Transaction costs on turnover
        trade_size = abs(target_weight - current_weight)
        cost = trade_size * (transaction_cost + bid_ask_spread / 2 + slippage)
        if trade_size > 0.001:
            n_trades += 1
            total_costs += cost * port_value

        # P&L: risky return * weight + cost
        gross_ret = current_weight * returns[t]
        net_ret = gross_ret - cost
        port_value *= (1 + net_ret)

        daily_returns_out.append(net_ret)
        portfolio_values.append(port_value)
        weights_out.append(current_weight)
        signals_out.append(signals[t])
        turnover_out.append(trade_size)

        current_weight = target_weight

    return BacktestResult(
        dates=list(range(T)),
        portfolio_value=portfolio_values,
        daily_returns=daily_returns_out,
        weights=weights_out,
        signals=signals_out,
        turnover=turnover_out,
        n_trades=n_trades,
        total_costs=total_costs,
    )

# ---------------------------------------------------------------------------
# 4. Performance metrics
# ---------------------------------------------------------------------------
def performance_metrics(result: BacktestResult, rf: float = 0.02) -> dict:
    rets = result.daily_returns
    if not rets:
        return {}

    n = len(rets)
    ann_factor = 252
    cum_ret = result.portfolio_value[-1] - 1.0
    ann_ret = (result.portfolio_value[-1]) ** (ann_factor / n) - 1
    ann_vol = std(rets) * math.sqrt(ann_factor)
    rf_daily = (1 + rf) ** (1 / ann_factor) - 1

    sharpe = (ann_ret - rf) / max(ann_vol, 1e-6)

    # Sortino
    downside = [min(r - rf_daily, 0.0)**2 for r in rets]
    sortino_vol = math.sqrt(mean(downside) * ann_factor)
    sortino = (ann_ret - rf) / max(sortino_vol, 1e-6)

    # Max drawdown
    peak = result.portfolio_value[0]
    max_dd = 0.0
    for v in result.portfolio_value:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)

    calmar = ann_ret / max(max_dd, 1e-6)

    # Win rate
    wins = sum(1 for r in rets if r > 0)
    win_rate = wins / n

    # Avg turnover
    avg_to = mean(result.turnover)
    ann_to = avg_to * ann_factor

    return {
        'cum_return': cum_ret,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'avg_daily_turnover': avg_to,
        'ann_turnover': ann_to,
        'n_trades': result.n_trades,
        'total_costs': result.total_costs,
    }

# ---------------------------------------------------------------------------
# 5. Signal decay analysis
# ---------------------------------------------------------------------------
def signal_decay_analysis(returns: list[float], signals: list[float],
                            max_lag: int = 20) -> list[tuple[int, float, float]]:
    """
    Information coefficient (IC) at each lag:
    IC(lag) = correlation(signal_t, return_{t+lag}).
    Returns list of (lag, IC, t-stat).
    """
    results = []
    n = len(returns)
    for lag in range(1, max_lag + 1):
        pairs = [(signals[t], returns[t + lag]) for t in range(n - lag)
                 if abs(signals[t]) > 0]
        if len(pairs) < 10:
            results.append((lag, 0.0, 0.0))
            continue
        s_vals = [p[0] for p in pairs]
        r_vals = [p[1] for p in pairs]
        s_m, r_m = mean(s_vals), mean(r_vals)
        num = sum((s - s_m) * (r - r_m) for s, r in zip(s_vals, r_vals))
        den = math.sqrt(sum((s - s_m)**2 for s in s_vals) * sum((r - r_m)**2 for r in r_vals))
        ic = num / max(den, 1e-10)
        t_stat = ic * math.sqrt(len(pairs) - 2) / math.sqrt(max(1 - ic**2, 1e-8))
        results.append((lag, ic, t_stat))
    return results

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("DAY 28: News-Driven Signal Backtesting Pipeline")
    print("=" * 65)

    rng = random.Random(42)
    T = 756  # 3 years
    rf = 0.02

    def randn():
        u = max(rng.random(), 1e-15)
        return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * rng.random())

    # Simulate price returns (with mild momentum + mean reversion)
    daily_vol = 0.015
    returns_sim = []
    z_prev = 0.0
    for t in range(T):
        eps = randn()
        z = 0.1 * z_prev + 0.99 * eps  # slight momentum in Z
        r = 0.0003 + daily_vol * z      # small positive drift + vol
        returns_sim.append(r)
        z_prev = z

    # Simulate sentiment series correlated with future returns
    sentiment_sim = []
    for t in range(T):
        # Sentiment has predictive power for next 5 days
        future_avg = mean(returns_sim[min(t + 1, T - 1): min(t + 6, T)]) if t + 1 < T else 0.0
        noise = randn() * 0.3
        sent = 3.0 * future_avg / daily_vol + noise  # ~IC 0.05-0.10
        sentiment_sim.append(sent)

    print(f"\nSimulation: T={T} days, daily_vol={daily_vol:.3f}")
    print(f"Annualized vol: {daily_vol * math.sqrt(252):.3f}")

    print("\n1. Signal Construction at each day")
    signals_mom = []
    signals_sent = []
    signals_vol = []
    signals_comp = []

    for t in range(T):
        mom = price_momentum_signal(returns_sim, t, lookback=21, skip=1)
        vr = ewma_vol_regime(returns_sim, t)
        sent = sentiment_signal(sentiment_sim, t, window=5)
        comp = composite_signal(mom, vr, sent)
        signals_mom.append(mom)
        signals_sent.append(sent)
        signals_vol.append(vr)
        signals_comp.append(comp)

    # Non-zero signal fraction
    nz_mom  = sum(1 for s in signals_mom  if abs(s) > 0.1) / T
    nz_sent = sum(1 for s in signals_sent if abs(s) > 0.1) / T
    nz_comp = sum(1 for s in signals_comp if abs(s) > 0.1) / T
    print(f"   Active signal fraction — Momentum: {nz_mom:.2%}  Sentiment: {nz_sent:.2%}  Composite: {nz_comp:.2%}")

    print("\n2. Backtest Comparison")
    strategies = [
        ("Momentum",  signals_mom),
        ("Sentiment", signals_sent),
        ("Composite", signals_comp),
        ("Buy-Hold",  [0.20] * T),  # constant 20% long
    ]

    metrics_all = {}
    for name, sigs in strategies:
        res = backtest(returns_sim, sigs, max_position=0.20, signal_scale=2.0,
                       transaction_cost=0.0005, rebal_freq=1)
        m = performance_metrics(res, rf)
        metrics_all[name] = m

    print(f"   {'Strategy':12} | {'Ann Ret':>8} | {'Ann Vol':>8} | {'Sharpe':>8} | {'Max DD':>8} | {'Ann TO':>8}")
    print("   " + "-" * 65)
    for name, m in metrics_all.items():
        print(f"   {name:12} | {m['ann_return']:>+8.4f} | {m['ann_vol']:>8.4f} | "
              f"{m['sharpe']:>8.3f} | {m['max_drawdown']:>8.4f} | {m['ann_turnover']:>8.2f}")

    print("\n3. Transaction Cost Sensitivity (Composite signal)")
    print(f"   {'Cost (bps)':>12} | {'Ann Ret':>8} | {'Sharpe':>8} | {'Net vs Gross':>14}")
    print("   " + "-" * 50)
    gross_res = backtest(returns_sim, signals_comp, transaction_cost=0.0, rebal_freq=1)
    gross_m = performance_metrics(gross_res, rf)
    for tc_bps in [0, 2, 5, 10, 20, 50]:
        tc = tc_bps / 10000
        res_tc = backtest(returns_sim, signals_comp, transaction_cost=tc, rebal_freq=1)
        m_tc = performance_metrics(res_tc, rf)
        net_vs_gross = m_tc['ann_return'] - gross_m['ann_return']
        print(f"   {tc_bps:>12} | {m_tc['ann_return']:>+8.4f} | {m_tc['sharpe']:>8.3f} | {net_vs_gross:>+14.4f}")

    print("\n4. Signal Decay (Information Coefficient by Lag)")
    decay = signal_decay_analysis(returns_sim, signals_comp, max_lag=15)
    print(f"   {'Lag':>5} | {'IC':>8} | {'t-stat':>8} | {'Decay bar':}")
    print("   " + "-" * 50)
    ic_max = max(abs(d[1]) for d in decay) if decay else 1.0
    for lag, ic, t in decay:
        bar_len = int(abs(ic) / max(ic_max, 0.001) * 20)
        bar = ('█' if ic > 0 else '░') * bar_len
        sig = '*' if abs(t) > 1.96 else ''
        print(f"   {lag:>5} | {ic:>+8.4f} | {t:>+8.3f} | {bar} {sig}")

    print("\n5. Rebalancing Frequency Sensitivity")
    print(f"   {'Rebal Freq':>12} | {'Ann Ret':>8} | {'Ann TO':>8} | {'Sharpe':>8}")
    print("   " + "-" * 48)
    for freq in [1, 5, 10, 21]:
        res_f = backtest(returns_sim, signals_comp, max_position=0.20,
                          transaction_cost=0.0005, rebal_freq=freq)
        m_f = performance_metrics(res_f, rf)
        print(f"   {freq:>12} | {m_f['ann_return']:>+8.4f} | {m_f['ann_turnover']:>8.2f} | {m_f['sharpe']:>8.3f}")

    print("\n[Done] Day 28: Signal Backtesting Pipeline complete.")
