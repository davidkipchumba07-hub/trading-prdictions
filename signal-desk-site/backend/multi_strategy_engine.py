"""
Multi-Strategy Ensemble Trading Engine
=======================================

WHAT THIS IS
------------
A rules-based system that combines several well-documented technical
trading approaches into one ensemble, instead of relying on a single
indicator. This mirrors how real systematic trading desks operate:

  1. TREND FOLLOWING       - SMA50/SMA200 golden/death cross, EMA stack
  2. MOMENTUM               - MACD histogram direction + rate of change
  3. MEAN REVERSION         - RSI extremes + Bollinger Band touches
  4. BREAKOUT               - Donchian channel (20-bar high/low break)
  5. MULTI-TIMEFRAME FILTER - higher-timeframe trend must agree
  6. REGIME DETECTION       - ADX decides whether trend or mean-reversion
                              strategies get more weight (trending vs
                              ranging market)
  7. CONFLUENCE REQUIREMENT - a trade only fires when enough independent
                              strategies agree, which trades win-rate
                              for trade frequency (fewer, higher-quality
                              signals)
  8. RISK MANAGEMENT        - ATR-based stops/targets, fixed-fractional
                              position sizing with a Kelly-fraction cap,
                              and a losing-streak circuit breaker

WHAT THIS IS NOT
----------------
- NOT a 90%-win-rate system, and it will not claim to be one, because that
  claim is not real. This prints its ACTUAL backtested numbers, whatever
  they are. If you see a tool anywhere claiming 90%+ win rate, be very
  skeptical of it.
- NOT financial advice, and not a guarantee of profit. Past performance in
  a backtest never guarantees future results.
- The "efficiency" that matters in real trading isn't win rate — it's
  expectancy (average $ won or lost per trade) and profit factor (gross
  wins / gross losses). A system that wins 40% of the time but makes 3x on
  winners and loses 1x on losers is far more "efficient" than one that
  wins 90% of the time but loses big on the 10%. This report shows both.

REQUIREMENTS
------------
    pip install pandas numpy yfinance --break-system-packages

USAGE
-----
    python multi_strategy_engine.py --demo
    python multi_strategy_engine.py --symbol EURUSD=X --interval 1h --period 60d
    python multi_strategy_engine.py --csv your_data.csv
    python multi_strategy_engine.py --demo --min-confluence 3 --risk-reward 2.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from signal_engine import (
    sma, ema, rsi, macd, bollinger_bands, atr, adx,
    load_from_yfinance, load_from_csv,
)


def generate_unbiased_demo_data(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """
    Honest synthetic OHLC data: a pure random walk with realistic
    volatility clustering (GARCH-like), and NO injected deterministic
    drift/cycle.

    signal_engine.py's generate_demo_data() bakes a smooth sine-wave into
    the returns, which trend-following/breakout strategies can partially
    "detect" - that's a demo-data artifact, not a real edge, and it
    inflates backtest results. This generator avoids that so backtest
    numbers here reflect what a strategy does against a market with no
    real predictable structure, which is the honest baseline to compare
    against real market data.
    """
    rng = np.random.default_rng(seed)
    vol = np.zeros(n)
    vol[0] = 0.0006
    for t in range(1, n):
        # simple volatility clustering: vol mean-reverts with random shocks
        vol[t] = 0.0006 + 0.9 * (vol[t - 1] - 0.0006) + rng.normal(0, 0.00005)
        vol[t] = max(vol[t], 0.0002)
    returns = rng.normal(0, 1, n) * vol  # zero mean - no drift, no cycle
    price = 1.1000 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(rng.normal(0, 0.0004, n)))
    low = price * (1 - np.abs(rng.normal(0, 0.0004, n)))
    open_ = np.roll(price, 1)
    open_[0] = price[0]
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": price, "volume": volume}
    )


# ======================================================================
# 1. EXTRA INDICATORS (beyond what signal_engine.py already has)
# ======================================================================

def donchian_channel(df: pd.DataFrame, length: int = 20):
    # shift(1) excludes the current bar from its own channel - without this,
    # "close >= upper" can almost never trigger, since today's high is
    # always >= today's close (this was the bug bias_diagnostics.py found:
    # breakout never firing).
    upper = df["high"].shift(1).rolling(length).max()
    lower = df["low"].shift(1).rolling(length).min()
    return upper, lower


def roc(series: pd.Series, length: int = 10) -> pd.Series:
    """Rate of change, % over `length` bars."""
    return (series / series.shift(length) - 1.0) * 100


def add_ensemble_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_20"] = sma(df["close"], 20)
    df["sma_50"] = sma(df["close"], 50)
    df["sma_200"] = sma(df["close"], 200)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["rsi_14"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd_line, signal_line, hist
    mid, upper, lower = bollinger_bands(df["close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = mid, upper, lower
    df["atr_14"] = atr(df, 14)
    df["adx_14"] = adx(df, 14)
    df["roc_10"] = roc(df["close"], 10)
    df["donchian_upper"], df["donchian_lower"] = donchian_channel(df, 20)
    # Higher-timeframe proxy: a slower-moving average acts as the
    # "higher timeframe trend" filter when a true separate timeframe
    # feed isn't available.
    df["htf_trend"] = sma(df["close"], 100)
    return df


# ======================================================================
# 2. INDIVIDUAL STRATEGIES
#    Each returns a vote in {-1, 0, +1} and a short reason string.
#    -1 = bearish, +1 = bullish, 0 = no opinion right now.
# ======================================================================

@dataclass
class StrategyVote:
    name: str
    vote: int
    reason: str


def strat_trend_following(row: pd.Series) -> StrategyVote:
    if pd.isna(row["sma_200"]):
        return StrategyVote("trend_following", 0, "insufficient history for SMA200")
    if row["sma_50"] > row["sma_200"] and row["close"] > row["sma_50"]:
        return StrategyVote("trend_following", 1, "price > SMA50 > SMA200 (uptrend structure)")
    if row["sma_50"] < row["sma_200"] and row["close"] < row["sma_50"]:
        return StrategyVote("trend_following", -1, "price < SMA50 < SMA200 (downtrend structure)")
    return StrategyVote("trend_following", 0, "no clear trend structure")


def strat_momentum(row: pd.Series) -> StrategyVote:
    if row["macd_hist"] > 0 and row["roc_10"] > 0:
        return StrategyVote("momentum", 1, "MACD histogram positive and price rising over 10 bars")
    if row["macd_hist"] < 0 and row["roc_10"] < 0:
        return StrategyVote("momentum", -1, "MACD histogram negative and price falling over 10 bars")
    return StrategyVote("momentum", 0, "momentum mixed")


def strat_mean_reversion(row: pd.Series) -> StrategyVote:
    # Only meaningful in ranging conditions - the regime gate downweights
    # this elsewhere, but the strategy itself still only fires at extremes.
    if row["rsi_14"] < 30 and row["close"] <= row["bb_lower"]:
        return StrategyVote("mean_reversion", 1, "RSI oversold + price at/below lower Bollinger Band")
    if row["rsi_14"] > 70 and row["close"] >= row["bb_upper"]:
        return StrategyVote("mean_reversion", -1, "RSI overbought + price at/above upper Bollinger Band")
    return StrategyVote("mean_reversion", 0, "no reversion extreme")


def strat_breakout(row: pd.Series) -> StrategyVote:
    if pd.isna(row["donchian_upper"]):
        return StrategyVote("breakout", 0, "insufficient history for Donchian channel")
    if row["close"] >= row["donchian_upper"]:
        return StrategyVote("breakout", 1, "price broke above 20-bar high (Donchian breakout)")
    if row["close"] <= row["donchian_lower"]:
        return StrategyVote("breakout", -1, "price broke below 20-bar low (Donchian breakdown)")
    return StrategyVote("breakout", 0, "price inside channel")


def strat_multi_timeframe_filter(row: pd.Series) -> StrategyVote:
    """Acts as a confirming filter: higher-timeframe trend must agree
    with the direction, otherwise it votes against / neutralizes."""
    if pd.isna(row["htf_trend"]):
        return StrategyVote("higher_tf_filter", 0, "insufficient history for higher-timeframe trend")
    if row["close"] > row["htf_trend"]:
        return StrategyVote("higher_tf_filter", 1, "price above higher-timeframe trend line")
    if row["close"] < row["htf_trend"]:
        return StrategyVote("higher_tf_filter", -1, "price below higher-timeframe trend line")
    return StrategyVote("higher_tf_filter", 0, "flat")


STRATEGIES = [
    strat_trend_following,
    strat_momentum,
    strat_mean_reversion,
    strat_breakout,
    strat_multi_timeframe_filter,
]


# ======================================================================
# 3. REGIME DETECTION + ENSEMBLE VOTE
# ======================================================================

def regime(row: pd.Series) -> str:
    """ADX > 25: trending market. ADX < 20: ranging market. Between: mixed."""
    if pd.isna(row["adx_14"]):
        return "unknown"
    if row["adx_14"] > 25:
        return "trending"
    if row["adx_14"] < 20:
        return "ranging"
    return "mixed"


# Regime-based weighting: in a trending market, trend/momentum/breakout
# strategies get more say; in a ranging market, mean reversion does.
# This is a standard technique real systematic traders use - the same
# strategy that works in a trend will bleed money in a range, and
# vice versa, so which one gets weight should depend on measured regime.
REGIME_WEIGHTS = {
    "trending": {
        "trend_following": 1.5, "momentum": 1.3, "mean_reversion": 0.3,
        "breakout": 1.4, "higher_tf_filter": 1.0,
    },
    "ranging": {
        "trend_following": 0.4, "momentum": 0.6, "mean_reversion": 1.6,
        "breakout": 0.5, "higher_tf_filter": 0.8,
    },
    "mixed": {
        "trend_following": 1.0, "momentum": 1.0, "mean_reversion": 1.0,
        "breakout": 1.0, "higher_tf_filter": 1.0,
    },
    "unknown": {
        "trend_following": 1.0, "momentum": 1.0, "mean_reversion": 1.0,
        "breakout": 1.0, "higher_tf_filter": 1.0,
    },
}


@dataclass
class EnsembleSignal:
    direction: str            # BUY / SELL / NEUTRAL
    confluence: int           # how many strategies agreed (of those with an opinion)
    n_opinions: int           # how many strategies had a non-zero vote
    weighted_score: float     # -100..100
    regime_label: str
    votes: list = field(default_factory=list)


def evaluate_row(row: pd.Series) -> EnsembleSignal:
    votes = [s(row) for s in STRATEGIES]
    reg = regime(row)
    weights = REGIME_WEIGHTS[reg]

    weighted_sum = 0.0
    weight_total = 0.0
    n_bullish, n_bearish, n_opinions = 0, 0, 0

    for v in votes:
        w = weights.get(v.name, 1.0)
        weight_total += w
        if v.vote != 0:
            n_opinions += 1
            weighted_sum += v.vote * w
            if v.vote > 0:
                n_bullish += 1
            else:
                n_bearish += 1

    weighted_score = (weighted_sum / weight_total) * 100 if weight_total else 0.0
    confluence = max(n_bullish, n_bearish)

    if weighted_score > 15 and n_bullish > n_bearish:
        direction = "BUY"
    elif weighted_score < -15 and n_bearish > n_bullish:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    return EnsembleSignal(
        direction=direction,
        confluence=confluence,
        n_opinions=n_opinions,
        weighted_score=round(weighted_score, 1),
        regime_label=reg,
        votes=votes,
    )


# ======================================================================
# 4. RISK MANAGEMENT: position sizing + losing-streak circuit breaker
# ======================================================================

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float, cap: float = 0.25) -> float:
    """
    Classic Kelly formula, capped hard at `cap` (e.g. 25% of the full-Kelly
    size). Full Kelly is usually far too aggressive for real trading, so
    professional use is almost always fractional Kelly - this defaults to
    a max of 25% of the theoretical optimum.
    """
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    f = win_rate - (1 - win_rate) / b
    return max(0.0, min(f, 1.0)) * cap


# ======================================================================
# 5. WALK-FORWARD BACKTEST WITH CONFLUENCE REQUIREMENT + RISK CONTROLS
# ======================================================================

@dataclass
class EnsembleBacktestResult:
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    total_return_pct: float
    max_drawdown_pct: float
    equity_curve: pd.Series
    circuit_breaker_pauses: int


def backtest_ensemble(
    df: pd.DataFrame,
    min_confluence: int = 3,
    atr_mult_sl: float = 1.5,
    risk_reward: float = 1.5,
    base_risk_pct: float = 1.0,
    max_kelly_cap: float = 0.25,
    losing_streak_pause: int = 3,
    pause_bars: int = 10,
    starting_equity: float = 10_000.0,
    lookahead: int = 50,
) -> EnsembleBacktestResult:
    df = add_ensemble_indicators(df)

    equity = starting_equity
    equity_curve = [equity]
    trades_log = []  # list of R-multiples (win/loss in units of risk)

    consecutive_losses = 0
    pause_until = -1

    i = 200  # need enough bars for SMA200 to be valid
    while i < len(df) - lookahead:
        row = df.iloc[i]

        if i < pause_until:
            equity_curve.append(equity)
            i += 1
            continue

        sig = evaluate_row(row)

        if sig.direction == "NEUTRAL" or sig.confluence < min_confluence:
            equity_curve.append(equity)
            i += 1
            continue

        entry = row["close"]
        a = row["atr_14"]
        if pd.isna(a) or a <= 0:
            equity_curve.append(equity)
            i += 1
            continue

        stop_dist = a * atr_mult_sl
        if sig.direction == "BUY":
            stop = entry - stop_dist
            target = entry + stop_dist * risk_reward
        else:
            stop = entry + stop_dist
            target = entry - stop_dist * risk_reward

        # walk forward to see what gets hit first
        outcome_r = None
        for j in range(i + 1, min(i + 1 + lookahead, len(df))):
            hi, lo = df.iloc[j]["high"], df.iloc[j]["low"]
            if sig.direction == "BUY":
                if lo <= stop:
                    outcome_r = -1.0
                    break
                if hi >= target:
                    outcome_r = risk_reward
                    break
            else:
                if hi >= stop:
                    outcome_r = -1.0
                    break
                if lo <= target:
                    outcome_r = risk_reward
                    break
        if outcome_r is None:
            equity_curve.append(equity)
            i += 1
            continue

        # position sizing: recompute Kelly fraction from trades so far
        # (walk-forward, not lookahead-biased) once we have enough sample
        if len(trades_log) >= 15:
            wins_so_far = [r for r in trades_log if r > 0]
            losses_so_far = [r for r in trades_log if r < 0]
            wr = len(wins_so_far) / len(trades_log)
            avg_w = np.mean(wins_so_far) if wins_so_far else 0
            avg_l = abs(np.mean(losses_so_far)) if losses_so_far else 1
            risk_pct = max(0.25, min(kelly_fraction(wr, avg_w, avg_l, max_kelly_cap) * 100, base_risk_pct * 2))
        else:
            risk_pct = base_risk_pct

        risk_amount = equity * (risk_pct / 100)
        pnl = risk_amount * outcome_r
        equity += pnl
        equity_curve.append(equity)
        trades_log.append(outcome_r)

        if outcome_r < 0:
            consecutive_losses += 1
            if consecutive_losses >= losing_streak_pause:
                pause_until = i + 1 + pause_bars  # circuit breaker: stand aside after a losing streak
                consecutive_losses = 0
        else:
            consecutive_losses = 0

        i += 1

    wins = [r for r in trades_log if r > 0]
    losses = [r for r in trades_log if r < 0]
    n_trades = len(trades_log)
    win_rate = round(100 * len(wins) / n_trades, 2) if n_trades else 0.0

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
    expectancy_r = round(np.mean(trades_log), 3) if trades_log else 0.0

    curve = pd.Series(equity_curve)
    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max * 100
    max_dd = round(drawdown.min(), 2) if len(drawdown) else 0.0
    total_return = round((equity / starting_equity - 1) * 100, 2)

    return EnsembleBacktestResult(
        trades=n_trades,
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy_r=expectancy_r,
        total_return_pct=total_return,
        max_drawdown_pct=max_dd,
        equity_curve=curve,
        circuit_breaker_pauses=0,
    )


# ======================================================================
# 6. REPORTING
# ======================================================================

def print_ensemble_signal(df: pd.DataFrame):
    df2 = add_ensemble_indicators(df)
    row = df2.iloc[-1]
    sig = evaluate_row(row)

    print("=" * 60)
    print("ENSEMBLE SIGNAL")
    print("=" * 60)
    print(f"Direction:        {sig.direction}")
    print(f"Weighted score:   {sig.weighted_score} (-100..100)")
    print(f"Confluence:       {sig.confluence} strategies agree")
    print(f"Market regime:    {sig.regime_label} (ADX {row['adx_14']:.1f})")
    print(f"Last close:       {row['close']:.5f}")
    print("-" * 60)
    print("Individual strategy votes:")
    for v in sig.votes:
        arrow = {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}[v.vote]
        print(f"  [{arrow:8s}] {v.name:18s} - {v.reason}")
    print("=" * 60)


def print_ensemble_backtest(res: EnsembleBacktestResult, min_confluence: int):
    print()
    print("=" * 60)
    print(f"ENSEMBLE BACKTEST (min confluence = {min_confluence} strategies)")
    print("=" * 60)
    print(f"Trades:            {res.trades}")
    print(f"Wins / Losses:     {res.wins} / {res.losses}")
    print(f"Win rate:          {res.win_rate}%")
    print(f"Profit factor:     {res.profit_factor}   (gross win / gross loss - >1.0 = net profitable)")
    print(f"Expectancy:        {res.expectancy_r} R per trade   (average result in units of risk)")
    print(f"Total return:      {res.total_return_pct}%")
    print(f"Max drawdown:      {res.max_drawdown_pct}%")
    print("-" * 60)
    print("READ THIS: win rate alone is a misleading number. A system can")
    print("win 35% of trades and still be profitable if winners are large")
    print("relative to losers (see profit factor and expectancy above), or")
    print("win 70% of trades and still lose money if losers are large. If")
    print("expectancy is <= 0 here, this parameter set / market has no real")
    print("edge and should not be traded live regardless of win rate.")
    print("=" * 60)


# ======================================================================
# 7. CLI
# ======================================================================

def main():
    p = argparse.ArgumentParser(description="Multi-strategy ensemble trading engine")
    p.add_argument("--symbol", type=str, help="Ticker, e.g. EURUSD=X")
    p.add_argument("--interval", type=str, default="1h")
    p.add_argument("--period", type=str, default="60d")
    p.add_argument("--csv", type=str, help="Path to OHLC CSV instead of yfinance")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    p.add_argument("--min-confluence", type=int, default=3,
                    help="Minimum number of agreeing strategies to trade (default 3 of 5)")
    p.add_argument("--risk-reward", type=float, default=1.5)
    p.add_argument("--atr-mult", type=float, default=1.5)
    p.add_argument("--risk-pct", type=float, default=1.0)
    args = p.parse_args()

    if args.csv:
        df = load_from_csv(args.csv)
    elif args.symbol:
        df = load_from_yfinance(args.symbol, args.interval, args.period)
    else:
        df = generate_unbiased_demo_data()
        print("(using unbiased synthetic random-walk data - pass --symbol or --csv for real data)\n")

    print_ensemble_signal(df)

    res = backtest_ensemble(
        df,
        min_confluence=args.min_confluence,
        atr_mult_sl=args.atr_mult,
        risk_reward=args.risk_reward,
        base_risk_pct=args.risk_pct,
    )
    print_ensemble_backtest(res, args.min_confluence)


if __name__ == "__main__":
    main()
