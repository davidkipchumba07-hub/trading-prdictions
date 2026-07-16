"""
Forex / Binary-Options Technical Analysis & Signal Engine
===========================================================

WHAT THIS IS
------------
A rules-based technical analysis engine that:
  1. Computes standard indicators (SMA/EMA, RSI, MACD, Bollinger Bands, ATR, ADX)
  2. Combines them into a scored BUY/SELL/CALL/PUT signal with a confidence level
  3. Suggests entry, stop-loss, and take-profit levels (for forex-style trades)
     or expiry-appropriate direction calls (for binary-style trades)
  4. Backtests the strategy on historical OHLC data so you can see real
     win-rate, drawdown, and expectancy numbers BEFORE risking money

WHAT THIS IS NOT
----------------
- Not a guarantee of profit. No indicator combination predicts the future
  reliably. Markets are noisy, and past performance in a backtest does not
  guarantee future results.
- Not financial advice. I'm not a licensed financial advisor, and this tool
  doesn't replace one. Use it to inform your own decisions, and treat any
  live money on binary options / leveraged forex as capital you can afford
  to lose in full.
- Binary options in particular are heavily associated with fraud and are
  banned or restricted for retail traders in many jurisdictions (US, EU,
  UK, Australia, etc.). Check your local regulations before using any
  broker. This tool only helps analyze price data — it doesn't execute
  trades or interact with any broker.

REQUIREMENTS
------------
    pip install pandas numpy yfinance --break-system-packages

USAGE
-----
    python signal_engine.py --symbol EURUSD=X --interval 1h --period 60d
    python signal_engine.py --csv my_data.csv          # use your own OHLC data
    python signal_engine.py --demo                       # run on synthetic data, no internet needed
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ======================================================================
# 1. INDICATORS
# ======================================================================

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, length: int = 20, std_mult: float = 2.0):
    mid = sma(series, length)
    std = series.rolling(length).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / length, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False).mean().fillna(0)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_20"] = sma(df["close"], 20)
    df["sma_50"] = sma(df["close"], 50)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["rsi_14"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    upper, mid, lower = bollinger_bands(df["close"])
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower
    df["atr_14"] = atr(df, 14)
    df["adx_14"] = adx(df, 14)
    return df


# ======================================================================
# 2. SIGNAL SCORING
# ======================================================================

@dataclass
class Signal:
    direction: str          # "BUY" / "SELL" / "NEUTRAL"
    confidence: float       # 0-100
    reasons: list = field(default_factory=list)
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    binary_call: Optional[str] = None   # "CALL" / "PUT" / "NO TRADE"


def score_row(row: pd.Series) -> Signal:
    """
    Combine trend (moving averages + ADX), momentum (RSI + MACD), and
    volatility (Bollinger position) into a single directional score.
    Score ranges roughly -100 (strong sell) to +100 (strong buy).
    """
    score = 0.0
    reasons = []

    # --- Trend component (weight: 35) ---
    if row["sma_20"] > row["sma_50"]:
        score += 15
        reasons.append("SMA20 > SMA50 (uptrend)")
    else:
        score -= 15
        reasons.append("SMA20 < SMA50 (downtrend)")

    if row["ema_12"] > row["ema_26"]:
        score += 10
        reasons.append("EMA12 > EMA26 (bullish cross state)")
    else:
        score -= 10
        reasons.append("EMA12 < EMA26 (bearish cross state)")

    # ADX confirms trend strength (doesn't add direction, scales confidence)
    trend_strength = min(row["adx_14"] / 50.0, 1.0)  # 0-1

    # --- Momentum component (weight: 40) ---
    if row["rsi_14"] < 30:
        score += 20
        reasons.append(f"RSI {row['rsi_14']:.1f} oversold")
    elif row["rsi_14"] > 70:
        score -= 20
        reasons.append(f"RSI {row['rsi_14']:.1f} overbought")

    if row["macd_hist"] > 0:
        score += 15
        reasons.append("MACD histogram positive")
    else:
        score -= 15
        reasons.append("MACD histogram negative")

    # --- Volatility / mean-reversion component (weight: 25) ---
    bb_range = row["bb_upper"] - row["bb_lower"]
    if bb_range > 0:
        bb_pos = (row["close"] - row["bb_lower"]) / bb_range  # 0-1
        if bb_pos < 0.1:
            score += 15
            reasons.append("Price near lower Bollinger Band")
        elif bb_pos > 0.9:
            score -= 15
            reasons.append("Price near upper Bollinger Band")

    # Base confidence is how much the components agree (max possible |score| = 100
    # when trend + momentum + volatility all point the same way).
    base_confidence = min(abs(score), 100)

    # ADX nudges confidence up/down modestly rather than crushing it — a weak
    # trend (low ADX) shaves confidence, a strong trend adds a small bonus.
    # Range: 0.8x (choppy) to 1.1x (strongly trending), instead of 0.5x-1.0x.
    adx_factor = 0.8 + 0.3 * trend_strength
    confidence = min(base_confidence * adx_factor, 100)

    if score > 20:
        direction = "BUY"
    elif score < -20:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    if trend_strength < 0.2:
        reasons.append(f"ADX weak ({row['adx_14']:.1f}) — choppy market, confidence reduced")

    return Signal(direction=direction, confidence=round(confidence, 1), reasons=reasons)


def build_trade_plan(signal: Signal, row: pd.Series, atr_mult_sl: float = 1.5,
                      risk_reward: float = 1.5) -> Signal:
    """Attach entry/stop/target for forex-style trades, and a CALL/PUT
    recommendation for binary-style trades, using ATR for sizing."""
    price = row["close"]
    a = row["atr_14"]

    if signal.direction == "BUY":
        signal.entry = price
        signal.stop_loss = price - atr_mult_sl * a
        signal.take_profit = price + atr_mult_sl * risk_reward * a
        signal.binary_call = "CALL" if signal.confidence >= 30 else "NO TRADE"
    elif signal.direction == "SELL":
        signal.entry = price
        signal.stop_loss = price + atr_mult_sl * a
        signal.take_profit = price - atr_mult_sl * risk_reward * a
        signal.binary_call = "PUT" if signal.confidence >= 30 else "NO TRADE"
    else:
        signal.binary_call = "NO TRADE"

    return signal


def latest_signal(df: pd.DataFrame) -> Signal:
    df = add_all_indicators(df)
    row = df.iloc[-1]
    sig = score_row(row)
    sig = build_trade_plan(sig, row)
    return sig


# ======================================================================
# 3. BACKTEST ENGINE
# ======================================================================

@dataclass
class BacktestResult:
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_r_multiple: float
    equity_curve: pd.Series


def backtest_forex_style(df: pd.DataFrame, atr_mult_sl: float = 1.5,
                          risk_reward: float = 1.5, min_confidence: float = 55.0,
                          starting_equity: float = 10_000.0,
                          risk_per_trade_pct: float = 1.0) -> BacktestResult:
    """
    Walk-forward backtest: at each bar, generate a signal using only data up
    to that bar, then simulate whether stop-loss or take-profit is hit first
    using subsequent bars' highs/lows. Position sized as a fixed % risk of
    equity per trade (compounding).
    """
    df = add_all_indicators(df).dropna().reset_index(drop=True)
    equity = starting_equity
    equity_curve = [equity]
    trades, wins, losses = 0, 0, 0
    r_multiples = []

    i = 0
    n = len(df)
    while i < n - 1:
        row = df.iloc[i]
        sig = score_row(row)
        sig = build_trade_plan(sig, row, atr_mult_sl, risk_reward)

        if sig.direction == "NEUTRAL" or sig.confidence < min_confidence:
            i += 1
            equity_curve.append(equity)
            continue

        entry = sig.entry
        stop = sig.stop_loss
        target = sig.take_profit
        direction = sig.direction

        risk_amount = equity * (risk_per_trade_pct / 100.0)
        risk_per_unit = abs(entry - stop)
        if risk_per_unit == 0:
            i += 1
            continue
        position_size = risk_amount / risk_per_unit

        # Look forward until stop or target is hit (max 50 bars ahead)
        outcome = None
        j = i + 1
        while j < min(i + 50, n):
            bar = df.iloc[j]
            if direction == "BUY":
                if bar["low"] <= stop:
                    outcome = "loss"
                    break
                if bar["high"] >= target:
                    outcome = "win"
                    break
            else:  # SELL
                if bar["high"] >= stop:
                    outcome = "loss"
                    break
                if bar["low"] <= target:
                    outcome = "win"
                    break
            j += 1

        trades += 1
        if outcome == "win":
            wins += 1
            pnl = position_size * abs(target - entry)
            r_multiples.append(risk_reward)
        elif outcome == "loss":
            losses += 1
            pnl = -position_size * abs(entry - stop)
            r_multiples.append(-1.0)
        else:
            # Neither hit within lookahead window -> close at last available price
            last_price = df.iloc[min(i + 49, n - 1)]["close"]
            pnl = position_size * (last_price - entry) if direction == "BUY" else position_size * (entry - last_price)
            r_multiples.append(pnl / risk_amount if risk_amount else 0)
            if pnl >= 0:
                wins += 1
            else:
                losses += 1

        equity += pnl
        equity_curve.append(equity)
        i = j if outcome else i + 1

    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100

    return BacktestResult(
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=round(100 * wins / trades, 2) if trades else 0.0,
        total_return_pct=round(100 * (equity - starting_equity) / starting_equity, 2),
        max_drawdown_pct=round(max_dd, 2),
        avg_r_multiple=round(float(np.mean(r_multiples)), 3) if r_multiples else 0.0,
        equity_curve=equity_series,
    )


# ======================================================================
# 4. DATA LOADING
# ======================================================================

def load_from_yfinance(symbol: str, interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit(
            "yfinance is not installed. Run: pip install yfinance --break-system-packages"
        )
    data = yf.download(symbol, interval=interval, period=period, progress=False)
    if data.empty:
        raise SystemExit(f"No data returned for {symbol}. Check the symbol/interval/period.")
    data = data.rename(columns=str.lower)
    data = data.rename(columns={"adj close": "adj_close"})
    return data[["open", "high", "low", "close", "volume"]]


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise SystemExit(f"CSV must contain columns: {required}. Found: {list(df.columns)}")
    return df


def generate_demo_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLC data (random-walk with drift + volatility clustering)
    so the tool can be demoed and sanity-checked with no internet access."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.0006, n) + np.sin(np.linspace(0, 20, n)) * 0.0003
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
# 5. CLI
# ======================================================================

def print_signal_report(sig: Signal, last_price: float):
    print("\n" + "=" * 60)
    print("LATEST SIGNAL")
    print("=" * 60)
    print(f"Direction        : {sig.direction}")
    print(f"Confidence       : {sig.confidence}/100")
    print(f"Binary call      : {sig.binary_call}")
    print(f"Last close price : {last_price:.5f}")
    if sig.entry is not None:
        print(f"Suggested entry  : {sig.entry:.5f}")
        print(f"Stop loss        : {sig.stop_loss:.5f}")
        print(f"Take profit      : {sig.take_profit:.5f}")
    print("\nReasons:")
    for r in sig.reasons:
        print(f"  - {r}")
    print("=" * 60)
    print(
        "NOTE: This is a technical-analysis output, not a guarantee. Position\n"
        "size according to your own risk tolerance and never risk more than\n"
        "you can afford to lose. Backtest before trading live."
    )


def print_backtest_report(res: BacktestResult):
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Trades taken       : {res.trades}")
    print(f"Wins / Losses      : {res.wins} / {res.losses}")
    print(f"Win rate           : {res.win_rate}%")
    print(f"Total return       : {res.total_return_pct}%")
    print(f"Max drawdown       : {res.max_drawdown_pct}%")
    print(f"Avg R-multiple     : {res.avg_r_multiple}")
    print("=" * 60)
    print(
        "Interpretation: win rate alone doesn't tell you if a strategy is\n"
        "profitable — a 40% win rate with 2:1 reward:risk can still be\n"
        "profitable, and a 60% win rate with poor risk:reward can still lose\n"
        "money. Look at total return and max drawdown together, and always\n"
        "test on out-of-sample data (a different period than you tuned on)\n"
        "before trusting these numbers."
    )


def main():
    parser = argparse.ArgumentParser(description="Forex/Binary technical signal & backtest engine")
    parser.add_argument("--symbol", type=str, help="Ticker for yfinance, e.g. EURUSD=X, GBPJPY=X")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval, e.g. 1h, 15m, 1d")
    parser.add_argument("--period", type=str, default="60d", help="History window, e.g. 60d, 6mo, 2y")
    parser.add_argument("--csv", type=str, help="Path to a CSV with open,high,low,close[,volume] columns")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic data (no internet needed)")
    parser.add_argument("--min-confidence", type=float, default=25.0, help="Min confidence to take a backtest trade")
    parser.add_argument("--risk-reward", type=float, default=1.5, help="Take-profit distance as a multiple of stop distance")
    parser.add_argument("--atr-mult", type=float, default=1.5, help="Stop-loss distance as a multiple of ATR")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Percent of equity risked per trade in backtest")
    args = parser.parse_args()

    if args.demo:
        df = generate_demo_data()
        label = "SYNTHETIC DEMO DATA"
    elif args.csv:
        df = load_from_csv(args.csv)
        label = args.csv
    elif args.symbol:
        df = load_from_yfinance(args.symbol, args.interval, args.period)
        label = args.symbol
    else:
        print("No data source given — defaulting to --demo. Use --symbol or --csv for real data.")
        df = generate_demo_data()
        label = "SYNTHETIC DEMO DATA"

    print(f"\nLoaded {len(df)} bars from: {label}")

    sig = latest_signal(df)
    print_signal_report(sig, df["close"].iloc[-1])

    res = backtest_forex_style(
        df,
        atr_mult_sl=args.atr_mult,
        risk_reward=args.risk_reward,
        min_confidence=args.min_confidence,
        risk_per_trade_pct=args.risk_pct,
    )
    print_backtest_report(res)


if __name__ == "__main__":
    main()
