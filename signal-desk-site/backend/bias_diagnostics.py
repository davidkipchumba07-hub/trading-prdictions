"""
Bias Diagnostics for the Ensemble Engine
==========================================

WHY THIS EXISTS
---------------
When backtesting multi_strategy_engine.py on a pure random-walk (no real
directional edge should exist), it showed a suspiciously high win rate
(~50-60%) and strongly positive expectancy across every random seed tested.
On a driftless random walk, a strategy with no real edge should hover
around breakeven expectancy - so a *consistent* positive result across
every seed means there's a look-ahead bug somewhere, not a real edge.

This script isolates the possible causes one at a time:
  1. RANDOM BASELINE  - random entries/directions through the same
                         stop/target simulation. If this alone is biased,
                         the bug is in the barrier-hit simulation itself.
  2. PER-STRATEGY      - each of the 5 strategies run alone (no ensemble,
                         no confluence requirement). If one strategy shows
                         a large, consistent edge on a random walk across
                         many seeds, that strategy's logic has a leak
                         (usually an accidental look-ahead, or an
                         indicator that reacts to information not really
                         available yet at decision time).

HOW TO USE THIS
---------------
Drop this file in the same folder as multi_strategy_engine.py and
signal_engine.py (backend/), then run:

    python bias_diagnostics.py --seeds 8

No changes to the other files are required - this only imports from them
and reports what it finds. Once the leaking strategy is identified, the
fix belongs in multi_strategy_engine.py itself (see the printed diagnosis
at the end of the run for a pointer to the likely cause).
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from multi_strategy_engine import (
    generate_unbiased_demo_data,
    add_ensemble_indicators,
    STRATEGIES,
)


def simulate_outcome(df: pd.DataFrame, i: int, direction: str,
                      atr_mult_sl: float, risk_reward: float, lookahead: int):
    row = df.iloc[i]
    entry = row["close"]
    a = row["atr_14"]
    if pd.isna(a) or a <= 0:
        return None
    stop_dist = a * atr_mult_sl
    if direction == "BUY":
        stop, target = entry - stop_dist, entry + stop_dist * risk_reward
    else:
        stop, target = entry + stop_dist, entry - stop_dist * risk_reward

    for j in range(i + 1, min(i + 1 + lookahead, len(df))):
        hi, lo = df.iloc[j]["high"], df.iloc[j]["low"]
        if direction == "BUY":
            if lo <= stop:
                return -1.0
            if hi >= target:
                return risk_reward
        else:
            if hi >= stop:
                return -1.0
            if lo <= target:
                return risk_reward
    return None


def random_baseline_test(df: pd.DataFrame, seed: int, entry_prob: float = 0.15,
                          atr_mult_sl: float = 1.5, risk_reward: float = 1.5,
                          lookahead: int = 50):
    """Random entries/directions through the same barrier simulation.
    Expected on a driftless random walk: win rate ~= 1/(1+risk_reward),
    expectancy ~= 0. If this comes back biased, the barrier simulation
    itself (not any strategy) is the problem."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for i in range(200, len(df) - lookahead):
        if rng.random() > entry_prob:
            continue
        direction = rng.choice(["BUY", "SELL"])
        outcome = simulate_outcome(df, i, direction, atr_mult_sl, risk_reward, lookahead)
        if outcome is not None:
            outcomes.append(outcome)
    return _summarize(outcomes)


def isolate_strategy_test(df: pd.DataFrame, strategy_fn, atr_mult_sl: float = 1.5,
                           risk_reward: float = 1.5, lookahead: int = 50):
    """Run a single strategy alone (its own votes only, no ensemble/confluence)
    through the same barrier simulation, so we can see if THIS strategy's
    filter correlates with real forward outcome on random-walk data."""
    outcomes = []
    for i in range(200, len(df) - lookahead):
        row = df.iloc[i]
        v = strategy_fn(row)
        if v.vote == 0:
            continue
        direction = "BUY" if v.vote > 0 else "SELL"
        outcome = simulate_outcome(df, i, direction, atr_mult_sl, risk_reward, lookahead)
        if outcome is not None:
            outcomes.append(outcome)
    return _summarize(outcomes)


def _summarize(outcomes):
    n = len(outcomes)
    if n == 0:
        return {"n": 0, "win_rate": None, "expectancy": None}
    wins = sum(1 for o in outcomes if o > 0)
    return {"n": n, "win_rate": round(100 * wins / n, 1), "expectancy": round(float(np.mean(outcomes)), 3)}


def main():
    p = argparse.ArgumentParser(description="Diagnose where a backtest's edge is really coming from")
    p.add_argument("--seeds", type=int, default=8, help="How many random-walk seeds to test")
    p.add_argument("--n-bars", type=int, default=2000)
    args = p.parse_args()

    theoretical_win_rate = round(100 / (1 + 1.5), 1)  # for default atr_mult=1.5, risk_reward=1.5
    print("=" * 70)
    print("BIAS DIAGNOSTICS")
    print("=" * 70)
    print(f"Theoretical fair win rate at risk_reward=1.5 (no edge): ~{theoretical_win_rate}%")
    print(f"Theoretical fair expectancy at that win rate: ~0.0R")
    print(f"Testing across {args.seeds} independent random-walk seeds.\n")

    print("-- 1. RANDOM BASELINE (random entries, no strategy) --")
    baseline_results = []
    for seed in range(1, args.seeds + 1):
        df = add_ensemble_indicators(generate_unbiased_demo_data(n=args.n_bars, seed=seed))
        r = random_baseline_test(df, seed=seed)
        baseline_results.append(r)
        print(f"  seed={seed:2d}  n={r['n']:4d}  win_rate={r['win_rate']}%  expectancy={r['expectancy']:+.3f}R")
    avg_exp = np.mean([r["expectancy"] for r in baseline_results if r["expectancy"] is not None])
    print(f"  AVERAGE expectancy across seeds: {avg_exp:+.3f}R "
          f"({'looks fair - barrier simulation OK' if abs(avg_exp) < 0.05 else 'BIASED - bug is in the barrier simulation itself'})\n")

    print("-- 2. PER-STRATEGY ISOLATION (each strategy alone, no confluence) --")
    for strat_fn in STRATEGIES:
        name = strat_fn.__name__.replace("strat_", "")
        results = []
        for seed in range(1, args.seeds + 1):
            df = add_ensemble_indicators(generate_unbiased_demo_data(n=args.n_bars, seed=seed))
            r = isolate_strategy_test(df, strat_fn)
            results.append(r)
        valid = [r for r in results if r["expectancy"] is not None]
        avg_exp = np.mean([r["expectancy"] for r in valid]) if valid else float("nan")
        avg_wr = np.mean([r["win_rate"] for r in valid]) if valid else float("nan")
        flag = "  <-- SUSPECT: consistent positive edge on a random walk" if avg_exp > 0.08 else ""
        print(f"  {name:20s}  avg_win_rate={avg_wr:5.1f}%  avg_expectancy={avg_exp:+.3f}R{flag}")

    print()
    print("=" * 70)
    print("HOW TO READ THIS")
    print("=" * 70)
    print("If part 1 (random baseline) is close to 0 expectancy, the")
    print("stop/target simulation logic is sound - trust it.")
    print("Whichever strategy in part 2 shows a large, consistent positive")
    print("expectancy on pure random-walk data (flagged 'SUSPECT' above) is")
    print("the one with a look-ahead leak. The usual culprits are:")
    print("  - an indicator computed with a centered/backward-looking window")
    print("    that accidentally includes the current or a future bar")
    print("  - a persistence effect: a smoothed indicator (like a long SMA)")
    print("    stays 'bullish' for many consecutive bars simply because it's")
    print("    a lagging filter, not because it predicts anything - so a")
    print("    strategy that never resets/exits can look profitable purely")
    print("    from the initial random swing that triggered it, without a")
    print("    real forward edge. Check bar-by-bar whether the flagged")
    print("    strategy re-evaluates its condition fresh each bar (it should)")
    print("    vs. drifting on stale state.")
    print("=" * 70)


if __name__ == "__main__":
    main()
