"""
Multi-timeframe backtest runner.
Generates realistic forex price data for each timeframe and compares results.

Usage:
    python -m forex_signal_engine.run_backtest
    python -m forex_signal_engine.run_backtest --symbol GBPUSD
    python -m forex_signal_engine.run_backtest --mt5   # use real MT5 data
"""

import argparse
import logging
import sys
import io

import numpy as np
import pandas as pd

from forex_signal_engine.config import Config
from forex_signal_engine.backtester import Backtester, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Backtest")

# Timeframe configs: (name, bars_per_day, total_bars to simulate ~2 years, noise_scale)
TIMEFRAMES = {
    "15M": {"bars_per_day": 96, "total_bars": 8000, "noise": 0.00030, "sl": 15, "tp": 30},
    "30M": {"bars_per_day": 48, "total_bars": 6000, "noise": 0.00045, "sl": 20, "tp": 40},
    "1H":  {"bars_per_day": 24, "total_bars": 5000, "noise": 0.00065, "sl": 30, "tp": 60},
    "4H":  {"bars_per_day": 6,  "total_bars": 3000, "noise": 0.00120, "sl": 50, "tp": 100},
}


def generate_realistic_candles(
    n: int,
    base_price: float = 1.1000,
    noise_scale: float = 0.0006,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic candles that mimic real forex behavior:
    - Trending phases (up and down)
    - Ranging/consolidation phases
    - Volatility clusters
    - Realistic OHLC relationships
    """
    rng = np.random.RandomState(seed)

    # Build price path with regime switching
    returns = np.zeros(n)
    regime = 0  # 0=trend_up, 1=trend_down, 2=range
    regime_len = 0

    for i in range(n):
        regime_len += 1
        switch_prob = min(0.01, regime_len / 5000)
        if rng.random() < switch_prob:
            regime = rng.choice([0, 1, 2], p=[0.35, 0.35, 0.30])
            regime_len = 0

        if regime == 0:  # trend up
            drift = noise_scale * 0.15
        elif regime == 1:  # trend down
            drift = -noise_scale * 0.15
        else:  # range
            drift = 0.0

        # Volatility clustering via GARCH-like effect
        vol_mult = 1.0 + 0.5 * abs(returns[i - 1]) / noise_scale if i > 0 else 1.0
        vol_mult = min(vol_mult, 3.0)

        returns[i] = drift + rng.normal(0, noise_scale * vol_mult)

    close = base_price * np.exp(np.cumsum(returns))

    # Build OHLC from close
    intrabar_vol = noise_scale * 0.6
    high = close + rng.uniform(0.2, 1.0, n) * intrabar_vol
    low = close - rng.uniform(0.2, 1.0, n) * intrabar_vol
    open_ = np.roll(close, 1) + rng.normal(0, noise_scale * 0.1, n)
    open_[0] = base_price

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(500, 5000, n)

    # Generate timestamps (5 trading days per week)
    times = pd.date_range(start="2023-01-02", periods=n, freq="h")

    return pd.DataFrame({
        "time": times[:n],
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def run_multi_timeframe(symbol: str = "EURUSD", use_mt5: bool = False):
    """Run backtest across all timeframes and rank results."""
    results: list[BacktestResult] = []

    for tf_name, tf_cfg in TIMEFRAMES.items():
        logger.info(f"Running {symbol} {tf_name}...")

        config = Config(
            symbols=[symbol],
            timeframe=tf_name,
            default_sl_pips=tf_cfg["sl"],
            default_tp_pips=tf_cfg["tp"],
        )

        backtester = Backtester(
            config=config,
            initial_balance=10_000.0,
            spread_pips=1.5,
            lot_size=0.1,
        )

        if use_mt5:
            from forex_signal_engine.mt5_executor import MT5Executor
            executor = MT5Executor(config)
            if not executor.connect():
                logger.error("MT5 connection failed, falling back to synthetic data")
                df = generate_realistic_candles(
                    tf_cfg["total_bars"],
                    noise_scale=tf_cfg["noise"],
                    seed=hash(tf_name) % 2**31,
                )
            else:
                df = executor.get_candles(symbol, tf_cfg["total_bars"])
                executor.disconnect()
                if df is None:
                    logger.warning(f"No MT5 data for {tf_name}, using synthetic")
                    df = generate_realistic_candles(
                        tf_cfg["total_bars"],
                        noise_scale=tf_cfg["noise"],
                        seed=hash(tf_name) % 2**31,
                    )
        else:
            base = 1.1000
            if "GBP" in symbol:
                base = 1.2700
            elif "JPY" in symbol:
                base = 150.00

            df = generate_realistic_candles(
                tf_cfg["total_bars"],
                base_price=base,
                noise_scale=tf_cfg["noise"],
                seed=hash(tf_name + symbol) % 2**31,
            )

        result = backtester.run(symbol, tf_name, df)
        results.append(result)

    # Print all results
    print("\n")
    for r in results:
        print(r.summary())
        print()

    # Rank by composite score: weighted net profit, profit factor, drawdown, Sharpe
    print("=" * 60)
    print("  TIMEFRAME RANKING (best to worst)")
    print("=" * 60)

    def score(r: BacktestResult) -> float:
        pf_score = min(r.profit_factor, 5.0) * 20  # cap at 5
        dd_score = max(0, 50 - r.max_drawdown_pct * 5)  # lower DD = better
        sharpe_score = r.sharpe_ratio * 15
        return_score = (r.net_profit / r.initial_balance * 100) * 2
        wr_score = r.win_rate * 0.5
        return pf_score + dd_score + sharpe_score + return_score + wr_score

    ranked = sorted(results, key=score, reverse=True)

    for i, r in enumerate(ranked, 1):
        s = score(r)
        medal = ["  1st", "  2nd", "  3rd", "  4th"][i - 1] if i <= 4 else f"  {i}th"
        print(
            f"{medal}  {r.timeframe:>4s}  |  "
            f"P&L: ${r.net_profit:>+10,.2f}  |  "
            f"WR: {r.win_rate:5.1f}%  |  "
            f"PF: {r.profit_factor:5.2f}  |  "
            f"DD: {r.max_drawdown_pct:5.2f}%  |  "
            f"Sharpe: {r.sharpe_ratio:+5.2f}  |  "
            f"Score: {s:.1f}"
        )

    best = ranked[0]
    print(f"\n  >>> BEST TIMEFRAME: {best.timeframe} <<<")
    print(f"      Net P&L: ${best.net_profit:+,.2f} | "
          f"Win Rate: {best.win_rate:.1f}% | "
          f"Profit Factor: {best.profit_factor:.2f} | "
          f"Max DD: {best.max_drawdown_pct:.2f}%")
    print()

    # Trade log for best timeframe
    print(f"\n{'─'*60}")
    print(f"  LAST 20 TRADES — {best.symbol} {best.timeframe}")
    print(f"{'─'*60}")
    print(f"  {'#':>3}  {'DIR':>5}  {'ENTRY':>10}  {'EXIT':>10}  {'P&L':>8}  {'REASON':>7}  {'BARS':>4}")
    for idx, t in enumerate(best.trades[-20:], 1):
        pnl_str = f"{t.pnl_pips:+.1f}p"
        print(
            f"  {idx:>3}  {t.direction.value:>5}  "
            f"{t.entry_price:>10.5f}  {t.exit_price:>10.5f}  "
            f"{pnl_str:>8}  {t.exit_reason:>7}  {t.bars_held:>4}"
        )

    return ranked


def main():
    parser = argparse.ArgumentParser(description="QMP Filter Multi-Timeframe Backtest")
    parser.add_argument("--symbol", default="EURUSD", help="Forex pair (default: EURUSD)")
    parser.add_argument("--mt5", action="store_true", help="Use real MT5 data instead of synthetic")
    args = parser.parse_args()

    # Handle Windows terminal encoding
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    run_multi_timeframe(symbol=args.symbol, use_mt5=args.mt5)


if __name__ == "__main__":
    main()
