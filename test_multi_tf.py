"""Compare 2H, 3H, and 4H timeframes with structure SL/TP + HIGH filter."""
import os, sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from forex_signal_engine.config import Config
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.backtester import Backtester

mt5.initialize(
    path=os.getenv("MT5_PATH", ""),
    login=int(os.getenv("MT5_LOGIN", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
)

info = mt5.account_info()
print(f"Connected: {info.server}")
print()

TF_MAP = {
    "2H": mt5.TIMEFRAME_H2,
    "3H": mt5.TIMEFRAME_H3,
    "4H": mt5.TIMEFRAME_H4,
}

print("=" * 90)
print("  MULTI-TIMEFRAME COMPARISON: 2H vs 3H vs 4H  |  Structure SL/TP + HIGH filter")
print("=" * 90)
print()
print(f"  {'Symbol':>6s}  {'TF':>3s}  {'Bars':>5s}  {'Trades':>6s}  {'WR':>6s}  {'Net':>9s}  {'PF':>5s}"
      f"  {'MaxDD':>6s}  {'Sharpe':>7s}  {'AvgRR':>6s}  {'AvgWin':>6s}  {'AvgLoss':>7s}  {'WStrk':>5s}  {'LStrk':>5s}")
print(f"  {'------':>6s}  {'---':>3s}  {'-----':>5s}  {'------':>6s}  {'------':>6s}  {'---------':>9s}  {'-----':>5s}"
      f"  {'------':>6s}  {'-------':>7s}  {'------':>6s}  {'------':>6s}  {'-------':>7s}  {'-----':>5s}  {'-----':>5s}")

for symbol in ["GBPUSD", "USDJPY"]:
    for tf_name, tf_const in TF_MAP.items():
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 5000)
        if rates is None or len(rates) < 300:
            print(f"  {symbol:>6s}  {tf_name:>3s}  insufficient data")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        cfg = Config(symbols=[symbol], timeframe=tf_name, default_sl_pips=50, default_tp_pips=100)
        sf = SignalFilter(min_confidence="HIGH")
        bt = Backtester(config=cfg, initial_balance=25000.0, spread_pips=1.5, lot_size=0.1, signal_filter=sf)
        r = bt.run(symbol, tf_name, df)

        pnl = f"${r.net_profit:+,.0f}"
        print(
            f"  {symbol:>6s}  {tf_name:>3s}  {r.total_bars:>5d}  {r.total_trades:>6d}  "
            f"{r.win_rate:>5.1f}%  {pnl:>9s}  {r.profit_factor:>5.2f}  "
            f"{r.max_drawdown_pct:>5.2f}%  {r.sharpe_ratio:>+7.2f}  {r.avg_rr_achieved:>6.2f}  "
            f"{r.avg_win_pips:>5.1f}p  {r.avg_loss_pips:>6.1f}p  {r.longest_win_streak:>5d}  {r.longest_lose_streak:>5d}"
        )

        # Daily P&L breakdown for prop firm analysis
        if r.trades:
            daily_pnl = {}
            for t in r.trades:
                if hasattr(t.exit_time, 'date'):
                    day = t.exit_time.date()
                else:
                    day = str(t.exit_time)[:10]
                daily_pnl[day] = daily_pnl.get(day, 0) + t.pnl_dollars

            daily_vals = list(daily_pnl.values())
            best_day = max(daily_vals)
            worst_day = min(daily_vals)
            total_profit = sum(d for d in daily_vals if d > 0)
            best_day_pct = (best_day / total_profit * 100) if total_profit > 0 else 0

            # Check prop firm rules against $25k
            balance = 25000.0
            max_daily_loss_hit = any(d < -750 for d in daily_vals)  # 3% of 25k
            best_day_rule_pass = best_day_pct <= 50

            print(
                f"         {'':>3s}  {'':>5s}  {'':>6s}  {'':>6s}  {'':>9s}  {'':>5s}  {'':>6s}  {'':>7s}"
                f"  Best day: ${best_day:+,.0f} ({best_day_pct:.0f}% of profit)"
                f"  Worst day: ${worst_day:+,.0f}"
                f"  {'PASS' if best_day_rule_pass else 'FAIL'} 50% rule"
                f"  Daily loss limit: {'PASS' if not max_daily_loss_hit else 'FAIL'}"
            )

    print()

mt5.shutdown()
print("Done.")
