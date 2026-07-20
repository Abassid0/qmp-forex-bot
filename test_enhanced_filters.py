"""Test enhanced 6-layer filters vs original 3-layer on 4H."""
import os, sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
from forex_signal_engine.config import Config
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.backtester import Backtester

mt5.initialize(
    path=os.getenv("MT5_PATH", ""),
    login=int(os.getenv("MT5_LOGIN", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
)

print("=" * 90)
print("  ENHANCED FILTERS BACKTEST (6 layers) | $25k account | 4H")
print("=" * 90)
print()

print(f"  {'Symbol':>6s}  {'Filter':>14s}  {'Trades':>6s}  {'WR':>6s}  {'Net':>9s}  {'PF':>5s}"
      f"  {'MaxDD':>6s}  {'Sharpe':>7s}  {'AvgRR':>6s}  {'AvgWin':>6s}  {'AvgLoss':>7s}")
print(f"  {'------':>6s}  {'--------------':>14s}  {'------':>6s}  {'------':>6s}  {'---------':>9s}  {'-----':>5s}"
      f"  {'------':>6s}  {'-------':>7s}  {'------':>6s}  {'------':>6s}  {'-------':>7s}")

for symbol in ["GBPUSD", "USDJPY"]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    for label, min_conf in [("NO FILTERS", None), ("HIGH (6-layer)", "HIGH"), ("MEDIUM (6-layer)", "MEDIUM")]:
        cfg = Config(symbols=[symbol], timeframe="4H", default_sl_pips=50, default_tp_pips=100)

        if min_conf:
            sf = SignalFilter(min_confidence=min_conf)
        else:
            sf = None

        bt = Backtester(config=cfg, initial_balance=25000.0, spread_pips=1.5, lot_size=0.1, signal_filter=sf)
        r = bt.run(symbol, "4H", df)

        pnl = f"${r.net_profit:+,.0f}"
        print(
            f"  {symbol:>6s}  {label:>14s}  {r.total_trades:>6d}  "
            f"{r.win_rate:>5.1f}%  {pnl:>9s}  {r.profit_factor:>5.2f}  "
            f"{r.max_drawdown_pct:>5.2f}%  {r.sharpe_ratio:>+7.2f}  {r.avg_rr_achieved:>6.2f}  "
            f"{r.avg_win_pips:>5.1f}p  {r.avg_loss_pips:>6.1f}p"
        )
    print()

# Prop firm simulation on best config
print("=" * 90)
print("  PROP FIRM SIMULATION: $25k | 4H | HIGH 6-layer filter")
print("=" * 90)

for symbol in ["GBPUSD", "USDJPY"]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    cfg = Config(symbols=[symbol], timeframe="4H", default_sl_pips=50, default_tp_pips=100)
    sf = SignalFilter(min_confidence="HIGH")
    bt = Backtester(config=cfg, initial_balance=25000.0, spread_pips=1.5, lot_size=0.1, signal_filter=sf)
    r = bt.run(symbol, "4H", df)

    # Simulate prop firm rules
    balance = 25000.0
    hwm = 25000.0  # high water mark
    max_loss_floor = 25000.0 - 2500.0  # 10% trailing
    daily_pnl = {}
    daily_profit = {}
    target_hit_trade = None
    blown_trade = None
    daily_loss_breach = None

    for t in r.trades:
        balance += t.pnl_dollars

        if hasattr(t.exit_time, 'date'):
            day = str(t.exit_time.date())
        else:
            day = str(t.exit_time)[:10]

        daily_pnl[day] = daily_pnl.get(day, 0) + t.pnl_dollars
        if t.pnl_dollars > 0:
            daily_profit[day] = daily_profit.get(day, 0) + t.pnl_dollars

        # EOD trailing (simplified - check after each trade)
        if balance > hwm:
            hwm = balance
            max_loss_floor = hwm - 2500.0

        if balance >= 27500 and target_hit_trade is None:
            target_hit_trade = len([x for x in r.trades if x.exit_time <= t.exit_time])

        if balance <= max_loss_floor and blown_trade is None:
            blown_trade = len([x for x in r.trades if x.exit_time <= t.exit_time])

        if daily_pnl[day] <= -750 and daily_loss_breach is None:
            daily_loss_breach = day

    # Best day analysis
    total_profit = sum(v for v in daily_profit.values())
    best_day_val = max(daily_profit.values()) if daily_profit else 0
    best_day_pct = (best_day_val / total_profit * 100) if total_profit > 0 else 0
    worst_day_val = min(daily_pnl.values()) if daily_pnl else 0

    print(f"\n  {symbol} 4H | {r.total_trades} trades over {r.total_bars} bars")
    print(f"  Final balance: ${balance:,.2f} | Net: ${balance - 25000:+,.2f}")
    print(f"  Target ($27,500): {'HIT at trade #' + str(target_hit_trade) if target_hit_trade else 'NOT reached'}")
    print(f"  Max loss breach: {'YES at trade #' + str(blown_trade) if blown_trade else 'NO (safe)'}")
    print(f"  Daily loss breach: {daily_loss_breach if daily_loss_breach else 'NONE'}")
    print(f"  Best day: ${best_day_val:+,.0f} ({best_day_pct:.0f}% of profit) | {'PASS' if best_day_pct <= 50 else 'FAIL'} 50% rule")
    print(f"  Worst day: ${worst_day_val:+,.0f} | {'PASS' if worst_day_val > -750 else 'FAIL'} 3% daily limit")
    print(f"  Max drawdown: {r.max_drawdown_pct:.2f}%")

mt5.shutdown()
print("\nDone.")
