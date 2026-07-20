"""Final backtest: market-structure SL/TP with per-symbol optimized params."""
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
from forex_signal_engine.market_structure import SYMBOL_DEFAULTS

mt5.initialize(
    path=os.getenv("MT5_PATH", ""),
    login=int(os.getenv("MT5_LOGIN", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
)

info = mt5.account_info()
print(f"Connected: {info.server}")
print()
print("=" * 70)
print("  FINAL BACKTEST: Structure SL/TP + HIGH filter (per-symbol tuned)")
print("=" * 70)

for symbol in ["GBPUSD", "USDJPY"]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    defaults = SYMBOL_DEFAULTS.get(symbol, {})
    cfg = Config(symbols=[symbol], timeframe="4H", default_sl_pips=50, default_tp_pips=100)
    sf = SignalFilter(min_confidence="HIGH")
    bt = Backtester(config=cfg, initial_balance=10000.0, spread_pips=1.5, lot_size=0.1, signal_filter=sf)
    r = bt.run(symbol, "4H", df)

    print(f"\n  {symbol} 4H  |  MaxSL={defaults.get('max_sl_pips','?')}  MinRR={defaults.get('min_rr','?')}")
    print(f"  {r.start_date[:10]} to {r.end_date[:10]}  |  {r.total_bars} bars")
    print(f"  Trades: {r.total_trades} | Winners: {r.winning_trades} ({r.win_rate:.1f}%)")
    print(f"  Net P&L: ${r.net_profit:+,.0f} | PF: {r.profit_factor:.2f}")
    print(f"  Max DD: {r.max_drawdown_pct:.2f}% | Sharpe: {r.sharpe_ratio:+.2f}")
    print(f"  Avg Win: {r.avg_win_pips:.1f}p | Avg Loss: {r.avg_loss_pips:.1f}p | R:R: {r.avg_rr_achieved:.2f}")
    print(f"  Win streak: {r.longest_win_streak} | Lose streak: {r.longest_lose_streak}")

    print(f"\n  Last 10 trades:")
    print(f"  {'#':>3s}  {'DIR':>5s}  {'ENTRY':>10s}  {'SL':>10s}  {'TP':>10s}  {'EXIT':>10s}  {'P&L':>8s}  {'Why':>7s}")
    for idx, t in enumerate(r.trades[-10:], 1):
        print(
            f"  {idx:>3d}  {t.direction.value:>5s}  {t.entry_price:>10.5f}  "
            f"{t.stop_loss:>10.5f}  {t.take_profit:>10.5f}  "
            f"{t.exit_price:>10.5f}  {t.pnl_pips:>+7.1f}p  {t.exit_reason:>7s}"
        )
    print()

mt5.shutdown()
print("Done.")
