"""Tune market-structure SL/TP parameters."""
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

print("Testing structure SL/TP parameter combinations (min_rr now propagated)...\n")
print(f"  {'Symbol':>6s}  {'MaxSL':>5s}  {'MinRR':>5s}  {'Trades':>6s}  {'WR':>6s}  {'Net':>8s}  {'PF':>5s}  {'DD':>6s}  {'Sharpe':>7s}  {'AvgRR':>6s}")
print(f"  {'------':>6s}  {'-----':>5s}  {'-----':>5s}  {'------':>6s}  {'------':>6s}  {'--------':>8s}  {'-----':>5s}  {'------':>6s}  {'-------':>7s}  {'------':>6s}")

for symbol in ["GBPUSD", "USDJPY"]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    for max_sl in [35, 40, 50]:
        for min_rr in [2.0, 2.5, 3.0]:
            cfg = Config(symbols=[symbol], timeframe="4H", default_sl_pips=max_sl, default_tp_pips=100, min_rr=min_rr)
            sf = SignalFilter(min_confidence="HIGH")
            bt = Backtester(config=cfg, initial_balance=10000.0, spread_pips=1.5, lot_size=0.1, signal_filter=sf)
            r = bt.run(symbol, "4H", df)

            pnl = f"${r.net_profit:+,.0f}"
            print(
                f"  {symbol:>6s}  {max_sl:>5d}  {min_rr:>5.1f}  {r.total_trades:>6d}  "
                f"{r.win_rate:>5.1f}%  {pnl:>8s}  {r.profit_factor:>5.2f}  "
                f"{r.max_drawdown_pct:>5.2f}%  {r.sharpe_ratio:>+7.2f}  {r.avg_rr_achieved:>6.2f}"
            )
    print()

mt5.shutdown()
print("Done.")
