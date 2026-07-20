"""Test signal engine with filters against live MT5 data for GBPUSD and USDJPY."""
import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
from forex_signal_engine.config import Config
from forex_signal_engine.signal_engine import SignalEngine
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.backtester import Backtester

mt5.initialize(
    path=os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe"),
    login=int(os.getenv("MT5_LOGIN", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
)

info = mt5.account_info()
if info is None:
    print(f"MT5 connection failed: {mt5.last_error()}")
    sys.exit(1)

print(f"Connected: {info.server} | Account: {info.login} | Balance: {info.balance} {info.currency}")
print()

# ── Part 1: Live signal check ──
print("=" * 70)
print("  LIVE SIGNAL CHECK (current 4H bar)")
print("=" * 70)

config = Config(
    symbols=["GBPUSD", "USDJPY"],
    timeframe="4H",
    default_sl_pips=50,
    default_tp_pips=100,
)

sf = SignalFilter(min_confidence="HIGH")
engine = SignalEngine(config, signal_filter=sf)

for symbol in ["GBPUSD", "USDJPY"]:
    tf = mt5.TIMEFRAME_H4
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    print(f"\n  {symbol} 4H")
    print(f"  Last bar: {df.iloc[-1]['time']}  Close: {df.iloc[-1]['close']:.5f}")

    signal = engine.analyze(symbol, df)
    if signal:
        print(f"  >>> SIGNAL: {signal.signal_type.value}")
        print(f"      Entry: {signal.entry_price:.5f}")
        print(f"      SL: {signal.stop_loss:.5f}  TP: {signal.take_profit:.5f}")
        print(f"      Confidence: {signal.confidence}")
    else:
        print(f"      No signal on current bar")

    result = engine.qmp.calculate(df)
    last = result.iloc[-1]
    macd_pos = "ABOVE" if last["macd_line"] > last["macd_signal"] else "BELOW"
    qqe_dir = "BULLISH" if last["qqe_bullish"] else "BEARISH"
    ema_trend = "UP" if last["ema_50"] > last["ema_100"] else "DOWN"

    print(f"      MACD: {last['macd_line']:.6f} ({macd_pos} signal)")
    print(f"      QQE:  RSI={last['qqe_rsi_ma']:.1f} ({qqe_dir})")
    print(f"      Trend: EMA50 {'>' if ema_trend == 'UP' else '<'} EMA100 ({ema_trend})")
    print(f"      Price vs EMA50: {'above' if last['close'] > last['ema_50'] else 'below'}")

print()

# ── Part 2: Backtest comparison ──
print("=" * 70)
print("  BACKTEST: GBPUSD & USDJPY 4H  (no filters vs HIGH filter)")
print("=" * 70)

for symbol in ["GBPUSD", "USDJPY"]:
    tf = mt5.TIMEFRAME_H4
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, 5000)
    if rates is None or len(rates) < 300:
        print(f"  {symbol}: insufficient data ({len(rates) if rates else 0} bars)")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    sl = 50
    tp = 100

    cfg = Config(symbols=[symbol], timeframe="4H", default_sl_pips=sl, default_tp_pips=tp)

    # No filters
    bt_raw = Backtester(config=cfg, initial_balance=10000.0, spread_pips=1.5, lot_size=0.1)
    r_raw = bt_raw.run(symbol, "4H", df)

    # HIGH filter
    sf_high = SignalFilter(min_confidence="HIGH")
    bt_high = Backtester(config=cfg, initial_balance=10000.0, spread_pips=1.5, lot_size=0.1, signal_filter=sf_high)
    r_high = bt_high.run(symbol, "4H", df)

    print(f"\n  {symbol} 4H  |  {r_raw.start_date[:10]} to {r_raw.end_date[:10]}  |  {r_raw.total_bars} bars")
    print(f"  {'':>18s}  {'Trades':>7s}  {'WR':>6s}  {'Net P&L':>10s}  {'PF':>5s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'AvgRR':>6s}")
    print(f"  {'':>18s}  {'-------':>7s}  {'------':>6s}  {'----------':>10s}  {'-----':>5s}  {'-------':>7s}  {'-------':>7s}  {'------':>6s}")

    for label, r in [("NO FILTERS", r_raw), ("HIGH filter", r_high)]:
        pnl = f"${r.net_profit:+,.0f}"
        print(
            f"  {label:>18s}  {r.total_trades:>7d}  {r.win_rate:>5.1f}%  {pnl:>10s}"
            f"  {r.profit_factor:>5.2f}  {r.max_drawdown_pct:>6.2f}%  {r.sharpe_ratio:>+7.2f}  {r.avg_rr_achieved:>6.2f}"
        )

    improvement = r_high.net_profit - r_raw.net_profit
    print(f"\n  Filter impact: {'+' if improvement >= 0 else ''}${improvement:,.0f} improvement")
    print(f"  Trades reduced: {r_raw.total_trades} -> {r_high.total_trades} ({(1 - r_high.total_trades/r_raw.total_trades)*100:.0f}% filtered)")
    print(f"  Drawdown: {r_raw.max_drawdown_pct:.2f}% -> {r_high.max_drawdown_pct:.2f}%")

    # Last 10 trades
    print(f"\n  Last 10 trades ({symbol} HIGH filter):")
    print(f"  {'#':>3s}  {'DIR':>5s}  {'ENTRY':>10s}  {'EXIT':>10s}  {'P&L':>8s}  {'REASON':>7s}  {'BARS':>4s}")
    for idx, t in enumerate(r_high.trades[-10:], 1):
        pnl_str = f"{t.pnl_pips:+.1f}p"
        print(
            f"  {idx:>3d}  {t.direction.value:>5s}  {t.entry_price:>10.5f}"
            f"  {t.exit_price:>10.5f}  {pnl_str:>8s}  {t.exit_reason:>7s}  {t.bars_held:>4d}"
        )
    print()

mt5.shutdown()
print("Done.")
