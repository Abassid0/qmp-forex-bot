"""Tune structure SL/TP params for all 5 pairs + backtest EURUSD/GBPUSD."""
import os, sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

mt5.initialize(
    path=os.getenv("MT5_PATH", ""),
    login=int(os.getenv("MT5_LOGIN", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
)

from forex_signal_engine.indicators import QMPFilter
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.market_structure import calculate_structure_sl_tp

qmp = QMPFilter()
sf = SignalFilter(min_confidence="HIGH")

ALL_PAIRS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "EURJPY", "GBPJPY"]

pairs_data = {}
for symbol in ALL_PAIRS:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    if rates is None:
        print(f"  {symbol}: no data")
        continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    sdf = qmp.calculate(df)
    pairs_data[symbol] = sdf
    print(f"  Loaded {symbol}: {len(sdf)} bars")

mt5.shutdown()


def backtest_pair(symbol, sdf, max_sl, min_rr):
    pip = 0.01 if "JPY" in symbol else 0.0001
    spread_pips = 1.5
    balance = 25000.0
    trades = []
    open_trade = None

    # Get filtered signals
    filtered = []
    for i in range(300, len(sdf)):
        bar = sdf.iloc[i]
        if not (bar["buy_signal"] or bar["sell_signal"]):
            continue
        direction = "BUY" if bar["buy_signal"] else "SELL"
        window = sdf.iloc[max(0, i-250):i+1]
        try:
            filt = sf.evaluate(window, direction, symbol, bar["time"])
            if filt.passed:
                filtered.append((i, direction))
        except Exception:
            pass

    for bar_idx, direction in filtered:
        bar = sdf.iloc[bar_idx]

        # Close existing
        if open_trade is not None:
            for j in range(open_trade["idx"] + 1, bar_idx + 1):
                b = sdf.iloc[j]
                hit = False
                if open_trade["dir"] == "BUY":
                    if b["low"] <= open_trade["sl"]:
                        pnl_pips = (open_trade["sl"] - open_trade["entry"]) / pip - spread_pips
                        hit = True
                    elif b["high"] >= open_trade["tp"]:
                        pnl_pips = (open_trade["tp"] - open_trade["entry"]) / pip - spread_pips
                        hit = True
                else:
                    if b["high"] >= open_trade["sl"]:
                        pnl_pips = (open_trade["entry"] - open_trade["sl"]) / pip - spread_pips
                        hit = True
                    elif b["low"] <= open_trade["tp"]:
                        pnl_pips = (open_trade["entry"] - open_trade["tp"]) / pip - spread_pips
                        hit = True
                if hit:
                    pip_val = 10.0 if "JPY" not in symbol else (10.0 / (open_trade["entry"] / 100))
                    pnl = pnl_pips * pip_val * open_trade["lots"]
                    balance += pnl
                    trades.append(pnl)
                    open_trade = None
                    break

            if open_trade is not None:
                close_p = bar["close"]
                if open_trade["dir"] == "BUY":
                    pnl_pips = (close_p - open_trade["entry"]) / pip - spread_pips
                else:
                    pnl_pips = (open_trade["entry"] - close_p) / pip - spread_pips
                pip_val = 10.0 if "JPY" not in symbol else (10.0 / (open_trade["entry"] / 100))
                pnl = pnl_pips * pip_val * open_trade["lots"]
                balance += pnl
                trades.append(pnl)
                open_trade = None

        try:
            sl, tp, sl_pips, tp_pips = calculate_structure_sl_tp(
                sdf, bar_idx, direction, symbol,
                max_sl_pips=max_sl, min_rr=min_rr
            )
        except Exception:
            continue

        risk_amount = balance * 0.01
        pip_val = 10.0 if "JPY" not in symbol else (10.0 / (bar["close"] / 100))
        lots = risk_amount / (sl_pips * pip_val)
        lots = max(0.01, round(lots, 2))

        open_trade = {
            "dir": direction, "entry": bar["close"],
            "sl": sl, "tp": tp, "lots": lots, "idx": bar_idx,
        }

    # Close last
    if open_trade is not None:
        last = sdf.iloc[-1]
        close_p = last["close"]
        if open_trade["dir"] == "BUY":
            pnl_pips = (close_p - open_trade["entry"]) / pip - spread_pips
        else:
            pnl_pips = (open_trade["entry"] - close_p) / pip - spread_pips
        pip_val = 10.0 if "JPY" not in symbol else (10.0 / (open_trade["entry"] / 100))
        pnl = pnl_pips * pip_val * open_trade["lots"]
        balance += pnl
        trades.append(pnl)

    total = len(trades)
    wins = sum(1 for t in trades if t > 0)
    net = balance - 25000
    wr = (wins / total * 100) if total else 0

    eq = np.array([25000] + [25000 + sum(trades[:i+1]) for i in range(len(trades))])
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / peak * 100)
    max_dd = dd.max() if len(dd) else 0

    return total, wins, wr, net, balance, max_dd


# Parameter sweep for each pair
print()
print("=" * 90)
print("  PARAMETER SWEEP: max_sl x min_rr | 4H | HIGH filter | 1% risk")
print("=" * 90)

best_params = {}

for symbol in ALL_PAIRS:
    if symbol not in pairs_data:
        continue

    sdf = pairs_data[symbol]
    print(f"\n  {symbol}:")
    print(f"  {'MaxSL':>6s}  {'MinRR':>5s}  {'Trades':>6s}  {'WR':>5s}  {'Net':>10s}  {'MaxDD':>6s}  {'Final':>10s}")
    print(f"  {'------':>6s}  {'-----':>5s}  {'------':>6s}  {'-----':>5s}  {'----------':>10s}  {'------':>6s}  {'----------':>10s}")

    best_net = -999999
    best_combo = None

    sl_range = [25, 30, 35, 40, 50] if "JPY" not in symbol else [30, 40, 50, 60, 80]
    rr_range = [1.5, 2.0, 2.5, 3.0, 3.5]

    for max_sl in sl_range:
        for min_rr in rr_range:
            total, wins, wr, net, final, max_dd = backtest_pair(symbol, sdf, max_sl, min_rr)
            if total < 5:
                continue

            marker = ""
            if net > best_net:
                best_net = net
                best_combo = (max_sl, min_rr, total, wr, net, max_dd)
                marker = " <<<"

            print(
                f"  {max_sl:>5.0f}p  {min_rr:>5.1f}  {total:>6d}  {wr:>4.0f}%  "
                f"${net:>+9,.0f}  {max_dd:>5.1f}%  ${final:>9,.0f}{marker}"
            )

    if best_combo:
        best_params[symbol] = best_combo
        print(f"\n  >>> BEST: max_sl={best_combo[0]}p, min_rr={best_combo[1]}, "
              f"{best_combo[2]} trades, WR {best_combo[3]:.0f}%, "
              f"net ${best_combo[4]:+,.0f}, DD {best_combo[5]:.1f}%")

# Summary
print()
print("=" * 90)
print("  OPTIMAL PARAMETERS SUMMARY")
print("=" * 90)
print(f"\n  {'Pair':>6s}  {'MaxSL':>6s}  {'MinRR':>5s}  {'Trades':>6s}  {'WR':>5s}  {'Net $':>10s}  {'MaxDD':>6s}  {'Verdict':>10s}")
print(f"  {'------':>6s}  {'------':>6s}  {'-----':>5s}  {'------':>6s}  {'-----':>5s}  {'----------':>10s}  {'------':>6s}  {'----------':>10s}")

combined_net = 0
for symbol in ALL_PAIRS:
    if symbol in best_params:
        bp = best_params[symbol]
        verdict = "TRADE" if bp[4] > 0 else "SKIP"
        combined_net += bp[4]
        print(
            f"  {symbol:>6s}  {bp[0]:>5.0f}p  {bp[1]:>5.1f}  {bp[2]:>6d}  {bp[3]:>4.0f}%  "
            f"${bp[4]:>+9,.0f}  {bp[5]:>5.1f}%  {verdict:>10s}"
        )

print(f"\n  Combined net (best params): ${combined_net:+,.0f}")
print(f"\nDone.")
