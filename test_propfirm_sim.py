"""Full prop firm simulation: both pairs combined, dynamic position sizing, $25k."""
import os, sys, io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import date
from forex_signal_engine.config import Config
from forex_signal_engine.indicators import QMPFilter
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.market_structure import calculate_structure_sl_tp

mt5.initialize(
    path=os.getenv("MT5_PATH", ""),
    login=int(os.getenv("MT5_LOGIN", "0")),
    password=os.getenv("MT5_PASSWORD", ""),
    server=os.getenv("MT5_SERVER", ""),
)

STARTING = 25000.0
TARGET = STARTING * 1.10         # $27,500
MAX_LOSS = STARTING * 0.10       # $2,500 trailing
MAX_DAILY_LOSS = STARTING * 0.03 # $750
BEST_DAY_PCT = 50
RISK_PCT = 1.0                   # 1% per trade

# Load both pairs
pairs_data = {}
for symbol in ["GBPUSD", "USDJPY"]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    qmp = QMPFilter()
    signals_df = qmp.calculate(df)
    pairs_data[symbol] = signals_df

mt5.shutdown()

# Merge both pairs by time and simulate chronologically
sf = SignalFilter(min_confidence="HIGH")

all_events = []
for symbol, sdf in pairs_data.items():
    for i in range(300, len(sdf)):
        bar = sdf.iloc[i]
        if bar["buy_signal"] or bar["sell_signal"]:
            direction = "BUY" if bar["buy_signal"] else "SELL"
            all_events.append((bar["time"], symbol, direction, i, sdf))

all_events.sort(key=lambda x: x[0])

# Simulation
balance = STARTING
hwm = STARTING
equity_curve = [STARTING]
open_positions = {}  # symbol -> {direction, entry, sl, tp, lots, time}
closed_trades = []
daily_pnl = {}
daily_profit = {}

target_reached = None
blown = None

spread_pips = 1.5

for event_time, symbol, direction, bar_idx, sdf in all_events:
    pip = 0.01 if "JPY" in symbol else 0.0001
    spread = spread_pips * pip
    bar = sdf.iloc[bar_idx]

    # First: check SL/TP on open positions using this bar's high/low
    for sym in list(open_positions.keys()):
        pos = open_positions[sym]
        # Find this timestamp in the symbol's data
        sym_df = pairs_data[sym]
        time_mask = sym_df["time"] == event_time
        if not time_mask.any():
            continue

        sym_bar = sym_df[time_mask].iloc[0]
        sym_pip = 0.01 if "JPY" in sym else 0.0001

        hit_sl = hit_tp = False
        exit_price = 0

        if pos["direction"] == "BUY":
            if sym_bar["low"] <= pos["sl"]:
                hit_sl, exit_price = True, pos["sl"]
            elif sym_bar["high"] >= pos["tp"]:
                hit_tp, exit_price = True, pos["tp"]
        else:
            if sym_bar["high"] >= pos["sl"]:
                hit_sl, exit_price = True, pos["sl"]
            elif sym_bar["low"] <= pos["tp"]:
                hit_tp, exit_price = True, pos["tp"]

        if hit_sl or hit_tp:
            if pos["direction"] == "BUY":
                pnl_pips = (exit_price - pos["entry"]) / sym_pip - spread_pips
            else:
                pnl_pips = (pos["entry"] - exit_price) / sym_pip - spread_pips

            pip_value = 10.0 if "JPY" not in sym else (10.0 / (pos["entry"] / 100))
            pnl = pnl_pips * pip_value * pos["lots"]
            balance += pnl

            day = str(event_time.date())
            daily_pnl[day] = daily_pnl.get(day, 0) + pnl
            if pnl > 0:
                daily_profit[day] = daily_profit.get(day, 0) + pnl

            closed_trades.append({
                "symbol": sym, "direction": pos["direction"],
                "entry": pos["entry"], "exit": exit_price,
                "lots": pos["lots"], "pnl_pips": pnl_pips, "pnl": pnl,
                "reason": "SL" if hit_sl else "TP",
                "entry_time": pos["time"], "exit_time": event_time,
            })

            del open_positions[sym]

            if balance > hwm:
                hwm = balance

    equity_curve.append(balance)

    # Check prop firm limits
    if balance >= TARGET and target_reached is None:
        target_reached = {"trade": len(closed_trades), "balance": balance, "time": event_time}

    trailing_floor = hwm - MAX_LOSS
    if balance <= trailing_floor and blown is None:
        blown = {"trade": len(closed_trades), "balance": balance, "time": event_time}
        break

    day = str(event_time.date())
    if daily_pnl.get(day, 0) <= -MAX_DAILY_LOSS:
        continue  # skip new trades today

    # Filter the signal
    window = sdf.iloc[max(0, bar_idx - 250):bar_idx + 1]
    filt = sf.evaluate(window, direction, symbol, event_time)
    if not filt.passed:
        continue

    # Close opposite position if exists
    if symbol in open_positions:
        pos = open_positions[symbol]
        if pos["direction"] != direction:
            close_price = bar["close"]
            sym_pip = pip
            if pos["direction"] == "BUY":
                pnl_pips = (close_price - pos["entry"]) / sym_pip - spread_pips
            else:
                pnl_pips = (pos["entry"] - close_price) / sym_pip - spread_pips

            pip_value = 10.0 if "JPY" not in symbol else (10.0 / (pos["entry"] / 100))
            pnl = pnl_pips * pip_value * pos["lots"]
            balance += pnl

            daily_pnl[day] = daily_pnl.get(day, 0) + pnl
            if pnl > 0:
                daily_profit[day] = daily_profit.get(day, 0) + pnl

            closed_trades.append({
                "symbol": symbol, "direction": pos["direction"],
                "entry": pos["entry"], "exit": close_price,
                "lots": pos["lots"], "pnl_pips": pnl_pips, "pnl": pnl,
                "reason": "REVERSE", "entry_time": pos["time"], "exit_time": event_time,
            })
            del open_positions[symbol]

            if balance > hwm:
                hwm = balance
        else:
            continue

    # Check max open trades
    if len(open_positions) >= 3:
        continue

    # Calculate structure SL/TP
    sl, tp, sl_pips, tp_pips = calculate_structure_sl_tp(sdf, bar_idx, direction, symbol)

    # Position sizing: 1% of balance
    risk_amount = balance * (RISK_PCT / 100)
    pip_value = 10.0 if "JPY" not in symbol else (10.0 / (bar["close"] / 100))
    lots = risk_amount / (sl_pips * pip_value)
    lots = max(0.01, round(lots, 2))

    entry = bar["close"] + (spread / 2 if direction == "BUY" else -spread / 2)

    open_positions[symbol] = {
        "direction": direction, "entry": entry,
        "sl": sl, "tp": tp, "lots": lots, "time": event_time,
    }

# Close remaining positions at last bar
for sym in list(open_positions.keys()):
    pos = open_positions[sym]
    last = pairs_data[sym].iloc[-1]
    close_price = last["close"]
    sym_pip = 0.01 if "JPY" in sym else 0.0001

    if pos["direction"] == "BUY":
        pnl_pips = (close_price - pos["entry"]) / sym_pip - spread_pips
    else:
        pnl_pips = (pos["entry"] - close_price) / sym_pip - spread_pips

    pip_value = 10.0 if "JPY" not in sym else (10.0 / (pos["entry"] / 100))
    pnl = pnl_pips * pip_value * pos["lots"]
    balance += pnl
    closed_trades.append({
        "symbol": sym, "direction": pos["direction"],
        "entry": pos["entry"], "exit": close_price,
        "lots": pos["lots"], "pnl_pips": pnl_pips, "pnl": pnl,
        "reason": "END", "entry_time": pos["time"], "exit_time": last["time"],
    })

# Results
print()
print("=" * 90)
print("  PROP FIRM CHALLENGE SIMULATION")
print(f"  Account: $25,000 | GBPUSD + USDJPY combined | 4H | 1% risk/trade")
print(f"  6-layer HIGH filter | Market structure SL/TP")
print("=" * 90)

total_trades = len(closed_trades)
winners = [t for t in closed_trades if t["pnl"] > 0]
losers = [t for t in closed_trades if t["pnl"] <= 0]
gross_profit = sum(t["pnl"] for t in winners)
gross_loss = abs(sum(t["pnl"] for t in losers))
net = balance - STARTING
wr = (len(winners) / total_trades * 100) if total_trades > 0 else 0
pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

eq = np.array(equity_curve)
peak = np.maximum.accumulate(eq)
dd = (peak - eq) / peak * 100
max_dd = dd.max()

print(f"\n  Trades: {total_trades} | Winners: {len(winners)} ({wr:.1f}%) | Losers: {len(losers)}")
print(f"  Net P&L: ${net:+,.2f}")
print(f"  Final balance: ${balance:,.2f}")
print(f"  Profit factor: {pf:.2f}")
print(f"  Max drawdown: {max_dd:.2f}%")

# Per-symbol breakdown
for sym in ["GBPUSD", "USDJPY"]:
    sym_trades = [t for t in closed_trades if t["symbol"] == sym]
    sym_pnl = sum(t["pnl"] for t in sym_trades)
    sym_wins = sum(1 for t in sym_trades if t["pnl"] > 0)
    sym_wr = (sym_wins / len(sym_trades) * 100) if sym_trades else 0
    print(f"    {sym}: {len(sym_trades)} trades, ${sym_pnl:+,.2f}, WR {sym_wr:.0f}%")

# Prop firm rules
print(f"\n  ---- PROP FIRM RULES ----")
print(f"  Profit target (10%): {'PASSED - hit at trade #' + str(target_reached['trade']) + ' (' + str(target_reached['time'].date()) + ')' if target_reached else 'NOT HIT ($' + f'{net:+,.0f} / $2,500)'}")
print(f"  Max loss (10% trailing): {'BLOWN at trade #' + str(blown['trade']) if blown else 'SAFE (HWM: $' + f'{hwm:,.0f})'}")

worst_day = min(daily_pnl.values()) if daily_pnl else 0
daily_breach_days = [d for d, v in daily_pnl.items() if v <= -MAX_DAILY_LOSS]
print(f"  Max daily loss (3%): {'BREACHED on ' + str(daily_breach_days) if daily_breach_days else 'SAFE (worst: $' + f'{worst_day:+,.0f})'}")

total_p = sum(v for v in daily_profit.values())
if total_p > 0:
    best_day_v = max(daily_profit.values())
    best_day_p = best_day_v / total_p * 100
    print(f"  Best day rule (50%): {'FAIL' if best_day_p > 50 else 'PASS'} (best day: ${best_day_v:+,.0f} = {best_day_p:.0f}% of profit)")
else:
    print(f"  Best day rule: N/A (no profit days)")

# Last 15 trades
print(f"\n  Last 15 trades:")
print(f"  {'#':>3s}  {'Sym':>6s}  {'DIR':>5s}  {'Lots':>5s}  {'Entry':>10s}  {'Exit':>10s}  {'P&L':>10s}  {'Why':>7s}")
for i, t in enumerate(closed_trades[-15:], 1):
    pnl_str = f"${t['pnl']:+,.2f}"
    print(
        f"  {i:>3d}  {t['symbol']:>6s}  {t['direction']:>5s}  {t['lots']:>5.2f}  {t['entry']:>10.5f}"
        f"  {t['exit']:>10.5f}  {pnl_str:>10s}  {t['reason']:>7s}"
    )

print(f"\nDone.")
