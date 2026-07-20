"""Correlation analysis: DXY vs AUDUSD, JPY crosses (USDJPY/EURJPY/GBPJPY)."""
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

# Check what's available on FTMO
print("Checking available symbols...")
all_symbols = mt5.symbols_get()
dxy_candidates = [s.name for s in all_symbols if "DX" in s.name.upper() or "USDX" in s.name.upper() or "DOLLAR" in s.name.upper()]
print(f"  DXY-like symbols: {dxy_candidates if dxy_candidates else 'NONE'}")

target_pairs = ["AUDUSD", "USDJPY", "EURJPY", "GBPJPY"]
available = {}
for pair in target_pairs:
    info = mt5.symbol_info(pair)
    if info:
        available[pair] = True
        print(f"  {pair}: available (spread: {info.spread})")
    else:
        # Try with suffix
        for suffix in [".i", ".a", "_m", ""]:
            info = mt5.symbol_info(pair + suffix)
            if info:
                available[pair] = pair + suffix
                print(f"  {pair}: available as {pair + suffix}")
                break
        if pair not in available:
            print(f"  {pair}: NOT FOUND")

# Also check GBPUSD and EURUSD for constructing synthetic DXY proxy
for pair in ["GBPUSD", "EURUSD"]:
    info = mt5.symbol_info(pair)
    if info:
        available[pair] = True
        print(f"  {pair}: available")

print()

# Pull 4H data for all available pairs
pairs_data = {}
for pair in ["AUDUSD", "USDJPY", "EURJPY", "GBPJPY", "GBPUSD", "EURUSD"]:
    if pair in available:
        sym = available[pair] if isinstance(available[pair], str) else pair
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 0, 5000)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            pairs_data[pair] = df
            print(f"  Loaded {pair}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

mt5.shutdown()

if len(pairs_data) < 3:
    print("Not enough data. Exiting.")
    sys.exit(1)

# Build returns DataFrame
returns = pd.DataFrame()
closes = pd.DataFrame()
for pair, df in pairs_data.items():
    closes[pair] = df["close"]
    returns[pair] = df["close"].pct_change()

returns = returns.dropna()
closes = closes.dropna()

# Create synthetic USD strength index from available pairs
# DXY proxy: average of USD-positive moves (inverse EURUSD, inverse GBPUSD, USDJPY)
if all(p in returns.columns for p in ["EURUSD", "GBPUSD", "USDJPY"]):
    returns["USD_STRENGTH"] = (-returns["EURUSD"] - returns["GBPUSD"] + returns["USDJPY"]) / 3
    closes["USD_STRENGTH"] = (1 / closes["EURUSD"] * 1 / closes["GBPUSD"] * closes["USDJPY"]).apply(np.cbrt)
    print("\n  Built USD Strength proxy from EURUSD, GBPUSD, USDJPY")

print()
print("=" * 90)
print("  CORRELATION ANALYSIS | 4H candles")
print("=" * 90)

# 1. Full-period correlation matrix
print("\n  ── Full Period Correlation (returns) ──")
corr = returns.corr()
cols = [c for c in ["USD_STRENGTH", "AUDUSD", "USDJPY", "EURJPY", "GBPJPY", "GBPUSD", "EURUSD"] if c in corr.columns]
corr_display = corr.loc[cols, cols]

header = f"  {'':>14s}"
for c in cols:
    header += f"  {c[:7]:>7s}"
print(header)

for row in cols:
    line = f"  {row:>14s}"
    for col in cols:
        v = corr_display.loc[row, col]
        line += f"  {v:>+7.3f}"
    print(line)

# 2. Rolling correlation (to see if it's stable)
print(f"\n  ── Rolling 50-bar Correlation Stability ──")
key_pairs_corr = [
    ("USD_STRENGTH", "AUDUSD", "USD vs AUDUSD"),
    ("USDJPY", "EURJPY", "USDJPY vs EURJPY"),
    ("USDJPY", "GBPJPY", "USDJPY vs GBPJPY"),
    ("EURJPY", "GBPJPY", "EURJPY vs GBPJPY"),
]

for p1, p2, label in key_pairs_corr:
    if p1 in returns.columns and p2 in returns.columns:
        rolling = returns[p1].rolling(50).corr(returns[p2])
        mean_c = rolling.mean()
        min_c = rolling.min()
        max_c = rolling.max()
        stable = rolling.std()
        print(f"  {label:>22s}: mean={mean_c:+.3f}  min={min_c:+.3f}  max={max_c:+.3f}  std={stable:.3f}")

# 3. QMP signals analysis across pairs
print()
print("=" * 90)
print("  QMP FILTER SIGNAL ANALYSIS ACROSS PAIRS")
print("=" * 90)

from forex_signal_engine.indicators import QMPFilter
from forex_signal_engine.filters import SignalFilter

qmp = QMPFilter()
sf = SignalFilter(min_confidence="HIGH")

signals_by_pair = {}
for pair in ["AUDUSD", "USDJPY", "EURJPY", "GBPJPY", "GBPUSD"]:
    if pair not in pairs_data:
        continue

    df = pairs_data[pair].reset_index()
    sdf = qmp.calculate(df)

    raw_buys = sdf["buy_signal"].sum()
    raw_sells = sdf["sell_signal"].sum()

    # Run through filters
    filtered_signals = []
    for i in range(300, len(sdf)):
        bar = sdf.iloc[i]
        if not (bar["buy_signal"] or bar["sell_signal"]):
            continue
        direction = "BUY" if bar["buy_signal"] else "SELL"
        window = sdf.iloc[max(0, i-250):i+1]

        # For new pairs, use default filter (some filters may not have symbol-specific config)
        try:
            filt = sf.evaluate(window, direction, pair, bar["time"])
            if filt.passed:
                filtered_signals.append({
                    "time": bar["time"], "direction": direction,
                    "confidence": filt.confidence, "bar_idx": i,
                })
        except Exception:
            pass

    signals_by_pair[pair] = {
        "raw_buys": raw_buys, "raw_sells": raw_sells,
        "filtered": filtered_signals, "df": sdf,
    }

    print(f"\n  {pair}:")
    print(f"    Raw signals: {raw_buys} buys, {raw_sells} sells ({raw_buys + raw_sells} total)")
    print(f"    After 6-layer filter: {len(filtered_signals)} trades")

# 4. Signal coincidence analysis — do JPY crosses signal together?
print()
print("=" * 90)
print("  SIGNAL COINCIDENCE (same 4H bar or ±1 bar)")
print("=" * 90)

jpy_pairs = [p for p in ["USDJPY", "EURJPY", "GBPJPY"] if p in signals_by_pair]

for i, p1 in enumerate(jpy_pairs):
    for p2 in jpy_pairs[i+1:]:
        sigs1 = signals_by_pair[p1]["filtered"]
        sigs2 = signals_by_pair[p2]["filtered"]

        coincident = 0
        same_dir = 0
        opposite_dir = 0

        for s1 in sigs1:
            for s2 in sigs2:
                diff = abs((s1["time"] - s2["time"]).total_seconds()) / 3600
                if diff <= 4:  # within 1 bar (4H)
                    coincident += 1
                    if s1["direction"] == s2["direction"]:
                        same_dir += 1
                    else:
                        opposite_dir += 1
                    break

        pct1 = (coincident / len(sigs1) * 100) if sigs1 else 0
        pct2 = (coincident / len(sigs2) * 100) if sigs2 else 0
        print(f"  {p1} vs {p2}: {coincident} coincident signals ({pct1:.0f}% of {p1}, {pct2:.0f}% of {p2})")
        print(f"    Same direction: {same_dir} | Opposite: {opposite_dir}")

# 5. Backtest the new pairs with structure SL/TP
print()
print("=" * 90)
print("  BACKTEST: NEW PAIRS | 4H | HIGH filter | Market structure SL/TP")
print("=" * 90)

from forex_signal_engine.market_structure import calculate_structure_sl_tp, SYMBOL_DEFAULTS

for pair in ["AUDUSD", "EURJPY", "GBPJPY"]:
    if pair not in signals_by_pair:
        continue

    sdf = signals_by_pair[pair]["df"]
    filtered = signals_by_pair[pair]["filtered"]
    pip = 0.01 if "JPY" in pair else 0.0001
    spread_pips = 1.5

    balance = 25000.0
    trades = []
    open_trade = None

    for sig in filtered:
        bar_idx = sig["bar_idx"]
        direction = sig["direction"]
        bar = sdf.iloc[bar_idx]

        # Close existing trade at this bar
        if open_trade is not None:
            # Check bars since entry
            for j in range(open_trade["bar_idx"] + 1, bar_idx + 1):
                b = sdf.iloc[j]
                hit = False
                if open_trade["direction"] == "BUY":
                    if b["low"] <= open_trade["sl"]:
                        exit_p = open_trade["sl"]
                        pnl_pips = (exit_p - open_trade["entry"]) / pip - spread_pips
                        hit = True
                    elif b["high"] >= open_trade["tp"]:
                        exit_p = open_trade["tp"]
                        pnl_pips = (exit_p - open_trade["entry"]) / pip - spread_pips
                        hit = True
                else:
                    if b["high"] >= open_trade["sl"]:
                        exit_p = open_trade["sl"]
                        pnl_pips = (open_trade["entry"] - exit_p) / pip - spread_pips
                        hit = True
                    elif b["low"] <= open_trade["tp"]:
                        exit_p = open_trade["tp"]
                        pnl_pips = (open_trade["entry"] - exit_p) / pip - spread_pips
                        hit = True

                if hit:
                    pip_val = 10.0 if "JPY" not in pair else (10.0 / (open_trade["entry"] / 100))
                    pnl = pnl_pips * pip_val * open_trade["lots"]
                    balance += pnl
                    trades.append({"pnl": pnl, "pnl_pips": pnl_pips})
                    open_trade = None
                    break

            if open_trade is not None:
                # Close at current bar (reverse)
                close_p = bar["close"]
                if open_trade["direction"] == "BUY":
                    pnl_pips = (close_p - open_trade["entry"]) / pip - spread_pips
                else:
                    pnl_pips = (open_trade["entry"] - close_p) / pip - spread_pips
                pip_val = 10.0 if "JPY" not in pair else (10.0 / (open_trade["entry"] / 100))
                pnl = pnl_pips * pip_val * open_trade["lots"]
                balance += pnl
                trades.append({"pnl": pnl, "pnl_pips": pnl_pips})
                open_trade = None

        # Open new trade
        try:
            sl, tp, sl_pips, tp_pips = calculate_structure_sl_tp(sdf, bar_idx, direction, pair)
        except Exception:
            continue

        risk_amount = balance * 0.01
        pip_val = 10.0 if "JPY" not in pair else (10.0 / (bar["close"] / 100))
        lots = risk_amount / (sl_pips * pip_val)
        lots = max(0.01, round(lots, 2))

        open_trade = {
            "direction": direction, "entry": bar["close"],
            "sl": sl, "tp": tp, "lots": lots, "bar_idx": bar_idx,
        }

    # Close last trade
    if open_trade is not None:
        last = sdf.iloc[-1]
        close_p = last["close"]
        if open_trade["direction"] == "BUY":
            pnl_pips = (close_p - open_trade["entry"]) / pip - spread_pips
        else:
            pnl_pips = (open_trade["entry"] - close_p) / pip - spread_pips
        pip_val = 10.0 if "JPY" not in pair else (10.0 / (open_trade["entry"] / 100))
        pnl = pnl_pips * pip_val * open_trade["lots"]
        balance += pnl
        trades.append({"pnl": pnl, "pnl_pips": pnl_pips})

    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    net = balance - 25000
    wr = (wins / total * 100) if total else 0
    avg_win = np.mean([t["pnl_pips"] for t in trades if t["pnl"] > 0]) if wins else 0
    avg_loss = np.mean([t["pnl_pips"] for t in trades if t["pnl"] <= 0]) if (total - wins) else 0

    print(f"\n  {pair}: {total} trades | WR: {wr:.0f}% | Net: ${net:+,.2f} | Final: ${balance:,.2f}")
    print(f"    Avg win: {avg_win:+.1f} pips | Avg loss: {avg_loss:+.1f} pips")

    defaults = SYMBOL_DEFAULTS.get(pair)
    if defaults:
        print(f"    Using tuned params: max_sl={defaults['max_sl_pips']}p, min_rr={defaults['min_rr']}")
    else:
        print(f"    Using DEFAULT params: max_sl=50p, min_rr=2.0 (needs tuning!)")

print(f"\nDone.")
