"""
Backtest: QQE Overbought/Oversold Mean-Reversion Strategy.

Logic:
  - SELL when QQE RSI crosses ABOVE overbought threshold (exhaustion)
  - BUY when QQE RSI crosses BELOW oversold threshold (exhaustion)
  - Only trade in RANGING markets (ADX < threshold)
  - Tight fixed SL with configurable RR
  - Separate from QMP trend system — catches reversals

Tests multiple parameter combos to find optimal setup.
"""
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


def calculate_adx(df, period=14):
    """Average Directional Index — measures trend strength."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    tr = np.zeros(len(df))

    for i in range(1, len(df)):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

    return adx


def calculate_qqe_rsi(df, rsi_period=8, smoothing=1, wilders_period=3.0):
    """QQE RSI calculation — returns smoothed RSI values."""
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    # Smooth RSI
    smoothed_rsi = rsi.ewm(span=smoothing, adjust=False).mean()
    return smoothed_rsi.values


def calculate_atr(df, period=14):
    """ATR for SL sizing."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


def backtest_qqe_obos(symbol, df, ob_level=70, os_level=30, adx_max=30,
                       sl_pips=20, rr=2.0, risk_pct=0.30, use_atr_sl=False):
    """
    Backtest QQE OB/OS mean-reversion.

    Entry:
      - BUY when QQE RSI crosses below os_level (oversold bounce)
      - SELL when QQE RSI crosses above ob_level (overbought rejection)

    Filters:
      - ADX must be below adx_max (ranging market)
      - Candle must show rejection (wick > body for counter-trend)

    Exit: fixed SL/TP based on pips or ATR multiple.
    """
    pip = 0.01 if "JPY" in symbol else 0.0001
    spread_pips = 1.5
    spread = spread_pips * pip

    qqe_rsi = calculate_qqe_rsi(df)
    adx = calculate_adx(df)
    atr = calculate_atr(df)

    balance = 25000.0
    trades = []
    open_trade = None

    for i in range(50, len(df)):
        bar = df.iloc[i]
        prev_rsi = qqe_rsi[i - 1]
        curr_rsi = qqe_rsi[i]
        curr_adx = adx[i]
        curr_atr = atr[i]

        # Check open trade SL/TP
        if open_trade is not None:
            hit = False
            if open_trade["dir"] == "BUY":
                if bar["low"] <= open_trade["sl"]:
                    exit_p = open_trade["sl"]
                    pnl_pips = (exit_p - open_trade["entry"]) / pip - spread_pips
                    hit = True
                elif bar["high"] >= open_trade["tp"]:
                    exit_p = open_trade["tp"]
                    pnl_pips = (exit_p - open_trade["entry"]) / pip - spread_pips
                    hit = True
            else:
                if bar["high"] >= open_trade["sl"]:
                    exit_p = open_trade["sl"]
                    pnl_pips = (open_trade["entry"] - exit_p) / pip - spread_pips
                    hit = True
                elif bar["low"] <= open_trade["tp"]:
                    exit_p = open_trade["tp"]
                    pnl_pips = (open_trade["entry"] - exit_p) / pip - spread_pips
                    hit = True

            if hit:
                pip_val = 10.0 if "JPY" not in symbol else (10.0 / (open_trade["entry"] / 100))
                pnl = pnl_pips * pip_val * open_trade["lots"]
                balance += pnl
                trades.append({
                    "pnl": pnl, "pnl_pips": pnl_pips, "dir": open_trade["dir"],
                    "rsi_entry": open_trade["rsi"], "adx_entry": open_trade["adx"],
                })
                open_trade = None

        if open_trade is not None:
            continue

        # Ranging market filter
        if curr_adx > adx_max:
            continue

        # Determine SL in pips
        if use_atr_sl:
            actual_sl_pips = max(10, min(sl_pips, curr_atr / pip * 1.0))
        else:
            actual_sl_pips = sl_pips

        tp_pips = actual_sl_pips * rr
        sl_dist = actual_sl_pips * pip
        tp_dist = tp_pips * pip

        signal = None

        # OVERSOLD: QQE RSI was below OS level and starts turning up (buy the bounce)
        if prev_rsi <= os_level and curr_rsi > os_level:
            # Rejection candle check: lower wick should be significant
            body = abs(bar["close"] - bar["open"])
            lower_wick = min(bar["open"], bar["close"]) - bar["low"]
            candle_range = bar["high"] - bar["low"]
            if candle_range > 0 and lower_wick / candle_range > 0.3:
                signal = "BUY"

        # OVERBOUGHT: QQE RSI was above OB level and starts turning down (sell the rejection)
        elif prev_rsi >= ob_level and curr_rsi < ob_level:
            body = abs(bar["close"] - bar["open"])
            upper_wick = bar["high"] - max(bar["open"], bar["close"])
            candle_range = bar["high"] - bar["low"]
            if candle_range > 0 and upper_wick / candle_range > 0.3:
                signal = "SELL"

        if signal is None:
            continue

        # Position sizing
        risk_amount = balance * (risk_pct / 100)
        pip_val = 10.0 if "JPY" not in symbol else (10.0 / (bar["close"] / 100))
        lots = risk_amount / (actual_sl_pips * pip_val)
        lots = max(0.01, round(lots, 2))

        entry = bar["close"]
        if signal == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        open_trade = {
            "dir": signal, "entry": entry, "sl": sl, "tp": tp,
            "lots": lots, "rsi": curr_rsi, "adx": curr_adx,
        }

    # Close last trade
    if open_trade is not None:
        last = df.iloc[-1]
        if open_trade["dir"] == "BUY":
            pnl_pips = (last["close"] - open_trade["entry"]) / pip - spread_pips
        else:
            pnl_pips = (open_trade["entry"] - last["close"]) / pip - spread_pips
        pip_val = 10.0 if "JPY" not in symbol else (10.0 / (open_trade["entry"] / 100))
        pnl = pnl_pips * pip_val * open_trade["lots"]
        balance += pnl
        trades.append({"pnl": pnl, "pnl_pips": pnl_pips, "dir": open_trade["dir"],
                        "rsi_entry": open_trade["rsi"], "adx_entry": open_trade["adx"]})

    total = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    net = balance - 25000
    wr = (wins / total * 100) if total else 0

    buys = [t for t in trades if t["dir"] == "BUY"]
    sells = [t for t in trades if t["dir"] == "SELL"]
    buy_wr = (sum(1 for t in buys if t["pnl"] > 0) / len(buys) * 100) if buys else 0
    sell_wr = (sum(1 for t in sells if t["pnl"] > 0) / len(sells) * 100) if sells else 0

    # Max drawdown
    eq = [25000]
    for t in trades:
        eq.append(eq[-1] + t["pnl"])
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak * 100
    max_dd = dd.max() if len(dd) > 0 else 0

    # Profit factor
    gross_w = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_l = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    pf = (gross_w / gross_l) if gross_l > 0 else float('inf')

    return {
        "total": total, "wins": wins, "wr": wr, "net": net, "balance": balance,
        "max_dd": max_dd, "pf": pf, "buys": len(buys), "buy_wr": buy_wr,
        "sells": len(sells), "sell_wr": sell_wr,
    }


# Load data
ALL_PAIRS = ["GBPUSD", "USDJPY", "GBPJPY"]
pairs_data = {}
for symbol in ALL_PAIRS:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 5000)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    pairs_data[symbol] = df
    print(f"  Loaded {symbol}: {len(df)} bars")

mt5.shutdown()

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Parameter sweep
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 95)
print("  QQE OB/OS MEAN-REVERSION BACKTEST | 4H | 0.30% risk | $25k")
print("=" * 95)

for symbol in ALL_PAIRS:
    df = pairs_data[symbol]

    print(f"\n  ── {symbol} Parameter Sweep ──")
    print(f"  {'OB/OS':>7s}  {'ADXmax':>6s}  {'SL':>4s}  {'RR':>4s}  {'Trades':>6s}  {'WR':>5s}  "
          f"{'Net':>10s}  {'PF':>5s}  {'MaxDD':>6s}  {'Buys':>5s}  {'BWR':>4s}  {'Sells':>5s}  {'SWR':>4s}")
    print(f"  {'-------':>7s}  {'------':>6s}  {'----':>4s}  {'----':>4s}  {'------':>6s}  {'-----':>5s}  "
          f"{'----------':>10s}  {'-----':>5s}  {'------':>6s}  {'-----':>5s}  {'----':>4s}  {'-----':>5s}  {'----':>4s}")

    best_net = -999999
    best_combo = None

    for ob, os_lev in [(70, 30), (75, 25), (80, 20)]:
        for adx_max in [20, 25, 30, 35]:
            for sl_pips in [15, 20, 25]:
                for rr in [1.5, 2.0, 2.5, 3.0]:
                    r = backtest_qqe_obos(symbol, df, ob_level=ob, os_level=os_lev,
                                          adx_max=adx_max, sl_pips=sl_pips, rr=rr)

                    if r["total"] < 5:
                        continue

                    marker = ""
                    if r["net"] > best_net:
                        best_net = r["net"]
                        best_combo = (ob, os_lev, adx_max, sl_pips, rr, r)
                        marker = " <<<"

                    # Only print profitable or near-profitable combos to keep output readable
                    if r["net"] > -500 or marker:
                        print(
                            f"  {ob}/{os_lev:>2d}    {adx_max:>5d}  {sl_pips:>3d}p  {rr:>4.1f}  "
                            f"{r['total']:>6d}  {r['wr']:>4.0f}%  ${r['net']:>+9,.0f}  "
                            f"{r['pf']:>5.2f}  {r['max_dd']:>5.1f}%  {r['buys']:>5d}  {r['buy_wr']:>3.0f}%"
                            f"  {r['sells']:>5d}  {r['sell_wr']:>3.0f}%{marker}"
                        )

    if best_combo:
        ob, os_lev, adx, sl, rr, r = best_combo
        print(f"\n  >>> BEST {symbol}: OB/OS={ob}/{os_lev}, ADX<{adx}, SL={sl}p, RR={rr}")
        print(f"      {r['total']} trades, WR {r['wr']:.0f}%, Net ${r['net']:+,.0f}, "
              f"PF {r['pf']:.2f}, DD {r['max_dd']:.1f}%")
    else:
        print(f"\n  >>> {symbol}: No profitable configuration found.")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Combined simulation with QMP + QQE OB/OS
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 95)
print("  COMBINED SYSTEM: QMP Trend + QQE OB/OS Mean-Reversion")
print("=" * 95)

# Run QMP trend backtest for comparison
from forex_signal_engine.indicators import QMPFilter
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.market_structure import calculate_structure_sl_tp

qmp = QMPFilter()
sf = SignalFilter(min_confidence="HIGH")

for symbol in ALL_PAIRS:
    df = pairs_data[symbol]
    sdf = qmp.calculate(df)

    # QMP trend trades
    qmp_balance = 25000.0
    qmp_trades = []
    qmp_open = None

    for i in range(300, len(sdf)):
        bar = sdf.iloc[i]

        # Check open trade
        if qmp_open is not None:
            pip = 0.01 if "JPY" in symbol else 0.0001
            hit = False
            if qmp_open["dir"] == "BUY":
                if bar["low"] <= qmp_open["sl"]:
                    pnl_pips = (qmp_open["sl"] - qmp_open["entry"]) / pip - 1.5
                    hit = True
                elif bar["high"] >= qmp_open["tp"]:
                    pnl_pips = (qmp_open["tp"] - qmp_open["entry"]) / pip - 1.5
                    hit = True
            else:
                if bar["high"] >= qmp_open["sl"]:
                    pnl_pips = (qmp_open["entry"] - qmp_open["sl"]) / pip - 1.5
                    hit = True
                elif bar["low"] <= qmp_open["tp"]:
                    pnl_pips = (qmp_open["entry"] - qmp_open["tp"]) / pip - 1.5
                    hit = True
            if hit:
                pip_val = 10.0 if "JPY" not in symbol else (10.0 / (qmp_open["entry"] / 100))
                pnl = pnl_pips * pip_val * qmp_open["lots"]
                qmp_balance += pnl
                qmp_trades.append(pnl)
                qmp_open = None

        if not (bar["buy_signal"] or bar["sell_signal"]):
            continue
        direction = "BUY" if bar["buy_signal"] else "SELL"
        window = sdf.iloc[max(0, i-250):i+1]
        try:
            filt = sf.evaluate(window, direction, symbol, bar["time"])
            if not filt.passed:
                continue
        except:
            continue

        if qmp_open is not None:
            continue

        try:
            sl, tp, sl_pips_v, tp_pips_v = calculate_structure_sl_tp(sdf, i, direction, symbol)
        except:
            continue

        pip = 0.01 if "JPY" in symbol else 0.0001
        risk_amount = qmp_balance * 0.003
        pip_val = 10.0 if "JPY" not in symbol else (10.0 / (bar["close"] / 100))
        lots = risk_amount / (sl_pips_v * pip_val)
        lots = max(0.01, round(lots, 2))

        qmp_open = {"dir": direction, "entry": bar["close"], "sl": sl, "tp": tp, "lots": lots}

    qmp_net = qmp_balance - 25000
    qmp_total = len(qmp_trades)
    qmp_wins = sum(1 for t in qmp_trades if t > 0)
    qmp_wr = (qmp_wins / qmp_total * 100) if qmp_total else 0

    # Best QQE OB/OS for this pair (re-run)
    best_qqe = None
    best_qqe_net = -999999
    for ob, os_lev in [(70, 30), (75, 25), (80, 20)]:
        for adx_max in [20, 25, 30, 35]:
            for sl_pips in [15, 20, 25]:
                for rr in [1.5, 2.0, 2.5, 3.0]:
                    r = backtest_qqe_obos(symbol, df, ob_level=ob, os_level=os_lev,
                                          adx_max=adx_max, sl_pips=sl_pips, rr=rr)
                    if r["total"] >= 5 and r["net"] > best_qqe_net:
                        best_qqe_net = r["net"]
                        best_qqe = r

    print(f"\n  {symbol}:")
    print(f"    QMP Trend only:   {qmp_total} trades, WR {qmp_wr:.0f}%, Net ${qmp_net:+,.0f}")
    if best_qqe:
        print(f"    QQE OB/OS only:   {best_qqe['total']} trades, WR {best_qqe['wr']:.0f}%, Net ${best_qqe['net']:+,.0f}")
        combined = qmp_net + best_qqe["net"]
        combined_trades = qmp_total + best_qqe["total"]
        print(f"    COMBINED:         {combined_trades} trades, Net ${combined:+,.0f}")
        print(f"    Added value:      ${best_qqe['net']:+,.0f} from {best_qqe['total']} extra trades")
    else:
        print(f"    QQE OB/OS:        No profitable config found")

print(f"\nDone.")
