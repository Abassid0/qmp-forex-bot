"""
Market structure analysis: swing highs/lows for dynamic SL/TP placement.

SL goes behind the nearest swing point that protects the trade.
TP targets the next opposing structure level or a measured move.
"""

import numpy as np
import pandas as pd


def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Mark bars where high is the highest in a window of 2*lookback+1 bars."""
    highs = df["high"]
    swing = pd.Series(False, index=df.index)
    for i in range(lookback, len(highs) - lookback):
        window = highs.iloc[i - lookback : i + lookback + 1]
        if highs.iloc[i] == window.max():
            swing.iloc[i] = True
    return swing


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Mark bars where low is the lowest in a window of 2*lookback+1 bars."""
    lows = df["low"]
    swing = pd.Series(False, index=df.index)
    for i in range(lookback, len(lows) - lookback):
        window = lows.iloc[i - lookback : i + lookback + 1]
        if lows.iloc[i] == window.min():
            swing.iloc[i] = True
    return swing


def get_recent_swing_levels(
    df: pd.DataFrame, bar_index: int, lookback: int = 5, max_swings: int = 10
) -> tuple[list[float], list[float]]:
    """
    Return lists of recent swing high prices and swing low prices
    looking backwards from bar_index.
    """
    highs = df["high"].values
    lows = df["low"].values

    swing_highs = []
    swing_lows = []

    start = max(lookback, 0)
    end = min(bar_index - lookback, len(df) - lookback)

    for i in range(end, start - 1, -1):
        if len(swing_highs) >= max_swings and len(swing_lows) >= max_swings:
            break

        lo = max(i - lookback, 0)
        hi_bound = min(i + lookback + 1, len(df))
        window_high = highs[lo:hi_bound]
        window_low = lows[lo:hi_bound]

        if len(swing_highs) < max_swings and highs[i] == window_high.max():
            swing_highs.append(highs[i])

        if len(swing_lows) < max_swings and lows[i] == window_low.min():
            swing_lows.append(lows[i])

    return swing_highs, swing_lows


SYMBOL_DEFAULTS = {
    "GBPUSD": {"max_sl_pips": 40.0, "min_rr": 3.5},
    "USDJPY": {"max_sl_pips": 40.0, "min_rr": 2.0},
    "GBPJPY": {"max_sl_pips": 60.0, "min_rr": 3.5},
    "EURUSD": {"max_sl_pips": 40.0, "min_rr": 2.0},
}


def calculate_structure_sl_tp(
    df: pd.DataFrame,
    bar_index: int,
    direction: str,
    symbol: str,
    min_sl_pips: float = 15.0,
    max_sl_pips: float | None = None,
    min_rr: float | None = None,
    swing_lookback: int = 5,
) -> tuple[float, float, float, float]:
    """
    Calculate SL and TP based on market structure.

    For BUY:
      - SL below the nearest swing low (with a small buffer)
      - TP at the nearest swing high above entry, or a measured move

    For SELL:
      - SL above the nearest swing high (with a small buffer)
      - TP at the nearest swing low below entry, or a measured move

    Returns: (stop_loss, take_profit, sl_pips, tp_pips)
    """
    defaults = SYMBOL_DEFAULTS.get(symbol, {"max_sl_pips": 40.0, "min_rr": 2.0})
    if max_sl_pips is None:
        max_sl_pips = defaults["max_sl_pips"]
    if min_rr is None:
        min_rr = defaults["min_rr"]

    pip = 0.01 if "JPY" in symbol else 0.0001
    buffer_pips = 5
    buffer = buffer_pips * pip

    entry = df.iloc[bar_index]["close"]

    swing_highs, swing_lows = get_recent_swing_levels(
        df, bar_index, lookback=swing_lookback, max_swings=10
    )

    if direction == "BUY":
        sl = _find_buy_sl(entry, swing_lows, buffer, pip, min_sl_pips, max_sl_pips)
        sl_distance = entry - sl
        sl_pips = sl_distance / pip

        tp = _find_buy_tp(entry, swing_highs, sl_distance, min_rr)
        tp_pips = (tp - entry) / pip

    else:
        sl = _find_sell_sl(entry, swing_highs, buffer, pip, min_sl_pips, max_sl_pips)
        sl_distance = sl - entry
        sl_pips = sl_distance / pip

        tp = _find_sell_tp(entry, swing_lows, sl_distance, min_rr)
        tp_pips = (entry - tp) / pip

    return sl, tp, sl_pips, tp_pips


def _find_buy_sl(
    entry: float,
    swing_lows: list[float],
    buffer: float,
    pip: float,
    min_sl_pips: float,
    max_sl_pips: float,
) -> float:
    """Find the best swing low to place SL below for a long trade."""
    candidates = [s for s in swing_lows if s < entry]
    candidates.sort(reverse=True)  # nearest first

    for level in candidates:
        sl = level - buffer
        distance_pips = (entry - sl) / pip
        if min_sl_pips <= distance_pips <= max_sl_pips:
            return sl

    # Fallback: use the nearest swing low even if slightly outside range,
    # but clamp to min/max
    if candidates:
        sl = candidates[0] - buffer
        distance_pips = (entry - sl) / pip
        if distance_pips < min_sl_pips:
            return entry - min_sl_pips * pip
        if distance_pips > max_sl_pips:
            return entry - max_sl_pips * pip
        return sl

    return entry - 30 * pip  # absolute fallback


def _find_sell_sl(
    entry: float,
    swing_highs: list[float],
    buffer: float,
    pip: float,
    min_sl_pips: float,
    max_sl_pips: float,
) -> float:
    """Find the best swing high to place SL above for a short trade."""
    candidates = [s for s in swing_highs if s > entry]
    candidates.sort()  # nearest first

    for level in candidates:
        sl = level + buffer
        distance_pips = (sl - entry) / pip
        if min_sl_pips <= distance_pips <= max_sl_pips:
            return sl

    if candidates:
        sl = candidates[0] + buffer
        distance_pips = (sl - entry) / pip
        if distance_pips < min_sl_pips:
            return entry + min_sl_pips * pip
        if distance_pips > max_sl_pips:
            return entry + max_sl_pips * pip
        return sl

    return entry + 30 * pip


def _find_buy_tp(
    entry: float,
    swing_highs: list[float],
    sl_distance: float,
    min_rr: float,
) -> float:
    """
    Find TP for a long trade. Uses the LARGER of:
    - Nearest swing high that gives at least min_rr
    - Measured move (min_rr * SL distance)
    This ensures winners always run at least min_rr.
    """
    measured = entry + sl_distance * min_rr

    candidates = [s for s in swing_highs if s > measured]
    candidates.sort()

    if candidates:
        return candidates[0]

    return measured


def _find_sell_tp(
    entry: float,
    swing_lows: list[float],
    sl_distance: float,
    min_rr: float,
) -> float:
    """
    Find TP for a short trade. Uses the LARGER of:
    - Nearest swing low that gives at least min_rr
    - Measured move (min_rr * SL distance)
    """
    measured = entry - sl_distance * min_rr

    candidates = [s for s in swing_lows if s < measured]
    candidates.sort(reverse=True)

    if candidates:
        return candidates[0]

    return measured
