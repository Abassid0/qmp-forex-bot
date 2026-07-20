"""
QQE Overbought/Oversold mean-reversion strategy.

Complements the QMP trend system by trading reversals in ranging markets.
Sells at overbought exhaustion, buys at oversold exhaustion.
Only active when ADX is low (no trend).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_signal_engine.signal_engine import SignalType, TradeSignal

logger = logging.getLogger(__name__)

SYMBOL_PARAMS = {
    "GBPUSD": {"ob": 75, "os": 25, "adx_max": 35, "sl_pips": 25, "rr": 3.0},
    "USDJPY": {"ob": 75, "os": 25, "adx_max": 35, "sl_pips": 25, "rr": 2.5},
    "GBPJPY": {"ob": 75, "os": 25, "adx_max": 35, "sl_pips": 20, "rr": 2.0},
}

DEFAULT_PARAMS = {"ob": 75, "os": 25, "adx_max": 30, "sl_pips": 20, "rr": 2.0}


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    tr = np.zeros(len(df))

    for i in range(1, len(df)):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    safe_atr = np.where(atr > 0, atr, 1)
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / safe_atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / safe_atr

    di_sum = plus_di + minus_di
    dx = 100 * np.abs(plus_di - minus_di) / np.where(di_sum > 0, di_sum, 1)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return adx


def _calculate_qqe_rsi(df: pd.DataFrame, rsi_period: int = 8, smoothing: int = 1) -> np.ndarray:
    close = df["close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)

    smoothed = rsi.ewm(span=smoothing, adjust=False).mean()
    return smoothed.values


class QQEObOsEngine:
    """Detects QQE overbought/oversold setups in ranging markets."""

    def __init__(self):
        self._last_signal_bar: dict[str, int] = {}

    def analyze(self, symbol: str, df: pd.DataFrame, timeframe: str = "4H") -> TradeSignal | None:
        if len(df) < 100:
            return None

        params = SYMBOL_PARAMS.get(symbol, DEFAULT_PARAMS)
        ob = params["ob"]
        os_level = params["os"]
        adx_max = params["adx_max"]
        sl_pips = params["sl_pips"]
        rr = params["rr"]

        pip = 0.01 if "JPY" in symbol else 0.0001

        qqe_rsi = _calculate_qqe_rsi(df)
        adx = _calculate_adx(df)

        bar_idx = len(df) - 1

        if self._last_signal_bar.get(symbol) == bar_idx:
            return None

        curr_rsi = qqe_rsi[bar_idx]
        prev_rsi = qqe_rsi[bar_idx - 1]
        curr_adx = adx[bar_idx]

        if curr_adx > adx_max:
            return None

        bar = df.iloc[bar_idx]
        candle_range = bar["high"] - bar["low"]
        if candle_range == 0:
            return None

        signal_type = None

        # Oversold bounce: RSI crosses back above OS level + lower wick rejection
        if prev_rsi <= os_level and curr_rsi > os_level:
            lower_wick = min(bar["open"], bar["close"]) - bar["low"]
            if lower_wick / candle_range > 0.3:
                signal_type = SignalType.BUY

        # Overbought rejection: RSI crosses back below OB level + upper wick rejection
        elif prev_rsi >= ob and curr_rsi < ob:
            upper_wick = bar["high"] - max(bar["open"], bar["close"])
            if upper_wick / candle_range > 0.3:
                signal_type = SignalType.SELL

        if signal_type is None:
            return None

        self._last_signal_bar[symbol] = bar_idx

        entry = bar["close"]
        sl_dist = sl_pips * pip
        tp_dist = sl_pips * rr * pip

        if signal_type == SignalType.BUY:
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        logger.info(
            f"{symbol}: QQE OB/OS {signal_type.value} — "
            f"RSI {curr_rsi:.1f}, ADX {curr_adx:.1f}, "
            f"SL {sl_pips}p, TP {sl_pips * rr:.0f}p"
        )

        return TradeSignal(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=timeframe,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            lot_size=0.01,
            timestamp=bar.get("time", None),
            macd_value=0.0,
            qqe_rsi_ma=curr_rsi,
            qqe_trend=0.0,
            confidence=f"QQE_OBOS (ADX={curr_adx:.0f})",
        )
