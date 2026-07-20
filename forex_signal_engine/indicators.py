"""
Forex indicators translated from PineScript:
- MACD Platinum (Zero-Lag MACD)
- QQE ADV (Quantitative Qualitative Estimation)
- QMP Filter (Combined signal system)
"""

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    return _crossover(a, b) | _crossunder(a, b)


def _barssince(condition: pd.Series) -> pd.Series:
    """Number of bars since condition was last True."""
    result = pd.Series(np.nan, index=condition.index)
    count = np.nan
    for i in range(len(condition)):
        if condition.iloc[i]:
            count = 0
        elif not np.isnan(count):
            count += 1
        result.iloc[i] = count
    return result


class MACDPlatinum:
    """Zero-Lag MACD indicator."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Zero-lag fast EMA
        ma1 = _ema(close, self.fast)
        ma2 = _ema(ma1, self.fast)
        zerolag_fast = 2 * ma1 - ma2

        # Zero-lag slow EMA
        mas1 = _ema(close, self.slow)
        mas2 = _ema(mas1, self.slow)
        zerolag_slow = 2 * mas1 - mas2

        # MACD line
        macd_line = zerolag_fast - zerolag_slow

        # Zero-lag signal line
        emasig1 = _ema(macd_line, self.signal)
        emasig2 = _ema(emasig1, self.signal)
        signal_line = 2 * emasig1 - emasig2

        histogram = macd_line - signal_line

        result = df.copy()
        result["macd_line"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_hist"] = histogram
        result["macd_cross_up"] = _crossover(macd_line, signal_line)
        result["macd_cross_down"] = _crossunder(macd_line, signal_line)
        return result


class QQEADV:
    """Quantitative Qualitative Estimation indicator."""

    def __init__(self, rsi_period: int = 10, sf: int = 14, wp: float = 2.0):
        self.rsi_period = rsi_period
        self.sf = sf
        self.wp = wp

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        wilders_period = self.rsi_period * 2 - 1

        rsi = _rsi(close, self.rsi_period)
        rsi_ma = _ema(rsi, self.sf)

        atr_rsi = (rsi_ma.shift(1) - rsi_ma).abs()
        ma_atr_rsi = _ema(atr_rsi, wilders_period)
        dar = _ema(ma_atr_rsi, wilders_period) * self.wp

        # Iterative calculation for ratcheting bands and trend
        n = len(df)
        rsi0 = np.zeros(n)  # short band (upper)
        rsi1 = np.zeros(n)  # long band (lower)
        trend = np.ones(n)
        rsi_ma_vals = rsi_ma.values
        dar_vals = dar.values

        for i in range(1, n):
            if np.isnan(rsi_ma_vals[i]) or np.isnan(dar_vals[i]):
                rsi0[i] = np.nan
                rsi1[i] = np.nan
                trend[i] = np.nan
                continue

            new_short = rsi_ma_vals[i] + dar_vals[i]
            new_long = rsi_ma_vals[i] - dar_vals[i]

            # Short band ratchets down
            if rsi_ma_vals[i - 1] < rsi0[i - 1] and rsi_ma_vals[i] < rsi0[i - 1]:
                rsi0[i] = min(rsi0[i - 1], new_short)
            else:
                rsi0[i] = new_short

            # Long band ratchets up
            if rsi_ma_vals[i - 1] > rsi1[i - 1] and rsi_ma_vals[i] > rsi1[i - 1]:
                rsi1[i] = max(rsi1[i - 1], new_long)
            else:
                rsi1[i] = new_long

            # Trend detection
            prev_rsi0 = rsi0[i - 1]
            prev_rsi1 = rsi1[i - 1]
            crossed_short = (
                (rsi_ma_vals[i] > prev_rsi0 and rsi_ma_vals[i - 1] <= prev_rsi0)
                or (rsi_ma_vals[i] < prev_rsi0 and rsi_ma_vals[i - 1] >= prev_rsi0)
            )
            crossed_long = (
                (rsi_ma_vals[i] > prev_rsi1 and rsi_ma_vals[i - 1] <= prev_rsi1)
                or (rsi_ma_vals[i] < prev_rsi1 and rsi_ma_vals[i - 1] >= prev_rsi1)
            )

            if crossed_short:
                trend[i] = 1
            elif crossed_long:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1] if not np.isnan(trend[i - 1]) else 1

        second_rsi = np.where(trend == 1, rsi1, rsi0)

        result = df.copy()
        result["qqe_rsi_ma"] = rsi_ma
        result["qqe_second_line"] = second_rsi
        result["qqe_trend"] = trend
        result["qqe_bullish"] = rsi_ma > pd.Series(second_rsi, index=df.index)
        result["qqe_bearish"] = rsi_ma < pd.Series(second_rsi, index=df.index)
        return result


class QMPFilter:
    """
    Combined QMP Filter signal system.
    Uses MACD Platinum for momentum + QQE for trend confirmation.
    Note: QMP uses different QQE settings than standalone QQE ADV.
    """

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        qqe_rsi_period: int = 8,
        qqe_sf: int = 1,
        qqe_wp: float = 3.0,
        bb_length: int = 80,
        bb_mult: float = 3.0,
        sma_period: int = 20,
        ema_50: int = 50,
        ema_100: int = 100,
        wma_240: int = 240,
    ):
        self.macd = MACDPlatinum(macd_fast, macd_slow, macd_signal)
        self.qqe = QQEADV(qqe_rsi_period, qqe_sf, qqe_wp)
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.sma_period = sma_period
        self.ema_50 = ema_50
        self.ema_100 = ema_100
        self.wma_240 = wma_240

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = self.macd.calculate(df)

        qqe_result = self.qqe.calculate(df)
        for col in ["qqe_rsi_ma", "qqe_second_line", "qqe_trend", "qqe_bullish", "qqe_bearish"]:
            result[col] = qqe_result[col]

        # barssince logic for QMP signals
        bars_since_cross_up = _barssince(result["macd_cross_up"])
        bars_since_cross_down = _barssince(result["macd_cross_down"])

        long_situation = (bars_since_cross_up < bars_since_cross_down) & result["qqe_bullish"]
        short_situation = (bars_since_cross_up > bars_since_cross_down) & result["qqe_bearish"]

        # Signal fires on the first bar where condition aligns
        bars_since_long = _barssince(long_situation)
        bars_since_short = _barssince(short_situation)

        result["buy_signal"] = long_situation & (bars_since_long.shift(1) > bars_since_short.shift(1))
        result["sell_signal"] = short_situation & (bars_since_short.shift(1) > bars_since_long.shift(1))

        # Moving averages for trend context
        close = df["close"]
        result["sma_20"] = _sma(close, self.sma_period)
        result["ema_50"] = _ema(close, self.ema_50)
        result["ema_100"] = _ema(close, self.ema_100)
        result["wma_240"] = _wma(close, self.wma_240)

        # Bollinger Bands
        basis = _sma(close, self.bb_length)
        dev = close.rolling(window=self.bb_length).std() * self.bb_mult
        result["bb_upper"] = basis + dev
        result["bb_lower"] = basis - dev
        result["bb_basis"] = basis

        return result
