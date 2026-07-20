"""
Signal filters derived from real MT5 backtest analysis.
Each filter can reject a signal before it reaches the executor.

Findings from 3+ years of EURUSD/GBPUSD 4H data:
- Asian session (00:00-04:00 UTC) signals lose on both pairs
- London+NY overlap (08:00-16:00 UTC) is consistently profitable
- GBPUSD profits WITH the trend (EMA50>EMA100), EURUSD is neutral
- Monday signals underperform; Friday sells on GBPUSD are losers
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from forex_signal_engine.indicators import _ema, _sma

logger = logging.getLogger(__name__)


class SessionWindow(Enum):
    ASIAN = "ASIAN"          # 00:00-04:00 UTC
    LONDON_EARLY = "LONDON"  # 04:00-08:00 UTC
    OVERLAP = "OVERLAP"      # 08:00-16:00 UTC (London + NY)
    NY_LATE = "NY_LATE"      # 16:00-20:00 UTC
    DEAD = "DEAD"            # 20:00-00:00 UTC


def get_session(hour: int) -> SessionWindow:
    if 0 <= hour < 4:
        return SessionWindow.ASIAN
    elif 4 <= hour < 8:
        return SessionWindow.LONDON_EARLY
    elif 8 <= hour < 16:
        return SessionWindow.OVERLAP
    elif 16 <= hour < 20:
        return SessionWindow.NY_LATE
    return SessionWindow.DEAD


ALLOWED_SESSIONS = {
    "EURUSD": {SessionWindow.OVERLAP, SessionWindow.NY_LATE},
    "GBPUSD": {SessionWindow.OVERLAP, SessionWindow.NY_LATE},
    "USDJPY": {SessionWindow.ASIAN, SessionWindow.OVERLAP},
    "GBPJPY": {SessionWindow.ASIAN, SessionWindow.OVERLAP},
}

BLOCKED_DAYS = {
    "EURUSD": set(),
    "GBPUSD": {4},
    "USDJPY": set(),
    "GBPJPY": {4},
}


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    confidence: str = "LOW"
    trend_direction: str = "NEUTRAL"
    session: str = ""
    score: int = 0


class TrendFilter:
    def check(self, df: pd.DataFrame, signal_direction: str, symbol: str) -> tuple[bool, str, str]:
        last = df.iloc[-1]
        close = last["close"]

        ema_20 = last.get("sma_20", np.nan)
        ema_50 = last.get("ema_50", np.nan)
        ema_100 = last.get("ema_100", np.nan)

        if any(np.isnan(v) for v in [ema_20, ema_50, ema_100]):
            return True, "NEUTRAL", "insufficient EMA data"

        bullish_score = 0
        if close > ema_20:
            bullish_score += 1
        if close > ema_50:
            bullish_score += 1
        if close > ema_100:
            bullish_score += 1
        if ema_20 > ema_50:
            bullish_score += 1
        if ema_50 > ema_100:
            bullish_score += 1

        ema_200 = _ema(df["close"], 200)
        ema_200_val = ema_200.iloc[-1]
        if not np.isnan(ema_200_val):
            if close > ema_200_val:
                bullish_score += 1

        if bullish_score >= 5:
            trend = "STRONG_UP"
        elif bullish_score >= 4:
            trend = "UP"
        elif bullish_score <= 1:
            trend = "STRONG_DOWN"
        elif bullish_score <= 2:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        if symbol == "GBPUSD":
            if signal_direction == "BUY" and trend in ("STRONG_DOWN", "DOWN"):
                return False, trend, "GBPUSD buy rejected: against downtrend"
            if signal_direction == "SELL" and trend in ("STRONG_UP", "UP"):
                return False, trend, "GBPUSD sell rejected: against uptrend"

        if signal_direction == "BUY" and trend == "STRONG_DOWN":
            return False, trend, "buy rejected: strong downtrend"
        if signal_direction == "SELL" and trend == "STRONG_UP":
            return False, trend, "sell rejected: strong uptrend"

        return True, trend, "trend aligned"


class TimeFilter:
    def check(self, bar_time, signal_direction: str, symbol: str) -> tuple[bool, str]:
        if hasattr(bar_time, 'hour'):
            hour = bar_time.hour
            dow = bar_time.dayofweek if hasattr(bar_time, 'dayofweek') else bar_time.weekday()
        else:
            return True, "no timestamp"

        session = get_session(hour)

        allowed = ALLOWED_SESSIONS.get(symbol, {SessionWindow.OVERLAP})
        if session not in allowed:
            return False, f"rejected: {session.value} session ({hour:02d}:00 UTC)"

        blocked = BLOCKED_DAYS.get(symbol, set())
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if dow in blocked:
            return False, f"rejected: {days[dow]} signals underperform for {symbol}"

        return True, f"{session.value} session OK"


class DivergenceFilter:
    """
    Detects MACD-price divergence — a leading reversal signal.
    If price makes higher highs but MACD makes lower highs = bearish divergence (skip buys).
    If price makes lower lows but MACD makes higher lows = bullish divergence (skip sells).
    """

    def check(self, df: pd.DataFrame, signal_direction: str, lookback: int = 20) -> tuple[bool, str, int]:
        if len(df) < lookback + 5:
            return True, "insufficient data for divergence", 0

        recent = df.iloc[-lookback:]
        close = recent["close"].values
        macd = recent["macd_line"].values

        if any(np.isnan(macd)):
            return True, "MACD has NaN", 0

        # Find two most recent swing highs and lows in price
        score_bonus = 0

        if signal_direction == "BUY":
            # Check for hidden bullish divergence (price lower low, MACD higher low = continuation)
            lows_idx = []
            for i in range(2, len(close) - 2):
                if close[i] <= close[i - 1] and close[i] <= close[i - 2] and close[i] <= close[i + 1] and close[i] <= close[i + 2]:
                    lows_idx.append(i)

            if len(lows_idx) >= 2:
                i1, i2 = lows_idx[-2], lows_idx[-1]
                # Bearish divergence: price higher low but MACD lower low = weakness
                if close[i2] > close[i1] and macd[i2] < macd[i1]:
                    return False, "bearish divergence on MACD (buy rejected)", 0
                # Bullish confluence: price lower low but MACD higher low = hidden strength
                if close[i2] < close[i1] and macd[i2] > macd[i1]:
                    score_bonus = 1

        else:
            highs_idx = []
            for i in range(2, len(close) - 2):
                if close[i] >= close[i - 1] and close[i] >= close[i - 2] and close[i] >= close[i + 1] and close[i] >= close[i + 2]:
                    highs_idx.append(i)

            if len(highs_idx) >= 2:
                i1, i2 = highs_idx[-2], highs_idx[-1]
                # Bullish divergence: price lower high but MACD higher high = weakness
                if close[i2] < close[i1] and macd[i2] > macd[i1]:
                    return False, "bullish divergence on MACD (sell rejected)", 0
                # Bearish confluence: price higher high but MACD lower high = hidden weakness
                if close[i2] > close[i1] and macd[i2] < macd[i1]:
                    score_bonus = 1

        return True, "no divergence conflict", score_bonus


class CandleStrengthFilter:
    """
    Rejects signals on weak candles (dojis, spinning tops).
    A strong signal candle should have a body at least 40% of its total range.
    """

    def check(self, df: pd.DataFrame, signal_direction: str) -> tuple[bool, str, int]:
        last = df.iloc[-1]
        body = abs(last["close"] - last["open"])
        total_range = last["high"] - last["low"]

        if total_range == 0:
            return False, "zero-range candle", 0

        body_ratio = body / total_range
        score_bonus = 0

        if body_ratio < 0.25:
            return False, f"doji/spinning top (body {body_ratio:.0%} of range)", 0

        # Check candle direction matches signal
        bullish_candle = last["close"] > last["open"]
        if signal_direction == "BUY" and not bullish_candle:
            if body_ratio > 0.4:
                return False, "bearish candle on buy signal", 0

        if signal_direction == "SELL" and bullish_candle:
            if body_ratio > 0.4:
                return False, "bullish candle on sell signal", 0

        # Strong momentum candle = bonus point
        if body_ratio > 0.65:
            score_bonus = 1

        return True, f"candle body {body_ratio:.0%}", score_bonus


class VolatilityFilter:
    """
    ATR-based volatility filter.
    Rejects signals during very low volatility (choppy/ranging) or
    extremely high volatility (news spikes/blow-off moves).
    """

    def check(self, df: pd.DataFrame, period: int = 14) -> tuple[bool, str, int]:
        if len(df) < period + 50:
            return True, "insufficient data for ATR", 0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-50:].mean()

        if np.isnan(current_atr) or np.isnan(avg_atr) or avg_atr == 0:
            return True, "ATR calculation failed", 0

        atr_ratio = current_atr / avg_atr
        score_bonus = 0

        if atr_ratio < 0.5:
            return False, f"low volatility (ATR {atr_ratio:.1f}x avg) — choppy market", 0

        if atr_ratio > 2.5:
            return False, f"extreme volatility (ATR {atr_ratio:.1f}x avg) — news spike", 0

        # Normal-to-elevated volatility is ideal for trending
        if 0.8 <= atr_ratio <= 1.8:
            score_bonus = 1

        return True, f"ATR {atr_ratio:.1f}x normal", score_bonus


class SRProximityFilter:
    """
    Rejects buys near resistance and sells near support.
    Uses recent swing highs/lows as S/R levels.
    """

    def check(self, df: pd.DataFrame, signal_direction: str, symbol: str, lookback: int = 5) -> tuple[bool, str]:
        if len(df) < 50:
            return True, "insufficient data for S/R"

        pip = 0.01 if "JPY" in symbol else 0.0001
        close = df.iloc[-1]["close"]
        min_clearance = 15 * pip  # need at least 15 pips clearance from S/R

        highs = df["high"].values
        lows = df["low"].values

        # Find recent swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(lookback, len(df) - lookback - 1):
            lo = max(i - lookback, 0)
            hi = min(i + lookback + 1, len(df))
            if highs[i] == highs[lo:hi].max():
                swing_highs.append(highs[i])
            if lows[i] == lows[lo:hi].min():
                swing_lows.append(lows[i])

        if signal_direction == "BUY":
            # Check if price is too close to resistance (recent swing highs)
            for sh in swing_highs[-10:]:
                if 0 < (sh - close) < min_clearance:
                    return False, f"buy too close to resistance {sh:.5f} ({(sh - close) / pip:.0f}p away)"

        else:
            # Check if price is too close to support (recent swing lows)
            for sl in swing_lows[-10:]:
                if 0 < (close - sl) < min_clearance:
                    return False, f"sell too close to support {sl:.5f} ({(close - sl) / pip:.0f}p away)"

        return True, "clear of S/R"


class ConfidenceFilter:
    """
    Enhanced confidence scoring with weighted factors.
    9 base points + up to 3 bonus from new filters = 12 possible.
    """

    def score(self, df: pd.DataFrame, signal_direction: str, trend: str, bonus: int = 0) -> tuple[str, int]:
        last = df.iloc[-1]
        close = last["close"]
        points = 0

        # 1. Trend alignment (0-3 points)
        if signal_direction == "BUY":
            if trend in ("STRONG_UP", "UP"):
                points += 3
            elif trend == "NEUTRAL":
                points += 1
        else:
            if trend in ("STRONG_DOWN", "DOWN"):
                points += 3
            elif trend == "NEUTRAL":
                points += 1

        # 2. MACD histogram momentum (0-2 points)
        hist = last.get("macd_hist", 0)
        prev_hist = df.iloc[-2].get("macd_hist", 0) if len(df) > 1 else 0
        if signal_direction == "BUY" and hist > 0 and hist > prev_hist:
            points += 2
        elif signal_direction == "SELL" and hist < 0 and hist < prev_hist:
            points += 2
        elif (signal_direction == "BUY" and hist > 0) or (signal_direction == "SELL" and hist < 0):
            points += 1

        # 3. QQE strength (0-2 points)
        qqe_rsi = last.get("qqe_rsi_ma", 50)
        if signal_direction == "BUY" and qqe_rsi > 55:
            points += 1
            if qqe_rsi > 60:
                points += 1
        elif signal_direction == "SELL" and qqe_rsi < 45:
            points += 1
            if qqe_rsi < 40:
                points += 1

        # 4. Not at Bollinger extreme (0-1 point)
        bb_upper = last.get("bb_upper", np.nan)
        bb_lower = last.get("bb_lower", np.nan)
        if not np.isnan(bb_upper) and not np.isnan(bb_lower):
            if bb_lower < close < bb_upper:
                points += 1

        # 5. Volume confirmation (0-1 point)
        if "volume" in df.columns:
            avg_vol = df["volume"].rolling(20).mean().iloc[-1]
            if not np.isnan(avg_vol) and last["volume"] > avg_vol:
                points += 1

        # 6. Bonus from divergence, candle strength, volatility filters
        points += bonus

        if points >= 7:
            return "HIGH", points
        elif points >= 4:
            return "MEDIUM", points
        return "LOW", points


class SignalFilter:
    """Combines all filters into a single pass/reject decision."""

    def __init__(self, min_confidence: str = "MEDIUM"):
        self.trend_filter = TrendFilter()
        self.time_filter = TimeFilter()
        self.divergence_filter = DivergenceFilter()
        self.candle_filter = CandleStrengthFilter()
        self.volatility_filter = VolatilityFilter()
        self.sr_filter = SRProximityFilter()
        self.confidence_filter = ConfidenceFilter()
        self._min_confidence = min_confidence
        self._confidence_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    def evaluate(
        self, df: pd.DataFrame, signal_direction: str, symbol: str, bar_time=None
    ) -> FilterResult:
        # 1. Time filter
        if bar_time is not None:
            time_pass, time_reason = self.time_filter.check(bar_time, signal_direction, symbol)
            if not time_pass:
                return FilterResult(passed=False, reason=time_reason)

        # 2. Trend filter
        trend_pass, trend_dir, trend_reason = self.trend_filter.check(df, signal_direction, symbol)
        if not trend_pass:
            return FilterResult(passed=False, reason=trend_reason, trend_direction=trend_dir)

        # 3. Divergence filter
        div_pass, div_reason, div_bonus = self.divergence_filter.check(df, signal_direction)
        if not div_pass:
            return FilterResult(passed=False, reason=div_reason, trend_direction=trend_dir)

        # 4. Candle strength filter
        candle_pass, candle_reason, candle_bonus = self.candle_filter.check(df, signal_direction)
        if not candle_pass:
            return FilterResult(passed=False, reason=candle_reason, trend_direction=trend_dir)

        # 5. Volatility filter
        vol_pass, vol_reason, vol_bonus = self.volatility_filter.check(df)
        if not vol_pass:
            return FilterResult(passed=False, reason=vol_reason, trend_direction=trend_dir)

        # 6. S/R proximity filter
        sr_pass, sr_reason = self.sr_filter.check(df, signal_direction, symbol)
        if not sr_pass:
            return FilterResult(passed=False, reason=sr_reason, trend_direction=trend_dir)

        # 7. Confidence filter (with bonus points from new filters)
        total_bonus = div_bonus + candle_bonus + vol_bonus
        confidence, score = self.confidence_filter.score(df, signal_direction, trend_dir, total_bonus)
        min_level = self._confidence_order.get(self._min_confidence, 1)
        cur_level = self._confidence_order.get(confidence, 0)

        if cur_level < min_level:
            return FilterResult(
                passed=False,
                reason=f"confidence {confidence} (score {score}/12) below minimum {self._min_confidence}",
                confidence=confidence,
                trend_direction=trend_dir,
                score=score,
            )

        session = ""
        if bar_time is not None and hasattr(bar_time, 'hour'):
            session = get_session(bar_time.hour).value

        return FilterResult(
            passed=True,
            reason=f"all filters passed (score {score}/12)",
            confidence=confidence,
            trend_direction=trend_dir,
            session=session,
            score=score,
        )
