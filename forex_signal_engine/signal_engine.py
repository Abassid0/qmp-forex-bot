"""
Signal engine: runs QMP Filter on live data, emits trade signals with risk parameters.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from forex_signal_engine.config import Config
from forex_signal_engine.indicators import QMPFilter
from forex_signal_engine.market_structure import calculate_structure_sl_tp

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradeSignal:
    signal_type: SignalType
    symbol: str
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    timestamp: datetime
    macd_value: float
    qqe_rsi_ma: float
    qqe_trend: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"

    def __str__(self) -> str:
        return (
            f"{'🟢' if self.signal_type == SignalType.BUY else '🔴'} "
            f"{self.signal_type.value} {self.symbol} @ {self.entry_price:.5f}\n"
            f"SL: {self.stop_loss:.5f} | TP: {self.take_profit:.5f}\n"
            f"Lot: {self.lot_size} | Confidence: {self.confidence}\n"
            f"MACD: {self.macd_value:.6f} | QQE RSI: {self.qqe_rsi_ma:.2f}"
        )


class SignalEngine:
    def __init__(self, config: Config, signal_filter=None):
        self.config = config
        self.signal_filter = signal_filter
        self.qmp = QMPFilter(
            macd_fast=config.macd_fast,
            macd_slow=config.macd_slow,
            macd_signal=config.macd_signal,
            qqe_rsi_period=config.qqe_rsi_period,
            qqe_sf=config.qqe_sf,
            qqe_wp=config.qqe_wp,
            bb_length=config.bb_length,
            bb_mult=config.bb_mult,
        )
        self._last_signal_bar: dict[str, int] = {}

    def analyze(self, symbol: str, df: pd.DataFrame) -> TradeSignal | None:
        """
        Run QMP Filter on candle data and return a TradeSignal if the latest
        confirmed bar triggers a buy or sell.

        Args:
            symbol: e.g. "EURUSD"
            df: DataFrame with columns: open, high, low, close, volume, time
                Must have at least ~300 rows for indicators to warm up.

        Returns:
            TradeSignal if a new signal fires on the last confirmed bar, else None.
        """
        if len(df) < 300:
            logger.warning(f"{symbol}: need at least 300 bars, got {len(df)}")
            return None

        result = self.qmp.calculate(df)
        last = result.iloc[-1]
        bar_index = len(result) - 1

        if self._last_signal_bar.get(symbol) == bar_index:
            return None

        if not last["buy_signal"] and not last["sell_signal"]:
            return None

        self._last_signal_bar[symbol] = bar_index

        signal_type = SignalType.BUY if last["buy_signal"] else SignalType.SELL

        # Apply filters before generating signal
        if self.signal_filter is not None:
            bar_time = last.get("time", None)
            filt = self.signal_filter.evaluate(result, signal_type.value, symbol, bar_time)
            if not filt.passed:
                logger.info(f"{symbol}: signal filtered out — {filt.reason}")
                return None
            confidence = filt.confidence
            trend_dir = filt.trend_direction
        else:
            confidence = self._assess_confidence(last, signal_type)
            trend_dir = ""

        entry = last["close"]

        sl, tp, sl_pips, tp_pips = calculate_structure_sl_tp(
            df=result,
            bar_index=len(result) - 1,
            direction=signal_type.value,
            symbol=symbol,
        )

        return TradeSignal(
            signal_type=signal_type,
            symbol=symbol,
            timeframe=self.config.timeframe,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            lot_size=0.01,  # placeholder — risk manager overrides this
            timestamp=last.get("time", datetime.now()),
            macd_value=last["macd_line"],
            qqe_rsi_ma=last["qqe_rsi_ma"],
            qqe_trend=last["qqe_trend"],
            confidence=confidence,
        )

    def _assess_confidence(self, bar: pd.Series, signal_type: SignalType) -> str:
        """Rate signal confidence based on confluence of indicators."""
        score = 0

        # Trend alignment with moving averages
        close = bar["close"]
        if signal_type == SignalType.BUY:
            if close > bar["ema_50"]:
                score += 1
            if close > bar["ema_100"]:
                score += 1
            if bar["ema_50"] > bar["ema_100"]:
                score += 1
        else:
            if close < bar["ema_50"]:
                score += 1
            if close < bar["ema_100"]:
                score += 1
            if bar["ema_50"] < bar["ema_100"]:
                score += 1

        # Not at Bollinger Band extreme (counter-trend risk)
        if bar["bb_lower"] < close < bar["bb_upper"]:
            score += 1

        if score >= 3:
            return "HIGH"
        elif score >= 2:
            return "MEDIUM"
        return "LOW"

    def _pip_value(self, symbol: str) -> float:
        if "JPY" in symbol:
            return 0.01
        return 0.0001
