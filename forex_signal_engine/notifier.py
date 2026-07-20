"""Telegram notification sender for trade signals."""

import logging
import urllib.request
import urllib.parse
import json

from forex_signal_engine.config import Config
from forex_signal_engine.signal_engine import TradeSignal, SignalType

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, config: Config):
        self.token = config.telegram_bot_token
        self.chat_id = config.telegram_chat_id
        self._enabled = bool(self.token and self.chat_id)
        if not self._enabled:
            logger.info("Telegram notifications disabled (no token/chat_id)")

    def send_signal(self, signal: TradeSignal, executed: bool = False) -> bool:
        if not self._enabled:
            return False

        pip = 0.01 if "JPY" in signal.symbol else 0.0001
        if signal.signal_type == SignalType.BUY:
            sl_pips = (signal.entry_price - signal.stop_loss) / pip
            tp_pips = (signal.take_profit - signal.entry_price) / pip
        else:
            sl_pips = (signal.stop_loss - signal.entry_price) / pip
            tp_pips = (signal.entry_price - signal.take_profit) / pip

        rr = tp_pips / sl_pips if sl_pips > 0 else 0
        arrow = "\U0001f7e2" if signal.signal_type == SignalType.BUY else "\U0001f534"
        status = "EXECUTED" if executed else "PAPER"

        message = (
            f"{arrow} *{signal.signal_type.value} {signal.symbol}* [{status}]\n"
            f"\n"
            f"Entry: `{signal.entry_price:.5f}`\n"
            f"SL: `{signal.stop_loss:.5f}` ({sl_pips:.0f} pips)\n"
            f"TP: `{signal.take_profit:.5f}` ({tp_pips:.0f} pips)\n"
            f"R:R = 1:{rr:.1f}\n"
            f"\n"
            f"Confidence: *{signal.confidence}*\n"
            f"Timeframe: {signal.timeframe}\n"
            f"MACD: {signal.macd_value:.6f} | QQE RSI: {signal.qqe_rsi_ma:.1f}\n"
            f"\n"
            f"_SL/TP based on market structure_"
        )
        return self._send(message)

    def send_close(self, symbol: str, direction: str, exit_price: float,
                   exit_reason: str, pnl_pips: float, pnl_dollars: float) -> bool:
        if not self._enabled:
            return False

        marker = "✅" if pnl_pips > 0 else "❌"
        message = (
            f"{marker} *CLOSED {symbol} {direction}*\n"
            f"\n"
            f"Exit: `{exit_price:.5f}` ({exit_reason})\n"
            f"P&L: {pnl_pips:+.1f} pips (${pnl_dollars:+.2f})\n"
        )
        return self._send(message)

    def send_message(self, text: str) -> bool:
        if not self._enabled:
            return False
        return self._send(text)

    def _send(self, text: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
