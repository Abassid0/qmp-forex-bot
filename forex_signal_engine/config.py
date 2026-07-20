"""Configuration for the forex signal engine."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # Pairs to monitor
    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    timeframe: str = "1H"

    # Risk management
    risk_per_trade_pct: float = 1.0  # % of account balance risked per trade
    max_open_trades: int = 3
    default_sl_pips: float = 30.0
    default_tp_pips: float = 60.0  # 2:1 reward-to-risk
    trailing_stop_pips: float = 20.0
    min_rr: float = 2.0  # minimum reward-to-risk for structure TP

    # QMP Filter indicator settings (matching PineScript defaults)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    qqe_rsi_period: int = 8
    qqe_sf: int = 1
    qqe_wp: float = 3.0
    bb_length: int = 80
    bb_mult: float = 3.0

    # MT5 connection
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str = ""

    # Telegram notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Engine settings
    candle_lookback: int = 500  # bars of history to load
    poll_interval_seconds: int = 60
