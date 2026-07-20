"""
MetaTrader 5 integration: data feed + order execution.
Requires the MetaTrader5 package: pip install MetaTrader5
Only works on Windows with MT5 terminal installed.
"""

import logging
from datetime import datetime

import pandas as pd

from forex_signal_engine.config import Config
from forex_signal_engine.signal_engine import TradeSignal, SignalType

logger = logging.getLogger(__name__)

# MT5 timeframe mapping
TIMEFRAME_MAP = {
    "1M": "TIMEFRAME_M1",
    "5M": "TIMEFRAME_M5",
    "15M": "TIMEFRAME_M15",
    "30M": "TIMEFRAME_M30",
    "1H": "TIMEFRAME_H1",
    "4H": "TIMEFRAME_H4",
    "1D": "TIMEFRAME_D1",
}


class MT5Executor:
    def __init__(self, config: Config):
        self.config = config
        self._connected = False

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False

        init_kwargs = {}
        if self.config.mt5_path:
            init_kwargs["path"] = self.config.mt5_path
        if self.config.mt5_login:
            init_kwargs["login"] = self.config.mt5_login
            init_kwargs["password"] = self.config.mt5_password
            init_kwargs["server"] = self.config.mt5_server

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        account = mt5.account_info()
        logger.info(f"Connected to MT5: {account.server}, balance: {account.balance}")
        self._connected = True
        return True

    def disconnect(self):
        import MetaTrader5 as mt5
        mt5.shutdown()
        self._connected = False

    def get_candles(self, symbol: str, count: int = 500) -> pd.DataFrame | None:
        import MetaTrader5 as mt5

        tf_name = TIMEFRAME_MAP.get(self.config.timeframe.upper())
        if not tf_name:
            logger.error(f"Unknown timeframe: {self.config.timeframe}")
            return None

        tf = getattr(mt5, tf_name)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"No data for {symbol}: {mt5.last_error()}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def get_account_balance(self) -> float:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        return info.balance if info else 0.0

    def get_open_trade_count(self) -> int:
        import MetaTrader5 as mt5
        positions = mt5.positions_total()
        return positions if positions is not None else 0

    def execute_signal(self, signal: TradeSignal) -> bool:
        import MetaTrader5 as mt5

        symbol_info = mt5.symbol_info(signal.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {signal.symbol} not found in MT5")
            return False

        if not symbol_info.visible:
            mt5.symbol_select(signal.symbol, True)

        tick = mt5.symbol_info_tick(signal.symbol)
        if tick is None:
            logger.error(f"No tick data for {signal.symbol}")
            return False

        if signal.signal_type == SignalType.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.lot_size,
            "type": order_type,
            "price": price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "deviation": 20,
            "magic": 202501,
            "comment": f"QMP {signal.signal_type.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = result.comment if result else mt5.last_error()
            logger.error(f"Order failed for {signal.symbol}: {error}")
            return False

        logger.info(
            f"Order filled: {signal.signal_type.value} {signal.symbol} "
            f"{signal.lot_size} lots @ {result.price}, ticket #{result.order}"
        )
        return True
