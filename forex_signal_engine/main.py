"""
Main entry point: connects to MT5, polls for signals, manages paper/live trades.

Usage:
    python -m forex_signal_engine              # paper mode (virtual trades on demo)
    python -m forex_signal_engine --live       # live execution on MT5
    python -m forex_signal_engine --backtest   # backtest on historical data
"""

import argparse
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

from forex_signal_engine.config import Config
from forex_signal_engine.signal_engine import SignalEngine, SignalType, TradeSignal
from forex_signal_engine.risk_manager import RiskManager, RiskLimits
from forex_signal_engine.mt5_executor import MT5Executor
from forex_signal_engine.notifier import TelegramNotifier
from forex_signal_engine.filters import SignalFilter
from forex_signal_engine.qqe_obos import QQEObOsEngine

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QMP")


def load_config() -> tuple[Config, RiskLimits]:
    import os
    from dotenv import load_dotenv
    load_dotenv()

    config = Config(
        symbols=os.getenv("QMP_SYMBOLS", "EURUSD,GBPUSD,USDJPY").split(","),
        timeframe=os.getenv("QMP_TIMEFRAME", "4H"),
        risk_per_trade_pct=float(os.getenv("QMP_RISK_PCT", "1.0")),
        max_open_trades=int(os.getenv("QMP_MAX_TRADES", "3")),
        default_sl_pips=float(os.getenv("QMP_SL_PIPS", "50")),
        default_tp_pips=float(os.getenv("QMP_TP_PIPS", "100")),
        mt5_login=int(os.getenv("MT5_LOGIN", "0")),
        mt5_password=os.getenv("MT5_PASSWORD", ""),
        mt5_server=os.getenv("MT5_SERVER", ""),
        mt5_path=os.getenv("MT5_PATH", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        poll_interval_seconds=int(os.getenv("QMP_POLL_INTERVAL", "60")),
    )

    limits = RiskLimits(
        starting_balance=float(os.getenv("STARTING_BALANCE", "25000")),
        profit_target_pct=float(os.getenv("PROFIT_TARGET_PCT", "10")),
        max_loss_pct=float(os.getenv("MAX_LOSS_PCT", "10")),
        max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "3")),
        best_day_rule_pct=float(os.getenv("BEST_DAY_RULE_PCT", "50")),
        risk_per_trade_pct=config.risk_per_trade_pct,
        max_open_trades=config.max_open_trades,
    )

    return config, limits


@dataclass
class PaperTrade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_time: datetime
    confidence: str = ""

    def pip_value(self) -> float:
        return 0.01 if "JPY" in self.symbol else 0.0001

    def current_pnl_pips(self, current_price: float) -> float:
        pv = self.pip_value()
        if self.direction == "BUY":
            return (current_price - self.entry_price) / pv
        return (self.entry_price - current_price) / pv

    def check_exit(self, high: float, low: float) -> tuple[bool, str, float]:
        if self.direction == "BUY":
            if low <= self.stop_loss:
                return True, "SL", self.stop_loss
            if high >= self.take_profit:
                return True, "TP", self.take_profit
        else:
            if high >= self.stop_loss:
                return True, "SL", self.stop_loss
            if low <= self.take_profit:
                return True, "TP", self.take_profit
        return False, "", 0.0


@dataclass
class PaperTradeLog:
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_dollars: float
    exit_reason: str
    entry_time: datetime
    exit_time: datetime


class PaperTrader:
    """Tracks virtual positions against live MT5 prices."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.open_trades: dict[str, PaperTrade] = {}
        self.closed_trades: list[PaperTradeLog] = []

    def open(self, signal: TradeSignal) -> PaperTrade:
        trade = PaperTrade(
            symbol=signal.symbol,
            direction=signal.signal_type.value,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            lot_size=signal.lot_size,
            entry_time=signal.timestamp,
            confidence=signal.confidence,
        )
        self.open_trades[signal.symbol] = trade
        return trade

    def check_and_close(self, symbol: str, high: float, low: float, current_price: float) -> PaperTradeLog | None:
        trade = self.open_trades.get(symbol)
        if trade is None:
            return None

        hit, reason, exit_price = trade.check_exit(high, low)
        if not hit:
            return None

        return self._close(trade, exit_price, reason)

    def close_on_reverse(self, symbol: str, current_price: float) -> PaperTradeLog | None:
        trade = self.open_trades.get(symbol)
        if trade is None:
            return None
        return self._close(trade, current_price, "REVERSE")

    def _close(self, trade: PaperTrade, exit_price: float, reason: str) -> PaperTradeLog:
        pnl_pips = trade.current_pnl_pips(exit_price)
        pip_value_per_lot = 10.0
        pnl_dollars = pnl_pips * pip_value_per_lot * trade.lot_size

        self.balance += pnl_dollars
        del self.open_trades[trade.symbol]

        log = PaperTradeLog(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            pnl_pips=pnl_pips,
            pnl_dollars=pnl_dollars,
            exit_reason=reason,
            entry_time=trade.entry_time,
            exit_time=datetime.now(),
        )
        self.closed_trades.append(log)
        return log

    def summary(self) -> str:
        total = len(self.closed_trades)
        winners = sum(1 for t in self.closed_trades if t.pnl_pips > 0)
        net = self.balance - self.initial_balance
        wr = (winners / total * 100) if total > 0 else 0

        lines = [
            f"Paper Trading Summary",
            f"  Trades: {total} | Wins: {winners} ({wr:.0f}%) | Net: ${net:+,.2f}",
            f"  Balance: ${self.balance:,.2f}",
        ]

        if self.open_trades:
            lines.append(f"  Open positions: {len(self.open_trades)}")
            for sym, t in self.open_trades.items():
                lines.append(f"    {sym}: {t.direction} @ {t.entry_price:.5f}")

        return "\n".join(lines)


def run_paper(config: Config):
    """Paper trading loop — tracks virtual positions against live MT5 prices."""
    signal_filter = SignalFilter(min_confidence="HIGH")
    engine = SignalEngine(config, signal_filter=signal_filter)
    executor = MT5Executor(config)
    notifier = TelegramNotifier(config)
    paper = PaperTrader(initial_balance=10000.0)

    if not executor.connect():
        logger.error("Failed to connect to MT5. Exiting.")
        return

    import MetaTrader5 as mt5
    account = mt5.account_info()

    print()
    print("=" * 70)
    print("  QMP FILTER - PAPER TRADING MODE")
    print("=" * 70)
    print(f"  Account: {account.login} ({account.server})")
    print(f"  Real Balance: {account.balance:,.0f} {account.currency}")
    print(f"  Paper Balance: ${paper.balance:,.2f}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  SL/TP: Market structure (swing highs/lows)")
    print(f"    GBPUSD: MaxSL=35p, MinRR=1:3")
    print(f"    USDJPY: MaxSL=40p, MinRR=1:2")
    print(f"  Filter: HIGH confidence only")
    print(f"  Poll interval: {config.poll_interval_seconds}s")
    print("=" * 70)
    print()
    print("  Watching for signals... (Ctrl+C to stop)")
    print()

    notifier.send_message(
        "QMP Bot started [PAPER]\n"
        f"Symbols: {', '.join(config.symbols)}\n"
        f"Filter: HIGH confidence"
    )

    last_bar_time: dict[str, str] = {}
    cycle = 0

    try:
        while True:
            cycle += 1
            now = datetime.now().strftime("%H:%M:%S")

            for symbol in config.symbols:
                df = executor.get_candles(symbol, config.candle_lookback)
                if df is None:
                    continue

                last = df.iloc[-1]
                bar_key = str(last["time"])

                # Check open trade SL/TP on every poll
                if symbol in paper.open_trades:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        closed = paper.check_and_close(
                            symbol, tick.ask, tick.bid,
                            tick.bid if paper.open_trades[symbol].direction == "BUY" else tick.ask,
                        )
                        if closed:
                            marker = "WIN" if closed.pnl_pips > 0 else "LOSS"
                            print(
                                f"  [{now}] CLOSED {closed.symbol} {closed.direction} "
                                f"@ {closed.exit_price:.5f} ({closed.exit_reason}) "
                                f"{closed.pnl_pips:+.1f} pips (${closed.pnl_dollars:+.2f}) [{marker}]"
                            )
                            print(f"           Paper balance: ${paper.balance:,.2f}")
                            notifier.send_close(
                                closed.symbol, closed.direction, closed.exit_price,
                                closed.exit_reason, closed.pnl_pips, closed.pnl_dollars,
                            )

                # Only check for new signals on new bars
                if bar_key == last_bar_time.get(symbol):
                    continue
                last_bar_time[symbol] = bar_key

                # Check for signal
                signal = engine.analyze(symbol, df)
                if signal is None:
                    continue

                # Close opposite trade if exists
                if symbol in paper.open_trades:
                    existing = paper.open_trades[symbol]
                    if existing.direction != signal.signal_type.value:
                        tick = mt5.symbol_info_tick(symbol)
                        close_price = tick.bid if existing.direction == "BUY" else tick.ask
                        closed = paper.close_on_reverse(symbol, close_price)
                        if closed:
                            marker = "WIN" if closed.pnl_pips > 0 else "LOSS"
                            print(
                                f"  [{now}] REVERSED {closed.symbol} {closed.direction} "
                                f"@ {closed.exit_price:.5f} "
                                f"{closed.pnl_pips:+.1f} pips [{marker}]"
                            )
                    else:
                        continue  # same direction, skip

                # Open new paper trade
                trade = paper.open(signal)
                print(
                    f"  [{now}] SIGNAL {signal.signal_type.value} {signal.symbol} "
                    f"@ {signal.entry_price:.5f} | SL: {signal.stop_loss:.5f} "
                    f"TP: {signal.take_profit:.5f} | Conf: {signal.confidence}"
                )
                notifier.send_signal(signal, executed=False)

            # Periodic status update every 10 cycles
            if cycle % 10 == 0:
                open_str = ""
                for sym, t in paper.open_trades.items():
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        cur = tick.bid if t.direction == "BUY" else tick.ask
                        pnl = t.current_pnl_pips(cur)
                        open_str += f"  {sym} {t.direction}: {pnl:+.1f}p | "

                status = f"  [{now}] Cycle {cycle} | Balance: ${paper.balance:,.2f}"
                if open_str:
                    status += f" | Open: {open_str.rstrip(' | ')}"
                if paper.closed_trades:
                    net = paper.balance - paper.initial_balance
                    status += f" | Net: ${net:+,.2f} ({len(paper.closed_trades)} trades)"
                print(status)

            time.sleep(config.poll_interval_seconds)

    except KeyboardInterrupt:
        print()
        print("-" * 70)
        print(f"  {paper.summary()}")
        if paper.closed_trades:
            print()
            print("  Trade log:")
            print(f"  {'#':>3s}  {'Symbol':>6s}  {'DIR':>5s}  {'Entry':>10s}  {'Exit':>10s}  {'P&L':>8s}  {'$':>8s}  {'Why':>5s}")
            for i, t in enumerate(paper.closed_trades, 1):
                print(
                    f"  {i:>3d}  {t.symbol:>6s}  {t.direction:>5s}  {t.entry_price:>10.5f}  "
                    f"{t.exit_price:>10.5f}  {t.pnl_pips:>+7.1f}p  ${t.pnl_dollars:>+7.2f}  {t.exit_reason:>5s}"
                )
        print("-" * 70)

    finally:
        executor.disconnect()
        notifier.send_message("QMP Bot stopped")


def run_live(config: Config, limits: RiskLimits | None = None):
    """
    Live auto-trading loop:
    - Executes real trades on MT5 (demo or live)
    - Monitors open positions and detects SL/TP closes
    - Enforces prop firm risk rules
    - Sends Telegram alerts on every entry and exit
    """
    import MetaTrader5 as mt5

    signal_filter = SignalFilter(min_confidence="HIGH")
    engine = SignalEngine(config, signal_filter=signal_filter)
    qqe_engine = QQEObOsEngine()
    risk_mgr = RiskManager(config, limits)
    executor = MT5Executor(config)
    notifier = TelegramNotifier(config)

    if not executor.connect():
        logger.error("Failed to connect to MT5. Exiting.")
        return

    account = mt5.account_info()
    magic = 202501

    # Sync starting balance with actual MT5 account balance
    if abs(risk_mgr.limits.starting_balance - account.balance) > 100:
        logger.warning(
            f"Config starting_balance (${risk_mgr.limits.starting_balance:,.0f}) "
            f"doesn't match MT5 balance (${account.balance:,.0f}). Using MT5 balance."
        )
        risk_mgr.limits.starting_balance = account.balance
        risk_mgr._highest_eod_equity = account.balance

    lim = risk_mgr.limits

    print()
    print("=" * 70)
    print("  QMP FILTER - AUTO TRADING MODE")
    print("=" * 70)
    print(f"  Account: {account.login} ({account.server})")
    print(f"  Balance: {account.balance:,.0f} {account.currency}")
    print(f"  Symbols: {', '.join(config.symbols)}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  SL/TP: Market structure (swing highs/lows)")
    print(f"  Strategy 1: QMP Trend (6-layer HIGH filter)")
    print(f"  Strategy 2: QQE OB/OS Mean-Reversion (ranging markets)")
    print(f"  Risk: {lim.risk_per_trade_pct}% per trade | Max open: {lim.max_open_trades}")
    print(f"  ---- Prop Firm Rules (${lim.starting_balance:,.0f} account) ----")
    print(f"  Profit target:  {lim.profit_target_pct}% (${lim.profit_target:,.0f})")
    print(f"  Max loss (EOD): {lim.max_loss_pct}% (${lim.max_loss_amount:,.0f} trailing)")
    print(f"  Max daily loss: {lim.max_daily_loss_pct}% (${lim.max_daily_loss:,.0f})")
    print(f"  Best day rule:  {lim.best_day_rule_pct}%")
    print(f"  Poll interval: {config.poll_interval_seconds}s")
    print("=" * 70)
    print()
    print("  Bot is trading automatically. Ctrl+C to stop.")
    print()

    notifier.send_message(
        "QMP Bot started [AUTO TRADING]\n"
        f"Account: {account.login}\n"
        f"Symbols: {', '.join(config.symbols)}\n"
        f"Risk: {lim.risk_per_trade_pct}% per trade\n"
        f"Strategy 1: QMP Trend (6-layer filter)\n"
        f"Strategy 2: QQE OB/OS (mean-reversion)\n"
        f"--- Risk rules ---\n"
        f"Target: {lim.profit_target_pct}% (${lim.profit_target:,.0f})\n"
        f"Max loss: {lim.max_loss_pct}% trailing\n"
        f"Daily loss: {lim.max_daily_loss_pct}%\n"
        f"Best day: {lim.best_day_rule_pct}%"
    )

    # Track positions opened by this bot (by magic number)
    known_tickets: set[int] = set()
    # Snapshot current bot positions on startup
    positions = mt5.positions_get()
    if positions:
        for p in positions:
            if p.magic == magic:
                known_tickets.add(p.ticket)
                print(f"  Existing position: {p.symbol} {'BUY' if p.type == 0 else 'SELL'} {p.volume} lots @ {p.price_open:.5f} (#{p.ticket})")

    last_bar_time: dict[str, str] = {}
    cycle = 0
    total_pnl = 0.0
    closed_count = 0
    last_risk_alert: str = ""  # prevent spam — only alert when reason changes

    try:
        while True:
            cycle += 1
            now = datetime.now().strftime("%H:%M:%S")

            # ── 1. Check for closed positions (SL/TP hit by MT5) ──
            current_tickets = set()
            positions = mt5.positions_get()
            if positions:
                for p in positions:
                    if p.magic == magic:
                        current_tickets.add(p.ticket)

            closed_tickets = known_tickets - current_tickets
            for ticket in closed_tickets:
                # Position was closed — find it in deal history
                deals = mt5.history_deals_get(
                    datetime(2025, 1, 1), datetime.now(), group="*"
                )
                if deals:
                    for deal in reversed(deals):
                        if deal.position_id == ticket and deal.entry == 1:  # entry==1 means exit deal
                            symbol = deal.symbol
                            pip = 0.01 if "JPY" in symbol else 0.0001
                            pnl = deal.profit + deal.commission + deal.swap
                            direction = "SELL" if deal.type == 0 else "BUY"  # exit deal type is opposite

                            print(
                                f"  [{now}] CLOSED #{ticket} {symbol} {direction} "
                                f"@ {deal.price:.5f} | P&L: ${pnl:+.2f}"
                            )

                            total_pnl += pnl
                            closed_count += 1

                            risk_mgr.record_trade_close(pnl, datetime.now())

                            notifier.send_close(
                                symbol, direction, deal.price,
                                "SL/TP", pnl / (10 * 0.1), pnl,
                            )
                            break

            known_tickets = current_tickets

            # ── 2. Check for new signals on new bars ──
            for symbol in config.symbols:
                df = executor.get_candles(symbol, config.candle_lookback)
                if df is None:
                    continue

                last = df.iloc[-1]
                bar_key = str(last["time"])

                if bar_key == last_bar_time.get(symbol):
                    continue
                last_bar_time[symbol] = bar_key

                # Try QMP trend signal first, then QQE OB/OS
                signal = engine.analyze(symbol, df)
                strategy_tag = "QMP"

                if signal is None:
                    signal = qqe_engine.analyze(symbol, df, config.timeframe)
                    strategy_tag = "QQE_OBOS"

                if signal is None:
                    continue

                # Check if we already have a position on this symbol
                has_position = False
                position_direction = None
                if positions:
                    for p in positions:
                        if p.magic == magic and p.symbol == symbol:
                            has_position = True
                            position_direction = "BUY" if p.type == 0 else "SELL"
                            break

                # QQE OB/OS should not reverse existing QMP positions
                if has_position and strategy_tag == "QQE_OBOS":
                    continue

                # If we have an opposite position, close it first (QMP only)
                if has_position and position_direction != signal.signal_type.value:
                    for p in positions:
                        if p.magic == magic and p.symbol == symbol:
                            close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
                            tick = mt5.symbol_info_tick(symbol)
                            close_price = tick.bid if p.type == 0 else tick.ask

                            close_request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": symbol,
                                "volume": p.volume,
                                "type": close_type,
                                "position": p.ticket,
                                "price": close_price,
                                "deviation": 20,
                                "magic": magic,
                                "comment": "QMP REVERSE",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            result = mt5.order_send(close_request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                pnl = p.profit + p.swap + p.commission
                                total_pnl += pnl
                                closed_count += 1
                                print(
                                    f"  [{now}] REVERSED #{p.ticket} {symbol} {position_direction} "
                                    f"@ {close_price:.5f} | P&L: ${pnl:+.2f}"
                                )
                                risk_mgr.record_trade_close(pnl, datetime.now())

                                notifier.send_close(
                                    symbol, position_direction, close_price,
                                    "REVERSE", p.profit / (10 * p.volume), pnl,
                                )
                                known_tickets.discard(p.ticket)
                            else:
                                err = result.comment if result else mt5.last_error()
                                logger.error(f"Failed to close {symbol}: {err}")
                            break
                    has_position = False

                elif has_position:
                    continue  # same direction, skip

                # Size and execute the new trade
                balance = executor.get_account_balance()
                open_count = len(known_tickets)
                sized_signal = risk_mgr.size_position(signal, balance, open_count)

                if sized_signal is None:
                    continue

                executed = executor.execute_signal(sized_signal)
                if executed:
                    # Find the new ticket
                    new_positions = mt5.positions_get(symbol=symbol)
                    if new_positions:
                        for p in new_positions:
                            if p.magic == magic and p.ticket not in known_tickets:
                                known_tickets.add(p.ticket)
                                break

                    pip = 0.01 if "JPY" in symbol else 0.0001
                    sl_pips = abs(signal.entry_price - signal.stop_loss) / pip
                    tp_pips = abs(signal.take_profit - signal.entry_price) / pip

                    print(
                        f"  [{now}] [{strategy_tag}] OPENED {signal.signal_type.value} {symbol} "
                        f"{sized_signal.lot_size} lots @ {signal.entry_price:.5f} "
                        f"| SL: {sl_pips:.0f}p TP: {tp_pips:.0f}p"
                    )

                notifier.send_signal(sized_signal, executed=executed)

            # ── 3. Periodic status ──
            if cycle % 10 == 0:
                account = mt5.account_info()
                risk_mgr.update_eod_equity(account.equity)

                open_str = ""
                positions = mt5.positions_get()
                if positions:
                    for p in positions:
                        if p.magic == magic:
                            direction = "BUY" if p.type == 0 else "SELL"
                            open_str += f"  {p.symbol} {direction}: ${p.profit:+.2f} |"

                status = f"  [{now}] Cycle {cycle} | Equity: {account.equity:,.0f} {account.currency}"
                if open_str:
                    status += f" | {open_str.rstrip(' |')}"
                if closed_count > 0:
                    status += f" | Closed: {closed_count} (${total_pnl:+,.2f})"

                # Check risk limits
                can_trade, risk_reason = risk_mgr.check_limits(account.equity)
                if not can_trade:
                    status += f"\n  >>> {risk_reason} <<<"
                    if risk_reason != last_risk_alert:
                        notifier.send_message(f"RISK ALERT: {risk_reason}")
                        last_risk_alert = risk_reason

                print(status)

            time.sleep(config.poll_interval_seconds)

    except KeyboardInterrupt:
        print()
        print("-" * 70)
        account = mt5.account_info()
        print(f"  Final equity: {account.equity:,.0f} {account.currency}")
        print(f"  Trades closed this session: {closed_count}")
        print(f"  Session P&L: ${total_pnl:+,.2f}")

        positions = mt5.positions_get()
        bot_positions = [p for p in (positions or []) if p.magic == magic]
        if bot_positions:
            print(f"  Open positions left: {len(bot_positions)}")
            for p in bot_positions:
                direction = "BUY" if p.type == 0 else "SELL"
                print(f"    {p.symbol} {direction} {p.volume} lots | P&L: ${p.profit:+.2f}")
        print("-" * 70)

    finally:
        executor.disconnect()
        notifier.send_message("QMP Bot stopped")


def run_backtest(config: Config):
    """Backtest QMP signals on historical MT5 data."""
    engine = SignalEngine(config)
    executor = MT5Executor(config)

    if not executor.connect():
        logger.error("Failed to connect to MT5. Exiting.")
        return

    logger.info("=== QMP Filter Backtest ===")

    for symbol in config.symbols:
        df = executor.get_candles(symbol, count=5000)
        if df is None:
            continue

        result = engine.qmp.calculate(df)
        buys = result[result["buy_signal"]].copy()
        sells = result[result["sell_signal"]].copy()

        logger.info(f"\n--- {symbol} ({config.timeframe}) ---")
        logger.info(f"Total bars: {len(df)}")
        logger.info(f"Buy signals: {len(buys)}")
        logger.info(f"Sell signals: {len(sells)}")

        if len(buys) > 0:
            logger.info(f"Last buy:  {buys.iloc[-1]['time']} @ {buys.iloc[-1]['close']:.5f}")
        if len(sells) > 0:
            logger.info(f"Last sell: {sells.iloc[-1]['time']} @ {sells.iloc[-1]['close']:.5f}")

    executor.disconnect()


def main():
    parser = argparse.ArgumentParser(description="QMP Filter Forex Signal Bot")
    parser.add_argument("--live", action="store_true", help="Execute real trades on MT5")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on historical data")
    args = parser.parse_args()

    config, limits = load_config()

    if args.backtest:
        run_backtest(config)
    elif args.live:
        run_live(config, limits)
    else:
        run_paper(config)


if __name__ == "__main__":
    main()
