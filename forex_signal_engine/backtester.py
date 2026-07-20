"""
Backtesting engine for QMP Filter signals.
Simulates trades with proper SL/TP execution, spread, and slippage.
Tracks full P&L, drawdown, win rate, profit factor, and per-trade log.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from forex_signal_engine.config import Config
from forex_signal_engine.indicators import QMPFilter
from forex_signal_engine.market_structure import calculate_structure_sl_tp

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class BacktestTrade:
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str = ""
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    total_bars: int
    start_date: str
    end_date: str
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_dollars: float
    avg_win_pips: float
    avg_loss_pips: float
    avg_rr_achieved: float
    sharpe_ratio: float
    avg_bars_held: float
    longest_win_streak: int
    longest_lose_streak: int
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  BACKTEST: {self.symbol} | {self.timeframe}",
            f"  {self.start_date} -> {self.end_date} ({self.total_bars} bars)",
            f"{'='*60}",
            f"  Initial Balance:    ${self.initial_balance:,.2f}",
            f"  Final Balance:      ${self.final_balance:,.2f}",
            f"  Net Profit:         ${self.net_profit:,.2f} ({self.net_profit/self.initial_balance*100:+.2f}%)",
            f"{'─'*60}",
            f"  Total Trades:       {self.total_trades}",
            f"  Winners:            {self.winning_trades} ({self.win_rate:.1f}%)",
            f"  Losers:             {self.losing_trades}",
            f"  Profit Factor:      {self.profit_factor:.2f}",
            f"{'─'*60}",
            f"  Avg Win:            {self.avg_win_pips:.1f} pips",
            f"  Avg Loss:           {self.avg_loss_pips:.1f} pips",
            f"  Avg R:R Achieved:   {self.avg_rr_achieved:.2f}",
            f"  Avg Bars Held:      {self.avg_bars_held:.1f}",
            f"{'─'*60}",
            f"  Max Drawdown:       ${self.max_drawdown_dollars:,.2f} ({self.max_drawdown_pct:.2f}%)",
            f"  Sharpe Ratio:       {self.sharpe_ratio:.2f}",
            f"  Win Streak:         {self.longest_win_streak}",
            f"  Lose Streak:        {self.longest_lose_streak}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class Backtester:
    """
    Walk-forward backtester: processes each bar sequentially,
    checks for signal entry, and manages open trades with SL/TP
    hit detection on high/low of each bar.
    """

    def __init__(
        self,
        config: Config,
        initial_balance: float = 10_000.0,
        spread_pips: float = 1.5,
        lot_size: float = 0.1,
        risk_pct: float = 1.0,
        signal_filter=None,
    ):
        self.config = config
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.lot_size = lot_size
        self.risk_pct = risk_pct
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

    def run(self, symbol: str, timeframe: str, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on a DataFrame with columns: open, high, low, close, volume.
        Optionally includes 'time' column.
        """
        if len(df) < 300:
            raise ValueError(f"Need at least 300 bars, got {len(df)}")

        pip = 0.01 if "JPY" in symbol else 0.0001
        spread = self.spread_pips * pip
        pip_value_per_lot = 10.0  # ~$10 per pip per standard lot for majors

        # Calculate all indicators up front
        signals_df = self.qmp.calculate(df)

        balance = self.initial_balance
        equity_curve = [balance]
        trades: list[BacktestTrade] = []
        open_trade: BacktestTrade | None = None

        warmup = 300

        for i in range(warmup, len(signals_df)):
            bar = signals_df.iloc[i]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]
            bar_time = bar.get("time", i)

            # --- Manage open trade: check SL/TP on this bar's high/low ---
            if open_trade is not None:
                open_trade.bars_held += 1
                hit_sl = False
                hit_tp = False

                if open_trade.direction == TradeDirection.LONG:
                    if low <= open_trade.stop_loss:
                        hit_sl = True
                        exit_price = open_trade.stop_loss
                    elif high >= open_trade.take_profit:
                        hit_tp = True
                        exit_price = open_trade.take_profit
                else:
                    if high >= open_trade.stop_loss:
                        hit_sl = True
                        exit_price = open_trade.stop_loss
                    elif low <= open_trade.take_profit:
                        hit_tp = True
                        exit_price = open_trade.take_profit

                if hit_sl or hit_tp:
                    open_trade.exit_time = bar_time
                    open_trade.exit_price = exit_price
                    open_trade.exit_reason = "SL" if hit_sl else "TP"

                    if open_trade.direction == TradeDirection.LONG:
                        pnl_pips = (exit_price - open_trade.entry_price) / pip
                    else:
                        pnl_pips = (open_trade.entry_price - exit_price) / pip

                    pnl_pips -= self.spread_pips
                    open_trade.pnl_pips = pnl_pips
                    open_trade.pnl_dollars = pnl_pips * pip_value_per_lot * self.lot_size
                    balance += open_trade.pnl_dollars
                    trades.append(open_trade)
                    open_trade = None

                # Opposite signal closes the trade
                elif open_trade.direction == TradeDirection.LONG and bar["sell_signal"]:
                    open_trade.exit_time = bar_time
                    open_trade.exit_price = close - spread / 2
                    open_trade.exit_reason = "REVERSE"
                    pnl_pips = (open_trade.exit_price - open_trade.entry_price) / pip - self.spread_pips
                    open_trade.pnl_pips = pnl_pips
                    open_trade.pnl_dollars = pnl_pips * pip_value_per_lot * self.lot_size
                    balance += open_trade.pnl_dollars
                    trades.append(open_trade)
                    open_trade = None

                elif open_trade.direction == TradeDirection.SHORT and bar["buy_signal"]:
                    open_trade.exit_time = bar_time
                    open_trade.exit_price = close + spread / 2
                    open_trade.exit_reason = "REVERSE"
                    pnl_pips = (open_trade.entry_price - open_trade.exit_price) / pip - self.spread_pips
                    open_trade.pnl_pips = pnl_pips
                    open_trade.pnl_dollars = pnl_pips * pip_value_per_lot * self.lot_size
                    balance += open_trade.pnl_dollars
                    trades.append(open_trade)
                    open_trade = None

            # --- Entry: only if no open trade ---
            if open_trade is None:
                direction = None
                if bar["buy_signal"]:
                    direction = "BUY"
                elif bar["sell_signal"]:
                    direction = "SELL"

                if direction and self.signal_filter is not None:
                    window = signals_df.iloc[max(0, i - 250):i + 1]
                    filt = self.signal_filter.evaluate(window, direction, symbol, bar_time)
                    if not filt.passed:
                        direction = None

                if direction == "BUY":
                    entry = close + spread / 2
                    sl, tp, _, _ = calculate_structure_sl_tp(
                        signals_df, i, "BUY", symbol,
                    )
                    open_trade = BacktestTrade(
                        direction=TradeDirection.LONG,
                        entry_time=bar_time,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                    )
                elif direction == "SELL":
                    entry = close - spread / 2
                    sl, tp, _, _ = calculate_structure_sl_tp(
                        signals_df, i, "SELL", symbol,
                    )
                    open_trade = BacktestTrade(
                        direction=TradeDirection.SHORT,
                        entry_time=bar_time,
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit=tp,
                    )

            equity_curve.append(balance)

        # Close any remaining open trade at last close
        if open_trade is not None:
            last = signals_df.iloc[-1]
            open_trade.exit_time = last.get("time", len(signals_df) - 1)
            open_trade.exit_price = last["close"]
            open_trade.exit_reason = "END"
            if open_trade.direction == TradeDirection.LONG:
                pnl_pips = (last["close"] - open_trade.entry_price) / pip - self.spread_pips
            else:
                pnl_pips = (open_trade.entry_price - last["close"]) / pip - self.spread_pips
            open_trade.pnl_pips = pnl_pips
            open_trade.pnl_dollars = pnl_pips * pip_value_per_lot * self.lot_size
            balance += open_trade.pnl_dollars
            trades.append(open_trade)
            equity_curve.append(balance)

        return self._compile_result(symbol, timeframe, df, trades, equity_curve)

    def _compile_result(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        trades: list[BacktestTrade],
        equity_curve: list[float],
    ) -> BacktestResult:
        total = len(trades)
        winners = [t for t in trades if t.pnl_pips > 0]
        losers = [t for t in trades if t.pnl_pips <= 0]
        w_count = len(winners)
        l_count = len(losers)

        gross_profit = sum(t.pnl_dollars for t in winners)
        gross_loss = abs(sum(t.pnl_dollars for t in losers))

        # Drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        drawdown = peak - eq
        max_dd_dollars = drawdown.max() if len(drawdown) > 0 else 0.0
        max_dd_pct = (drawdown / peak * 100).max() if len(peak) > 0 and peak.max() > 0 else 0.0

        # Sharpe ratio (annualized, assuming daily returns for simplicity)
        if total > 1:
            returns = [t.pnl_dollars / self.initial_balance for t in trades]
            avg_r = np.mean(returns)
            std_r = np.std(returns)
            sharpe = (avg_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        # Streaks
        win_streak = lose_streak = max_win = max_lose = 0
        for t in trades:
            if t.pnl_pips > 0:
                win_streak += 1
                lose_streak = 0
                max_win = max(max_win, win_streak)
            else:
                lose_streak += 1
                win_streak = 0
                max_lose = max(max_lose, lose_streak)

        avg_win_pips = np.mean([t.pnl_pips for t in winners]) if winners else 0.0
        avg_loss_pips = abs(np.mean([t.pnl_pips for t in losers])) if losers else 0.0
        avg_rr = (avg_win_pips / avg_loss_pips) if avg_loss_pips > 0 else 0.0

        start_date = str(df.iloc[0].get("time", "N/A"))
        end_date = str(df.iloc[-1].get("time", "N/A"))

        return BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(df),
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=equity_curve[-1] if equity_curve else self.initial_balance,
            total_trades=total,
            winning_trades=w_count,
            losing_trades=l_count,
            win_rate=(w_count / total * 100) if total > 0 else 0.0,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=equity_curve[-1] - self.initial_balance if equity_curve else 0.0,
            profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            max_drawdown_pct=max_dd_pct,
            max_drawdown_dollars=max_dd_dollars,
            avg_win_pips=avg_win_pips,
            avg_loss_pips=avg_loss_pips,
            avg_rr_achieved=avg_rr,
            sharpe_ratio=sharpe,
            avg_bars_held=np.mean([t.bars_held for t in trades]) if trades else 0.0,
            longest_win_streak=max_win,
            longest_lose_streak=max_lose,
            trades=trades,
            equity_curve=equity_curve,
        )
