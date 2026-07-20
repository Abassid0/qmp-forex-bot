"""
Position sizing and risk management with prop-firm challenge rules.

Rules enforced:
- Profit target: configurable % of starting balance
- Max loss (EOD trailing): trails from highest end-of-day equity
- Max daily loss: hard cap on single-day losses
- Best day rule: no single day > X% of total realized profit
- Per-trade risk: % of current balance
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime

from forex_signal_engine.config import Config
from forex_signal_engine.signal_engine import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    starting_balance: float = 25000.0
    profit_target_pct: float = 10.0       # 10% = $2,500
    max_loss_pct: float = 10.0            # 10% trailing from highest EOD equity
    max_daily_loss_pct: float = 3.0       # 3% = $750 per day
    best_day_rule_pct: float = 50.0       # no single day > 50% of total profit
    risk_per_trade_pct: float = 1.0       # 1% of balance per trade
    max_open_trades: int = 3

    @property
    def profit_target(self) -> float:
        return self.starting_balance * (self.profit_target_pct / 100)

    @property
    def max_loss_amount(self) -> float:
        return self.starting_balance * (self.max_loss_pct / 100)

    @property
    def max_daily_loss(self) -> float:
        return self.starting_balance * (self.max_daily_loss_pct / 100)


class RiskManager:
    def __init__(self, config: Config, limits: RiskLimits | None = None):
        self.config = config
        self.limits = limits or RiskLimits(
            starting_balance=25000.0,
            risk_per_trade_pct=config.risk_per_trade_pct,
            max_open_trades=config.max_open_trades,
        )

        self._highest_eod_equity = self.limits.starting_balance
        self._daily_pnl: dict[str, float] = {}
        self._daily_profit: dict[str, float] = {}
        self._total_realized_profit = 0.0
        self._target_hit = False
        self._blown = False

    def record_trade_close(self, pnl: float, close_time: datetime | None = None):
        """Call after every trade close to update daily P&L tracking."""
        today = (close_time.strftime("%Y-%m-%d") if close_time else
                 date.today().isoformat())

        self._daily_pnl[today] = self._daily_pnl.get(today, 0) + pnl

        if pnl > 0:
            self._daily_profit[today] = self._daily_profit.get(today, 0) + pnl
            self._total_realized_profit += pnl

    def update_eod_equity(self, equity: float):
        """Call at end of each trading day to update trailing max loss."""
        if equity > self._highest_eod_equity:
            self._highest_eod_equity = equity

    def check_limits(self, current_equity: float) -> tuple[bool, str]:
        """
        Check all prop firm rules. Returns (can_trade, reason).
        Call before opening any new position.
        """
        # 1. Profit target reached
        profit = current_equity - self.limits.starting_balance
        if profit >= self.limits.profit_target:
            self._target_hit = True
            return False, (
                f"PROFIT TARGET HIT: ${profit:+,.2f} >= "
                f"${self.limits.profit_target:,.2f} ({self.limits.profit_target_pct}%)"
            )

        # 2. Max loss (EOD trailing)
        trailing_floor = self._highest_eod_equity - self.limits.max_loss_amount
        if current_equity <= trailing_floor:
            self._blown = True
            return False, (
                f"MAX LOSS BREACHED: equity ${current_equity:,.2f} <= "
                f"trailing floor ${trailing_floor:,.2f} "
                f"(HWM: ${self._highest_eod_equity:,.2f} - ${self.limits.max_loss_amount:,.2f})"
            )

        # 3. Max daily loss
        today = date.today().isoformat()
        today_pnl = self._daily_pnl.get(today, 0)
        if today_pnl <= -self.limits.max_daily_loss:
            return False, (
                f"DAILY LOSS LIMIT: today's P&L ${today_pnl:+,.2f} exceeds "
                f"-${self.limits.max_daily_loss:,.2f} ({self.limits.max_daily_loss_pct}%)"
            )

        # 4. Best day rule — preemptive check
        if self._total_realized_profit > 0:
            today_profit = self._daily_profit.get(today, 0)
            best_day_limit = self._total_realized_profit * (self.limits.best_day_rule_pct / 100)
            if today_profit > best_day_limit and self._total_realized_profit > 100:
                return False, (
                    f"BEST DAY RULE: today's profit ${today_profit:,.2f} is "
                    f"{today_profit / self._total_realized_profit * 100:.0f}% of total "
                    f"${self._total_realized_profit:,.2f} (limit: {self.limits.best_day_rule_pct}%)"
                )

        return True, "all limits OK"

    def size_position(self, signal: TradeSignal, account_balance: float, open_trades: int) -> TradeSignal | None:
        """Calculate lot size based on risk rules. Returns None to reject."""
        # Check prop firm limits first
        can_trade, reason = self.check_limits(account_balance)
        if not can_trade:
            logger.warning(f"RISK BLOCK: {reason}")
            return None

        if open_trades >= self.limits.max_open_trades:
            logger.info(f"Skipping {signal.symbol}: max open trades ({self.limits.max_open_trades}) reached")
            return None

        # Check if this trade would risk more than daily loss allows
        today = date.today().isoformat()
        today_pnl = self._daily_pnl.get(today, 0)
        remaining_daily_risk = self.limits.max_daily_loss + today_pnl  # positive = room left
        if remaining_daily_risk <= 0:
            logger.info(f"Skipping: daily loss limit already reached")
            return None

        risk_amount = account_balance * (self.limits.risk_per_trade_pct / 100.0)

        # Cap risk to remaining daily allowance
        risk_amount = min(risk_amount, remaining_daily_risk)

        sl_distance = abs(signal.entry_price - signal.stop_loss)
        if sl_distance == 0:
            logger.warning(f"Skipping {signal.symbol}: zero SL distance")
            return None

        pip_value = 0.01 if "JPY" in signal.symbol else 0.0001
        sl_pips = sl_distance / pip_value

        pip_cost_per_lot = 10.0 if "JPY" not in signal.symbol else (10.0 / (signal.entry_price / 100))
        lot_size = risk_amount / (sl_pips * pip_cost_per_lot)

        lot_size = max(0.01, round(lot_size, 2))

        signal.lot_size = lot_size
        logger.info(
            f"{signal.symbol}: risking ${risk_amount:.2f} "
            f"({self.limits.risk_per_trade_pct}%) -> {lot_size} lots, "
            f"SL={sl_pips:.1f} pips"
        )
        return signal

    def status_summary(self, current_equity: float) -> str:
        profit = current_equity - self.limits.starting_balance
        trailing_floor = self._highest_eod_equity - self.limits.max_loss_amount
        today = date.today().isoformat()
        today_pnl = self._daily_pnl.get(today, 0)
        daily_remaining = self.limits.max_daily_loss + today_pnl

        lines = [
            f"Risk status:",
            f"  Equity: ${current_equity:,.2f} | Profit: ${profit:+,.2f} / ${self.limits.profit_target:,.2f} target",
            f"  Trailing floor: ${trailing_floor:,.2f} (HWM: ${self._highest_eod_equity:,.2f})",
            f"  Today P&L: ${today_pnl:+,.2f} | Daily loss room: ${daily_remaining:,.2f}",
        ]
        if self._target_hit:
            lines.append("  >>> PROFIT TARGET REACHED <<<")
        if self._blown:
            lines.append("  >>> ACCOUNT BLOWN <<<")
        return "\n".join(lines)
