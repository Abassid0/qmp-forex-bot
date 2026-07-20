# QMP Forex Bot

Auto-trading bot built on the **QMP Filter** system (Zero-Lag MACD Platinum + QQE ADV) with a dual-strategy architecture designed for prop firm challenges.

## What It Does

The bot connects to MetaTrader 5, scans for high-probability setups on 4H candles, and executes trades automatically with full risk management.

**Strategy 1 — QMP Trend Following**
- Zero-Lag MACD (double-EMA: `2*EMA - EMA(EMA)`) eliminates traditional MACD lag
- QQE ADV (volatility-adaptive RSI with ratcheting bands) confirms momentum
- Both indicators must fire within a `barssince()` window — no single-indicator entries
- 6-layer signal filter rejects 87% of raw signals, keeping only HIGH confidence setups

**Strategy 2 — QQE OB/OS Mean-Reversion**
- Trades reversals when QQE RSI hits extreme levels (overbought/oversold)
- Only active in ranging markets (low ADX) — never fights strong trends
- Requires rejection candle confirmation (wick > 30% of range)
- Complements the trend system by capturing profits in sideways markets

## Signal Filters (6 Layers)

| Layer | What It Does |
|-------|-------------|
| Time | Only trades during high-liquidity sessions (London/Overlap/NY) |
| Trend | 6-point EMA ribbon scoring, blocks counter-trend trades |
| Divergence | Detects MACD-price divergence, rejects weakening signals |
| Candle Strength | Rejects dojis/spinning tops, rewards momentum candles |
| Volatility | ATR-based, rejects choppy markets and news spikes |
| S/R Proximity | Won't buy near resistance or sell near support |

## Risk Management

Built for prop firm challenges with configurable rules:

- **Profit Target** — Stops trading when target % is reached
- **Max Loss (EOD Trailing)** — Trails from highest end-of-day equity
- **Daily Loss Cap** — Hard limit on single-day losses
- **Best Day Rule** — No single day exceeds X% of total profit
- **Dynamic Position Sizing** — Calculates lots based on SL distance and risk %

## Market Structure SL/TP

No fixed pip targets. Stop loss and take profit are placed at actual market structure:

- **SL**: Behind the nearest swing high/low + buffer, clamped to per-symbol max
- **TP**: The larger of the measured move (min R:R ratio) or the next swing target

Per-symbol optimization:

| Pair | Max SL | Min R:R | Backtest Net |
|------|--------|---------|-------------|
| GBPUSD | 40 pips | 3.5:1 | +$2,530 |
| USDJPY | 40 pips | 2.0:1 | +$7,092 |
| GBPJPY | 60 pips | 3.5:1 | +$4,577 |

## Setup

### Requirements

- Windows (MetaTrader 5 is Windows-only)
- Python 3.11+
- MetaTrader 5 terminal installed and logged in
- A demo or live MT5 account

### Installation

```bash
git clone https://github.com/Abassid0/qmp-forex-bot.git
cd qmp-forex-bot
pip install -e ".[mt5]"
```

### Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your MT5 login, password, server, terminal path, and Telegram bot token.

### Running

```bash
# Auto-trading (executes real trades on MT5)
python -u -m forex_signal_engine.main --live

# Paper trading (virtual trades, no execution)
python -u -m forex_signal_engine.main

# Backtest on historical data
python -u -m forex_signal_engine.main --backtest
```

## Architecture

```
forex_signal_engine/
  indicators.py       # Zero-Lag MACD + QQE ADV calculations
  filters.py          # 6-layer signal filter (time/trend/divergence/candle/volatility/S&R)
  qqe_obos.py         # QQE overbought/oversold mean-reversion strategy
  signal_engine.py    # QMP trend signal generation
  market_structure.py # Swing high/low detection, dynamic SL/TP placement
  risk_manager.py     # Prop firm risk rules + position sizing
  mt5_executor.py     # MetaTrader 5 order execution
  notifier.py         # Telegram alerts (entry/exit/risk breach)
  main.py             # Main loop: paper, live, backtest modes
  config.py           # Configuration dataclass
  backtester.py       # Historical backtesting engine
```

## Backtest Results

Tested on 5000 4H bars (~3.4 years) with 0.30% risk per trade on a $25,000 account:

**QMP Trend + QQE OB/OS Combined:**

| Pair | QMP Trades | QQE Trades | Combined Net |
|------|-----------|------------|-------------|
| GBPUSD | 53 | 10 | +$844 |
| USDJPY | 62 | 14 | +$1,917 |
| GBPJPY | 64 | 11 | +$2,502 |
| **Total** | **179** | **35** | **+$5,263** |

## Correlation Analysis

The bot exploits currency correlations to diversify risk:

- EURJPY and GBPJPY are 87% correlated — bot trades GBPJPY only (higher volume, more profit)
- USDJPY fires independently from JPY crosses (12-16% signal overlap) — real diversification
- AUDUSD and EURUSD showed no edge with QMP Filter — excluded

## Notifications

The bot sends Telegram alerts for:
- Trade entries (with SL/TP levels and lot size)
- Trade exits (with P&L)
- Risk limit breaches (daily loss, max loss, profit target)
- Startup/shutdown status

## Disclaimer

This software is for educational and research purposes. Trading forex involves substantial risk of loss. Past backtest performance does not guarantee future results. Always test on a demo account before using real capital.

## License

MIT
