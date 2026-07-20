"""Smoke test: run all indicators on synthetic data to verify calculations."""

import numpy as np
import pandas as pd
from forex_signal_engine.indicators import MACDPlatinum, QQEADV, QMPFilter


def make_synthetic_candles(n: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    # Simulate a trending-then-reversing price series
    trend = np.concatenate([
        np.linspace(1.1000, 1.1500, n // 3),
        np.linspace(1.1500, 1.1200, n // 3),
        np.linspace(1.1200, 1.1600, n - 2 * (n // 3)),
    ])
    noise = np.random.normal(0, 0.001, n)
    close = trend + noise
    high = close + np.random.uniform(0.0005, 0.002, n)
    low = close - np.random.uniform(0.0005, 0.002, n)
    open_ = close + np.random.normal(0, 0.0005, n)
    volume = np.random.randint(100, 10000, n).astype(float)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
    })


def test_macd():
    df = make_synthetic_candles()
    macd = MACDPlatinum()
    result = macd.calculate(df)
    assert "macd_line" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns
    assert result["macd_cross_up"].any(), "Expected at least one MACD cross up"
    assert result["macd_cross_down"].any(), "Expected at least one MACD cross down"
    print(f"  MACD Platinum: {result['macd_cross_up'].sum()} crosses up, {result['macd_cross_down'].sum()} crosses down")


def test_qqe():
    df = make_synthetic_candles()
    qqe = QQEADV()
    result = qqe.calculate(df)
    assert "qqe_rsi_ma" in result.columns
    assert "qqe_second_line" in result.columns
    assert "qqe_trend" in result.columns
    valid = result.dropna(subset=["qqe_rsi_ma"])
    assert len(valid) > 0
    bull = result["qqe_bullish"].sum()
    bear = result["qqe_bearish"].sum()
    print(f"  QQE ADV: {bull} bullish bars, {bear} bearish bars")


def test_qmp():
    df = make_synthetic_candles()
    qmp = QMPFilter()
    result = qmp.calculate(df)
    buys = result["buy_signal"].sum()
    sells = result["sell_signal"].sum()
    assert "buy_signal" in result.columns
    assert "sell_signal" in result.columns
    assert "bb_upper" in result.columns
    assert "ema_50" in result.columns
    print(f"  QMP Filter: {buys} buy signals, {sells} sell signals")


if __name__ == "__main__":
    print("Running indicator smoke tests...")
    test_macd()
    test_qqe()
    test_qmp()
    print("All tests passed!")
