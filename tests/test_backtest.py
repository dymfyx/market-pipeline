import numpy as np
import pandas as pd
from market_pipeline.backtest.backtest import run_backtest, compute_signal

def test_signal_holds_when_uptrend():
    # 一直涨的价格：短均线必然在长均线之上 → 信号应该全 1
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    signal = compute_signal(close, short_window=2, long_window=4)
    assert (signal.iloc[3:] == 1).all(), "上涨趋势下信号应全为 1"

def test_signal_empty_when_downtrend():
    close = pd.Series([8, 7, 6, 5, 4, 3, 2, 1])
    signal = compute_signal(close, short_window=2, long_window=4)
    assert (signal.iloc[3:] == 0).all(), "下跌趋势下信号应全为 0"

def test_position_shifted_one_day():
    close = pd.Series([6, 5, 4, 3, 2, 1, 5, 10])
    position = run_backtest(close, short_window=2, long_window=4)['position']
    assert position.iloc[6] == 0, "仓位应延迟一天生效，防止未来函数"
    assert position.iloc[7] == 1, "仓位应延迟一天生效，防止未来函数"

def test_cost_reduces_returns():
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    result_no_cost = run_backtest(close, short_window=2, long_window=4, cost=0.0)
    result_with_cost = run_backtest(close, short_window=2, long_window=4, cost=0.01)
    assert result_with_cost['cumulative_strategy'].iloc[-1] < result_no_cost['cumulative_strategy'].iloc[-1], "手续费应降低策略收益"