import numpy as np
import pandas as pd
from market_pipeline.rotation_backtest import run_backtest, select_positions, filter_limit_up

def test_filter_limit_up_excludes_limit_up_stock():
    # A 当天 +10%（涨停），B、C 正常
    day_returns = pd.Series({'A': 0.10, 'B': 0.02, 'C': 0.01})
    selected = filter_limit_up(['A', 'B', 'C'], day_returns, top_n=2)
    assert 'A' not in selected      # 涨停的 A 被剔除
    assert selected == ['B', 'C']   # 剩下的按候选顺序选满

def test_filter_limit_up_skips_missing_stock():
    # 候选里有只股票当天没数据（停牌）→ 跳过它
    day_returns = pd.Series({'B': 0.02})
    selected = filter_limit_up(['A', 'B'], day_returns, top_n=1)
    assert selected == ['B']   # A 无数据被跳过，B 顶上

def test_filter_limit_up_stops_at_top_n():
    day_returns = pd.Series({'A': 0.01, 'B': 0.02, 'C': 0.03})
    selected = filter_limit_up(['A', 'B', 'C'], day_returns, top_n=2)
    assert len(selected) == 2      # 选满 2 只就停，不贪多

def test_positions_are_shifted_one_day():
    """今天的仓位必须来自昨天的信号——防未来函数的核心断言。"""
    # 构造动量数据：A 稳步涨、B 稳步跌、C 剧烈震荡
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    #freq = "D" 表示按天生成日期索引
    momentum = pd.DataFrame({
        "A": 10 * (1.01 ** np.arange(30)),
        "B": 30 * (0.99 ** np.arange(30)),
        "C": 20 + 10 * np.sin(np.arange(30))
    }, index=idx)

    # 每天调仓(top_n=1)是最严苛的测试：信号每天都变，最容易暴露未来函数
    returns = momentum.pct_change()
    positions = select_positions(momentum, returns, rebalance_every=1, top_n=1, limit_up_threshold=1.0)  # 涨停阈值设大，保证不会过滤掉任何股票

    for t in range(21, 28):  # 取动量窗口已满的几天
        yesterday_best = momentum.iloc[t - 1].dropna().idxmax()
        assert positions.iloc[t][yesterday_best] == 1.0, \
            f"第{t}天应持有昨天得分最高的{yesterday_best}"

