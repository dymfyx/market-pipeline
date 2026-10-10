import numpy as np
import pandas as pd
from market_pipeline.multi_factor_backtest import compute_factors, select_positions, run_backtest

def _make_close(n=30):
    """确定性数据集：A 稳步涨、B 稳步跌、C 剧烈震荡。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    a = 10 * (1.01 ** np.arange(n))
    b = 30 * (0.99 ** np.arange(n))
    c = 20 + 10 * np.sin(np.arange(n))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)

def test_positions_are_shifted_one_day():
    """今天的仓位必须来自昨天的信号——防未来函数的核心断言。"""
    close = _make_close()
    # 每天调仓(top_n=1)是最严苛的测试：信号每天都变，最容易暴露未来函数
    positions = select_positions(compute_factors(close), rebalance_every=1, top_n=1)

    # 涨得最稳的 A 在第 21 天之后（动量窗口满了）应该几乎每天都是得分第一
    # 反转检验：把仓位反向 shift 回去，应该和"当天的选择"对得上
    # 断言方式：第 t 天的仓位，只能由 <= t-1 天的得分决定
    scores = compute_factors(close)
    for t in range(21, 28):  # 取动量窗口已满的几天
        yesterday_best = scores.iloc[t - 1].dropna().idxmax()
        assert positions.iloc[t][yesterday_best] == 1.0, \
            f"第{t}天应持有昨天得分最高的{yesterday_best}"

def test_weight_changes_score():
    """测试不同权重下的综合得分变化。"""
    close = _make_close()

    excepted_momentum_rank = (close / close.shift(20) - 1).rank(axis=1, pct=True)
    excepted_volatility_rank = ((-close.pct_change().rolling(window=20).std())).rank(axis=1, pct=True)

    score1 = compute_factors(close, w_momentum=1.0, w_volatility=0.0)
    score2 = compute_factors(close, w_momentum=0.0, w_volatility=1.0)

    # 断言：两种权重下的综合得分不完全相同
    pd.testing.assert_frame_equal(score1, excepted_momentum_rank)
    pd.testing.assert_frame_equal(score2, excepted_volatility_rank)