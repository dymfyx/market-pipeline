import numpy as np
import pandas as pd

from market_pipeline.predict_returns import build_features_and_target


def test_target_is_future_return():
    """目标必须是"未来收益"（shift(-target_window)），且动量/波动率数值正确。

    数据：6 天、1 只股票、价格每天 +1：1,2,3,4,5,6
    窗口：momentum_window=2, target_window=1

    有效样本 = 第 2、3、4 天：
    - 第 0、1 天：动量（shift(2)）无值 → 删
    - 第 5 天：目标（未来 1 天）无值 → 删
    """
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)

    X, y = build_features_and_target(close, momentum_window=2, target_window=1)

    # 边界：剩 3 行有效样本
    assert X.shape == (3, 2)
    assert y.shape == (3,)

    # 目标 = 未来 1 天收益：4/3-1, 5/4-1, 6/5-1
    np.testing.assert_allclose(y, [1 / 3, 1 / 4, 1 / 5])

    # 动量 = close/close.shift(2) - 1：3/1-1, 4/2-1, 5/3-1
    np.testing.assert_allclose(X[:, 0], [2.0, 1.0, 2 / 3])

    # 波动率 = pct_change().rolling(2).std()（pandas 默认 ddof=1）
    # 前 3 天涨跌幅: [NaN, 1.0, 0.5, 1/3, 0.25, 0.2]
    # 窗口 [1.0,0.5] -> 0.353553；[0.5,1/3] -> 0.117851；[1/3,0.25] -> 0.058926
    np.testing.assert_allclose(X[:, 1], [0.353553, 0.117851, 0.058926], atol=1e-5)


def test_drops_unpredictable_rows():
    """dropna 边界：前 momentum_window 天无动量、后 target_window 天无目标，都应被删。"""
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = pd.DataFrame({"A": np.arange(30, dtype=float) + 1}, index=idx)

    X, y = build_features_and_target(close)  # 默认 20/5

    # 有效样本 = 30 - 20（前动量）- 5（后目标）= 5 行
    assert X.shape == (5, 2)
    assert y.shape == (5,)
