import numpy as np
import pandas as pd

from market_pipeline.linear_regression import fit_ols


def build_features_and_target(close, momentum_window=20, target_window=5):
    """从收盘价宽表构造 特征 X 和 目标 y（横截面堆叠成一行一条样本）。

    参数
    ----
    close : pd.DataFrame
        收盘价宽表（行=日期，列=股票代码），来自 merge_data。
    momentum_window : int
        动量/波动率的计算窗口（默认 20 日）。
    target_window : int
        预测未来几天的收益（默认 5 日）。

    返回
    ----
    X : np.ndarray, shape (n_samples, 2)
        两列分别是 动量、波动率。
    y : np.ndarray, shape (n_samples,)
        未来 target_window 日收益。
    """
    # 动量因子：20 日收益率
    momentum = close / close.shift(momentum_window) - 1

    # 波动率因子：20 日收益率的滚动标准差
    volatility = close.pct_change().rolling(momentum_window).std()

    # 目标：未来 5 天收益。shift(-5) 把"5 天后的价格"拉到今天这一行
    target = close.shift(-target_window) / close - 1

    # 把三张宽表"摊平"成一行一条样本（(日期, 股票) 组合）
    # 各表分别 stack（只保留各自有数据的格子），pandas 会用索引并集自动对齐，
    # 任一方缺失的位置变成 NaN，最后统一 dropna 删掉
    data = pd.DataFrame({
        "momentum": momentum.stack(),
        "volatility": volatility.stack(),
        "target": target.stack(),
    })

    # 删掉缺数据的行：前 20 天没有动量、最后 5 天没有未来收益，都会在这里被删掉
    data = data.dropna()

    X = data[["momentum", "volatility"]].to_numpy()
    y = data["target"].to_numpy()
    return X, y


def main():
    # ===== I/O 层：读数据、打印，都只在 main 里 =====
    df = pd.read_parquet("data/merged_close.parquet")

    X, y = build_features_and_target(df)
    print(f"样本数: {X.shape[0]}（≈ 交易日 × 股票数，去掉缺数据后）")
    print(f"特征: 动量、波动率 | 目标: 未来5日收益")

    # 用闭式解学出 w = [截距, 动量系数, 波动率系数]
    w = fit_ols(X, y)
    print("\n=== 学到的系数 ===")
    print(f"截距       : {w[0]:.6f}")
    print(f"动量系数   : {w[1]:.6f}  （正 = 涨得多的股票未来收益更高）")
    print(f"波动率系数 : {w[2]:.6f}  （负 = 波动小的股票未来收益更高）")

    # 预测质量：预测值和真实值的相关系数
    pred = X @ w[1:] + w[0]
    corr = np.corrcoef(pred, y)[0, 1]
    print(f"\n预测-真实相关系数: {corr:.4f}")
    print("（A股横截面收益预测的相关性通常很低，0.01~0.1 已经算有信号）")


if __name__ == "__main__":
    main()
