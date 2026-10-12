"""
多因子轮动回测：动量 + 低波动
结构：纯函数（可测试）+ main（读文件 / 打印 / 画图）
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def compute_factors(close, w_momentum=0.2, w_volatility=0.8):
    """计算动量 + 低波动双因子综合得分。

    参数
    ----
    close : pd.DataFrame
        收盘价宽表，行 = 日期，列 = 股票代码。
    w_momentum / w_volatility : float
        动量 / 低波动因子权重（相加应为 1）。

    返回
    ----
    pd.DataFrame
        综合得分表，index/columns 与 close 一致，值为每日横截面百分位的加权和。
    """
    returns = close.pct_change()

    # 因子1：20日动量（越大越好）
    momentum = close / close.shift(20) - 1

    # 因子2：20日波动率（越小越好，取负后排名时波动小的排前面）
    volatility = returns.rolling(window=20).std()
    volatility_score = -volatility

    # 横截面百分位排名，0~1
    momentum_rank = momentum.rank(axis=1, pct=True)
    volatility_rank = volatility_score.rank(axis=1, pct=True)

    # 加权合成（真实权重：20% 动量 + 80% 低波动，对应你的实验记录三）
    return w_momentum * momentum_rank + w_volatility * volatility_rank
 

def select_positions(combined_score, rebalance_every=20, top_n=10):
    """按调仓周期轮动选股，输出延迟一天生效的仓位表。

    参数
    ----
    combined_score : pd.DataFrame
        compute_factors 的输出。
    rebalance_every : int
        每多少个交易日调仓一次。
    top_n : int
        每次持有股票数（等权，每只 1/top_n）。

    返回
    ----
    pd.DataFrame
        仓位表，已 shift(1) 防未来函数。
    """
    positions = pd.DataFrame(0.0, index=combined_score.index,
                             columns=combined_score.columns)
    rebalance_days = combined_score.index[::rebalance_every]

    for i, day in enumerate(rebalance_days):
        scores = combined_score.loc[day].dropna()
        if len(scores) < top_n:
            continue
        top = scores.nlargest(top_n).index
        if i + 1 < len(rebalance_days):
            next_day = rebalance_days[i + 1]
            positions.loc[day:next_day, top] = 1 / top_n

    # 关键：防止未来函数，仓位延迟一天
    return positions.shift(1).fillna(0)

def run_backtest(close, w_momentum=0.2, w_volatility=0.8,
                 rebalance_every=20, top_n=10, cost=0.0015):
    """跑一轮双因子轮动回测，返回策略与基准的每日收益、净值。

    参数
    ----
    close : pd.DataFrame
        收盘价宽表。
    cost : float
        单边手续费+滑点（默认万分之十五）。

    返回
    ----
    dict
        含 strategy_returns / benchmark_returns / 两个净值序列 / turnover。
    """
    returns = close.pct_change()
    combined_score = compute_factors(close, w_momentum, w_volatility)
    positions = select_positions(combined_score, rebalance_every, top_n)

    turnover = positions.diff().abs().sum(axis=1) / 2
    strategy_returns = (positions * returns).sum(axis=1) - turnover * cost
    benchmark_returns = returns.mean(axis=1)

    return {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'strategy_net_value': (1 + strategy_returns).cumprod(),
        'benchmark_net_value': (1 + benchmark_returns).cumprod(),
        'turnover': turnover,
    }

def main():
    df = pd.read_parquet("data/merged_close.parquet")
    result = run_backtest(df)

    print(f"总交易日: {len(df)} 天，共调仓 {len(df.index[::20])} 次")
    print("\n=== 双因子轮动回测结果 ===")
    print(f"策略最终净值: {result['strategy_net_value'].iloc[-1]:.4f}")
    print(f"等权基准最终净值: {result['benchmark_net_value'].iloc[-1]:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(result['strategy_net_value'], label='双因子轮动(动量+低波动)', color='purple')
    plt.plot(result['benchmark_net_value'], label='等权基准', color='gray', alpha=0.7)
    plt.title("双因子轮动策略 vs 基准")
    plt.xlabel("日期")
    plt.ylabel("累计净值")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

# ====== 实验记录（保留） ======
# 2026-09-28：0.5动量 + 0.5低波动，5天调仓，净值 1.2448，跑输基准。
# 结论：A股低波动因子与动量因子冲突，且高频调仓成本过高。
#
# 实验记录二：权重 80%动量 + 20%低波动
# 5天调仓: 1.2719（被手续费严重拖累）| 10天: 1.2175 | 20天: 1.6170
#
# 实验记录三：权重 20%动量 + 80%低波动
# 5天调仓: 1.2793 | 10天: 1.3333 | 20天: 1.3859