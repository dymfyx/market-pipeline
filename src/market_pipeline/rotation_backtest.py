"""
动量轮动回测：20日动量选股 + 涨停过滤 + 沪深300基准对比

结构：纯函数（可测试）+ main（读文件 / 打印 / 画图）
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def compute_momentum(close, window=20):
    """计算动量因子：N日收益率（默认20日）。

    参数
    ----
    close : pd.DataFrame
        收盘价宽表，行 = 日期，列 = 股票代码。
    window : int
        动量窗口天数。

    返回
    ----
    pd.DataFrame
        动量因子表，index/columns 与 close 一致。
    """
    return close / close.shift(window) - 1


def filter_limit_up(candidates, day_returns, threshold=0.095, top_n=10):
    """从候选股票中剔除涨停股，选满 top_n 只。

    规则：只有"当天有数据且涨幅 < 阈值"的股票才可买入；
    涨停（涨幅 >= 阈值）或当天无数据（停牌）都买不进。

    参数
    ----
    candidates : Index / list
        按动量排名靠前的候选股票代码。
    day_returns : pd.Series
        调仓日当天所有股票的涨跌幅。
    threshold : float
        涨停判断阈值（默认 9.5%）。
    top_n : int
        需要选出的股票数量。

    返回
    ----
    list
        最终可买入的股票代码列表。
    """
    selected = []
    for code in candidates:
        # 有数据且没涨停 → 可买入；否则（涨停 / 停牌无数据）跳过
        if code in day_returns.index and day_returns[code] < threshold:
            selected.append(code)
        if len(selected) == top_n:  # 选满 top_n 只就停
            break
    return selected


def select_positions(momentum, returns, rebalance_every=5, top_n=10,
                     candidate_pool=20, limit_up_threshold=0.095):
    """按调仓周期轮动选股（含涨停过滤），输出延迟一天生效的仓位表。

    参数
    ----
    momentum : pd.DataFrame
        compute_momentum 的输出。
    returns : pd.DataFrame
        每日收益率宽表。
    rebalance_every : int
        每多少个交易日调仓一次。
    top_n : int
        每次持有股票数量（等权，每只 1/top_n）。
    candidate_pool : int
        候选池大小：先取动量前 candidate_pool 名，再过滤涨停
        （留出备选，防止前 top_n 里有人涨停买不进）。
    limit_up_threshold : float
        涨停判断阈值。

    返回
    ----
    pd.DataFrame
        仓位表，已 shift(1) 防未来函数。
    """
    positions = pd.DataFrame(0.0, index=momentum.index, columns=momentum.columns)
    # 每隔 rebalance_every 天取一个调仓日
    rebalance_days = momentum.index[::rebalance_every]

    for i, day in enumerate(rebalance_days):
        if day not in momentum.index:
            continue
        scores = momentum.loc[day].dropna()
        if len(scores) < top_n:  # 当天有效股票太少，跳过这次调仓
            continue
        # 先取动量前 candidate_pool 名作为备选池
        top_candidates = scores.nlargest(candidate_pool).index
        day_returns = returns.loc[day]
        # 过滤涨停，选出最终可买入的 top_n 只
        selected = filter_limit_up(top_candidates, day_returns,
                                   threshold=limit_up_threshold, top_n=top_n)

        if i + 1 < len(rebalance_days):
            next_day = rebalance_days[i + 1]
            positions.loc[day:next_day, selected] = 1 / top_n  # 等权

    # 关键：防止未来函数，仓位延迟一天（今天收盘选股，明天开盘才能买）
    return positions.shift(1).fillna(0)


def run_backtest(close, rebalance_every=5, top_n=10, cost=0.0015):
    """跑一轮动量轮动回测，返回策略与等权基准的收益、净值。

    参数
    ----
    close : pd.DataFrame
        收盘价宽表。
    rebalance_every : int
        调仓间隔（交易日）。
    top_n : int
        持有股票数。
    cost : float
        单边手续费+滑点（默认万分之十五）。

    返回
    ----
    dict
        含 strategy_returns / benchmark_returns / 两个净值序列 / turnover。
    """
    returns = close.pct_change()
    momentum = compute_momentum(close)
    positions = select_positions(momentum, returns,
                                 rebalance_every=rebalance_every, top_n=top_n)

    # 换手率：仓位变化绝对值之和的一半（买卖各算一次）
    turnover = positions.diff().abs().sum(axis=1) / 2
    # 策略收益 = 持仓收益 - 换手率 * 交易成本
    strategy_returns = (positions * returns).sum(axis=1) - turnover * cost
    # 等权基准：所有股票等权持有（不调仓，无手续费）
    benchmark_returns = returns.mean(axis=1)

    return {
        'strategy_returns': strategy_returns,
        'benchmark_returns': benchmark_returns,
        'strategy_net_value': (1 + strategy_returns).cumprod(),
        'benchmark_net_value': (1 + benchmark_returns).cumprod(),
        'turnover': turnover,
    }


def main():
    # ===== I/O 层：读文件、打印、画图都只在这里 =====
    df = pd.read_parquet("data/merged_close.parquet")
    result = run_backtest(df, rebalance_every=5, top_n=10)

    # 沪深300基准（外部数据文件，属于 I/O，不进纯函数）
    bench = pd.read_parquet("data/benchmark.parquet")
    bench = bench.set_index('date')
    bench = bench.loc[df.index.min():df.index.max()]  # 对齐到策略日期范围
    bench_returns = bench['close'].pct_change()
    cumulative_benchmark_300 = (1 + bench_returns).cumprod()
    cumulative_benchmark_300.iloc[0] = 1.0  # 第一天是 NaN，置为 1.0

    cumulative_strategy = result['strategy_net_value']
    cumulative_benchmark = result['benchmark_net_value']

    print("\n=== 动量轮动回测结果 ===")
    print(f"策略最终净值: {cumulative_strategy.iloc[-1]:.4f}")
    print(f"等权基准最终净值: {cumulative_benchmark.iloc[-1]:.4f}")
    print(f"沪深300基准最终净值: {cumulative_benchmark_300.iloc[-1]:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_strategy, label='20日动量轮动策略(前10)', color='red')
    plt.plot(cumulative_benchmark, label='等权持有全部股票基准', color='gray', alpha=0.7)
    plt.plot(cumulative_benchmark_300, label='沪深300基准', color='green', alpha=0.7)
    plt.title("多股票动量轮动策略 vs 基准")
    plt.xlabel("日期")
    plt.ylabel("累计净值")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
