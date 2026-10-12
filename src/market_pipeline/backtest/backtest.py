import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def compute_signal(close, short_window, long_window):
    """计算双均线交易信号。

    参数
    ----
    close : pd.Series
        收盘价序列。
    short_window : int
        短期均线窗口大小。
    long_window : int
        长期均线窗口大小。

    返回
    ----
    pd.Series
        交易信号序列，1 表示持有，0 表示空仓。
    """
    ma_short = close.rolling(window=short_window).mean()
    #mean() 计算均值，rolling(window=short_window) 计算滚动窗口的均值，window=short_window 表示取前 short_window 天的均值
    ma_long = close.rolling(window=long_window).mean()
    signal = (ma_short > ma_long).astype(int)
    #astype(int) 将布尔值转换为整数，True 转为 1，False 转为 0
    return signal

def run_backtest(close, short_window=5, long_window=20, cost=0.0006):
    """运行双均线策略回测。

    参数
    ----
    close : pd.Series
        收盘价序列。
    short_window : int
        短期均线窗口大小。
    long_window : int
        长期均线窗口大小。
    cost : float
        单边手续费率（默认万分之六）。

    返回
    ----
    pd.DataFrame
        包含每日收益、累计净值等信息的 DataFrame。
    """
    df = pd.DataFrame(index=close.index)
    df['date'] = close.index
    df['close'] = close
    df['returns'] = close.pct_change()
    df['signal'] = compute_signal(close, short_window, long_window)
    df['position'] = df['signal'].shift(1).fillna(0)  # 防止未来函数
    df['trade'] = df['position'].diff().abs()
    df['strategy_returns'] = df['position'] * df['returns'] - df['trade'] * cost
    df['cumulative_market'] = (1 + df['returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    return df

def main():
    df = pd.read_parquet("data/raw/000001.parquet")
    result = run_backtest(df['close'])

    print("=== 回测结果 ===")
    print(f"买入并持有最终净值: {result['cumulative_market'].iloc[-1]:.4f}")
    print(f"双均线策略最终净值: {result['cumulative_strategy'].iloc[-1]:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(result['date'], result['cumulative_market'], label='买入并持有', color='blue')
    plt.plot(result['date'], result['cumulative_strategy'], label='双均线策略', color='orange')
    plt.title("000001 净值走势对比")
    plt.xlabel("日期")
    plt.ylabel("累计净值")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()




