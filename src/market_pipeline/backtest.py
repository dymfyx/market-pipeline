import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_parquet("data/raw/000001.parquet")

# 2. 计算收益率
df['returns'] = df['close'].pct_change()

# 3. 计算均线 (短期5日，长期20日)
df['ma5'] = df['close'].rolling(window=3).mean()
df['ma20'] = df['close'].rolling(window=10).mean()

# 4. 生成交易信号
# 如果 ma5 > ma20，信号为 1（持有），否则为 0（空仓）
df['signal'] = 0
df.loc[df['ma5'] > df['ma20'], 'signal'] = 1

# 5. 关键一步：把信号向后移一天！
# 因为今天收盘后才知道今天的均线，你的交易要在明天才能生效，否则就是“未来函数”
df['position'] = df['signal'].shift(1)

# 6. 计算策略每天的实际收益
# 加上万分之三的单边手续费（买卖各收一次），也就是千分之六
df['trade'] = df['position'].diff().abs() # 换仓时=1，否则=0
fee = 0.0006 # 手续费率
df['strategy_returns'] = df['position'] * df['returns'] - df['trade'] * fee

df['cumulative_market'] = (1 + df['returns']).cumprod()
df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()

# 7. 计算累计净值 (假设初始净值是1)
df['cumulative_market'] = (1 + df['returns']).cumprod() # 买入并持有
df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod() # 策略

# 8. 打印回测结果
print("=== 回测结果 ===")
print(f"买入并持有最终净值: {df['cumulative_market'].iloc[-1]:.4f}")
print(f"双均线策略最终净值: {df['cumulative_strategy'].iloc[-1]:.4f}")

# 9. 画净值对比图
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['cumulative_market'], label='买入并持有', color='blue')
plt.plot(df['date'], df['cumulative_strategy'], label='双均线策略', color='orange')
plt.title("000001 净值走势对比")
plt.xlabel("日期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(True)
plt.show()