import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 读取合并后的大表 =================
df = pd.read_parquet("data/merged_close.parquet")

# ================= 2. 计算每日收益率 =================
# 计算所有股票每天的涨跌幅
returns = df.pct_change()

# ================= 3. 计算动量（20日收益率） =================
# 动量 = 今天的价格 / 20天前的价格 - 1
momentum = df / df.shift(20) - 1

# ================= 4. 设定调仓周期：每5个交易日（一周）调仓一次 =================
# 找出每隔5天的调仓日索引
rebalance_days = df.index[::5] 
print(f"总交易日: {len(df)} 天，共调仓 {len(rebalance_days)} 次")

# ================= 5. 循环每个调仓日，挑选排名前10的股票 =================
# 为了简化，我们先假设每次买入前10名，等权重分配（每只股票10%仓位）
positions = pd.DataFrame(0.0, index=df.index, columns=df.columns) # 持仓矩阵

for i, day in enumerate(rebalance_days):
    if day not in momentum.index:
        continue
    
    # 获取该调仓日所有股票的动量排名
    scores = momentum.loc[day].dropna()
    if len(scores) < 10:
        continue
        
    # 挑出动量最高的前 10 只股票
    top_10 = scores.nlargest(10).index
    
    # 在下一次调仓日之前，一直持有这10只股票
    if i + 1 < len(rebalance_days):
        next_day = rebalance_days[i+1]
        positions.loc[day:next_day, top_10] = 0.1  # 每只股票分配10%的仓位

# 关键：仓位必须向后 shift(1) 天，防止未来函数（今天收盘选股，明天开盘才能买）
positions = positions.shift(1).fillna(0)

# ================= 6. 计算策略每日收益（带手续费和滑点） =================
# 计算换手率：仓位变化绝对值之和的一半（因为买卖对冲）
turnover = positions.diff().abs().sum(axis=1) / 2

# 设定单边手续费+滑点 = 万分之十五 (0.0015)
cost = 0.0015 

# 策略收益 = 持仓收益 - 换手率 * 交易成本
strategy_returns = (positions * returns).sum(axis=1) - turnover * cost

# 计算累计净值
cumulative_strategy = (1 + strategy_returns).cumprod()
benchmark_returns = returns.mean(axis=1)
cumulative_benchmark = (1 + benchmark_returns).cumprod()

# ================= 7. 打印结果 =================
print("\n=== 动量轮动回测结果 ===")
print(f"策略最终净值: {cumulative_strategy.iloc[-1]:.4f}")
print(f"等权基准最终净值: {cumulative_benchmark.iloc[-1]:.4f}")

# ================= 8. 画图 =================
plt.figure(figsize=(12, 6))
plt.plot(cumulative_strategy, label='20日动量轮动策略(前10)', color='red')
plt.plot(cumulative_benchmark, label='等权持有全部股票基准', color='gray', alpha=0.7)
plt.title("多股票动量轮动策略 vs 基准")
plt.xlabel("日期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(True)
plt.show()