import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 读取合并后的大表 =================
df = pd.read_parquet("data/merged_close.parquet")
returns = df.pct_change()

# ================= 2. 计算因子 (Factor) =================
# 因子1：20日动量 (越大越好)
momentum = df / df.shift(20) - 1

# 因子2：20日波动率 (越小越好，所以我们的因子值用负号，负得越多说明波动越小)
volatility = returns.rolling(window=20).std()
#.std() 计算标准差，20日滚动窗口
volatility_score = -volatility  # 取负值，这样排名时，波动小的反而排在前面

# ================= 3. 因子标准化 (百分位排名) =================
# 每天的因子值在横截面上排名，转换为0到1之间的百分比
momentum_rank = momentum.rank(axis=1, pct=True)
#rank(axis=1, pct=True) 按行排名，pct=True 表示返回百分位数
volatility_rank = volatility_score.rank(axis=1, pct=True)

# ================= 4. 合成综合得分 =================
# 各占50%权重
combined_score = 0.2 * momentum_rank + 0.8 * volatility_rank

# ================= 5. 设定调仓周期 =================
rebalance_days = df.index[::20]  # 每20个交易日调仓一次
print(f"总交易日: {len(df)} 天，共调仓 {len(rebalance_days)} 次")

# ================= 6. 轮动选股与计算仓位 =================
positions = pd.DataFrame(0.0, index=df.index, columns=df.columns)

for i, day in enumerate(rebalance_days):
    if day not in combined_score.index:
        continue
    
    scores = combined_score.loc[day].dropna()
    if len(scores) < 10:
        continue
        
    # 挑出综合得分最高的前 10 只股票（既动量高，又波动低）
    top_10 = scores.nlargest(10).index
    
    if i + 1 < len(rebalance_days):
        next_day = rebalance_days[i+1]
        positions.loc[day:next_day, top_10] = 0.1

# 关键：防止未来函数，仓位延迟一天
positions = positions.shift(1).fillna(0)

# ================= 7. 计算真实收益（带手续费和滑点！） =================
turnover = positions.diff().abs().sum(axis=1) / 2
cost = 0.0015  # 单边手续费+滑点万分之十五

strategy_returns = (positions * returns).sum(axis=1) - turnover * cost
#sum(axis=1) 按行求和，得到每天的策略收益，减去换手率*交易成本

cumulative_strategy = (1 + strategy_returns).cumprod()

# 基准：等权持有全部股票（也要算基准的手续费，但调仓为0，所以无手续费）
benchmark_returns = returns.mean(axis=1)
cumulative_benchmark = (1 + benchmark_returns).cumprod()

# ================= 8. 打印结果 =================
print("\n=== 双因子轮动回测结果 ===")
print(f"策略最终净值: {cumulative_strategy.iloc[-1]:.4f}")
print(f"等权基准最终净值: {cumulative_benchmark.iloc[-1]:.4f}")

# ================= 9. 画图 =================
plt.figure(figsize=(12, 6))
plt.plot(cumulative_strategy, label='双因子轮动(动量+低波动)', color='purple')
plt.plot(cumulative_benchmark, label='等权基准', color='gray', alpha=0.7)
plt.title("双因子轮动策略 vs 基准")
plt.xlabel("日期")
plt.ylabel("累计净值")
plt.legend()
plt.grid(True)
plt.show()




# 实验记录：2026-09-28
# 0.5动量 + 0.5低波动，5天调仓，净值为1.2448，跑输基准。
# 结论：A股低波动因子与动量因子冲突，且高频调仓成本过高。


# 实验记录二：
# 权重 80%动量 + 20%低波动
# 5天调仓: 净值 1.2719（被手续费严重拖累）
# 10天调仓: 净值 1.2175
# 20天调仓: 净值 1.6170

# 实验记录三：
# 权重 20%动量 + 80%低波动
# 5天调仓: 净值 1.2793（被手续费严重拖累）
# 10天调仓: 净值 1.3333
# 20天调仓: 净值 1.3859