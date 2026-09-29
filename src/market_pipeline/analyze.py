import pandas as pd
import matplotlib.pyplot as plt

# 解决画图时中文显示乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 读取数据
df = pd.read_parquet("data/raw/000001.parquet")

# 2. 计算日收益率 (今天收盘价 / 昨天收盘价 - 1)
df['returns'] = df['close'].pct_change()

# 3. 打印统计指标
print("=== 收益率统计 ===")
print(f"日均收益率: {df['returns'].mean():.4%}")
print(f"日收益率标准差(波动率): {df['returns'].std():.4%}")
print(f"最大单日涨幅: {df['returns'].max():.4%}")
print(f"最大单日跌幅: {df['returns'].min():.4%}")

# 4. 画图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 上图：收盘价走势
ax1.plot(df['date'], df['close'], color='blue')
ax1.set_title("000001 收盘价走势")
ax1.set_ylabel("价格")

# 下图：日收益率波动
ax2.plot(df['date'], df['returns'], color='red', alpha=0.5)
ax2.set_title("000001 日收益率")
ax2.set_ylabel("收益率")

plt.tight_layout() # 自动调整布局，防止重叠
plt.show()         # 弹出窗口显示图表