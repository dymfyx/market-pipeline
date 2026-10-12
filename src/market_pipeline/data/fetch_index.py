import akshare as ak
import pandas as pd
import os

def fetch_benchmark():
    """
    抓取沪深300指数（000300）作为策略基准，保存为 parquet 文件
    """
    print("正在抓取沪深300指数数据...")
    
    # 1. 抓取数据（akshare 的指数接口）
    # 注意：这里用 sh000300 代表上证指数系列下的沪深300
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
    except Exception as e:
        print(f"抓取失败，尝试备用接口: {e}")
        # 备用接口（东财）
        df = ak.index_zh_a_hist(symbol="000300", period="daily", start_date="20240101", end_date="20260920")
        # 东财接口列名是中文，需要重命名
        df = df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'})

    # 2. 数据清洗与格式化
    # 确保日期列是 datetime 类型
    df['date'] = pd.to_datetime(df['date'])
    # 按日期升序排列
    df = df.sort_values('date').reset_index(drop=True)
    # 只保留和策略回测需要对齐的列
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

    # 3. 检查数据范围（和你的股票数据对齐）
    print(f"数据起止日期：{df['date'].min()} 至 {df['date'].max()}")
    print(f"共计 {len(df)} 个交易日")

    # 4. 创建目录并保存
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "benchmark.parquet")
    
    df.to_parquet(file_path, index=False)
    print(f"✅ 沪深300基准数据已保存到 {file_path}")

if __name__ == "__main__":
    fetch_benchmark()