import os
import pandas as pd

def merge_all_data(data_dir="data/raw", output_file="data/merged_close.parquet"):
    """
    读取 raw 文件夹下的所有 parquet 文件，提取收盘价，拼成一张大表
    """
    # 1. 找出所有 parquet 文件
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    print(f"找到 {len(files)} 个数据文件，开始合并...")

    # 2. 用字典暂存每只股票的收盘价序列
    data_dict = {}
    for file in files:
        stock_code = file.replace(".parquet", "") # 去掉后缀当作股票代码
        file_path = os.path.join(data_dir, file)
        
        # 读取数据
        df = pd.read_parquet(file_path)
        
        # 把 date 设为索引，方便后续对齐
        df = df.set_index('date')
        
        # 只取 close 列，并以股票代码命名
        data_dict[stock_code] = df['close']
    
    # 3. 将字典转为 DataFrame
    # pandas 会自动按日期对齐，缺失的日期会自动填上 NaN
    merged_df = pd.DataFrame(data_dict)
    
    # 4. 处理缺失值（非常重要！）
    # 因为有些股票上市晚，或者停牌，会有 NaN
    # 我们使用“向前填充”（用昨天的价格填充今天的空值），防止报错
    merged_df = merged_df.ffill()
    
    # 5. 删除仍然有缺失值的行（比如最早期的数据）
    merged_df = merged_df.dropna()
    
    # 6. 打印大表信息
    print("\n=== 合并后的大表信息 ===")
    print(f"行数(交易日): {merged_df.shape[0]}")
    print(f"列数(股票数): {merged_df.shape[1]}")
    print("\n前5行数据预览：")
    print(merged_df.head())
    
    # 7. 保存合并后的大表
    merged_df.to_parquet(output_file)
    print(f"\n✅ 合并完成，已保存到 {output_file}")

    # 8. 返回合并结果，供调用方继续使用（如后续因子计算、回测）
    return merged_df

if __name__ == "__main__":
    merge_all_data()