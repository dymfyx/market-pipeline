import pandas as pd

file_path = r"C:\Users\dym54\Desktop\market-pipeline\data\merged_close.parquet"
df = pd.read_parquet(file_path)

print("===前五行数据===")
print(df.head())

print("===数据基本信息===")
print(df.info())

print("\n===日期范围===")
print(f"最早日期：{df.index.min()}")
print(f"最晚日期：{df.index.max()}")