import akshare as ak
import pandas as pd
import time

def fetch_stock_data(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data for a given stock code and date range.

    Parameters:
    - stock_code (str): The stock code to fetch data for.
    - start_date (str): The start date in 'YYYY-MM-DD' format.
    - end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: A DataFrame containing the historical stock data.
    """
    # Fetch the historical stock data using akshare
    print(f"正在抓取{stock_code} ...")
    stock_data = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="qfq")
    
    return stock_data

if __name__ == "__main__":
    codes = ["000001", "000002", "600000", "600036", "601318"]

    start = time.perf_counter()
    for code in codes:
        try:
            df = fetch_stock_data(code, "20240101", "20260921")
            print(f"{code}: {len(df)} 行")
        except Exception as e:
            print(f"{code} 失败: {e}")
    elapsed = time.perf_counter() - start
    print(f"\n同步抓 {len(codes)} 只股票，耗时 {elapsed:.2f} 秒")