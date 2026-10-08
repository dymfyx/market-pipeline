import os
import asyncio #python异步库
import sys #判断系统操作
import time
import random
import akshare as ak #爬数据
import pandas as pd #存表格
from concurrent.futures import ThreadPoolExecutor #线程池

# 关键修复：Windows 下强制使用 Selector 事件循环,以使用管道
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 限制同时最多 3 个并发请求，防止被封 IP
sem = asyncio.Semaphore(3)

# 显式创建一个受控的线程池，避免 Python 底层默认线程池在 Windows 上崩溃
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ak_fetch")

async def fetch_with_retry(stock_code: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
    """
    带重试、随机休眠的生产级抓取函数（显式线程池版）。
    """
    symbol = f"sz{stock_code}" if stock_code.startswith("0") else f"sh{stock_code}"
    loop = asyncio.get_running_loop()

    for attempt in range(1, max_retries + 1):
        try:
            async with sem:#控制同时进程数
                print(f"正在抓取 {stock_code} (第 {attempt} 次尝试) ...")

                # 使用我们自建的 executor，而不是 asyncio.to_thread
                df = await loop.run_in_executor(
                    executor,
                    lambda: ak.stock_zh_a_daily(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",#前复权，抹平历史价格断层
                    ),
                )
                await asyncio.sleep(random.uniform(0.5, 1.5))#随机休眠，防止被封 IP
                return df

        except Exception as e:
            print(f"{stock_code} 第 {attempt} 次尝试失败: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 * attempt)#指数退避


def validate_data(df, stock_code):
    """
    数据校验：检查数据是不是空的，有没有明显错误
    """
    if df is None or df.empty:#防御性写法，先看有没有这个数据，再看数据是否非空
        print(f"⚠️ {stock_code} 数据为空！")
        return False
    
    # 检查关键列是否存在
    required_cols = ['date', 'open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ {stock_code} 缺少列: {col}")
            return False
            
    # 检查最高价是否大于等于最低价（逻辑校验）
    if (df['high'] < df['low']).any():
        print(f"⚠️ {stock_code} 存在最高价低于最低价的异常数据！")
        return False
        
    # 检查日期是否重复
    if df['date'].duplicated().any():
        print(f"⚠️ {stock_code} 存在重复日期！")
        return False
        
    return True

def save_data(df, stock_code, output_dir="data/raw"):
    """
    保存数据为 parquet 文件
    """
    # 创建文件夹（如果不存在）存在则静默跳过
    os.makedirs(output_dir, exist_ok=True)
    
    # 把日期列转成标准的时间格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)#行号重排，不乱加回之前行号
    
    # 保存文件
    file_path = os.path.join(output_dir, f"{stock_code}.parquet")#拼接路径
    df.to_parquet(file_path, index=False)#去除行号
    print(f"💾 {stock_code} 已保存到 {file_path}")


async def main() -> None:
    codes = [
    "000001", "000002", "000063", "000100", "000157", "000333", "000338", "000425", "000538", "000568",
    "000596", "000625", "000651", "000725", "000768", "000776", "000858", "000876", "000895", "000938",
    "000977", "001979", "002001", "002007", "002027", "002049", "002050", "002129", "002142", "002179",
    "002230", "002236", "002241", "002271", "002304", "002311", "002352", "002371", "002415", "002459",
    "002460", "002466", "002475", "002493", "002594", "002601", "002607", "002714", "002736", "002812",
    "002821", "002841", "002916", "002938", "003816", "300014", "300015", "300033", "300059", "300122",
    "300124", "300142", "300274", "300316", "300347", "300408", "300413", "300433", "300450", "300498",
    "300628", "300661", "300750", "300759", "300760", "300782", "300896", "300919", "300999", "600000",
    "600009", "600016", "600028", "600030", "600031", "600036", "600048", "600050", "600089", "600104",
    "600111", "600115", "600150", "600176", "600196", "600276", "600309", "600346", "600406", "600438"
]

    start = time.perf_counter()

    tasks = [fetch_with_retry(code, "20240101", "20260920") for code in codes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start

    for code, res in zip(codes, results):
        if isinstance(res, Exception):
            print(f"❌ {code} 最终失败: {res}")
        else:
            if validate_data(res , code):
                save_data(res , code)
                print(f"✅ {code}: 成功处理{len(res)} 行")
            else:
                print(f"❌ {code} :数据校验不通过，跳过保存")
        #     print(f"✅ {code}: {len(res)} 行")
        # if code == "000001":
        #     print("\n=== 000001 股票数据预览 ===")
        #     print(res.head(3))  # 打印前 3 行数据，方便调试
        #     print("\n===数据类型===")
        #     print(res.dtypes)  # 打印数据类型，方便调试

    print(f"\n异步抓 {len(codes)} 只股票，耗时 {elapsed:.2f} 秒")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # 主程序结束时，安全关闭线程池
        executor.shutdown(wait=True)