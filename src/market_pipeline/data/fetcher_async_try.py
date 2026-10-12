import asyncio
import sys
import time
import random
import akshare as ak
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sem = asyncio.Semaphore(3)

executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ak_fetch")

async def fetch_with_retry(stock_code: str, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:

    symbol = f"sz{stock_code}" if stock_code.startswith("0") else f"sh{stock_code}"
    loop = asyncio.get_running_loop()

    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                print(f"正在抓取{stock_code} (第 {attempt} 次尝试) ...")

                df = await loop.run_in_executor(
                    executor,
                    lambda: ak.stock_zh_a_daily(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq",
                    ),
                )

                await asyncio.sleep(random.uniform(0.5, 1.5))
                return df
            
        except Exception as e:
            print(f"{stock_code} 第 {attempt} 次尝试失败: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 * attempt)

async def main() -> None:
    codes = ["000001", "000002", "600000", "600036", "601318"]

    start = time.perf_counter()

    tasks = [fetch_with_retry(code, "20240101", "20260920") for code in codes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - start

    for code, res in zip(codes, results):
        if isinstance(res, Exception):
            print(f"❌ {code} 最终失败: {res}")
        else:
            print(f"✅ {code}: {len(res)} 行")

    print(f"\n异步抓 {len(codes)} 只股票，耗时 {elapsed:.2f} 秒")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        executor.shutdown(wait=True)