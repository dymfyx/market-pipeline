# market-pipeline

## 📖 项目简介
本项目是一个基于 Python 的 A 股多因子选股与量化回测系统。从数据源获取到策略回测，形成了一套完整的量化研究闭环。

**当前版本：v0.5**

## ✨ 核心功能
- **异步数据抓取**：使用 `asyncio` 与 `ThreadPoolExecutor` 并发抓取 A 股行情数据，内置 `Semaphore` 限流与指数退避重试机制，高效且安全。
- **数据清洗与质量校验**：自动校验空数据、逻辑异常（如最高价低于最低价），处理缺失值并对齐多股票时间序列。
- **多因子选股模型**：支持动量因子与低波动因子的横截面打分合成，自动选出 Top N 股票进行轮动。
- **参数敏感性测试**：支持调整调仓周期（5/10/20天）、因子权重、持仓数量等参数，并输出对比结果。
- **真实约束回测**：包含手续费、滑点计算，严格防范未来函数（`shift(1)`），支持与等权基准对比。

## 🛠️ 技术栈
- **语言**：Python 3.11+
- **数据处理**：Pandas, NumPy
- **数据源**：akshare
- **异步框架**：asyncio, concurrent.futures
- **可视化**：Matplotlib

## 📁 目录结构
```text
market-pipeline/
├── src/market_pipeline/
│   ├── config.py             # 参数配置
│   ├── fetcher_async.py      # 异步数据抓取
│   ├── models.py             # 数据模型定义
├── data/                     # 本地数据存储（git忽略）
│   └── raw/                  # 原始行情数据
│   └── merged_close.parquet  # 对齐后的收盘价大表
├── notes/                    # 学习与复盘笔记
│   ├── day1.md
│   └── ...
├── read_data.py              # 数据预览脚本
├── merge_data.py             # 数据合并脚本
├── rotation_backtest.py      # 轮动策略回测
├── multi_factor_backtest.py  # 多因子策略回测
└── README.md