import pandas as pd

from market_pipeline.merge_data import merge_all_data


def _write_stock(tmp_path, code: str, dates, closes) -> None:
    """把一只股票的假数据写成 parquet 文件，模拟 raw 目录中的真实数据。"""
    df = pd.DataFrame({"date": dates, "close": closes})
    df.to_parquet(tmp_path / f"{code}.parquet", index=False)


def test_merge_two_stocks_basic(tmp_path) -> None:
    """两只股票正常合并：列数=股票数，行数=对齐后的交易日数。"""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    _write_stock(tmp_path, "000001", dates, [10.0, 11.0, 12.0, 13.0, 14.0])
    _write_stock(tmp_path, "000002", dates, [20.0, 21.0, 22.0, 23.0, 24.0])

    merged = merge_all_data(data_dir=str(tmp_path), output_file=str(tmp_path / "merged.parquet"))

    # 列数 = 2 只股票
    assert merged.shape[1] == 2
    # 行数 = 5 个交易日
    assert merged.shape[0] == 5
    # 列名正确（股票代码）
    assert list(merged.columns) == ["000001", "000002"]
    # 数值正确：第二列第一行应为 20.0
    assert merged["000002"].iloc[0] == 20.0


def test_merge_fills_missing_dates_with_ffill(tmp_path) -> None:
    """一只股票缺了中间一个交易日，合并后应被前一天的收盘价填充。"""
    all_dates = pd.date_range("2024-01-01", periods=4, freq="D")
    # 股票 B 缺少第 2 天（2024-01-02）
    b_dates = all_dates[[0, 2, 3]]
    _write_stock(tmp_path, "000001", all_dates, [10.0, 11.0, 12.0, 13.0])
    _write_stock(tmp_path, "000002", b_dates, [20.0, 22.0, 24.0])

    merged = merge_all_data(data_dir=str(tmp_path), output_file=str(tmp_path / "merged.parquet"))

    # 行数 = 4 个交易日
    assert merged.shape[0] == 4
    # 股票 B 缺的那天（第 2 行）应被 ffill 填充为 20.0
    assert merged["000002"].iloc[1] == 20.0


def test_merge_drops_rows_with_unrecoverable_nan(tmp_path) -> None:
    """股票 A 首日就缺数据（无法前向填充），该行应被 dropna 删除。"""
    all_dates = pd.date_range("2024-01-01", periods=3, freq="D")
    # 股票 B 晚一天上市：缺第 1 天，且第 1 天没有任何前值可填充
    b_dates = all_dates[1:]
    _write_stock(tmp_path, "000001", all_dates, [10.0, 11.0, 12.0])
    _write_stock(tmp_path, "000002", b_dates, [20.0, 21.0])

    merged = merge_all_data(data_dir=str(tmp_path), output_file=str(tmp_path / "merged.parquet"))

    # 第 1 天股票 B 为 NaN 且无法 ffill，整行被删除 → 剩 2 行
    assert merged.shape[0] == 2
    assert merged["000002"].isna().sum() == 0


def test_merge_empty_dir_returns_empty(tmp_path) -> None:
    """数据目录为空时：当前实现是返回空 DataFrame 而不是报错。

    注意：这是一个"待改进点"——空目录静默成功意味着调用方可能拿到一张空表
    继续算因子而不自知（后续回测会在 NaN 上崩溃）。如果希望改成抛错，
    需要同时改 merge_data.py 并在本测试同步更新断言。
    """
    merged = merge_all_data(data_dir=str(tmp_path), output_file=str(tmp_path / "merged.parquet"))

    # 当前真实行为：空表
    assert merged is not None
    assert merged.shape == (0, 0)
