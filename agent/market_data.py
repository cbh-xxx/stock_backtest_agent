import re
from dataclasses import dataclass
from datetime import date, timedelta
from threading import Lock

import baostock as bs
import pandas as pd

from agent.backtest_spec import (
    BENCHMARK_CODES,
    DEFAULT_STOCKS,
    EARLIEST_DATE,
    BacktestInputError,
    StrategySpec,
)


class MarketDataError(RuntimeError):
    pass


@dataclass
class MarketData:
    stocks: dict[str, pd.DataFrame]
    benchmark_code: str | None
    benchmark: pd.Series | None


PRICE_COLUMNS = ["open", "high", "low", "close"]
STOCK_COLUMNS = [*PRICE_COLUMNS, "volume"]
_SESSION_LOCK = Lock()


def _check_response(result, operation: str) -> None:
    if result.error_code != "0":
        raise MarketDataError(f"{operation}失败（{result.error_code}）：{result.error_msg}")


def _rows(result, operation: str) -> pd.DataFrame:
    _check_response(result, operation)
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    _check_response(result, operation)
    return pd.DataFrame(rows, columns=result.fields)


def _code(value: str, *, benchmark: bool = False) -> str | None:
    value = value.casefold()
    if match := re.fullmatch(r"(sh|sz)\.?([0-9]{6})", value):
        return f"{match[1]}.{match[2]}"
    if match := re.fullmatch(r"([0-9]{6})\.(sh|sz)", value):
        return f"{match[2]}.{match[1]}"
    if re.fullmatch(r"[0-9]{6}", value):
        if benchmark:
            return next((code for code in BENCHMARK_CODES.values() if code.endswith("." + value)), None)
        if value.startswith("6"):
            return f"sh.{value}"
        if value.startswith(("00", "30")):
            return f"sz.{value}"
        raise BacktestInputError(f"「{value}」不在沪深 A 股支持范围内。")
    return None


def _stock_info(value: str) -> pd.Series:
    code = _code(value)
    label = f"查询股票「{value}」"
    result = bs.query_stock_basic(code=code) if code else bs.query_stock_basic(code_name=value)
    rows = _rows(result, label)
    if not rows.empty:
        rows = rows[rows["type"].eq("1") & rows["code"].str.match(r"(sh\.6|sz\.(00|30))")]
        if not code:
            exact = rows[rows["code_name"].eq(value)]
            if not exact.empty:
                rows = exact
        rows = rows.drop_duplicates("code")
        if len(rows) == 1:
            return rows.iloc[0]
        if len(rows) > 1:
            options = "、".join(f"{row.code_name}（{row.code}）" for row in rows.head(5).itertuples())
            raise BacktestInputError(f"「{value}」匹配多个沪深股票：{options}。请提供明确代码。")
    raise BacktestInputError(f"无法将「{value}」识别为沪深 A 股，请提供明确的股票代码。")


def _calendar(start: date, end: date) -> set[date]:
    label = f"查询交易日历 {start} 至 {end}"
    rows = _rows(bs.query_trade_dates(start_date=str(start), end_date=str(end)), label)
    if rows.empty:
        raise MarketDataError(f"{label}没有返回记录。")
    dates = pd.to_datetime(rows["calendar_date"], errors="raise").dt.date
    return set(dates[rows["is_trading_day"].eq("1")])


def _bars(code: str, start: date, end: date, *, index: bool = False) -> pd.DataFrame:
    label = f"获取 {code} 日线（{start} 至 {end}）"
    columns = ["close"] if index else [*STOCK_COLUMNS, "tradestatus"]
    rows = _rows(
        bs.query_history_k_data_plus(
            code, ",".join(["date", "code", *columns]),
            start_date=str(start), end_date=str(end),
            frequency="d", adjustflag="3" if index else "2",
        ),
        label,
    )
    if rows.empty:
        raise MarketDataError(f"{label}没有返回行情。")
    try:
        frame = rows.set_index(pd.to_datetime(rows["date"], format="%Y-%m-%d")).sort_index()
        if frame.index.hasnans or frame.index.has_duplicates or not frame["code"].eq(code).all():
            raise ValueError("日期缺失、重复，或标的代码不一致")
        if frame.index[0].date() < start or frame.index[-1].date() > end:
            raise ValueError("返回日期超出请求区间")
    except (KeyError, ValueError) as exc:
        raise MarketDataError(f"{label}返回了无效数据：{exc}") from exc
    return frame


def _require_dates(code: str, frame: pd.DataFrame, expected: set[date]) -> None:
    missing = sorted(expected.difference(frame.index.date))
    if missing:
        sample = "、".join(map(str, missing[:3]))
        raise MarketDataError(f"{code} 缺少 {len(missing)} 个交易日的行情（例如 {sample}）。")


def _numeric_bars(code: str, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    price_columns = [column for column in columns if column != "volume"]
    if (
        numeric.isna().any().any()
        or numeric.abs().eq(float("inf")).any().any()
        or numeric[price_columns].le(0).any().any()
        or ("volume" in columns and numeric["volume"].lt(0).any())
    ):
        raise MarketDataError(f"{code} 行情包含缺失或无效的价格、成交量。")
    return numeric


def _stock_bars(
    stock: pd.Series, start: date, end: date, required: set[date], warmup: int
) -> pd.DataFrame:
    code = stock["code"]
    ipo = date.fromisoformat(stock["ipoDate"]) if stock["ipoDate"] else EARLIEST_DATE
    out = date.fromisoformat(stock["outDate"]) if stock["outDate"] else None
    if min(required) < ipo or (out and max(required) > out):
        raise MarketDataError(f"{code} 的上市区间（{ipo} 至 {out or '尚未退市'}）不覆盖请求区间。")

    # 若窗口内有长期停牌，向前扩展，直到找到足量可交易日线或到达上市日。
    earliest = max(ipo, EARLIEST_DATE)
    days = warmup * 3 + 30
    while True:
        fetch_start = max(earliest, start - timedelta(days=min(days, (start - EARLIEST_DATE).days)))
        frame = _bars(code, fetch_start, end)
        _require_dates(code, frame, required)
        if not frame["tradestatus"].isin(["0", "1"]).all():
            raise MarketDataError(f"{code} 行情包含无法识别的交易状态。")
        tradable = frame[frame["tradestatus"].eq("1")]
        history = tradable.loc[tradable.index.date < start].tail(warmup)
        in_period = tradable.loc[tradable.index.date >= start]
        if in_period.empty:
            raise MarketDataError(f"{code} 在 {start} 至 {end} 没有可交易日线。")
        if len(history) >= warmup:
            warmup_start = history.index[0].date()
            _require_dates(code, frame, _calendar(warmup_start, start - timedelta(days=1)))
            return _numeric_bars(code, pd.concat([history, in_period]), STOCK_COLUMNS)
        if fetch_start == earliest:
            raise MarketDataError(f"{code} 在 {start} 之前需要 {warmup} 根预热日线，只有 {len(history)} 根。")
        days *= 2


def load_market_data(spec: StrategySpec) -> MarketData:
    start, end = spec.start, spec.end
    benchmark_code = None
    if spec.benchmark:
        benchmark_name = spec.benchmark.replace(" ", "").replace("上证综指", "上证指数")
        if benchmark_name != "上证指数":
            benchmark_name = benchmark_name.removesuffix("指数")
        benchmark_code = BENCHMARK_CODES.get(benchmark_name) or _code(benchmark_name, benchmark=True)
        if benchmark_code not in BENCHMARK_CODES.values():
            raise BacktestInputError("基准仅支持上证指数、沪深300、中证500或深证成指。")
    warmup = max(signal.warmup() for signal in spec.entry)
    # BaoStock SDK 使用进程级连接，串行使用完整会话。
    with _SESSION_LOCK:
        _check_response(bs.login(), "连接 BaoStock")
        try:
            required = _calendar(start, end)
            if not required:
                raise MarketDataError(f"{start} 至 {end} 没有交易日。")
            stocks = {}
            for value in spec.stocks or DEFAULT_STOCKS:
                stock = _stock_info(value)
                code = stock["code"]
                if code not in stocks:
                    stocks[code] = _stock_bars(stock, start, end, required, warmup)
            benchmark = None
            if benchmark_code:
                index_frame = _bars(benchmark_code, start, end, index=True)
                _require_dates(benchmark_code, index_frame, required)
                benchmark = _numeric_bars(benchmark_code, index_frame, ["close"])["close"]
            return MarketData(stocks, benchmark_code, benchmark)
        finally:
            bs.logout()
