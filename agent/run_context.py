import json
from datetime import date

from agent.strategy_session import CompletedRun


# 用 UTF-8 字节作为保守的预算，并为工具定义及模型回复预留空间。
INPUT_BUDGET = 800_000
_RESERVE = 20_000
_DETAIL_PAGE_BUDGET = 32_000


class RunContextError(ValueError):
    pass


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _size(text: str) -> int:
    return len(text.encode("utf-8"))


def _index_entry(number: int, run: CompletedRun) -> dict:
    request = run.request
    dataset = request.dataset
    return {
        "run_number": number,
        "run_id": request.id,
        "code_version": request.version.id,
        "dataset_id": dataset.id,
        "start": str(dataset.start),
        "end": str(dataset.end),
        "symbols": list(dataset.market.stocks),
        "benchmark": dataset.market.benchmark_code,
        "starting_cash": request.starting_cash,
        "commission": request.commission,
        "metrics": run.metrics(),
    }


def _summary(number: int, run: CompletedRun) -> dict:
    return {
        **_index_entry(number, run),
        "rules": run.request.version.rules.model_dump(mode="json", exclude_none=True),
        "warmup_bars": run.request.version.warmup_bars,
        "coverage": run.request.dataset.coverage(),
        "holdings": run.result.holdings,
    }


def _market_rows(run: CompletedRun, symbol: str | None = None):
    market = run.request.dataset.market
    for code, frame in market.stocks.items():
        if symbol is None or symbol == code:
            for point in frame.itertuples():
                yield (
                    code, str(point.Index.date()), float(point.open), float(point.high),
                    float(point.low), float(point.close), float(point.volume),
                )
    if market.benchmark is not None and (symbol is None or symbol == market.benchmark_code):
        for day, close in market.benchmark.items():
            yield market.benchmark_code, str(day.date()), None, None, None, float(close), None


MARKET_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]
MARKET_NOTE = "实际供回测使用的 BaoStock 日线，含预热期。股票为前复权可交易日线，基准仅有收盘价。"


def _equity_rows(run: CompletedRun):
    for point, drawdown in zip(run.result.equity, run.result.drawdowns(run.request.starting_cash)):
        yield {**point.model_dump(mode="json"), "drawdown_fraction": drawdown}


def _full_details(number: int, run: CompletedRun) -> str:
    return _json({
        **_summary(number, run),
        "equity": list(_equity_rows(run)),
        "fills": [item.model_dump(mode="json", exclude_none=True) for item in run.result.fills],
        "unfilled": [item.model_dump(mode="json", exclude_none=True) for item in run.result.unfilled],
        "market": {"columns": MARKET_COLUMNS, "rows": list(_market_rows(run)), "note": MARKET_NOTE},
    })


def context_for_chat(
    system: str, messages: list, runs: dict[str, CompletedRun],
) -> tuple[str, list[dict]]:
    if not runs:
        return system, []
    history_size = sum(
        _size(_json({"content": message.content, "tool_calls": getattr(message, "tool_calls", [])})) + 64
        for message in messages
    )
    budget = INPUT_BUDGET - history_size - _size(system) - _RESERVE
    ordered = list(enumerate(runs.values(), start=1))
    index = [_index_entry(number, run) for number, run in ordered]
    omitted = [{"run_number": number, "run_id": run.request.id} for number, run in ordered]
    # 为所有摘要和可能省略的编号预留空间，再按新到旧追加整次运行。
    remaining = budget - _size(_json({"run_index": index, "omitted_details": omitted, "details": []}))
    if remaining < 0:
        index = omitted.copy()
        remaining = budget - _size(_json({"run_index": index, "omitted_details": omitted, "details": []}))
    if remaining < 0:
        raise RunContextError("本轮聊天历史和运行目录已超过可用上下文，无法继续发送。")
    details = []
    included = set()
    for number, run in reversed(ordered):
        detail = _full_details(number, run)
        cost = _size(detail) + 1
        if cost > remaining:
            continue
        details.append(detail)
        included.add(run.request.id)
        remaining -= cost
    omitted = [entry for entry in omitted if entry["run_id"] not in included]
    prefix = _json({"run_index": index, "omitted_details": omitted})
    context = prefix[:-1] + ',"details":[' + ",".join(reversed(details)) + "]}"
    return system + "\n\n当前会话已完成的回测（JSON 数据）：\n" + context, omitted


def _resolve_run(runs: dict[str, CompletedRun], reference: str) -> tuple[int, CompletedRun]:
    ordered = list(runs.values())
    if reference == "latest" and ordered:
        return len(ordered), ordered[-1]
    for number, run in enumerate(ordered, start=1):
        if reference in (str(number), run.request.id):
            return number, run
    raise ValueError("当前会话没有对应的已完成回测，请核对运行编号。")


def read_run_details(
    runs: dict[str, CompletedRun], reference: str, section: str, symbol: str | None,
    start: date | None, end: date | None, offset: int, limit: int,
) -> dict:
    number, run = _resolve_run(runs, reference)
    output = {"run_number": number, "run_id": run.request.id, "section": section}
    if section == "summary":
        return {**output, **_summary(number, run)}
    if start and end and start > end:
        raise ValueError("查询开始日不能晚于结束日。")
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("offset 须为非负数，limit 须在 1 到 200 之间。")
    if symbol is not None:
        symbol = symbol.strip().casefold()
        market = run.request.dataset.market
        if symbol not in market.stocks and symbol != market.benchmark_code:
            raise ValueError("请使用该次回测的完整标的代码。")
    if symbol is not None and section == "equity":
        raise ValueError("净值是整个账户的数据，查询时请省略 symbol。")

    def matches(day: date) -> bool:
        return (start is None or day >= start) and (end is None or day <= end)

    if section == "market":
        output.update(columns=MARKET_COLUMNS, note=MARKET_NOTE)
        rows = (row for row in _market_rows(run, symbol) if matches(date.fromisoformat(row[1])))
    elif section == "equity":
        rows = (row for row in _equity_rows(run) if matches(date.fromisoformat(row["date"])))
    elif section in ("fills", "unfilled"):
        records = run.result.fills if section == "fills" else run.result.unfilled
        rows = (
            order.model_dump(mode="json", exclude_none=True)
            for order in records
            if (symbol is None or order.symbol == symbol)
            and (matches(order.signal_date) or (section == "fills" and matches(order.fill_date)))
        )
    else:
        raise ValueError("未知明细类别。")
    page = []
    remaining = _DETAIL_PAGE_BUDGET
    next_offset = None
    for index, row in enumerate(rows):
        if index < offset:
            continue
        cost = _size(_json(row)) + 1
        if len(page) >= limit or (page and cost > remaining):
            next_offset = index
            break
        page.append(row)
        remaining -= cost
    return {**output, "offset": offset, "next_offset": next_offset, "rows": page}
