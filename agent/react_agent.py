import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool

from agent.backtest_spec import DEFAULT_STOCKS, EARLIEST_DATE
from agent.market_data import fetch_daily_market_data
from agent.run_context import RunContextError, context_for_chat, read_run_details
from agent.run_strategy import run_approved_strategy
from agent.strategy_codegen import write_strategy
from agent.strategy_compiler import compile_strategy
from agent.strategy_rules import MAX_WARMUP, StrategyRules, parse_rules
from agent.strategy_session import (
    DEFAULT_CASH, DEFAULT_COMMISSION, CodeVersion, CompletedRun, Dataset, RunRequest, StrategySession,
)
from utils.logger_handler import logger


_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "code_agent_prompt.txt"
_MAX_STEPS = 12
_BEIJING = timezone(timedelta(hours=8))
ChartType = Literal["equity", "drawdown"]
TimelineKind = Literal["user", "assistant", "error", "version", "dataset", "confirmation", "run"]


@dataclass
class TimelineEntry:
    kind: TimelineKind
    content: str | CodeVersion | Dataset | RunRequest | CompletedRun
    time: datetime = field(default_factory=lambda: datetime.now(_BEIJING))
    tools: list[tuple[str, str]] = field(default_factory=list)
    status: str = ""
    reason: str = ""


class ReactAgent:
    def __init__(self):
        self.session = StrategySession()
        self.messages = []
        self.timeline: list[TimelineEntry] = []
        self.tools = {
            handler.__name__: StructuredTool.from_function(handler)
            for handler in (
                self.generate_strategy_code, self.modify_strategy_code,
                self.fetch_market_data, self.execute_strategy_code, self.query_backtest_details,
            )
        }
        self.chart_tool = StructuredTool.from_function(self.render_backtest_charts)

    def add_timeline_entry(
        self, kind: TimelineKind, content: str | CodeVersion | Dataset | RunRequest | CompletedRun,
    ) -> TimelineEntry:
        entry = TimelineEntry(kind, content)
        self.timeline.append(entry)
        return entry

    def _save_version(
        self, rules: StrategyRules, requirements: str, references: tuple[str, ...],
        previous: CodeVersion | None = None,
    ) -> CodeVersion:
        source, warmup = compile_strategy(rules)
        state = self.session
        with state.lock:
            version = CodeVersion(
                id=f"v{state.version_count + 1}", rules=rules, source=source, warmup_bars=warmup,
                requirements=requirements, references=references, previous=previous or state.version,
            )
            state.version_count += 1
            state.version = version
            pending = state.pending
            state.pending = None
            self.add_timeline_entry("version", version)
        if pending is not None:
            self._finish_execution_message(pending.id, {"status": "cancelled", "reason": "代码版本已变化。"})
        return version

    def generate_strategy_code(self, requirements: str) -> dict:
        """生成一份可审阅的 Backtrader 策略。requirements 须包含用户完整、明确的交易逻辑。"""
        if not requirements.strip():
            raise ValueError("请提供要生成的策略规则。")
        rules, references = write_strategy(requirements)
        return self._save_version(rules, requirements, references).tool_output()

    def modify_strategy_code(self, code_version: str, change_request: str) -> dict:
        """按明确要求修改已有策略版本，保留用户未要求改动的交易规则。"""
        previous = self.session.version
        if previous is None or previous.id != code_version:
            raise ValueError("请指定页面当前展示的策略版本。")
        if not change_request.strip():
            raise ValueError("请说明要修改的策略规则。")
        rules, references = write_strategy(previous.requirements, previous.rules, change_request)
        requirements = previous.requirements + "\n后续修改：" + change_request
        return self._save_version(rules, requirements, references, previous).tool_output()

    def save_manual(self, code_version: str, rules_json: str) -> CodeVersion:
        previous = self.session.version
        if previous is None or previous.id != code_version:
            raise ValueError("编辑的版本已变更，请重新打开当前版本。")
        return self._save_version(
            parse_rules(rules_json), previous.requirements, previous.references, previous,
        )

    def fetch_market_data(
        self, start_date: str, end_date: str, warmup_bars: int,
        symbols: list[str] | None = None, benchmark: str | None = None,
    ) -> dict:
        """按 YYYY-MM-DD 区间取 BaoStock 日线。未指定股票时省略 symbols；指数只能是 benchmark。"""
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if not EARLIEST_DATE <= start <= end <= self._today():
            raise ValueError("日期须从 1990 年起，且开始日不晚于结束日，结束日不在未来。")
        if not 0 <= warmup_bars <= MAX_WARMUP:
            raise ValueError(f"预热日线数须在 0 到 {MAX_WARMUP} 之间。")
        if symbols is not None and (not symbols or any(not item.strip() for item in symbols)):
            raise ValueError("请提供完整股票清单，或省略 symbols 使用默认组合。")
        version = self.session.version
        if version is None or warmup_bars < version.warmup_bars:
            raise ValueError("请先生成策略，再按其预热需求取数。")
        market = fetch_daily_market_data(start, end, warmup_bars, symbols, benchmark)
        dataset = Dataset(f"d-{uuid4().hex[:10]}", start, end, warmup_bars, market)
        with self.session.lock:
            pending = self.session.pending
            self.session.pending = None
            self.session.dataset = dataset
            self.add_timeline_entry("dataset", dataset)
        if pending is not None:
            self._finish_execution_message(pending.id, {"status": "cancelled", "reason": "回测行情已变化。"})
        return {
            "dataset_id": dataset.id, "start_date": start, "end_date": end,
            "symbols": list(market.stocks), "benchmark": market.benchmark_code,
            "warmup_bars": warmup_bars, "coverage": dataset.coverage(),
        }

    def execute_strategy_code(
        self, code_version: str, dataset_id: str,
        starting_cash: float = DEFAULT_CASH, commission: float = DEFAULT_COMMISSION,
    ) -> dict:
        """提交待确认运行请求。手续费使用小数，例如 0.001 表示 0.1%。实际执行须由用户在页面确认。"""
        request = self._prepare_run(code_version, dataset_id, starting_cash, commission, show_confirmation=True)
        return {"status": "pending_confirmation", "run_request_id": request.id, "summary": request.describe()}

    def _prepare_run(
        self, code_version: str, dataset_id: str, starting_cash: float, commission: float,
        *, show_confirmation: bool,
    ) -> RunRequest:
        self._check_account(starting_cash, commission)
        state = self.session
        with state.lock:
            version, dataset = state.version, state.dataset
            if version is None or version.id != code_version:
                raise ValueError("请选择当前可审阅的代码版本。")
            if dataset is None or dataset.id != dataset_id or dataset.warmup_bars < version.warmup_bars:
                raise ValueError("请先为当前版本准备满足预热需求的行情。")
            if state.pending or state.in_progress:
                raise ValueError("已有待确认或正在运行的请求。")
            request = RunRequest(f"run-{uuid4().hex[:10]}", version, dataset, starting_cash, commission)
            state.pending = request
            state.starting_cash = starting_cash
            state.commission = commission
            if show_confirmation:
                self.add_timeline_entry("confirmation", request).status = "待确认"
        return request

    @staticmethod
    def _check_account(starting_cash: float, commission: float):
        if not isfinite(starting_cash) or starting_cash <= 0 or not 0 <= commission < 1:
            raise ValueError("初始资金须大于 0，手续费须在 0（含）到 100%（不含）之间。")

    def run_from_page(
        self, code_version: str, start: date, end: date, symbols: list[str] | None,
        benchmark: str | None, starting_cash: float, commission: float,
    ) -> CompletedRun:
        """由时间线中的执行按钮调用，一次点击确认本次显示的代码与参数。"""
        state = self.session
        version, dataset, pending = state.version, state.dataset, state.pending
        if state.in_progress or version is None or version.id != code_version:
            raise ValueError("策略版本已变化或已有运行，请刷新页面后重试。")
        if not EARLIEST_DATE <= start <= end <= self._today():
            raise ValueError("请检查回测起止日期，结束日期不能在未来。")
        self._check_account(starting_cash, commission)
        selected_symbols = tuple(dict.fromkeys(symbol.casefold() for symbol in (symbols or DEFAULT_STOCKS)))
        reuse_data = (
            dataset is not None and dataset.start == start and dataset.end == end
            and tuple(dataset.market.stocks) == selected_symbols
            and (dataset.market.benchmark_code or "").casefold() == (benchmark or "").casefold()
            and dataset.warmup_bars >= version.warmup_bars
        )
        if (
            reuse_data and pending is not None and pending.version is version and pending.dataset is dataset
            and pending.starting_cash == starting_cash and pending.commission == commission
        ):
            return self.approve(pending.id)
        self.cancel_pending()
        if reuse_data:
            dataset_id = dataset.id
        else:
            first_entry = len(self.timeline)
            try:
                prepared = self.fetch_market_data(
                    str(start), str(end), version.warmup_bars, symbols, benchmark,
                )
            except Exception as exc:
                message = f"行情准备失败：{exc}"
                self.add_timeline_entry("error", message).tools.append(("fetch_market_data", "失败"))
                raise ValueError(message) from exc
            self.timeline[first_entry].tools.append(("fetch_market_data", "完成"))
            dataset_id = prepared["dataset_id"]
        request = self._prepare_run(version.id, dataset_id, starting_cash, commission, show_confirmation=False)
        return self.approve(request.id)

    def render_backtest_charts(self, run_id: str, chart_types: list[ChartType]) -> dict:
        """为已完成的本次回测自动绘图。chart_types 同时传入 equity 和 drawdown。"""
        run = self.session.runs.get(run_id)
        if run is None or len(chart_types) != 2 or set(chart_types) != {"equity", "drawdown"}:
            raise ValueError("绘图须使用本次成功运行编号，并同时指定 equity 和 drawdown。")
        run.render_charts(chart_types)
        return {"run_id": run_id, "charts": list(run.charts)}

    def query_backtest_details(
        self, run: str, section: Literal["summary", "equity", "fills", "unfilled", "market"],
        symbol: str | None = None, start_date: str | None = None, end_date: str | None = None,
        offset: int = 0, limit: int = 100,
    ) -> dict:
        """只读查询已完成回测。run 为运行编号、从 1 开始的序号或 latest。

        symbol 使用完整代码，可省略；日期使用 YYYY-MM-DD。成交可按信号日或成交日匹配。
        返回 next_offset 时可用它继续取同一查询的下一页。此工具不会运行策略或重新下载行情。
        """
        return read_run_details(
            self.session.runs, run.strip(), section, symbol,
            date.fromisoformat(start_date) if start_date else None,
            date.fromisoformat(end_date) if end_date else None, offset, limit,
        )

    @staticmethod
    def _today() -> date:
        return datetime.now(_BEIJING).date()

    def _system(self):
        version, dataset = self.session.version, self.session.dataset
        snapshot = {
            "today": str(self._today()),
            "current_code_version": version.id if version else None,
            "current_rules": version.rules.model_dump(mode="json", exclude_none=True) if version else None,
            "warmup_bars": version.warmup_bars if version else None,
            "dataset_id": dataset.id if dataset else None,
            "dataset_summary": dataset.coverage() if dataset else [],
            "dataset_start": str(dataset.start) if dataset else None,
            "dataset_end": str(dataset.end) if dataset else None,
            "dataset_warmup_bars": dataset.warmup_bars if dataset else None,
            "benchmark": dataset.market.benchmark_code if dataset else None,
        }
        return SystemMessage(content=_PROMPT.read_text(encoding="utf-8") + "\n当前会话：" + json.dumps(
            snapshot, ensure_ascii=False,
        ))

    def cancel_pending(self):
        with self.session.lock:
            request = self.session.pending
            self.session.pending = None
        if request is not None:
            self._finish_execution_message(request.id, {"status": "cancelled"})

    def _finish_execution_message(self, request_id: str, outcome: dict):
        status = {"cancelled": "已取消", "completed": "已完成", "error": "执行失败"}[outcome["status"]]
        self._update_confirmation(request_id, status, outcome.get("reason", ""))
        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            if not isinstance(message, ToolMessage) or message.name != "execute_strategy_code":
                continue
            if json.loads(message.content).get("run_request_id") != request_id:
                continue
            self.messages[index] = ToolMessage(
                name=message.name, tool_call_id=message.tool_call_id,
                content=json.dumps({**outcome, "run_request_id": request_id}, ensure_ascii=False),
            )
            break

    def _update_confirmation(self, request_id: str, status: str, reason: str = ""):
        for entry in reversed(self.timeline):
            if entry.kind == "confirmation" and entry.content.id == request_id:
                entry.status = status
                entry.reason = reason
                break

    def _remember_messages(self, messages: list):
        # 当前轮次保留工具明细供模型回答。历史中只留查阅记录，避免反复附带同一批日线。
        remembered = []
        for message in messages:
            if isinstance(message, ToolMessage) and message.name == "query_backtest_details":
                outcome = json.loads(message.content)
                if "rows" in outcome:
                    rows = outcome.pop("rows")
                    outcome.update(row_count=len(rows), details_available_via="query_backtest_details")
                    message = message.model_copy(update={"content": json.dumps(outcome, ensure_ascii=False)})
            remembered.append(message)
        self.messages = remembered

    def execute_stream(self, query: str):
        self.add_timeline_entry("user", query)
        if self.session.in_progress:
            message = "当前回测正在运行。"
            self.add_timeline_entry("assistant", message)
            yield message
            return
        self.cancel_pending()
        # 暂存消息，避免一次 API 异常把未配对的工具调用留在下一轮历史里。
        turn = [*self.messages, HumanMessage(content=query)]
        omitted_runs: dict[str, int] = {}
        reply_tools: list[tuple[str, str]] = []

        def reply(message: str, *, error: bool = False) -> str:
            if omitted_runs:
                runs = "、".join(f"第 {number} 次（{run_id}）" for run_id, number in omitted_runs.items())
                message += f"\n\n> 本轮上下文曾缩减以下回测的完整明细：{runs}。具体记录可按运行或日期补查。"
            self.add_timeline_entry("error" if error else "assistant", message).tools.extend(reply_tools)
            return message

        try:
            from model.factory import chat_model

            model = chat_model.bind_tools(list(self.tools.values()))
            for _ in range(_MAX_STEPS):
                system, omitted = context_for_chat(self._system().content, turn, self.session.runs)
                omitted_runs.update((entry["run_id"], entry["run_number"]) for entry in omitted)
                response = model.invoke([SystemMessage(content=system), *turn])
                turn.append(response)
                if not response.tool_calls:
                    self._remember_messages(turn)
                    if response.invalid_tool_calls:
                        yield reply("DeepSeek 返回了无法读取的工具参数，请重新描述请求。", error=True)
                    else:
                        yield reply(StrOutputParser().invoke(response).strip() or "请提供策略规则和回测区间。")
                    return
                failure = None
                for call in response.tool_calls:
                    name = call["name"]
                    if self.session.pending or failure:
                        outcome = {"status": "skipped", "reason": "本轮正在等待确认或已有工具失败。"}
                        reply_tools.append((name, "跳过"))
                    else:
                        first_entry = len(self.timeline)
                        try:
                            if name not in self.tools:
                                raise ValueError("工具未注册。")
                            outcome = self.tools[name].invoke(call["args"])
                            status = "等待确认" if self.session.pending else "完成"
                            tools = self.timeline[first_entry].tools if len(self.timeline) > first_entry else reply_tools
                            tools.append((name, status))
                        except Exception as exc:
                            logger.exception("回测工具失败：%s", name)
                            failure = f"{name} 失败：{exc}"
                            outcome = {"status": "error", "reason": str(exc)}
                            reply_tools.append((name, "失败"))
                    turn.append(ToolMessage(
                        content=json.dumps(outcome, ensure_ascii=False, default=str),
                        name=name, tool_call_id=call["id"],
                    ))
                self._remember_messages(turn)
                if failure:
                    yield reply(failure, error=True)
                    return
                if self.session.pending:
                    yield reply("回测已准备好。请在对应策略条目审阅源码，再到待确认条目核对参数并点击「执行代码」。")
                    return
            yield reply("本轮工具调用已达到上限，已生成的版本和行情仍可在时间线查看。", error=True)
        except RunContextError as exc:
            yield reply(str(exc), error=True)
        except Exception as exc:
            logger.exception("DeepSeek 工具调度失败")
            # 上一条完整配对的消息已保留；失败的模型响应不会进入下轮。
            yield reply(f"DeepSeek 工具调度失败（{type(exc).__name__}）。请检查 API 配置或日志后重试。", error=True)

    def _automatic_charts(self, run: CompletedRun):
        from model.factory import chat_model

        model = chat_model.bind_tools([self.chart_tool], tool_choice=self.chart_tool.name)
        response = model.invoke([
            SystemMessage(content="请通过唯一绘图工具自动绘制净值和回撤图。参数必须使用给定的运行编号。"),
            HumanMessage(content=json.dumps({
                "run_id": run.request.id, "chart_types": ["equity", "drawdown"],
            })),
        ])
        calls = response.tool_calls
        if (
            len(calls) != 1 or calls[0]["name"] != self.chart_tool.name
            or calls[0]["args"].get("run_id") != run.request.id
        ):
            raise ValueError("DeepSeek 未返回本次运行的有效绘图参数。")
        self.chart_tool.invoke(calls[0]["args"])

    def approve(self, request_id: str) -> CompletedRun:
        """仅由 Streamlit 确认按钮调用，不提供给模型。"""
        state = self.session
        with state.lock:
            request = state.pending
            if request is None or request.id != request_id or state.in_progress:
                raise ValueError("运行请求已过期、已使用，或当前已有运行。")
            if state.version is not request.version or state.dataset is not request.dataset:
                state.pending = None
                reason = "代码或行情已变化，请重新提出运行请求。"
                self._finish_execution_message(request.id, {"status": "cancelled", "reason": reason})
                raise ValueError(reason)
            state.pending = None
            state.in_progress = True
            self._update_confirmation(request.id, "执行中")
        try:
            result = run_approved_strategy(
                request.version.source, request.version.warmup_bars, request.dataset.market,
                request.dataset.start, request.dataset.end, request.starting_cash, request.commission,
            )
            run = CompletedRun(request, result)
            state.runs[request.id] = run
            entry = self.add_timeline_entry("run", run)
            entry.tools.append(("execute_strategy_code", "执行完成"))
            self._finish_execution_message(request.id, {
                "status": "completed", "run_id": request.id,
            })
            try:
                self._automatic_charts(run)
                entry.tools.append(("render_backtest_charts", "完成"))
            except Exception as exc:
                logger.exception("自动绘图失败")
                run.chart_error = f"自动绘图失败（{type(exc).__name__}），报告仍可查看。"
                entry.tools.append(("render_backtest_charts", "失败"))
            return run
        except Exception as exc:
            self._finish_execution_message(request.id, {"status": "error", "reason": str(exc)})
            entry = self.add_timeline_entry("error", f"回测执行失败：{exc}")
            entry.tools.append(("execute_strategy_code", "执行失败"))
            raise
        finally:
            with state.lock:
                state.in_progress = False
