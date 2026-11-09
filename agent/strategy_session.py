from dataclasses import dataclass, field
from datetime import date
from difflib import unified_diff
from threading import Lock

import pandas as pd

from agent.market_data import MarketData
from agent.run_strategy import RunResult
from agent.strategy_rules import StrategyRules


DEFAULT_CASH = 100_000.0
DEFAULT_COMMISSION = 0.001


@dataclass(frozen=True)
class CodeVersion:
    id: str
    rules: StrategyRules
    source: str
    warmup_bars: int
    requirements: str
    references: tuple[str, ...]
    previous: "CodeVersion | None" = None

    def diff(self) -> str:
        if not self.previous:
            return ""
        return "\n".join(unified_diff(
            self.previous.source.splitlines(), self.source.splitlines(),
            fromfile=self.previous.id, tofile=self.id, lineterm="",
        ))

    def rule_diff(self) -> str:
        if not self.previous:
            return ""
        return "\n".join(unified_diff(
            self.previous.rules.as_json().splitlines(), self.rules.as_json().splitlines(),
            fromfile=f"{self.previous.id}.json", tofile=f"{self.id}.json", lineterm="",
        ))

    def tool_output(self) -> dict:
        return {
            "code_version": self.id, "source": self.source,
            "rules": self.rules.model_dump(mode="json", exclude_none=True),
            "warmup_bars": self.warmup_bars, "references": self.references,
            "diff": self.diff(),
        }


@dataclass(frozen=True)
class Dataset:
    id: str
    start: date
    end: date
    warmup_bars: int
    market: MarketData

    def coverage(self) -> list[dict]:
        coverage = []
        for code, frame in self.market.stocks.items():
            actual = frame.loc[frame.index.date >= self.start]
            coverage.append({
                "标的": code, "首日": str(actual.index[0].date()),
                "末日": str(actual.index[-1].date()), "可交易日线数": len(actual),
            })
        return coverage


@dataclass(frozen=True)
class RunRequest:
    id: str
    version: CodeVersion
    dataset: Dataset
    starting_cash: float
    commission: float

    def describe(self) -> str:
        data = self.dataset
        benchmark = f"，基准 {data.market.benchmark_code}" if data.market.benchmark_code else ""
        return (
            f"代码 {self.version.id} · 数据 {data.id} · "
            f"{'、'.join(data.market.stocks)}{benchmark} · {data.start} 至 {data.end} · "
            f"资金 ¥{self.starting_cash:,.2f} · 手续费 {self.commission:.3%}"
        )


@dataclass
class CompletedRun:
    request: RunRequest
    result: RunResult
    charts: dict[str, pd.DataFrame] = field(default_factory=dict)
    chart_error: str | None = None

    def metrics(self) -> dict:
        cash = self.request.starting_cash
        equity = self.result.equity[-1].value
        return {
            "final_equity": equity,
            "total_return": equity / cash - 1,
            "max_drawdown": max(self.result.drawdowns(cash)),
            "fill_count": len(self.result.fills),
            "fees": sum(fill.fee for fill in self.result.fills),
        }

    def render_charts(self, chart_types: list[str]) -> None:
        dates = pd.to_datetime([point.date for point in self.result.equity])
        cash = self.request.starting_cash
        series = {
            "equity": ("净值（初始 = 1）", [point.value / cash for point in self.result.equity]),
            "drawdown": ("回撤（%）", [100 * value for value in self.result.drawdowns(cash)]),
        }
        self.charts = {
            kind: pd.DataFrame({series[kind][0]: series[kind][1]}, index=dates)
            for kind in chart_types
        }

    def report(self) -> str:
        metrics = self.metrics()
        market = self.request.dataset.market
        lines = [
            f"### 回测报告 {self.request.id}", self.request.describe(), "",
            "- 行情来源为 BaoStock。股票使用前复权日线，下一可交易日开盘模拟成交。",
            f"- 期末权益 ¥{metrics['final_equity']:,.2f}，"
            f"总收益率 {metrics['total_return']:+.2%}，最大回撤 {metrics['max_drawdown']:.2%}。",
            f"- 成交 {metrics['fill_count']} 笔，手续费 ¥{metrics['fees']:,.2f}。",
        ]
        if market.benchmark is not None:
            benchmark = market.benchmark
            lines.append(f"- 基准 {market.benchmark_code} 首末收盘涨跌幅 {benchmark.iloc[-1] / benchmark.iloc[0] - 1:+.2%}。")
        holdings = "、".join(f"{code} {shares:g} 股" for code, shares in self.result.holdings.items())
        lines.append(f"- 期末持仓 {holdings or '无'}。")
        return "\n".join(lines)


class StrategySession:
    def __init__(self):
        self.versions: dict[str, CodeVersion] = {}
        self.version: CodeVersion | None = None
        self.dataset: Dataset | None = None
        self.pending: RunRequest | None = None
        self.starting_cash = DEFAULT_CASH
        self.commission = DEFAULT_COMMISSION
        self.runs: dict[str, CompletedRun] = {}
        self.in_progress = False
        self.lock = Lock()
