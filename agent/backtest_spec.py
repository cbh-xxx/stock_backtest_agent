from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_STOCKS = ("sh.600000", "sh.600519", "sz.000001")
EARLIEST_DATE = date(1990, 1, 1)
BENCHMARK_CODES = {
    "上证指数": "sh.000001",
    "沪深300": "sh.000300",
    "中证500": "sh.000905",
    "深证成指": "sz.399001",
}


class BacktestInputError(ValueError):
    pass


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class Signal(InputModel):
    kind: Literal["volume", "breakout", "ma_cross"]
    period: int | None = Field(default=None, ge=1, le=2520)
    multiple: float | None = Field(default=None, gt=0)
    fast: int | None = Field(default=None, ge=1, le=2520)
    slow: int | None = Field(default=None, ge=2, le=2520)

    def warmup(self) -> int:
        return self.slow if self.kind == "ma_cross" else self.period


class ExitRule(InputModel):
    kind: Literal["stop_loss", "take_profit", "max_hold"]
    value: float | None = Field(default=None, gt=0)


SIGNAL_NAMES = {"volume": "放量", "breakout": "价格突破", "ma_cross": "均线上穿"}
EXIT_NAMES = {"stop_loss": "止损", "take_profit": "止盈", "max_hold": "最长持有期"}


class StrategySpec(InputModel):
    stocks: list[str] | None = None
    benchmark: str | None = None
    start: date | None = None
    end: date | None = None
    entry: list[Signal] | None = None
    entry_logic: Literal["all", "any"] | None = None
    position_fraction: float | None = Field(default=None, gt=0, le=1)
    exits: list[ExitRule] | None = None
    hold_to_end: bool | None = None
    unsupported: list[str] = Field(default_factory=list)

    def input_issues(self, today: date) -> list[str]:
        issues = [f"暂不支持：{item}" for item in self.unsupported]
        if not self.start or not self.end:
            issues.append("请提供回测开始和结束日期。")
        elif not EARLIEST_DATE <= self.start <= self.end <= today:
            issues.append("开始日期须在 1990 年或之后且不晚于结束日期，结束日期不能在未来。")
        if self.stocks is not None and (not self.stocks or any(not s for s in self.stocks)):
            issues.append("请给出明确的股票代码或名称。")
        if self.benchmark is not None and not self.benchmark:
            issues.append("请指定指数基准的名称或代码。")
        if not self.entry:
            issues.append("请提供入场条件，例如放量、价格突破或均线上穿。")
        if self.entry and len(self.entry) > 1 and not self.entry_logic:
            issues.append("多个入场条件需要指定全部满足（且）或任一满足（或）。")
        for signal in self.entry or ():
            if signal.kind == "ma_cross":
                if signal.fast is None or signal.slow is None:
                    issues.append("均线上穿需要指定快线和慢线的周期。")
                elif signal.fast >= signal.slow:
                    issues.append("快线周期必须小于慢线周期。")
            else:
                if signal.period is None:
                    issues.append(f"请指定{SIGNAL_NAMES[signal.kind]}的观察周期。")
                if signal.kind == "volume" and signal.multiple is None:
                    issues.append("请指定成交量相对于历史均量的放量倍数。")
        if self.position_fraction is None:
            issues.append("请指定每只股票建仓时使用的账户权益比例。")
        if not self.exits and not self.hold_to_end:
            issues.append("请指定至少一项退出规则，或明确持有到回测结束。")
        if self.exits and self.hold_to_end:
            issues.append("请选择按退出规则卖出，或无条件持有到回测结束。")
        exit_kinds = [rule.kind for rule in self.exits or ()]
        if len(set(exit_kinds)) != len(exit_kinds):
            issues.append("每种退出规则只能指定一个阈值。")
        for rule in self.exits or ():
            if rule.value is None:
                issues.append(f"请指定{EXIT_NAMES[rule.kind]}的具体阈值。")
            elif rule.kind == "max_hold" and not rule.value.is_integer():
                issues.append("最长持有期必须是整数个交易日。")
            elif rule.kind == "stop_loss" and rule.value >= 1:
                issues.append("止损比例须小于 100%。")
        return list(dict.fromkeys(issues))


class StrategyUpdate(InputModel):
    action: Literal["new", "update", "unsupported"]
    changes: StrategySpec = Field(default_factory=StrategySpec)
