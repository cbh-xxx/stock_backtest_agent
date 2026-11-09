import json
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator


MAX_WARMUP = 2520
_MAX_RULE_TEXT = 16_000
_MAX_NODES = 64
_MAX_DEPTH = 6
Period = Annotated[int, Field(strict=True, ge=1, le=MAX_WARMUP)]
Compare = Literal["lt", "lte", "gt", "gte"]
OPERATORS = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}


class RuleFormatError(ValueError):
    pass


class RuleValidationError(ValueError):
    pass


class RuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Group(RuleModel):
    kind: Literal["all", "any"]
    conditions: tuple["Condition", ...] = Field(min_length=1, max_length=_MAX_NODES)


class RSI(RuleModel):
    kind: Literal["rsi"]
    period: int = Field(strict=True, ge=2, le=MAX_WARMUP)
    op: Compare
    threshold: float = Field(strict=True, ge=0, le=100)


class MACross(RuleModel):
    kind: Literal["ma_cross"]
    fast: Period
    slow: Period
    direction: Literal["up", "down"]

    @field_validator("slow")
    @classmethod
    def check_slow(cls, value: int, info: ValidationInfo) -> int:
        if "fast" in info.data and value <= info.data["fast"]:
            raise ValueError("慢线周期必须大于快线周期。")
        return value


class VolumeRatio(RuleModel):
    kind: Literal["volume_ratio"]
    period: Period
    op: Compare
    threshold: float = Field(strict=True, ge=0)


class Breakout(RuleModel):
    kind: Literal["breakout"]
    period: Period
    direction: Literal["above", "below"]


Condition = Annotated[Union[Group, RSI, MACross, VolumeRatio, Breakout], Field(discriminator="kind")]
Group.model_rebuild()


class StrategyRules(RuleModel):
    entry: Condition
    exit: Condition | None = None
    position_fraction: float = Field(strict=True, gt=0, le=1)
    stop_loss: float | None = Field(default=None, strict=True, gt=0, lt=1)
    take_profit: float | None = Field(default=None, strict=True, gt=0)
    max_hold_days: int | None = Field(default=None, strict=True, ge=1)
    hold_to_end: bool = Field(default=False, strict=True)

    def as_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)

    def summary(self) -> list[str]:
        exits = [_describe(self.exit)] if self.exit is not None else []
        if self.stop_loss is not None:
            exits.append(f"较实际买入价下跌 {self.stop_loss:.2%} 止损")
        if self.take_profit is not None:
            exits.append(f"较实际买入价上涨 {self.take_profit:.2%} 止盈")
        if self.max_hold_days is not None:
            exits.append(f"持有满 {self.max_hold_days} 根可交易日线（买入成交日计第 1 日）")
        return [
            f"买入：{_describe(self.entry)}",
            f"仓位：每只股票目标为账户权益的 {self.position_fraction:.2%}，受可用现金限制",
            "卖出：" + ("任一触发（" + "；".join(exits) + "）" if len(exits) > 1 else
                        exits[0] if exits else "持有到期末，按末日可用收盘价估值"),
        ]


class ModelIssue(RuleModel):
    field: str = Field(min_length=1, max_length=160)
    message: str = Field(min_length=1, max_length=400)


class ModelIssues(RuleModel):
    issues: tuple[ModelIssue, ...] = Field(min_length=1, max_length=12)


def _describe(condition: Condition) -> str:
    if isinstance(condition, Group):
        joiner = " 且 " if condition.kind == "all" else " 或 "
        return "(" + joiner.join(map(_describe, condition.conditions)) + ")"
    if isinstance(condition, RSI):
        return f"{condition.period} 日 RSI {OPERATORS[condition.op]} {condition.threshold:g}"
    if isinstance(condition, MACross):
        direction = "上穿" if condition.direction == "up" else "下穿"
        return f"{condition.fast} 日均线{direction} {condition.slow} 日均线"
    if isinstance(condition, VolumeRatio):
        return f"当日成交量 / 前 {condition.period} 根日线均量 {OPERATORS[condition.op]} {condition.threshold:g}"
    boundary = "最高价" if condition.direction == "above" else "最低价"
    compare = ">" if condition.direction == "above" else "<"
    return f"当日收盘价 {compare} 前 {condition.period} 根日线{boundary}"


def _pairs(pairs):
    data = {}
    for name, value in pairs:
        if name in data:
            raise RuleFormatError(f"JSON 存在重复字段 {name}。")
        data[name] = value
    return data


def _non_json_number(value):
    raise RuleFormatError(f"JSON 不支持数值 {value}。")


def _field_path(loc: tuple) -> str:
    tags = {"all", "any", "rsi", "ma_cross", "volume_ratio", "breakout"}
    path = ""
    for part in loc:
        if isinstance(part, int):
            path += f"[{part}]"
        elif part not in tags:
            path += ("." if path else "") + part
    return path or "规则"


def _check_tree(rules: StrategyRules):
    stack = [(rules.entry, "entry", 1)]
    if rules.exit is not None:
        stack.append((rules.exit, "exit", 1))
    count = 0
    while stack:
        condition, path, depth = stack.pop()
        count += 1
        if depth > _MAX_DEPTH:
            raise RuleValidationError(f"{path}：条件嵌套最多 {_MAX_DEPTH} 层。")
        if count > _MAX_NODES:
            raise RuleValidationError(f"{path}：买卖条件合计最多 {_MAX_NODES} 个节点。")
        if isinstance(condition, Group):
            stack.extend(
                (child, f"{path}.conditions[{index}]", depth + 1)
                for index, child in enumerate(condition.conditions)
            )


def parse_rules(text: str, *, from_model: bool = False) -> StrategyRules:
    if not text.strip():
        raise RuleFormatError("规则 JSON 为空。")
    if len(text) > _MAX_RULE_TEXT:
        raise RuleFormatError("规则 JSON 超过长度限制。")
    try:
        payload = json.loads(text, object_pairs_hook=_pairs, parse_constant=_non_json_number)
    except json.JSONDecodeError as exc:
        reason = "完整 JSON 后还有多余字符" if exc.msg == "Extra data" else exc.msg
        raise RuleFormatError(f"第 {exc.lineno} 行第 {exc.colno} 列：{reason}。") from None
    except RecursionError:
        raise RuleFormatError("规则 JSON 的嵌套过深。") from None
    except RuleFormatError:
        raise
    except ValueError:
        raise RuleFormatError("规则 JSON 包含无法读取的数值。") from None
    try:
        if from_model and isinstance(payload, dict) and "issues" in payload:
            issues = ModelIssues.model_validate(payload).issues
            raise RuleValidationError("；".join(f"{item.field}：{item.message}" for item in issues))
        rules = StrategyRules.model_validate(payload)
    except ValidationError as exc:
        details = []
        for error in exc.errors(include_input=False, include_url=False)[:8]:
            path = _field_path(error["loc"])
            if error["type"] == "missing":
                message = "缺少必填字段。"
            elif error["type"] == "extra_forbidden":
                message = "不支持此字段。"
            elif error["type"] in {"union_tag_invalid", "union_tag_not_found"}:
                path += ".kind"
                message = "仅支持 all、any、rsi、ma_cross、volume_ratio 或 breakout。"
            else:
                message = error["msg"]
            details.append(f"{path}：{message}")
        raise RuleValidationError("；".join(details)) from None
    has_exit = any(value is not None for value in (
        rules.exit, rules.stop_loss, rules.take_profit, rules.max_hold_days,
    ))
    if has_exit == rules.hold_to_end:
        if has_exit:
            raise RuleValidationError("hold_to_end：与 exit、止损、止盈或最长持有期不能同时启用。")
        raise RuleValidationError("exit：请指定退出条件或风控阈值，或明确设置 hold_to_end 为 true。")
    _check_tree(rules)
    return rules
