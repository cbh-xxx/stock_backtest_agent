from collections.abc import Iterator
from textwrap import indent

from agent.strategy_rules import OPERATORS, Breakout, Condition, Group, MACross, RSI, StrategyRules, VolumeRatio


def _leaves(condition: Condition) -> Iterator[RSI | MACross | VolumeRatio | Breakout]:
    if isinstance(condition, Group):
        for child in condition.conditions:
            yield from _leaves(child)
    else:
        yield condition


def _expression(condition: Condition) -> str:
    if isinstance(condition, Group):
        joiner = " and\n" if condition.kind == "all" else " or\n"
        return "(\n" + indent(joiner.join(map(_expression, condition.conditions)), "    ") + "\n)"
    if isinstance(condition, RSI):
        return f"self.rsi_{condition.period}[data][0] {OPERATORS[condition.op]} {condition.threshold!r}"
    if isinstance(condition, VolumeRatio):
        average = f"avg_volume_{condition.period}"
        return f"({average} > 0 and data.volume[0] / {average} {OPERATORS[condition.op]} {condition.threshold!r})"
    if isinstance(condition, MACross):
        fast, slow = condition.fast, condition.slow
        before, now = ("<=", ">") if condition.direction == "up" else (">=", "<")
        return f"(previous_sma_{fast} {before} previous_sma_{slow} and sma_{fast} {now} sma_{slow})"
    line, extreme, op = ("high", "max", ">") if condition.direction == "above" else ("low", "min", "<")
    return f"data.close[0] {op} {extreme}(data.{line}.get(ago=-1, size={condition.period}))"


def _signal_method(name: str, condition: Condition) -> str:
    leaves = tuple(_leaves(condition))
    body = [f"    def {name}(self, data):"]
    for period in sorted({node.period for node in leaves if isinstance(node, VolumeRatio)}):
        body.append(f"        avg_volume_{period} = mean(data.volume.get(ago=-1, size={period}))")
    sma = {period for node in leaves if isinstance(node, MACross) for period in (node.fast, node.slow)}
    for period in sorted(sma):
        body.extend([
            f"        previous_sma_{period} = mean(data.close.get(ago=-1, size={period}))",
            f"        sma_{period} = mean(data.close.get(size={period}))",
        ])
    body.append("        return " + _expression(condition).replace("\n", "\n        "))
    return "\n".join(body)


def _exit_method(rules: StrategyRules) -> str:
    body = ["    def _exit_reason(self, data, position):"]
    if rules.stop_loss is not None or rules.take_profit is not None:
        body.append("        change = data.close[0] / position.price - 1")
    if rules.stop_loss is not None:
        body.extend([
            f"        if change <= -{rules.stop_loss!r} + 1e-12:",
            '            return "止损"',
        ])
    if rules.take_profit is not None:
        body.extend([
            f"        if change >= {rules.take_profit!r} - 1e-12:",
            '            return "止盈"',
        ])
    if rules.max_hold_days is not None:
        body.extend([
            "        held_days = len(data) - self.entry_bars[data] + 1",
            f"        if held_days >= {rules.max_hold_days!r}:",
            '            return "最长持有期"',
        ])
    if rules.exit is not None:
        body.extend([
            "        if self._exit_signal(data):",
            '            return "退出条件满足"',
        ])
    body.append("        return None")
    return "\n".join(body)


_NEXT = '''    def next(self):
        for data in self.datas:
            bar = len(data)
            if bar <= self.warmup_bars or bar == self.seen.get(data):
                continue
            self.seen[data] = bar
            day = data.datetime.date(0)
            if not self.trade_start <= day <= self.trade_end or data in self.pending:
                continue
            position = self.getposition(data)
            if position.size:
                reason = self._exit_reason(data, position)
                if reason:
                    self.pending[data] = self.close(data=data, reason=reason)
                continue
            if self._entry_met(data):
                budget = min(self.broker.getvalue() * self.position_fraction, self.broker.getcash())
                commission = self.broker.getcommissioninfo(data).p.commission
                shares = int(budget / (data.close[0] * (1 + commission)))
                if shares > 0:
                    self.pending[data] = self.buy(data=data, size=shares, reason="入场条件满足")

    # 个别标的尚未到达首个回测日时，其余预热完成的标的仍可按自身日线运行。
    prenext = next
    nextstart = next

    def notify_order(self, order):
        if order.alive():
            return
        self.pending.pop(order.data, None)'''

_TRACK_HOLDING = '''
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_bars[order.data] = len(order.data)
            else:
                self.entry_bars.pop(order.data, None)'''


def compile_strategy(rules: StrategyRules) -> tuple[str, int]:
    """把已验证的规则展开成执行器可加载的策略；只插入枚举选择和数值字面量。"""
    leaves = list(_leaves(rules.entry))
    if rules.exit is not None:
        leaves.extend(_leaves(rules.exit))
    warmup = max(node.slow if isinstance(node, MACross) else node.period for node in leaves)
    imports = ["import backtrader as bt"]
    if any(isinstance(node, (MACross, VolumeRatio)) for node in leaves):
        imports = ["from statistics import mean", "", *imports]
    setup = [
        "class GeneratedStrategy(bt.Strategy):",
        f"    warmup_bars = {warmup}",
        f"    position_fraction = {rules.position_fraction!r}",
        "",
        "    def __init__(self):",
        "        self.pending = {}",
        "        self.seen = {}",
    ]
    if rules.max_hold_days is not None:
        setup.append("        self.entry_bars = {}")
    for period in sorted({node.period for node in leaves if isinstance(node, RSI)}):
        setup.extend([
            f"        self.rsi_{period} = {{",
            f"            data: bt.indicators.RSI_Safe(data.close, period={period}) for data in self.datas",
            "        }",
        ])
    parts = ["\n".join(imports), "\n".join(setup), _signal_method("_entry_met", rules.entry)]
    if rules.exit is not None:
        parts.append(_signal_method("_exit_signal", rules.exit))
    parts.append(_exit_method(rules))
    parts.append(_NEXT + (_TRACK_HOLDING if rules.max_hold_days is not None else ""))
    source = "\n\n".join(parts) + "\n"
    compile(source, "strategy.py", "exec")
    return source, warmup
