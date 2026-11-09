from dataclasses import dataclass
from datetime import date
from math import floor
from statistics import mean

import backtrader as bt

from agent.backtest_spec import Signal, StrategySpec
from agent.market_data import MarketData


STARTING_CASH = 100_000.0
COMMISSION = 0.001


@dataclass(frozen=True)
class OrderEvent:
    symbol: str
    side: str
    signal_date: date
    reason: str
    fill_date: date | None = None
    shares: int = 0
    price: float = 0.0
    fee: float = 0.0


def _entry_met(signal: Signal, data) -> bool:
    # ago=-1 从上一根有效日线取历史数据，突破和放量基准均排除当天。
    if signal.kind == "volume":
        average = mean(data.volume.get(ago=-1, size=signal.period))
        return average > 0 and data.volume[0] >= average * signal.multiple
    if signal.kind == "breakout":
        return data.close[0] > max(data.high.get(ago=-1, size=signal.period))
    previous_fast = mean(data.close.get(ago=-1, size=signal.fast))
    previous_slow = mean(data.close.get(ago=-1, size=signal.slow))
    current_fast = mean(data.close.get(size=signal.fast))
    current_slow = mean(data.close.get(size=signal.slow))
    return previous_fast <= previous_slow and current_fast > current_slow


class RuleStrategy(bt.Strategy):
    params = (("spec", None),)

    def __init__(self):
        self.spec: StrategySpec = self.p.spec
        self.seen_bars = {}
        self.pending = {}
        self.entry_bars = {}
        self.fills: list[OrderEvent] = []
        self.unfilled: list[OrderEvent] = []
        self.peak = STARTING_CASH
        self.max_drawdown = 0.0
        self.warmup = max(signal.warmup() for signal in self.spec.entry)

    def _exit_reason(self, data, position) -> str | None:
        change = data.close[0] / position.price - 1
        holding_days = len(data) - self.entry_bars[data] + 1
        rules = {rule.kind: rule.value for rule in self.spec.exits or ()}
        if "stop_loss" in rules and change <= -rules["stop_loss"] + 1e-12:
            return "止损"
        if "take_profit" in rules and change >= rules["take_profit"] - 1e-12:
            return "止盈"
        if "max_hold" in rules and holding_days >= rules["max_hold"]:
            return "最长持有期"
        return None

    def next(self):
        for data in self.datas:
            # 多标的日期不同步时，某一标的的旧日线不能重复触发信号。
            bar = len(data)
            if bar <= self.seen_bars.get(data, 0):
                continue
            self.seen_bars[data] = bar
            signal_day = data.datetime.date(0)
            if signal_day < self.spec.start or bar <= self.warmup or data in self.pending:
                continue
            position = self.getposition(data)
            if position.size:
                reason = self._exit_reason(data, position)
                if reason:
                    self.pending[data] = self.sell(data=data, size=position.size, reason=reason)
                continue
            combine = any if self.spec.entry_logic == "any" else all
            if combine(_entry_met(signal, data) for signal in self.spec.entry):
                budget = min(self.broker.getvalue() * self.spec.position_fraction, self.broker.getcash())
                shares = floor(budget / (data.close[0] * (1 + COMMISSION)))
                if shares > 0:
                    self.pending[data] = self.buy(data=data, size=shares, reason="入场条件满足")
                else:
                    self.unfilled.append(OrderEvent(data._name, "买入", signal_day, "资金不足以买入 1 股"))

        if self.datetime.date(0) >= self.spec.start:
            equity = self.broker.getvalue()
            self.peak = max(self.peak, equity)
            self.max_drawdown = max(self.max_drawdown, (self.peak - equity) / self.peak)

    prenext = next
    nextstart = next

    def notify_order(self, order):
        if order.alive():
            return
        data = order.data
        self.pending.pop(data, None)
        signal_day = bt.num2date(order.created.dt).date()
        side = "买入" if order.isbuy() else "卖出"
        if order.status == order.Completed:
            if order.isbuy():
                self.entry_bars[data] = len(data)
            else:
                self.entry_bars.pop(data, None)
            self.fills.append(OrderEvent(
                data._name, side, signal_day, order.info["reason"],
                bt.num2date(order.executed.dt).date(), abs(int(order.executed.size)),
                order.executed.price, order.executed.comm,
            ))
        else:
            reason = "成交时可用资金不足" if order.status == order.Margin else order.getstatusname()
            self.unfilled.append(OrderEvent(data._name, side, signal_day, reason))

    def stop(self):
        for order in self.pending.values():
            self.unfilled.append(OrderEvent(
                order.data._name, "买入" if order.isbuy() else "卖出",
                bt.num2date(order.created.dt).date(), "区间结束前没有下一根可成交日线",
            ))


def _describe_signal(signal: Signal) -> str:
    if signal.kind == "volume":
        return f"当日成交量达到此前 {signal.period} 根日线平均成交量的 {signal.multiple:g} 倍"
    if signal.kind == "breakout":
        return f"当日收盘价高于此前 {signal.period} 根日线的最高价"
    return f"{signal.fast} 日均线上穿 {signal.slow} 日均线"


def _report(strategy: RuleStrategy, spec: StrategySpec, market: MarketData) -> str:
    equity = strategy.broker.getvalue()
    combine = " 或 " if spec.entry_logic == "any" else " 且 "
    lines = [
        "### 回测结果",
        f"- **实际标的**：{'、'.join(market.stocks)}",
        f"- **行情来源**：BaoStock 在线历史日线。**请求区间**：{spec.start} 至 {spec.end}",
        f"- **入场**：{combine.join(map(_describe_signal, spec.entry))}",
        f"- **仓位**：按信号日收盘估算股数，每只股票目标为账户权益的 {spec.position_fraction:.2%}",
    ]
    exits = []
    for rule in spec.exits or ():
        if rule.kind == "max_hold":
            exits.append(f"持有满 {int(rule.value)} 个交易日（成交日计第 1 日）")
        else:
            name = "止损" if rule.kind == "stop_loss" else "止盈"
            exits.append(f"{name} {rule.value:.2%}")
    lines += [
        f"- **退出**：{'、'.join(exits) if exits else '持有到回测结束'}",
        "- **成交口径**：前复权价格，只做多。收盘判断，下一可交易日开盘成交。"
        "买卖各收 0.1% 手续费。未平仓持仓按期末可用收盘价估值。",
        f"- **资金**：¥{STARTING_CASH:,.2f} → ¥{equity:,.2f}",
        f"- **总收益**：¥{equity - STARTING_CASH:+,.2f}（{equity / STARTING_CASH - 1:+.2%}）",
        f"- **最大回撤**：{strategy.max_drawdown:.2%}（按日终账户权益）",
        f"- **成交笔数**：{len(strategy.fills)}；"
        f"**总手续费**：¥{sum(event.fee for event in strategy.fills):,.2f}",
        "",
        "#### 实际可交易日线覆盖",
        "",
        "| 标的 | 首日 | 末日 | 日线数 |",
        "|---|---|---|---:|",
    ]
    for code, frame in market.stocks.items():
        actual = frame.loc[frame.index.date >= spec.start]
        lines.append(f"| {code} | {actual.index[0].date()} | {actual.index[-1].date()} | {len(actual)} |")
    if market.benchmark is not None:
        index = market.benchmark
        lines += [
            "",
            f"**基准 {market.benchmark_code}**：{index.index[0].date()} 至 {index.index[-1].date()}，"
            f"首末收盘涨跌幅 {index.iloc[-1] / index.iloc[0] - 1:+.2%}，仅供比较。",
        ]
    holdings = [
        f"{data._name} {strategy.getposition(data).size} 股"
        for data in strategy.datas if strategy.getposition(data).size
    ]
    lines += ["", f"**期末持仓**：{'、'.join(holdings) if holdings else '无'}", "", "#### 成交明细", ""]
    if strategy.fills:
        lines += [
            "| 标的 | 方向 | 信号日 | 成交日 | 股数 | 前复权成交价 | 手续费 | 原因 |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
        for event in strategy.fills:
            lines.append(
                f"| {event.symbol} | {event.side} | {event.signal_date} | {event.fill_date} | "
                f"{event.shares} | ¥{event.price:,.4f} | ¥{event.fee:,.2f} | {event.reason} |"
            )
    else:
        lines.append("零笔成交。")
    if strategy.unfilled:
        lines += ["", "#### 未成交信号", "", "| 标的 | 方向 | 信号日 | 原因 |", "|---|---|---|---|"]
        for event in strategy.unfilled:
            lines.append(f"| {event.symbol} | {event.side} | {event.signal_date} | {event.reason} |")
    return "\n".join(lines)


def run_backtest(spec: StrategySpec, market: MarketData) -> str:
    engine = bt.Cerebro(stdstats=False)
    engine.broker.setcash(STARTING_CASH)
    engine.broker.setcommission(commission=COMMISSION)
    for code, frame in market.stocks.items():
        engine.adddata(bt.feeds.PandasData(dataname=frame), name=code)
    # 指数不进入交易引擎，成交标的始终是用户指定或默认的股票。
    engine.addstrategy(RuleStrategy, spec=spec)
    strategy = engine.run(runonce=False)[0]
    return _report(strategy, spec, market)
