"""独立子进程入口。只由确认后的运行请求启动。"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import backtrader as bt
import pandas as pd


class DailyBroker(bt.brokers.BackBroker):
    params = (("trade_start", None), ("trade_end", None))

    def init(self):
        super().init()
        self.active_by_data = {}

    def submit(self, order, check=True):
        reason = None
        day = bt.num2date(order.created.dt).date()
        active = self.active_by_data.get(order.data)
        if not self.p.trade_start <= day <= self.p.trade_end:
            reason = "预热期或回测区间外不下单"
        elif order.exectype != bt.Order.Market or order.parent or not order.transmit:
            reason = "执行器仅接受独立市价订单"
        elif active is not None and active.alive():
            reason = "同一标的有未结束的订单"
        elif order.issell() and abs(order.size) > self.getposition(order.data).size:
            reason = "只做多，卖出数量不能超过当前持仓"
        if reason:
            order.addinfo(runner_reason=reason)
            order.reject(self)
            self.notify(order)
            return order
        self.active_by_data[order.data] = order
        return super().submit(order, check=check)


class Capture(bt.Analyzer):
    def start(self):
        self.equity = {}
        self.fills = []
        self.unfilled = []
        self.pending = {}

    def next(self):
        day = self.strategy.datetime.date(0)
        if self.strategy.trade_start <= day <= self.strategy.trade_end:
            self.equity[day.isoformat()] = float(self.strategy.broker.getvalue())

    def notify_order(self, order):
        entry = {
            "symbol": order.data._name,
            "side": "买入" if order.isbuy() else "卖出",
            "signal_date": bt.num2date(order.created.dt).date().isoformat(),
            "reason": str(order.info.get("runner_reason") or order.info.get("reason") or "")[:160],
        }
        if order.alive():
            self.pending[order.ref] = entry
            return
        self.pending.pop(order.ref, None)
        if order.status == order.Completed:
            self.fills.append({
                **entry,
                "fill_date": bt.num2date(order.executed.dt).date().isoformat(),
                "shares": abs(float(order.executed.size)),
                "price": float(order.executed.price),
                "fee": float(order.executed.comm),
            })
        else:
            entry["reason"] = entry["reason"] or order.getstatusname()
            self.unfilled.append(entry)

    def get_analysis(self):
        return {
            "equity": [{"date": day, "value": value} for day, value in sorted(self.equity.items())],
            "fills": self.fills,
            "unfilled": self.unfilled + [
                {**entry, "reason": "回测区间结束，订单未成交"} for entry in self.pending.values()
            ],
            "holdings": {
                data._name: float(self.strategy.getposition(data).size)
                for data in self.strategy.datas if self.strategy.getposition(data).size
            },
        }


def main(workdir: Path):
    request = json.loads((workdir / "request.json").read_text(encoding="utf-8"))
    start = date.fromisoformat(request["start"])
    end = date.fromisoformat(request["end"])
    # 已确认的策略源码只在这个子进程中加载。
    spec = importlib.util.spec_from_file_location("user_strategy", workdir / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    strategy = getattr(module, "GeneratedStrategy", None)
    if not isinstance(strategy, type) or not issubclass(strategy, bt.Strategy):
        raise ValueError("GeneratedStrategy 必须继承 backtrader.Strategy。")
    if strategy.warmup_bars != request["warmup_bars"]:
        raise ValueError("策略运行时的预热日线数与已确认值不一致。")
    strategy.trade_start = start
    strategy.trade_end = end

    engine = bt.Cerebro(stdstats=False)
    broker = DailyBroker(trade_start=start, trade_end=end)
    broker.setcash(request["starting_cash"])
    broker.setcommission(commission=request["commission"])
    engine.setbroker(broker)
    for code, filename in request["stocks"].items():
        frame = pd.read_csv(workdir / filename, index_col="date", parse_dates=["date"])
        engine.adddata(bt.feeds.PandasData(dataname=frame), name=code)
    engine.addstrategy(strategy)
    engine.addanalyzer(Capture, _name="capture")
    finished = engine.run(runonce=False)[0]
    result = finished.analyzers.capture.get_analysis()
    (workdir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
