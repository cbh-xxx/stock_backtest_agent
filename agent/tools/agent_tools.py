from agent.backtest_engine import run_backtest
from agent.backtest_spec import BacktestInputError, StrategySpec
from agent.market_data import MarketDataError, load_market_data
from utils.logger_handler import logger


class BacktestStageError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def execute_backtest(spec: StrategySpec) -> str:
    """对已通过校验的规则取数并运行固定的 Python 回测引擎。"""
    try:
        market = load_market_data(spec)
    except BacktestInputError as exc:
        raise BacktestStageError("参数校验", str(exc)) from exc
    except MarketDataError as exc:
        raise BacktestStageError("行情获取", str(exc)) from exc
    except Exception as exc:
        logger.exception("BaoStock 行情获取失败")
        raise BacktestStageError("行情获取", f"BaoStock 请求或数据格式异常：{exc}") from exc

    try:
        return run_backtest(spec, market)
    except Exception as exc:
        logger.exception("Backtrader 执行失败")
        raise BacktestStageError("引擎执行", str(exc)) from exc
