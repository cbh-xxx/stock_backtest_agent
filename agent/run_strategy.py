import json
import os
import subprocess
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, ConfigDict, Field

from agent.market_data import MarketData


TIMEOUT_SECONDS = 60
_RUNNER = Path(__file__).with_name("strategy_runner.py").resolve()
_ENV_NAMES = {
    "APPDATA", "COMSPEC", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH",
    "PATHEXT", "PROGRAMDATA", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP",
    "USERPROFILE", "WINDIR",
}


class RunnerError(RuntimeError):
    pass


class ResultRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class EquityPoint(ResultRow):
    date: date
    value: float


class OrderRecord(ResultRow):
    symbol: str
    side: str
    signal_date: date
    reason: str
    order_ref: int | None = None
    status: str | None = None
    failure_reason: str | None = None
    signal_detail: dict | None = None


class Fill(OrderRecord):
    fill_date: date
    shares: float = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(ge=0)


class RunResult(ResultRow):
    equity: list[EquityPoint] = Field(min_length=1)
    fills: list[Fill]
    unfilled: list[OrderRecord]
    holdings: dict[str, float]

    def drawdowns(self, starting_cash: float) -> list[float]:
        peak = starting_cash
        values = []
        for point in self.equity:
            peak = max(peak, point.value)
            values.append((peak - point.value) / peak)
        return values


def _runner_python() -> Path:
    configured = os.environ.get("BACKTEST_RUNNER_PYTHON")
    if not configured:
        raise RunnerError("请配置 BACKTEST_RUNNER_PYTHON，指向已安装 Backtrader、pandas 和 NumPy 的 python.exe。")
    path = Path(configured).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise RunnerError("BACKTEST_RUNNER_PYTHON 必须指向存在的 Python 可执行文件的绝对路径。")
    return path.resolve()


def _runner_env(python: Path) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name.upper() in _ENV_NAMES}
    prefix = python.parent
    env["PATH"] = os.pathsep.join([
        str(prefix), str(prefix / "Library" / "bin"), str(prefix / "Scripts"),
        env.get("PATH", ""),
    ])
    return env


def _tail(path: Path) -> str:
    with path.open("rb") as log:
        size = log.seek(0, os.SEEK_END)
        log.seek(max(0, size - 2048))
        return log.read().decode("utf-8", errors="replace").strip()


def run_approved_strategy(
    source: str, warmup_bars: int, market: MarketData, start: date, end: date,
    starting_cash: float, commission: float,
) -> RunResult:
    python = _runner_python()
    with TemporaryDirectory(prefix="backtest-run-", ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        (root / "strategy.py").write_text(source, encoding="utf-8")
        stocks = {}
        for index, (code, frame) in enumerate(market.stocks.items()):
            filename = f"stock-{index}.csv"
            stocks[code] = filename
            frame.to_csv(root / filename, index_label="date", date_format="%Y-%m-%d")
        (root / "request.json").write_text(json.dumps({
            "start": str(start), "end": str(end), "warmup_bars": warmup_bars,
            "starting_cash": starting_cash, "commission": commission, "stocks": stocks,
        }), encoding="utf-8")
        log_path = root / "runner.log"
        try:
            with log_path.open("wb") as log:
                process = subprocess.run(
                    [str(python), "-I", "-X", "utf8", str(_RUNNER), str(root)],
                    cwd=root, env=_runner_env(python), timeout=TIMEOUT_SECONDS,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"回测超过 {TIMEOUT_SECONDS} 秒，已终止子进程。") from exc
        except OSError as exc:
            raise RunnerError(f"无法启动策略子进程：{exc}") from exc
        if process.returncode:
            raise RunnerError(_tail(log_path) or f"子进程退出码 {process.returncode}。")
        try:
            result = RunResult.model_validate_json((root / "result.json").read_bytes())
        except (ValueError, OSError) as exc:
            raise RunnerError("子进程没有返回有效的回测数据。") from exc
        days = [point.date for point in result.equity]
        if days != sorted(set(days)) or days[0] < start or days[-1] > end:
            raise RunnerError("回测权益序列与确认的日期区间不一致。")
        return result
