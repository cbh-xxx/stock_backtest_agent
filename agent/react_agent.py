import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.backtest_spec import StrategySpec, StrategyUpdate
from agent.tools.agent_tools import BacktestStageError, execute_backtest
from utils.logger_handler import logger
from utils.prompt_loader import load_backtest_prompts


class ReactAgent:
    def __init__(self):
        self._draft: StrategySpec | None = None

    def _parse(self, query: str, today: date) -> StrategyUpdate:
        from model.factory import chat_model

        schema = json.dumps(StrategyUpdate.model_json_schema(), ensure_ascii=False)
        context = json.dumps({
            "today": str(today),
            "draft": self._draft.model_dump(mode="json") if self._draft else None,
            "request": query,
        }, ensure_ascii=False)
        messages = [
            ("system", f"{load_backtest_prompts()}\n\nJSON Schema:\n{schema}"),
            ("human", context),
        ]
        # JSON 模式只提取参数，行情和回测步骤由程序直接调用。
        parser = chat_model.bind(
            response_format={"type": "json_object"}, max_tokens=8192,
        ) | StrOutputParser()
        response = parser.invoke(messages)
        # DeepSeek 的 JSON 模式偶尔返回空白内容，此时只重试一次。
        if not response.strip():
            response = parser.invoke(messages)
        return StrategyUpdate.model_validate_json(response)

    def execute_stream(self, query: str):
        today = datetime.now(timezone(timedelta(hours=8))).date()
        try:
            update = self._parse(query, today)
            if update.action == "unsupported":
                yield "这个入口用于股票策略回测。请描述要回测的标的、日期和交易规则。"
                return
            existing = self._draft if update.action == "update" else None
            fields = existing.model_dump() if existing else {}
            fields.update(update.changes.model_dump(exclude_unset=True))
            self._draft = StrategySpec.model_validate(fields)
        except ValidationError as exc:
            logger.warning("策略参数校验失败：%s", exc.errors(include_input=False))
            locations = "、".join(
                ".".join(map(str, issue["loc"])) or "策略 JSON"
                for issue in exc.errors(include_input=False)
            )
            yield f"**参数校验失败**：无法读取有效的策略参数（{locations}）。请检查输入后重试。"
            return
        except Exception as exc:
            logger.exception("DeepSeek 策略解析失败")
            yield f"**策略解析失败**：DeepSeek 未能返回参数（{type(exc).__name__}）。请检查 API 配置后重试。"
            return

        issues = self._draft.input_issues(today)
        if issues:
            yield "**需要补齐或调整回测参数**\n\n" + "\n".join(f"- {issue}" for issue in issues)
            return

        try:
            report = execute_backtest(self._draft)
        except BacktestStageError as exc:
            yield f"**{exc.stage}失败**：{exc}"
            return
        self._draft = None
        yield report


if __name__ == "__main__":
    agent = ReactAgent()
    while query := input("回测请求（空行退出）：").strip():
        for message in agent.execute_stream(query):
            print(message)
