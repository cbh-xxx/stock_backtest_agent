import json
import re
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from threading import Lock

from langchain_core.output_parsers import StrOutputParser

from agent.strategy_rules import RuleFormatError, StrategyRules, parse_rules
from utils.logger_handler import logger


PROJECT = Path(__file__).resolve().parents[1]
KNOWLEDGE = PROJECT / "data" / "strategy_codegen.md"
_INDEX_LOCK = Lock()


class CodegenError(ValueError):
    pass


@lru_cache(maxsize=2)
def _open_index(content: str):
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from model.factory import EmbeddingsFactory
    from utils.config_handler import rag_conf

    sections = re.split(r"^## ", content, flags=re.MULTILINE)[1:]
    docs = []
    for section in sections:
        heading, _, body = section.partition("\n")
        docs.append(Document(
            page_content=heading + "\n" + body.strip(),
            metadata={"source": f"data/strategy_codegen.md · {heading.strip()}"},
        ))
    if not docs:
        raise CodegenError("策略知识库没有可检索的章节。")
    version = sha256((rag_conf["embedding_model_name"] + "\n" + content).encode("utf-8")).hexdigest()[:16]
    store = Chroma(
        collection_name=f"strategy_codegen_{version}",
        persist_directory=str(PROJECT / "chroma_db"),
        embedding_function=EmbeddingsFactory().generator(),
    )
    ids = [f"section-{index}" for index in range(len(docs))]
    existing = set(store.get(ids=ids)["ids"])
    missing = [(doc_id, doc) for doc_id, doc in zip(ids, docs) if doc_id not in existing]
    if missing:
        store.add_documents([doc for _, doc in missing], ids=[doc_id for doc_id, _ in missing])
    return store


def _retrieve(query: str) -> list[dict[str, str]]:
    try:
        with _INDEX_LOCK:
            store = _open_index(KNOWLEDGE.read_text(encoding="utf-8"))
        docs = store.similarity_search(query, k=3)
        if not docs:
            raise CodegenError("策略知识库未返回参考资料。")
        return [{"source": doc.metadata["source"], "text": doc.page_content} for doc in docs]
    except ModuleNotFoundError as exc:
        raise CodegenError(
            f"RAG 缺少 Python 模块 `{exc.name}`。请在项目目录中使用启动 Streamlit 的 Python "
            "运行 `python -m pip install -r requirements.txt`，然后重启网页。"
        ) from exc
    except CodegenError:
        raise
    except Exception as exc:
        raise CodegenError(f"RAG 检索失败（{type(exc).__name__}）。请检查知识库及百炼密钥。") from exc


def write_strategy(
    requirements: str, previous: StrategyRules | None = None, change: str | None = None,
) -> tuple[StrategyRules, tuple[str, ...]]:
    # 检索始终由程序完成，模型只接收检索片段并输出规则 JSON。
    query = requirements
    if previous is not None:
        query = f"修改要求：{change}\n当前策略：" + "\n".join(previous.summary())
    refs = _retrieve(query[:4000])
    from model.factory import chat_model

    prompt = (PROJECT / "prompts" / "strategy_generator_prompt.txt").read_text(encoding="utf-8")
    prompt += "\n规则 JSON Schema：\n" + json.dumps(StrategyRules.model_json_schema(), ensure_ascii=False)
    payload = json.dumps({
        "strategy_requirements": requirements,
        "previous_rules": previous.model_dump(mode="json", exclude_none=True) if previous else None,
        "change_request": change, "retrieved_references": refs,
    }, ensure_ascii=False)
    writer = chat_model.bind(response_format={"type": "json_object"}, max_tokens=8192) | StrOutputParser()
    correction = ""
    for attempt in range(2):
        try:
            response = writer.invoke([("system", prompt + correction), ("human", payload)])
        except Exception as exc:
            raise CodegenError(f"DeepSeek 策略规则请求失败（{type(exc).__name__}）。") from exc
        try:
            rules = parse_rules(response, from_model=True)
        except RuleFormatError as exc:
            if attempt == 1:
                raise CodegenError(f"DeepSeek 重试后仍返回无效 JSON：{exc}") from exc
            logger.warning("DeepSeek 规则 JSON 格式错误，重试一次：%s", exc)
            correction = f"\n上一次返回的格式错误：{exc} 请重新输出一个完整 JSON 对象，不附加任何内容。"
        else:
            return rules, tuple(ref["source"] for ref in refs)
