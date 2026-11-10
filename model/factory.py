from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek

from utils.config_handler import rag_conf


def create_chat_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=rag_conf["chat_model_name"],
        base_url="https://api.deepseek.com",
        # 非思考模式的工具调用不需要回传 reasoning_content。
        extra_body={"thinking": {"type": "disabled"}},
    )


def create_embeddings() -> Embeddings:
    from langchain_community.embeddings import DashScopeEmbeddings

    return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = create_chat_model()
