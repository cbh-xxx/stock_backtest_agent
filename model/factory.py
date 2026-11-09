from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek

import os
import sys
# 添加项目根目录到 Python 路径，解决直接运行时的导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录
sys.path.insert(0, project_root)
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

# Agent、RAG 总结和策略生成共用此模型，密钥从 DEEPSEEK_API_KEY 读取。
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatDeepSeek(
            model=rag_conf["chat_model_name"],
            base_url="https://api.deepseek.com",
            # 使用非思考模式，工具调用后的请求无需回传 reasoning_content。
            extra_body={"thinking": {"type": "disabled"}},
        )

# 嵌入模型工厂
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
