"""
yaml
k: v
"""
import os
import sys
import yaml

# 添加项目根目录到 Python 路径，解决直接运行时的导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录
sys.path.insert(0, project_root)
from utils.path_tool import get_abs_path

# RAG配置文件
def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 向量库配置文件
def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 提示词配置文件
def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# agent配置文件
def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
