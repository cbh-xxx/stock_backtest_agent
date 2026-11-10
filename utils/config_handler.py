import yaml

from utils.path_tool import get_abs_path


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as config:
        return yaml.load(config, Loader=yaml.FullLoader)


rag_conf = load_rag_config()
