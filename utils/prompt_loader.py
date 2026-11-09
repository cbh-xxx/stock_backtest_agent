import os
import sys
# 添加项目根目录到 Python 路径，解决直接运行时的导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录
sys.path.insert(0, project_root)
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# 一共有三个分工不同的智能体，每个智能体都要有提示词定义各自的工作

# 客服agent提示词加载
def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

# 检索参考资料agent提示词加载
def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG总结提示词出错，{str(e)}")
        raise e

# 输出报告agent提示词加载
def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e
# 策略生成agent提示词加载
def load_strategy_generator_prompts():
    try:
        strategy_generator_prompt_path = get_abs_path(prompts_conf["strategy_generator_prompt_path"])
    except KeyError as e:
        logger.error(f"[strategy_generator_prompt_path]在yaml配置项中没有strategy_generator_prompt_path配置项")
        raise e

    try:
        return open(strategy_generator_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_strategy_generator_prompts]解析策略生成提示词出错，{str(e)}")
        raise e
    
# 回测agent提示词加载
def load_backtest_prompts():
    try:
        backtest_prompt_path = get_abs_path(prompts_conf["backtest_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_backtest_prompts]在yaml配置项中没有backtest_prompt_path配置项")
        raise e

    try:
        return open(backtest_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_backtest_prompts]解析回测提示词出错，{str(e)}")
        raise e

if __name__ == '__main__':
    print(load_report_prompts())

