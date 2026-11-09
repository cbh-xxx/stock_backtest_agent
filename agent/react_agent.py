import os
import sys
# 添加项目根目录到 Python 路径，解决直接运行时的导入问题
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录sys.path.insert(0, project_root)
sys.path.insert(0, project_root)
from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_backtest_prompts
from agent.tools.agent_tools import (parse_strategy_with_rag, execute_backtest, get_complete_market_data)
# from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
#                                      get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
        model=chat_model,
        system_prompt=load_backtest_prompts(),  # 需要创建回测专用提示词
        tools=[execute_backtest],
        middleware=[monitor_tool, log_before_model],)

        # self.agent = create_agent(
        #     model=chat_model,
        #     system_prompt=load_system_prompts(),
        #     tools=[rag_summarize, get_weather, get_user_location, get_user_id,
        #            get_current_month, fetch_external_data, fill_context_for_report],
        #     middleware=[monitor_tool, log_before_model, report_prompt_switch],)

    # 流式执行用户提示词
    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是middlewear.py中上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream('''我想测试一个基于成交量放量的突破策略。具体来说，我希望策略能够监控每只股票的成交量变化，当某只股票的成交量突然放大，并且当天的收盘价也突破了最近20个交易日的最高价时，就买入这只股票。买入时使用账户总资金的5%，并且设置止损条件，如果股价从买入价下跌了8%就卖出止损。如果没有触发止损，就持有股票直到股价上涨了10%时止盈卖出。如果股价既没有达到止损也没有达到止盈，就持有最多10个交易日，无论盈亏都卖出。回测的时间范围是从2023年6月1日到2023年12月31日。这个策略应该同时监控多只股票，但每只股票独立计算信号和持仓。'''):
        print(chunk, end="", flush=True)

