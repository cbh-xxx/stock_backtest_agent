import os
import sys
current_dir = os.path.dirname(os.path.dirname(__file__))
project_root = os.path.dirname(current_dir)  # Agent 项目目录sys.path.insert(0, project_root)
sys.path.insert(0, project_root)
from langchain.agents import create_agent
from model.factory import chat_model
from agent.tools.middleware import monitor_tool, log_before_model
from langchain_core.tools import tool


@tool(description="获取小明的英文名")
def get_X() -> str:
    return '杰克'

@tool(description="获取小王的英文名")
def get_Y() -> str:
    return '莱特'

@tool(description="使用小明和小王的英文名获取二人的关系")
def get_XY(X: str, Y: str) -> str:
    return f'{X}和{Y}是朋友'

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
        model=chat_model,
        tools=[get_X, get_Y, get_XY],
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


    for chunk in agent.execute_stream('''请告诉我小明和小王的英文名字，并且他们之间的关系是什么？'''):
        print(chunk, end="", flush=True)
