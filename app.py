import time

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("股票智能回测系统")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# 显示所有历史消息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 收集所有消息，包括思考过程
    all_messages = []

    with st.spinner("回测系统思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        # 创建一个容器来显示思考过程
        thinking_container = st.container()
        tool_container = st.container()
        result_container = st.container()

        for message in res_stream:
            # 显示思考过程
            if isinstance(message, str) and ("思考" in message or "分析" in message):
                with thinking_container:
                    st.chat_message("assistant").write(message)
                    st.session_state["message"].append({"role": "assistant", "content": message})

            # 显示工具调用信息
            elif isinstance(message, str) and ("调用" in message or "工具" in message):
                with tool_container:
                    st.chat_message("assistant").write(message)
                    st.session_state["message"].append({"role": "assistant", "content": message})

            # 显示最终结果
            elif isinstance(message, str):
                with result_container:
                    st.chat_message("assistant").write(message)
                    st.session_state["message"].append({"role": "assistant", "content": message})

    st.rerun()
