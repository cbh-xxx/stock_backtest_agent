import streamlit as st

from agent.react_agent import ReactAgent


st.title("股票智能回测系统")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "message" not in st.session_state:
    st.session_state["message"] = []

history = st.session_state["message"]
for message in history:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("描述你的回测策略"):
    st.chat_message("user").markdown(prompt)
    history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"), st.spinner("正在处理回测请求…"):
        for message in st.session_state["agent"].execute_stream(prompt):
            st.markdown(message)
            history.append({"role": "assistant", "content": message})
