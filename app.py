import re

import streamlit as st

from agent.backtest_spec import DEFAULT_STOCKS, EARLIEST_DATE
from agent.react_agent import ReactAgent


st.title("股票智能回测系统")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "message" not in st.session_state:
    st.session_state["message"] = []

agent = st.session_state["agent"]
session = agent.session
history = st.session_state["message"]
for message in history:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("描述你的回测策略"):
    st.chat_message("user").markdown(prompt)
    history.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"), st.spinner("正在处理回测请求…"):
        for message in agent.execute_stream(prompt):
            st.markdown(message)
            history.append({"role": "assistant", "content": message})

if agent.events:
    with st.expander("本轮工具调用"):
        for tool, status in agent.events:
            st.text(f"{tool} · {status}")

if version := session.version:
    st.subheader(f"策略规则 {version.id}")
    st.caption(f"预热 {version.warmup_bars} 根日线")
    for line in version.rules.summary():
        st.text(line)
    rules_json = version.rules.as_json()
    with st.expander("编辑策略规则 JSON"):
        edited = st.text_area(
            "规则 JSON", value=rules_json, height=340, key=f"rules-{version.id}",
        )
        dirty = edited != rules_json
        if st.button("保存为新版本", disabled=not dirty or session.in_progress, key=f"save-{version.id}"):
            try:
                saved = agent.save_manual(version.id, edited)
            except ValueError as exc:
                st.error(str(exc))
            else:
                history.append({"role": "assistant", "content": f"规则已保存为 {saved.id}，Python 已重新生成。"})
                st.rerun()
    with st.expander("查看本次实际执行的完整 Python"):
        st.code(version.source, language="python")
    if version.previous:
        with st.expander(f"与 {version.previous.id} 的差异"):
            rule_diff, source_diff = st.tabs(["规则 JSON", "Python"])
            rule_diff.code(version.rule_diff() or "规则内容相同。", language="diff")
            source_diff.code(version.diff() or "源码内容相同。", language="diff")
    if version.references:
        with st.expander("规则参考来源"):
            for reference in version.references:
                st.text(reference)

    dataset = session.dataset
    pending = session.pending
    form_key = f"{version.id}-{pending.id if pending else 'page'}"
    if dirty:
        st.warning("规则有未保存的修改。保存为新版本后才可执行。")
    with st.container():
        st.caption(f"回测参数 · 点击「执行代码」即确认运行 {version.id}")
        first, last = st.columns(2)
        start = first.date_input(
            "开始日期", value=dataset.start if dataset else None,
            min_value=EARLIEST_DATE, max_value=agent._today(), key=f"start-{form_key}",
        )
        end = last.date_input(
            "结束日期", value=dataset.end if dataset else None,
            min_value=EARLIEST_DATE, max_value=agent._today(), key=f"end-{form_key}",
        )
        stocks, index = st.columns(2)
        symbols_text = stocks.text_input(
            "股票（用逗号分隔）",
            value="、".join(dataset.market.stocks) if dataset else "、".join(DEFAULT_STOCKS),
            key=f"symbols-{form_key}",
        )
        benchmark = index.text_input(
            "指数基准（选填）", value=(dataset.market.benchmark_code or "") if dataset else "",
            key=f"benchmark-{form_key}",
        ).strip() or None
        cash_field, fee_field = st.columns(2)
        cash = cash_field.number_input(
            "初始资金（元）", min_value=0.0, value=session.starting_cash, step=1000.0, key=f"cash-{form_key}",
        )
        fee_percent = fee_field.number_input(
            "手续费（%）", min_value=0.0, max_value=100.0,
            value=session.commission * 100, step=0.01, format="%.5f", key=f"commission-{form_key}",
        )
        submitted = st.button(
            "执行代码", type="primary", disabled=dirty or session.in_progress, key=f"run-{form_key}",
        )
    if submitted:
        if start is None or end is None:
            st.error("请填写开始日期和结束日期后再执行。")
        else:
            symbols = [item for item in re.split(r"[\s,，、;；]+", symbols_text.strip()) if item]
            try:
                with st.spinner("正在准备行情、执行策略并生成图表…"):
                    agent.run_from_page(version.id, start, end, symbols or None, benchmark, cash, fee_percent / 100)
            except Exception as exc:
                st.error(f"回测失败：{exc}")
            else:
                st.rerun()
    if session.pending and st.button("取消待确认请求", key=f"cancel-{session.pending.id}"):
        agent.cancel_pending()
        st.rerun()

if session.dataset:
    with st.expander(f"已准备行情 {session.dataset.id}"):
        st.caption(f"BaoStock · {session.dataset.start} 至 {session.dataset.end}")
        st.dataframe(session.dataset.coverage(), hide_index=True)

if session.runs:
    run_ids = list(reversed(session.runs))
    run_numbers = {run_id: number for number, run_id in enumerate(session.runs, start=1)}
    run = session.runs[st.selectbox(
        "查看当前会话的回测结果", run_ids, key=f"reports-{run_ids[0]}",
        format_func=lambda run_id: f"第 {run_numbers[run_id]} 次 · {run_id}",
    )]
    st.markdown(run.report())
    with st.expander("本次运行使用的规则"):
        st.code(run.request.version.rules.as_json(), language="json")
    with st.expander("本次运行使用的完整源码"):
        st.code(run.request.version.source, language="python")
    with st.expander("本次行情覆盖"):
        st.dataframe(run.request.dataset.coverage(), hide_index=True)
    if run.result.fills:
        st.markdown("#### 成交明细")
        st.dataframe([
            fill.model_dump(mode="json", exclude={"signal_detail"}) for fill in run.result.fills
        ], hide_index=True)
    if run.result.unfilled:
        with st.expander("未成交订单"):
            st.dataframe([
                row.model_dump(mode="json", exclude={"signal_detail"}) for row in run.result.unfilled
            ], hide_index=True)
    for kind, title in (("equity", "净值图"), ("drawdown", "回撤图")):
        if kind in run.charts:
            st.markdown(f"#### {title}")
            st.line_chart(run.charts[kind])
    if run.chart_error:
        st.warning(run.chart_error)
