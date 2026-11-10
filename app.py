import re

import streamlit as st

from agent.backtest_spec import DEFAULT_STOCKS, EARLIEST_DATE
from agent.react_agent import ReactAgent
from agent.strategy_session import CodeVersion, CompletedRun, Dataset, RunRequest


def show_strategy(version: CodeVersion, editable: bool) -> bool:
    st.markdown(f"#### 策略规则 {version.id}")
    st.caption(f"预热 {version.warmup_bars} 根日线")
    for line in version.rules.summary():
        st.text(line)
    with st.expander("查看完整 Python 源码"):
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

    if not editable:
        with st.expander("查看规则 JSON"):
            st.code(version.rules.as_json(), language="json")
        return False

    rules_json = version.rules.as_json()
    with st.expander("编辑策略规则 JSON"):
        edited = st.text_area("规则 JSON", value=rules_json, height=340, key=f"rules-{version.id}")
        dirty = edited != rules_json
        if st.button("保存为新版本", disabled=not dirty or session.in_progress, key=f"save-{version.id}"):
            try:
                agent.save_manual(version.id, edited)
            except ValueError as exc:
                agent.add_timeline_entry("error", f"保存 {version.id} 失败：{exc}")
            st.rerun()
    return dirty


def show_run_controls(version: CodeVersion, dirty: bool, request: RunRequest | None = None):
    dataset = request.dataset if request else session.dataset
    form_key = f"{version.id}-{request.id if request else 'page'}"
    if dirty:
        st.warning("规则有未保存的修改。保存为新版本后才可执行。")
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
        "初始资金（元）", min_value=0.0,
        value=request.starting_cash if request else session.starting_cash,
        step=1000.0, key=f"cash-{form_key}",
    )
    fee_percent = fee_field.number_input(
        "手续费（%）", min_value=0.0, max_value=100.0,
        value=(request.commission if request else session.commission) * 100,
        step=0.01, format="%.5f", key=f"commission-{form_key}",
    )
    if st.button("执行代码", type="primary", disabled=dirty or session.in_progress, key=f"run-{form_key}"):
        if start is None or end is None:
            agent.add_timeline_entry("error", "请填写开始日期和结束日期后再执行。")
        else:
            symbols = [item for item in re.split(r"[\s,，、;；]+", symbols_text.strip()) if item]
            first_entry = len(agent.timeline)
            try:
                with st.spinner("正在准备行情、执行策略并生成图表…"):
                    agent.run_from_page(version.id, start, end, symbols or None, benchmark, cash, fee_percent / 100)
            except Exception as exc:
                # 取数或引擎错误在产生处记录，表单参数错误在这里补记。
                if not any(entry.kind == "error" for entry in agent.timeline[first_entry:]):
                    agent.add_timeline_entry("error", f"回测失败：{exc}")
        st.rerun()
    if request and st.button("取消待确认请求", key=f"cancel-{request.id}"):
        agent.cancel_pending()
        st.rerun()


def show_dataset(dataset: Dataset):
    st.markdown(f"#### 已准备行情 {dataset.id}")
    symbols = "、".join(dataset.market.stocks)
    benchmark = dataset.market.benchmark_code
    st.caption(f"BaoStock · {dataset.start} 至 {dataset.end} · 预热 {dataset.warmup_bars} 根日线")
    st.text(f"{symbols} · 基准 {benchmark}" if benchmark else symbols)
    with st.expander("查看行情覆盖"):
        st.dataframe(dataset.coverage(), hide_index=True)


def show_run(run: CompletedRun):
    st.markdown(run.report())
    with st.expander("查看本次规则与源码"):
        st.code(run.request.version.rules.as_json(), language="json")
        st.code(run.request.version.source, language="python")
    with st.expander("查看本次行情覆盖"):
        st.dataframe(run.request.dataset.coverage(), hide_index=True)
    with st.expander("查看订单明细"):
        for title, rows in (("成交明细", run.result.fills), ("未成交订单", run.result.unfilled)):
            st.markdown(f"**{title}**")
            if rows:
                st.dataframe([row.model_dump(mode="json", exclude={"signal_detail"}) for row in rows], hide_index=True)
            else:
                st.caption("无记录")
    if run.charts:
        with st.expander("查看净值图和回撤图"):
            for kind, title in (("equity", "净值图"), ("drawdown", "回撤图")):
                if kind in run.charts:
                    st.markdown(f"**{title}**")
                    st.line_chart(run.charts[kind])
    if run.chart_error:
        st.warning(run.chart_error)


st.title("股票智能回测系统")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
agent = st.session_state["agent"]
session = agent.session

dirty = False
run_number = 0
for entry in agent.timeline:
    content = entry.content
    timestamp = entry.time.strftime("%H:%M:%S")
    if entry.kind in ("user", "assistant"):
        container = st.chat_message(entry.kind)
    else:
        container = st.container(border=True)
    with container:
        if entry.kind == "run":
            run_number += 1
            st.caption(f"{timestamp} · 第 {run_number} 次回测")
        else:
            st.caption(timestamp)

        if entry.kind in ("user", "assistant"):
            st.markdown(content)
        elif entry.kind == "error":
            st.error(content)
        elif entry.kind == "version":
            current = content is session.version
            edited = show_strategy(content, editable=current)
            if current:
                dirty = edited
                if session.pending is None:
                    show_run_controls(content, dirty)
        elif entry.kind == "dataset":
            show_dataset(content)
        elif entry.kind == "confirmation":
            st.markdown(f"#### 回测确认 {content.id}")
            st.text(content.describe())
            st.caption(f"{entry.status} · {entry.reason}" if entry.reason else entry.status)
            if content is session.pending:
                show_run_controls(content.version, dirty, content)
        elif entry.kind == "run":
            show_run(content)
        if entry.tools:
            with st.expander("工具调用"):
                for tool, status in entry.tools:
                    st.text(f"{tool} · {status}")

if prompt := st.chat_input("描述你的回测策略"):
    st.chat_message("user").markdown(prompt)
    with st.spinner("正在处理回测请求…"):
        for _ in agent.execute_stream(prompt):
            pass
    st.rerun()
