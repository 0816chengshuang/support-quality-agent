"""Offline UI checks; no page server, launcher or model is started."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from openchatbi.streaming import StreamStep
from sample_ui.history_loader import current_assistant_answer
from sample_ui.localization import step_text_zh


@pytest.mark.parametrize(
    "kind,data,expected",
    [
        ("tool", {"tools": ["support_quality", "AskHuman"]}, "执行：工单指标统计、确认查询条件"),
        ("sub_agent", {"sub_agents": ["text2sql"]}, "执行：SQL 查询助手"),
        ("execute_sql", {}, "执行只读 SQL 查询。"),
        ("sql", {"sql": "SELECT max(id) FROM support_tickets"}, "SELECT max(id) FROM support_tickets"),
        ("tool_error", {"tool": "support_quality"}, "工单指标统计执行失败"),
    ],
)
def test_structured_execution_steps_are_chinese_without_changing_sql(kind, data, expected):
    text = step_text_zh(StreamStep("English upstream label", 0, "main", kind, data))
    assert expected in text
    assert "English upstream label" not in text


@pytest.mark.parametrize(
    "answer",
    [
        "整体超时率：40%（4/10）。",
        "SQL 执行依据：SELECT MAX(id)，15。",
        "规则原文；来源：support-quality-v1#超时率；chunk_id：actual。",
    ],
)
def test_program_reports_are_completed_answers(answer):
    assert current_assistant_answer([HumanMessage("question"), AIMessage(answer)]) == answer


@pytest.mark.parametrize(
    "last",
    [
        HumanMessage("new question"),
        AIMessage(content="", tool_calls=[{"name": "AskHuman", "args": {"question": "dates?"}, "id": "ask"}]),
        ToolMessage(content="tool data", name="support_quality", tool_call_id="query"),
    ],
)
def test_pending_turn_does_not_show_a_previous_answer(last):
    assert current_assistant_answer([AIMessage("old answer"), last]) == ""


@pytest.mark.parametrize("interrupted", [False, True])
def test_streamlit_renders_program_report_or_pending_feedback_without_tokens(interrupted):
    # The existing UI is imported inside Streamlit's in-process test runtime.
    script = f"""
import asyncio
from types import SimpleNamespace
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from openchatbi.config_loader import ConfigLoader
ConfigLoader._config = None
from sample_ui.streamlit_ui import process_user_message_stream

interrupted = {interrupted!r}
answer = "整体超时率：40%（4/10）。"
question = "请补充日期。"

class OfflineGraph:
    async def astream(self, *args, **kwargs):
        yield (), "updates", {{"llm_node": {{"messages": [AIMessage(answer)]}}}}

    async def aget_state(self, config):
        messages = [HumanMessage("new question")] if interrupted else [HumanMessage("question"), AIMessage(answer)]
        interrupts = [SimpleNamespace(value={{"text": question, "buttons": []}})] if interrupted else []
        return SimpleNamespace(values={{"messages": messages}}, interrupts=interrupts)

async def get_graph(provider):
    return OfflineGraph()

st.session_state.graph_manager = SimpleNamespace(_initialized=True, get_graph=get_graph)
st.session_state.session_interrupts = {{}}
result = asyncio.run(process_user_message_stream("question", "test", "one", None, st.container(), st.container()))
st.session_state.test_result = result
"""
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception
    result = app.session_state["test_result"]
    timing = result[5]
    assert timing["total_s"] >= timing["prepare_s"] >= 0
    assert timing["first_event_s"] is None
    if interrupted:
        assert result[4] == "请补充日期。"
        assert "等待你补充信息" in result[3]
        assert "Analysis complete" not in result[3]
        assert "[]" not in result[4]
        assert "40%" not in result[4]
    else:
        assert result[4] == "整体超时率：40%（4/10）。"
        assert any("40%" in markdown.value for markdown in app.markdown)
