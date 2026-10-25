"""Explicit task scoping is program logic, not model intelligence."""

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openchatbi.tool.support_routing import (
    relative_time_clarification,
    scoped_support_tools,
    support_task_mode,
    tool_attempted_this_turn,
)


@pytest.mark.parametrize(
    "question,mode",
    [
        ("统计明确日期的超时率", "metrics"),
        ("保持超时率口径，按类别拆开看", "metrics"),
        ("超时率分母包括未解决的工单吗，给规则出处", "knowledge"),
        ("平均处理时长怎么算", "knowledge"),
        ("计算平均处理时长", "metrics"),
        ("退款审核政策要几个工作日", "knowledge"),
        ("原始记录最大 id，包括测试记录，不限制日期", "audit"),
        ("请用 SQL 查询最小编号", "audit"),
        ("帮我看看质量", "clarification"),
        ("沿用刚才的范围按类别拆开看", "clarification"),
    ],
)
def test_conservative_explicit_modes(question, mode):
    assert support_task_mode([HumanMessage(question)]) == mode


def test_followup_only_inherits_a_successful_tool_result_in_its_own_history():
    previous = ToolMessage(name="support_quality", tool_call_id="scope", content=json.dumps({"status": "succeeded"}))
    assert support_task_mode([previous, HumanMessage("按类别拆开看")]) == "metrics"
    assert support_task_mode([HumanMessage("按类别拆开看")]) == "clarification"


def test_routing_never_enables_a_tool_missing_from_the_configured_allowlist():
    tools = [SimpleNamespace(name=name) for name in ("support_quality", "AskHuman", "text2sql", "search_knowledge")]
    active, mode = scoped_support_tools(tools, [HumanMessage("计算超时率")])
    assert mode == "metrics" and {tool.name for tool in active} == {"AskHuman", "support_quality"}
    active, mode = scoped_support_tools(tools, [HumanMessage("原始表 MAX(id)")])
    assert mode == "audit" and {tool.name for tool in active} == {"AskHuman", "text2sql"}
    with pytest.raises(ValueError):
        scoped_support_tools([SimpleNamespace(name="support_quality")], [HumanMessage("原始表 MAX(id)")])


def test_required_lookup_resets_at_new_user_turn():
    result = ToolMessage(name="search_knowledge", tool_call_id="lookup", content="{}")
    assert tool_attempted_this_turn([HumanMessage("policy"), result], "search_knowledge")
    assert not tool_attempted_this_turn([result, HumanMessage("new policy")], "search_knowledge")


def test_resume_feedback_becomes_authoritative_user_message_in_demo(monkeypatch):
    import openchatbi.agent_graph as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(support_task_routing=True))
    monkeypatch.setattr(module, "interrupt", lambda payload: "统计明确日期的超时率")
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "AskHuman", "id": "clarify", "args": {"question": "补充什么指标和日期？"}}],
            )
        ]
    }
    result = module.ask_human(state)
    assert isinstance(result["messages"][-1], HumanMessage)
    assert support_task_mode(result["messages"]) == "metrics"
    assert not tool_attempted_this_turn(result["messages"], "support_quality")


@pytest.mark.parametrize(
    "question",
    [
        "这周超时率多少？",
        "本周按类别拆开",
        "上周工单数量",
        "今天平均处理时长",
        "最近七天超时率",
        "Overdue rate this week?",
        "Raw tickets from last month",
        "Tickets in the past 7 days",
        "这周沿用刚才日期看看",
        "这周从 2026-10-01 开始",
        "这周 2026-99-01 到 2026-99-08",
    ],
)
def test_relative_time_does_not_inherit_historical_dates(question):
    history = [HumanMessage("统计 2026-10-01 到 2026-10-08 超时率"), AIMessage("40%"), HumanMessage(question)]
    clarification = relative_time_clarification(history, date(2026, 10, 8))
    assert "不能直接沿用历史查询日期" in clarification
    assert "2026-10-08" in clarification
    assert "40%" not in clarification


@pytest.mark.parametrize(
    "question",
    [
        "保持刚才日期和口径，按类别拆开看",
        "沿用上次日期统计超时率",
        "统计 2026-10-01 到 2026-10-08 超时率",
        "这周指 2026-10-19 到 2026-10-26，统计超时率",
        "这周是怎么定义的？",
    ],
)
def test_explicit_dates_and_ordinary_followups_are_not_blocked(question):
    if "定义" in question:
        assert support_task_mode([HumanMessage(question)]) == "knowledge"
    else:
        assert relative_time_clarification([HumanMessage(question)]) is None


def test_guard_only_inspects_current_human_turn():
    assert relative_time_clarification([HumanMessage("这周超时率"), HumanMessage("2026-10-01 到 2026-10-08")]) is None
    assert relative_time_clarification([HumanMessage("这周超时率"), AIMessage("等待补充")]) is None


@pytest.mark.parametrize("question,mode", [("这周超时率多少？", "metrics"), ("这周原始记录多少？", "audit")])
def test_relative_time_guard_precedes_model_and_query(monkeypatch, question, mode):
    import openchatbi.agent_graph as module
    from openchatbi.tool.ask_human import AskHuman

    settings = SimpleNamespace(
        support_task_routing=True, bi_config={"support_quality_skill": {"snapshot_date": "2026-10-08"}}
    )
    monkeypatch.setattr(module.config, "get", lambda: settings)
    monkeypatch.setattr(module, "get_agent_prompt_template", lambda: "Offline time scope check")
    model_call = Mock(side_effect=AssertionError("Ambiguous time must not call the model"))
    monkeypatch.setattr(module, "call_llm_chat_model_with_retry", model_call)
    tools = [AskHuman, SimpleNamespace(name="support_quality"), SimpleNamespace(name="text2sql")]
    node = module.agent_llm_call(Mock(), tools)
    result = node(
        {"messages": [HumanMessage("统计 2026-10-01 到 2026-10-08 超时率"), AIMessage("40%"), HumanMessage(question)]}
    )
    response = result["messages"][0]
    assert response.tool_calls[0]["name"] == "AskHuman"
    assert "不能直接沿用" in response.tool_calls[0]["args"]["question"]
    assert response.response_metadata["support_relative_time_guard"]
    assert response.response_metadata["support_tool_scope"]["mode"] == mode
    assert result["sends"][0].node == "ask_human"
    model_call.assert_not_called()


@pytest.mark.parametrize("question,tool", [("统计超时率", "support_quality"), ("超时率分母是什么", "search_knowledge")])
def test_model_cannot_answer_from_old_results_without_current_execution(monkeypatch, question, tool):
    import openchatbi.agent_graph as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(support_task_routing=True))
    monkeypatch.setattr(module, "get_agent_prompt_template", lambda: "Offline execution check")
    monkeypatch.setattr(module, "call_llm_chat_model_with_retry", lambda *args, **kwargs: AIMessage("这周超时率40%"))
    node = module.agent_llm_call(Mock(), [SimpleNamespace(name=tool)])
    history = [
        HumanMessage("old"),
        ToolMessage(name=tool, tool_call_id="old", content="{}"),
        AIMessage("40%"),
        HumanMessage(question),
    ]
    result = node({"messages": history})
    assert "尚未执行" in result["final_answer"]
    assert "40%" not in result["final_answer"]
    assert result["messages"][0].response_metadata["support_missing_execution"] == tool


def test_null_group_is_repaired_only_from_explicit_current_user_scope():
    from openchatbi.tool.support_quality import apply_support_scope_guard

    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "support_quality",
                "id": "null-group",
                "args": {
                    "metric": "overdue_rate",
                    "start_date": "2026-10-01",
                    "end_date": "2026-10-08",
                    "group_by": None,
                },
            }
        ],
    )
    fixed = apply_support_scope_guard(response, True, [HumanMessage("统计整体超时率")])
    assert fixed.tool_calls[0]["args"]["group_by"] == "overall"
    assert response.tool_calls[0]["args"]["group_by"] is None
    assert fixed.response_metadata["support_explicit_group_guard"]["source"] == "current_user_request"
    unchanged = apply_support_scope_guard(
        response, True, [HumanMessage("统计整体超时率"), HumanMessage("现在按类别拆开")]
    )
    assert unchanged.tool_calls[0]["args"]["group_by"] is None
    ambiguous = apply_support_scope_guard(response, True, [HumanMessage("不要整体，按类别统计")])
    assert ambiguous.tool_calls[0]["args"]["group_by"] is None
