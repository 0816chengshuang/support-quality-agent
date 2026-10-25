"""User scope is data for the SQL role, not a model-invented filter."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openchatbi.agent_graph import _format_sql_response
from openchatbi.constants import SQL_SUCCESS
from openchatbi.tool.sql_handoff import SQL_EVIDENCE_MARKER, anchor_sql_user_request, append_sql_evidence


def test_anchor_preserves_actual_user_scope_and_records_model_intervention():
    original = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "text2sql",
                "id": "audit",
                "args": {"context": "Invented filter: WHERE is_test=0", "visualize": False},
                "type": "tool_call",
            }
        ],
    )
    result = anchor_sql_user_request(original, [HumanMessage("包括全部测试记录，不做排除")])
    assert "包括全部测试记录，不做排除" in result.tool_calls[0]["args"]["context"]
    assert "WHERE is_test=0" not in result.tool_calls[0]["args"]["context"]
    assert result.response_metadata["sql_scope_anchor"]["original_tool_calls"] == original.tool_calls
    assert original.tool_calls[0]["args"]["context"] == "Invented filter: WHERE is_test=0"


def test_recent_user_history_remains_available_but_old_messages_are_bounded():
    response = AIMessage(content="", tool_calls=[{"name": "text2sql", "id": "followup", "args": {"context": "notes"}}])
    result = anchor_sql_user_request(
        response, [HumanMessage(value) for value in ("old", "orders table", "maximum id", "change to minimum")]
    )
    context = result.tool_calls[0]["args"]["context"]
    assert "orders table" in context and "change to minimum" in context
    assert '"old"' not in context


def test_scope_budget_fails_instead_of_truncating_the_user_request():
    response = AIMessage(content="", tool_calls=[{"name": "text2sql", "id": "long", "args": {"context": "notes"}}])
    with pytest.raises(ValueError, match="8000"):
        anchor_sql_user_request(response, [HumanMessage("x" * 8001)])


def test_other_tools_and_messages_without_user_context_are_unchanged():
    response = AIMessage(content="", tool_calls=[{"name": "search_schema", "id": "schema", "args": {}}])
    assert anchor_sql_user_request(response, [HumanMessage("schema")]) is response
    assert anchor_sql_user_request(response, []) is response


@pytest.mark.parametrize("result", [{}, {"sql": "SELECT 1"}, {"sql": "SELECT 1", "sql_execution_result": "failed"}])
def test_missing_execution_is_never_formatted_as_an_empty_data_result(result):
    answer = _format_sql_response(result, require_execution_evidence=True)
    assert "没有执行证据" in answer
    assert "Query Results" not in answer


def test_successful_execution_keeps_real_sql_and_csv():
    answer = _format_sql_response({"sql": "SELECT 2", "data": "value\n2", "sql_execution_result": SQL_SUCCESS}, True)
    assert "SELECT 2" in answer and "value\n2" in answer


def test_sql_footer_uses_this_turn_actual_artifact_and_deduplicates():
    messages = [
        HumanMessage("query"),
        ToolMessage(
            content="untrusted prose",
            name="text2sql",
            tool_call_id="sql",
            artifact={"status": SQL_SUCCESS, "sql": "SELECT 7", "data": "value\n7\n"},
        ),
    ]
    first = append_sql_evidence("查询完成", messages)
    second = append_sql_evidence(first, messages)
    assert first == second and first.count(SQL_EVIDENCE_MARKER) == 1
    assert "SELECT 7" in first and "value\n7" in first
    assert "untrusted prose" not in first
    assert append_sql_evidence("新问题", messages + [HumanMessage("next")]) == "新问题"


def test_missing_or_failed_artifacts_do_not_claim_success():
    message = ToolMessage(
        content="SQL shown but not executed",
        name="text2sql",
        tool_call_id="sql",
        artifact={"status": "failed", "sql": "SELECT 7", "data": ""},
    )
    assert append_sql_evidence("失败", [message]) == "失败"


def test_long_evidence_is_not_silently_truncated():
    message = ToolMessage(
        content="",
        name="text2sql",
        tool_call_id="sql",
        artifact={"status": SQL_SUCCESS, "sql": "SELECT 7", "data": "x" * 4001},
    )
    result = append_sql_evidence("完成", [message])
    assert "查看工具执行记录" in result and "```csv" not in result
