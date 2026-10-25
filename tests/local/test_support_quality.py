"""Real SQLite/ToolNode checks with scripted model decisions and no network."""

import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from openchatbi.analysis.support_quality import SupportQualityInput, analyze_support_quality
from openchatbi.catalog import CatalogStore
from openchatbi.tool.support_quality import (
    EVIDENCE_MARKER,
    append_support_evidence,
    apply_support_scope_guard,
    get_support_quality_tools,
    render_support_report,
    support_evidence_note,
)

SNAPSHOT = date(2026, 10, 8)
FIXTURE = Path(__file__).resolve().parents[2] / "example/support_quality/fixture.sql"


@pytest.fixture
def support_database(tmp_path):
    path = tmp_path / "support.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(FIXTURE.read_text(encoding="utf-8"))
    engine = create_engine(f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true")
    yield engine, path
    engine.dispose()


def request(**overrides):
    return SupportQualityInput.model_validate(
        {"metric": "overdue_rate", "start_date": "2026-10-01", "end_date": "2026-10-08", **overrides}
    )


@pytest.mark.parametrize(
    "metric,value,numerator,denominator",
    [
        ("ticket_count", 10, 10, None),
        ("avg_resolution_minutes", 800.0, 4800.0, 6),
        ("overdue_rate", 0.4, 4, 10),
    ],
)
def test_metrics_have_independent_known_answers(support_database, metric, value, numerator, denominator):
    engine, _ = support_database
    result = analyze_support_quality(engine, request(metric=metric), SNAPSHOT)
    assert result["status"] == "succeeded"
    row = result["rows"][0]
    assert row["ticket_count"] == 10
    assert row["resolved_count"] == 6
    assert row["overdue_count"] == 4
    assert row["value"] == pytest.approx(value)
    assert row["numerator"] == pytest.approx(numerator)
    assert row["denominator"] == denominator
    assert result["contract"]["timezone"] == "UTC"
    assert result["contract"]["source"].startswith("support-quality-v1#")


def test_category_breakdown_preserves_denominators(support_database):
    engine, _ = support_database
    result = analyze_support_quality(engine, request(group_by="category"), SNAPSHOT)
    assert [(r["group"], r["numerator"], r["denominator"], r["value"]) for r in result["rows"]] == [
        ("account", 3, 4, 0.75),
        ("billing", 0, 2, 0.0),
        ("delivery", 1, 4, 0.25),
    ]


@pytest.mark.parametrize("field", ["metric", "start_date", "end_date"])
def test_missing_scope_never_connects_to_database(field):
    engine = Mock()
    result = analyze_support_quality(engine, request(**{field: None}), SNAPSHOT)
    assert result["status"] == "needs_clarification"
    assert field in result["missing_fields"]
    assert result["rows"] == []
    engine.connect.assert_not_called()


@pytest.mark.parametrize(
    "start,end",
    [
        ("2026-10-08", "2026-10-08"),
        ("2026-10-08", "2026-10-01"),
        ("2026-10-01", "2026-10-09"),
        ("2026-01-01", "2026-10-08"),
    ],
)
def test_invalid_or_future_window_never_connects(start, end):
    engine = Mock()
    result = analyze_support_quality(engine, request(start_date=start, end_date=end), SNAPSHOT)
    assert result["status"] == "needs_clarification"
    engine.connect.assert_not_called()


@pytest.mark.parametrize(
    "fields",
    [
        {"metric": "quality"},
        {"group_by": "category; DROP TABLE support_tickets"},
        {"sql": "DROP TABLE support_tickets"},
        {"start_date": "not-a-date"},
    ],
)
def test_input_schema_rejects_unsupported_intent(fields):
    with pytest.raises(ValidationError):
        request(**fields)


def test_category_is_bound_not_interpolated(support_database):
    engine, _ = support_database
    malicious_category = "account' OR 1=1; DROP TABLE support_tickets; --"
    result = analyze_support_quality(engine, request(category=malicious_category), SNAPSHOT)
    assert result["status"] == "no_data"
    assert malicious_category not in result["evidence"]["sql"]
    assert result["evidence"]["parameters"]["category"] == malicious_category
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM support_tickets")).scalar_one() == 15


def test_empty_cohort_is_not_a_zero_percent_score(support_database):
    engine, _ = support_database
    result = analyze_support_quality(engine, request(start_date="2026-09-01", end_date="2026-09-02"), SNAPSHOT)
    assert result["status"] == "no_data"
    assert result["rows"] == []


def test_unresolved_average_is_null_not_zero(support_database):
    engine, _ = support_database
    result = analyze_support_quality(
        engine, request(metric="avg_resolution_minutes", start_date="2026-10-07"), SNAPSHOT
    )
    row = result["rows"][0]
    assert row["ticket_count"] == 3
    assert row["resolved_count"] == 0
    assert row["value"] is None


def test_deadline_and_resolution_exactly_at_snapshot_are_not_past_events(support_database):
    engine, path = support_database
    with sqlite3.connect(path) as connection:
        connection.execute("""INSERT INTO support_tickets
            (id, category, created_at, due_at, resolved_at) VALUES
            (16, 'boundary', '2026-10-07 12:00:00', '2026-10-08 00:00:00', '2026-10-08 00:00:00')""")
    result = analyze_support_quality(engine, request(category="boundary"), SNAPSHOT)
    assert result["rows"][0]["overdue_count"] == 0
    assert result["rows"][0]["resolved_count"] == 0


def test_too_many_groups_does_not_return_partial_success(support_database):
    engine, path = support_database
    with sqlite3.connect(path) as connection:
        connection.executemany(
            """INSERT INTO support_tickets (id, category, created_at, due_at)
            VALUES (?, ?, '2026-10-02 00:00:00', '2026-10-03 00:00:00')""",
            [(100 + index, f"group-{index}") for index in range(21)],
        )
    result = analyze_support_quality(engine, request(group_by="category"), SNAPSHOT)
    assert result["status"] == "result_limit_exceeded"
    assert result["rows"] == []


def test_database_failure_is_explicit_without_connection_details():
    engine = Mock()
    engine.dialect.name = "sqlite"
    engine.connect.side_effect = OperationalError("secret-host", {}, Exception("secret-password"))
    result = analyze_support_quality(engine, request(), SNAPSHOT)
    assert result["status"] == "execution_error"
    assert result["rows"] == []
    assert "secret" not in json.dumps(result)


def test_sqlite_only_is_explicit():
    engine = Mock()
    engine.dialect.name = "mysql"
    assert analyze_support_quality(engine, request(), SNAPSHOT)["status"] == "unsupported_database"
    engine.connect.assert_not_called()


def test_fixture_connection_is_read_only(support_database):
    engine, _ = support_database
    with engine.connect() as connection, pytest.raises(OperationalError, match="readonly"):
        connection.execute(text("DELETE FROM support_tickets"))


def test_evidence_fingerprint_is_reproducible_and_scope_sensitive(support_database):
    engine, _ = support_database
    first = analyze_support_quality(engine, request(), SNAPSHOT)["evidence"]
    again = analyze_support_quality(engine, request(), SNAPSHOT)["evidence"]
    changed = analyze_support_quality(engine, request(category="account"), SNAPSHOT)["evidence"]
    assert first == again
    assert first["query_fingerprint"] != changed["query_fingerprint"]


def test_skill_is_opt_in(monkeypatch):
    import openchatbi.tool.support_quality as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(bi_config={}))
    catalog = Mock(spec=CatalogStore)
    assert get_support_quality_tools(catalog) == []
    catalog.get_sql_engine.assert_not_called()


def test_scope_guard_interrupts_before_an_incomplete_tool_call():
    original = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "support_quality",
                "id": "missing",
                "args": {"metric": None, "start_date": None, "end_date": None, "group_by": None},
            }
        ],
    )
    guarded = apply_support_scope_guard(original, ask_human_enabled=True)
    assert guarded.tool_calls[0]["name"] == "AskHuman"
    assert guarded.tool_calls[0]["args"]["options"] == []
    assert set(guarded.response_metadata["support_scope_guard"]["missing_fields"]) == {
        "metric",
        "start_date",
        "end_date",
    }
    assert original.tool_calls[0]["name"] == "support_quality"
    assert apply_support_scope_guard(original, ask_human_enabled=False) is original


def test_complete_scope_is_not_rewritten():
    original = AIMessage(
        content="",
        tool_calls=[{"name": "support_quality", "id": "complete", "args": request().model_dump(mode="json")}],
    )
    assert apply_support_scope_guard(original, ask_human_enabled=True) is original


def test_evidence_note_is_from_current_turn_not_an_older_query(support_database):
    engine, _ = support_database
    result = analyze_support_quality(engine, request(), SNAPSHOT)
    evidence = ToolMessage(name="support_quality", tool_call_id="actual", content=json.dumps(result))
    note = support_evidence_note([HumanMessage("query"), evidence])
    assert "合成演示" in note and "4/10" in note
    assert "2026-10-01" in note and "support-quality-v1#overdue-rate" in note
    assert support_evidence_note([evidence, HumanMessage("a new unrelated question")]) == ""


@pytest.mark.parametrize(
    "metric,expected", [("overdue_rate", "40%"), ("avg_resolution_minutes", "800 分钟"), ("ticket_count", "10 条")]
)
def test_report_renders_actual_values_without_another_model_call(support_database, metric, expected):
    engine, _ = support_database
    result = analyze_support_quality(engine, request(metric=metric), SNAPSHOT)
    evidence = ToolMessage(name="support_quality", tool_call_id="actual", content=json.dumps(result))
    report = render_support_report([HumanMessage("query"), evidence])
    assert expected in report and report.count(EVIDENCE_MARKER) == 1
    assert render_support_report([evidence, HumanMessage("next query")]) is None


@pytest.mark.parametrize(
    "status,expected",
    [
        ("needs_clarification", "尚未执行"),
        ("no_data", "不能解释为 0%"),
        ("execution_error", "执行失败"),
        ("unsupported_database", "不支持"),
        ("result_limit_exceeded", "超过限制"),
    ],
)
def test_failure_reports_preserve_the_actual_tool_status(status, expected):
    evidence = ToolMessage(
        name="support_quality",
        tool_call_id="checked",
        content=json.dumps({"status": status, "message": "Exact validator detail", "rows": []}),
    )
    report = render_support_report([evidence])
    assert expected in report and "Exact validator detail" in report
    assert EVIDENCE_MARKER not in report


def test_model_copied_footer_is_replaced_not_duplicated(support_database):
    engine, _ = support_database
    result = analyze_support_quality(engine, request(), SNAPSHOT)
    evidence = ToolMessage(name="support_quality", tool_call_id="actual", content=json.dumps(result))
    answer = append_support_evidence("Summary" + EVIDENCE_MARKER + "wrong old evidence 99/100", [evidence])
    assert answer.count(EVIDENCE_MARKER) == 1
    assert "99/100" not in answer and "4/10" in answer


@pytest.mark.parametrize("content", ["not json", "[]", '{"status":"execution_error"}', '{"status":"succeeded"}'])
def test_invalid_or_failed_evidence_does_not_get_a_success_note(content):
    assert support_evidence_note([ToolMessage(name="support_quality", tool_call_id="failed", content=content)]) == ""


def test_configured_tool_has_skill_instructions_and_real_results(monkeypatch, support_database):
    import openchatbi.tool.support_quality as module

    engine, _ = support_database
    monkeypatch.setattr(
        module.config,
        "get",
        lambda: SimpleNamespace(bi_config={"support_quality_skill": {"snapshot_date": "2026-10-08"}}),
    )
    catalog = Mock(spec=CatalogStore)
    catalog.get_sql_engine.return_value = engine
    tool = get_support_quality_tools(catalog)[0]
    assert "AskHuman" in tool.description
    result = tool.invoke(request().model_dump(mode="json"))
    assert result["rows"][0]["value"] == 0.4


def make_offline_graph(monkeypatch, engine, sync_mode):
    import openchatbi.agent_graph as module
    import openchatbi.analysis.agent as analysis_agent

    class ScriptedModel:
        def bind_tools(self, tools, **kwargs):
            assert "support_quality" in [tool.name if hasattr(tool, "name") else tool.__name__ for tool in tools]
            return self

    outputs = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "support_quality", "args": request().model_dump(mode="json"), "id": "support-test-1"}
                ],
            ),
            AIMessage(content="Scripted offline completion; not a model-quality evaluation."),
        ]
    )
    monkeypatch.setattr(module, "get_llm", lambda provider=None: ScriptedModel())
    monkeypatch.setattr(module, "call_llm_chat_model_with_retry", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(module, "get_agent_prompt_template", lambda: "Offline test only")
    monkeypatch.setattr(module, "build_sql_graph", lambda *args, **kwargs: Mock())
    monkeypatch.setattr(module, "get_memory_tools", lambda *args, **kwargs: None)
    unused_tool = StructuredTool.from_function(func=lambda task: task, name="unused_analysis", description="Unused")
    monkeypatch.setattr(analysis_agent, "get_data_analysis_tool", lambda **kwargs: unused_tool)
    monkeypatch.setattr(
        module.config,
        "get",
        lambda: SimpleNamespace(bi_config={"support_quality_skill": {"snapshot_date": "2026-10-08"}}),
    )
    catalog = Mock(spec=CatalogStore)
    catalog.get_sql_engine.return_value = engine
    return module._build_graph_core(
        catalog,
        sync_mode=sync_mode,
        checkpointer=None,
        memory_store=None,
        memory_tools=None,
        mcp_tools=[],
        enable_context_management=False,
    )


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_existing_langgraph_executes_skill_and_returns_tool_evidence(monkeypatch, support_database, sync_mode):
    engine, _ = support_database
    graph = make_offline_graph(monkeypatch, engine, sync_mode)
    payload = {"messages": [HumanMessage("Use the explicitly provided metric and dates.")]}
    if sync_mode:
        result = graph.invoke(payload, config={"recursion_limit": 6})
    else:
        result = await graph.ainvoke(payload, config={"recursion_limit": 6})
    messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(messages) == 1
    evidence = json.loads(messages[0].content)
    assert evidence["status"] == "succeeded"
    assert evidence["rows"][0]["numerator"] == 4
    assert evidence["rows"][0]["denominator"] == 10
    assert evidence["contract"]["source"] == "support-quality-v1#overdue-rate"
