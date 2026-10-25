"""Original UI + real async graph/SQLite; scripted model, no browser server."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from streamlit.testing.v1 import AppTest

from openchatbi.config_loader import ConfigLoader
from openchatbi.support_demo import prepare_support_demo

ROOT = Path(__file__).resolve().parents[2]


class OfflineModel:
    def bind_tools(self, tools, **kwargs):
        return self

    def bind(self, **kwargs):
        return self


def scripted_call(model, messages, **kwargs):
    question = next(message.content for message in reversed(messages) if isinstance(message, HumanMessage))
    prior = [
        json.loads(message.content)
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "support_quality"
    ]
    has_scope = "2026-10-01" in question or any(result.get("status") == "succeeded" for result in prior)
    args = {
        "metric": "overdue_rate" if has_scope else None,
        "start_date": "2026-10-01" if has_scope else None,
        "end_date": "2026-10-08" if has_scope else None,
        "group_by": "category" if "类别" in question else (None if "整体" in question else "overall"),
    }
    return AIMessage(content="", tool_calls=[{"name": "support_quality", "args": args, "id": "offline-query"}])


@pytest.fixture
def configured_page(tmp_path, monkeypatch):
    import openchatbi.agent_graph as agent_module
    import sample_ui.async_graph_manager as manager_module

    for relative in (
        "pyproject.toml",
        "example/support_quality/fixture.sql",
        "example/support_quality/bi.yaml",
        "openchatbi/skills/support_quality/METRICS.md",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    prepared = prepare_support_demo(tmp_path)
    loader = ConfigLoader()
    loader.load(prepared["config_file"])
    configuration = loader.get()
    monkeypatch.setattr(agent_module.config, "get", lambda: configuration)
    monkeypatch.setattr(agent_module, "get_llm", lambda provider=None: OfflineModel())
    monkeypatch.setattr(agent_module, "call_llm_chat_model_with_retry", scripted_call)
    monkeypatch.setattr(
        manager_module,
        "AsyncSqliteSaver",
        SimpleNamespace(
            from_conn_string=lambda path: AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.db"))
        ),
    )
    app = AppTest.from_file(str(ROOT / "sample_ui/streamlit_ui.py"), default_timeout=60)
    try:
        app.run()
        assert not app.exception
        assert not app.error
        assert not app.chat_input[0].disabled
        yield app
    finally:
        loop = app.session_state["event_loop"]
        if loop is not None and not loop.is_closed():
            loop.run_until_complete(app.session_state["graph_manager"].cleanup())
            loop.close()
        configuration.catalog_store.get_sql_engine().dispose()


def final_answers(app):
    return [
        message.get("final_answer", message.get("content", ""))
        for message in app.session_state["messages"]
        if message["role"] == "assistant"
    ]


def test_original_page_statistics_followup_and_new_conversation(configured_page):
    app = configured_page
    app.chat_input[0].set_value("统计 2026-10-01 到 2026-10-08 超时率").run()
    assert not app.exception and not app.error
    assert "40%" in final_answers(app)[-1]
    app.chat_input[0].set_value("保持原范围，按类别拆分").run()
    assert not app.exception and not app.error
    assert all(rate in final_answers(app)[-1] for rate in ("75%", "0%", "25%"))

    original_thread = app.sidebar.text_input[1].value
    next(button for button in app.button if button.label == "新建对话").click().run()
    assert app.sidebar.text_input[1].value != original_thread
    assert app.session_state["messages"] == []
    assert any(button.key == f"history_{original_thread}" for button in app.button)
    app.chat_input[0].set_value("沿用刚才的条件，按类别拆分").run()
    assert "请补充" in final_answers(app)[-1]

    next(button for button in app.button if button.key == f"history_{original_thread}").click().run()
    assert app.sidebar.text_input[1].value == original_thread
    assert any("75%" in answer for answer in final_answers(app))


def test_pending_session_restores_question_and_resumes_original_graph(configured_page):
    app = configured_page
    app.chat_input[0].set_value("帮我看看工单质量").run()
    assert not app.exception and not app.error
    assert "请补充" in final_answers(app)[-1]
    original_thread = app.sidebar.text_input[1].value

    app.sidebar.text_input[1].set_value("other-thread").run()
    assert app.session_state["messages"] == []
    app.session_state["session_interrupts"] = {}
    app.sidebar.text_input[1].set_value(original_thread).run()
    assert "请补充" in final_answers(app)[-1]
    assert app.session_state["session_interrupts"][f"default-{original_thread}"]

    app.chat_input[0].set_value("统计 2026-10-01 到 2026-10-08 超时率").run()
    assert not app.exception and not app.error
    assert "40%" in final_answers(app)[-1]
    assert not app.session_state["session_interrupts"][f"default-{original_thread}"]


def test_history_sidebar_survives_new_page_and_does_not_list_other_user(configured_page):
    app = configured_page
    app.chat_input[0].set_value("统计 2026-10-01 到 2026-10-08 超时率").run()
    session = app.sidebar.text_input[1].value
    fresh = AppTest.from_file(str(ROOT / "sample_ui/streamlit_ui.py"), default_timeout=60).run()
    try:
        assert not fresh.exception and not fresh.error
        assert any(button.key == f"history_{session}" for button in fresh.button)
        fresh.sidebar.text_input[0].set_value("other-user").run()
        assert not any(button.key == f"history_{session}" for button in fresh.button)
        assert fresh.session_state["messages"] == []
        fresh.sidebar.text_input[0].set_value("default").run()
        next(button for button in fresh.button if button.key == f"history_{session}").click().run()
        assert "40%" in final_answers(fresh)[-1]
    finally:
        loop = fresh.session_state["event_loop"]
        loop.run_until_complete(fresh.session_state["graph_manager"].cleanup())
        loop.close()


def test_relative_week_after_history_asks_dates_and_resumes(configured_page):
    app = configured_page
    app.chat_input[0].set_value("统计 2026-10-01 到 2026-10-08 超时率").run()
    assert "40%" in final_answers(app)[-1]
    app.chat_input[0].set_value("这周超时率多少？").run()
    answer = final_answers(app)[-1]
    assert not app.exception and not app.error
    assert "不能直接沿用历史查询日期" in answer
    assert "40%" not in answer
    assert "观察截止为 2026-10-08" in answer
    thread = f"default-{app.sidebar.text_input[1].value}"
    assert app.session_state["session_interrupts"][thread]
    app.chat_input[0].set_value("沿用演示范围，2026-10-01 到 2026-10-08，UTC，统计整体超时率").run()
    assert not app.exception and not app.error
    assert "40%" in final_answers(app)[-1]
    assert not app.session_state["session_interrupts"][thread]


@pytest.mark.asyncio
async def test_graph_initialization_failure_releases_checkpoint_and_can_retry(monkeypatch):
    import sample_ui.async_graph_manager as module

    contexts = []

    class Context:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            self.closed = True

    def open_checkpoint(path):
        context = Context()
        context.closed = False
        contexts.append(context)
        return context

    monkeypatch.setattr(module, "AsyncSqliteSaver", SimpleNamespace(from_conn_string=open_checkpoint))
    monkeypatch.setattr(module, "setup_async_memory_store", AsyncMock())
    monkeypatch.setattr(module, "get_async_memory_store", AsyncMock(return_value=None))
    monkeypatch.setattr(module, "cleanup_async_memory_store", AsyncMock())
    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(catalog_store=None))
    graph = object()
    monkeypatch.setattr(
        module, "build_agent_graph_async", AsyncMock(side_effect=[RuntimeError("offline failure"), graph])
    )
    manager = module.AsyncGraphManager()

    with pytest.raises(RuntimeError, match="offline failure"):
        await manager.initialize()
    assert contexts[0].closed
    assert not manager._initialized and manager.checkpointer is None and manager.graphs == {}
    await manager.initialize()
    assert manager.graph is graph and manager._initialized
    await manager.cleanup()
    assert contexts[1].closed
