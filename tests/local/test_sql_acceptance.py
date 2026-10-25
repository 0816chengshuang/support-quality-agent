"""Offline checks of SQL handoff plumbing, not model-quality evidence."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from openchatbi.agent_graph import get_sql_tools
from openchatbi.llm.llm import bind_tools_for_profile
from openchatbi.sql_acceptance import csv_numbers
from openchatbi.tool.ask_human import AskHuman


def test_csv_results_are_parsed_as_data():
    assert csv_numbers("Query Results:\n```csv\nmax_id\n15\n```") == [15.0]


@pytest.mark.parametrize(
    "content",
    [
        "The answer is 15",
        "```csv\nmax_id\n15\n14\n```",
        "```csv\na,b\n15,14\n```",
        "```csv\nmax_id\nnot a number\n```",
    ],
)
def test_missing_or_ambiguous_csv_is_not_accepted(content):
    assert csv_numbers(content) == []


def test_trace_reason_is_not_required_to_execute_the_real_tool_wrapper():
    graph = Mock()
    graph.invoke.return_value = {"sql": "SELECT 1", "data": "v\n1"}
    tool = get_sql_tools(graph, sync_mode=True)
    result = tool.invoke({"context": "query", "visualize": False})
    assert "SELECT 1" in result
    assert graph.invoke.call_args.args[0] == {"messages": "query", "visualize": False}


def test_profile_can_disable_upstream_llm_retries(monkeypatch):
    import openchatbi.llm.llm as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(llm_max_attempts=1))
    model = Mock()
    model.invoke.side_effect = RuntimeError("offline failure")
    assert module.call_llm_chat_model_with_retry(model, [HumanMessage("offline")]) is None
    assert model.invoke.call_count == 1


def test_resource_prompts_decode_utf8_on_windows(monkeypatch):
    import openchatbi.prompts.system_prompt as module

    for name in (
        "_extraction_prompt_template_cache",
        "_table_selection_prompt_template_cache",
        "_text2sql_prompt_template_cache",
        "_visualization_prompt_template_cache",
        "_summary_prompt_template_cache",
    ):
        monkeypatch.setattr(module, name, None)
    assert "rewrite_question" in module.get_extraction_prompt_template()
    assert "Candidate Tables" in module.get_table_selection_prompt_template()
    assert module.get_text2sql_prompt_template()
    assert module.get_visualization_prompt_template()
    assert module.get_summary_prompt_template()


@pytest.mark.parametrize("strict", [True, False])
def test_server_profile_controls_strict_without_disabling_tools(monkeypatch, strict):
    import openchatbi.llm.llm as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(openai_strict_tools=strict))
    model = ChatOpenAI(model="offline", api_key="offline-placeholder-only")
    bound = bind_tools_for_profile(model, [AskHuman], json_output=True)
    tool = bound.kwargs["tools"][0]["function"]
    assert tool["name"] == "AskHuman"
    if strict:
        assert tool["strict"] is True
        assert bound.kwargs["response_format"] == {"type": "json_object"}
    else:
        assert "strict" not in tool
        assert "response_format" not in bound.kwargs


def test_default_profile_preserves_upstream_strict_mode(monkeypatch):
    import openchatbi.llm.llm as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace())
    model = ChatOpenAI(model="offline", api_key="offline-placeholder-only")
    assert bind_tools_for_profile(model, [AskHuman]).kwargs["tools"][0]["function"]["strict"] is True
