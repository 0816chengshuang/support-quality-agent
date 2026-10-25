"""Offline enforcement checks; model replies below are scripted, not quality evidence."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from openchatbi.local_graph_smoke import LocalRequestBudget
from openchatbi.support_demo import LOCAL_MODEL_NAME


def make_graph(monkeypatch, allowlist, responses=()):
    import openchatbi.agent_graph as module

    bound_names = []

    class Model:
        def bind_tools(self, tools, **kwargs):
            bound_names.extend(getattr(tool, "name", None) or tool.__name__ for tool in tools)
            return self

    def forbidden(*args, **kwargs):
        raise AssertionError("A disabled SQL/analysis/memory dependency was constructed")

    settings = SimpleNamespace(agent_tool_allowlist=allowlist, bi_config={})
    monkeypatch.setattr(module.config, "get", lambda: settings)
    monkeypatch.setattr(module, "get_llm", lambda *args: Model())
    monkeypatch.setattr(module, "build_sql_graph", forbidden)
    monkeypatch.setattr(module, "get_memory_tools", forbidden)
    monkeypatch.setattr(module, "get_agent_prompt_template", lambda: "Offline enforcement check")
    tool = StructuredTool.from_function(func=lambda metric: metric, name="support_quality", description="Offline stub")
    monkeypatch.setattr(module, "get_support_quality_tools", lambda catalog: [tool])
    outputs = iter(responses)
    monkeypatch.setattr(module, "call_llm_chat_model_with_retry", lambda *args, **kwargs: next(outputs))
    graph = module._build_graph_core(
        Mock(),
        sync_mode=True,
        checkpointer=None,
        memory_store=None,
        memory_tools=None,
        mcp_tools=[],
        enable_context_management=False,
    )
    return graph, bound_names


def test_allowlist_constrains_bound_tools_and_skips_disabled_dependencies(monkeypatch):
    _, names = make_graph(monkeypatch, ["support_quality", "search_knowledge", "AskHuman"])
    assert set(names) == {"support_quality", "search_knowledge", "AskHuman"}


def test_domain_prompt_is_explicit_and_keeps_template_replacements(monkeypatch):
    import openchatbi.prompts.system_prompt as module

    monkeypatch.setattr(module, "_agent_prompt_template_cache", None)
    monkeypatch.setattr(
        module.config,
        "get",
        lambda: SimpleNamespace(
            organization="Demo",
            bi_config={
                "agent_prompt_template": "For [organization]: [extra_tool_use_rule]",
                "extra_tool_use_rule": "Use only bound tools",
            },
        ),
    )
    assert module.get_agent_prompt_template() == "For Demo: Use only bound tools"


@pytest.mark.parametrize("value", ["", "  ", 42, ["not text"]])
def test_invalid_domain_prompt_is_not_silently_ignored(monkeypatch, value):
    import openchatbi.prompts.system_prompt as module

    monkeypatch.setattr(module, "_agent_prompt_template_cache", None)
    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(bi_config={"agent_prompt_template": value}))
    with pytest.raises(ValueError, match="agent_prompt_template"):
        module.get_agent_prompt_template()


def test_ask_human_can_be_excluded_too(monkeypatch):
    _, names = make_graph(monkeypatch, ["support_quality"])
    assert names == ["support_quality"]


def test_unknown_tool_name_fails_closed(monkeypatch):
    with pytest.raises(ValueError, match="Unknown or unavailable"):
        make_graph(monkeypatch, ["support_qualty"])


def test_disabled_tool_is_unavailable_to_actual_tool_node(monkeypatch):
    graph, _ = make_graph(
        monkeypatch,
        ["support_quality"],
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "run_python_code", "args": {"code": "raise AssertionError()"}, "id": "denied"}],
            ),
            AIMessage(content="Offline completion after denial"),
        ],
    )
    result = graph.invoke({"messages": [HumanMessage("Offline check")]}, config={"recursion_limit": 6})
    outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(outputs) == 1
    assert "not a valid tool" in outputs[0].content


def request(**changes):
    body = {"model": LOCAL_MODEL_NAME, "tools": [{"function": {"name": "support_quality"}}], **changes}
    return httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions", json=body)


def test_budget_stops_before_exceeding_total_requests():
    budget = LocalRequestBudget(max_requests=1)
    budget.check(request())
    budget.begin_turn()
    with pytest.raises(RuntimeError, match="budget"):
        budget.check(request())
    assert budget.requests == 1


def test_budget_stops_each_turn_and_can_start_a_new_turn():
    budget = LocalRequestBudget(per_turn=1)
    budget.check(request())
    with pytest.raises(RuntimeError, match="budget"):
        budget.check(request())
    budget.begin_turn()
    budget.check(request())
    assert budget.requests == 2


def test_expired_budget_stops_before_network():
    budget = LocalRequestBudget(max_seconds=-1)
    with pytest.raises(RuntimeError, match="budget"):
        budget.check(request())
    assert budget.requests == 0


@pytest.mark.parametrize(
    "body",
    [
        {"model": "a-different-model"},
        {"tools": [{"function": {"name": "run_python_code"}}]},
    ],
)
def test_budget_rejects_unapproved_model_and_tool(body):
    budget = LocalRequestBudget()
    with pytest.raises(ValueError):
        budget.check(request(**body))
    assert budget.requests == 0


def test_budget_rejects_paid_endpoint_before_counting_request():
    budget = LocalRequestBudget()
    call = httpx.Request(
        "POST", "https://api.gptgod.online/v1/chat/completions", content=json.dumps({"model": LOCAL_MODEL_NAME})
    )
    with pytest.raises(ValueError):
        budget.check(call)
    assert budget.requests == 0
