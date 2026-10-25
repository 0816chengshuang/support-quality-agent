"""Main agent graph construction and execution logic."""

import datetime
import json
import logging
import traceback
import uuid
from collections.abc import Callable
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import START
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer, Send, interrupt
from pydantic import BaseModel, Field

from openchatbi import config
from openchatbi.catalog import CatalogStore
from openchatbi.constants import SQL_SUCCESS, datetime_format
from openchatbi.context_config import get_context_config
from openchatbi.context_manager import ContextManager
from openchatbi.graph_state import AgentState, InputState, OutputState
from openchatbi.llm.llm import bind_tools_for_profile, call_llm_chat_model_with_retry, get_llm
from openchatbi.prompts.system_prompt import get_agent_prompt_template
from openchatbi.text2sql.sql_graph import build_sql_graph
from openchatbi.tool.ask_human import AskHuman
from openchatbi.tool.mcp_tools import create_mcp_tools_sync, get_mcp_tools_async
from openchatbi.tool.memory import get_memory_tools
from openchatbi.tool.run_python_code import run_python_code
from openchatbi.tool.save_report import save_report
from openchatbi.tool.search_knowledge import render_knowledge_report, search_knowledge, search_schema, show_schema
from openchatbi.tool.sql_handoff import anchor_sql_user_request, append_sql_evidence
from openchatbi.tool.support_quality import (
    append_support_evidence,
    apply_support_scope_guard,
    get_support_quality_tools,
    render_support_report,
)
from openchatbi.tool.support_routing import relative_time_clarification, scoped_support_tools, tool_attempted_this_turn
from openchatbi.utils import get_text_from_content, log, recover_incomplete_tool_calls

logger = logging.getLogger(__name__)


def _get_mcp_servers() -> list[Any]:
    """Get MCP servers from config with fallback for tests."""
    try:
        return config.get().mcp_servers
    except ValueError:
        return []


def ask_human(state: AgentState) -> dict[str, Any]:
    """Node function to ask human for additional information or clarification.

    Args:
        state (AgentState): The current graph state containing messages and context.

    Returns:
        dict: Updated state with human feedback as a tool message and user input.
    """
    last_message = cast(AIMessage, state["messages"][-1])
    tool_call = last_message.tool_calls[0]
    tool_call_id = tool_call["id"]
    args = tool_call["args"]
    user_feedback = interrupt({"text": args["question"], "buttons": args.get("options", None)})
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "name": "AskHuman", "content": user_feedback}]
    try:
        promote_feedback = getattr(config.get(), "support_task_routing", False) is True
    except ValueError:
        promote_feedback = False
    if promote_feedback:
        tool_message.append(HumanMessage(user_feedback))
    return {
        "messages": tool_message,
        "history_messages": [AIMessage(args["question"])] if promote_feedback else [AIMessage(args["question"]), HumanMessage(user_feedback)],
        "user_input": user_feedback,
    }


class CallSQLGraphInput(BaseModel):
    reasoning: str = Field(
        default="", description="Optional trace explanation; the required query is in context",
    )
    context: str | dict[str, Any] | list[dict[str, Any]] | list[str] = Field(
        description="""Context payload passed to Text2SQL.
        Prefer a structured format to preserve information while isolating scope:
        - Shared Context: stable business background, entities, metrics, key filters
        - Current Subtask (User's latest question): The ONLY specific query to execute NOW.
        - Carry-over From Previous Step: prior SQL/result and the exact delta to apply (optional)
        - Deferred Tasks: remaining stages to ignore for now (optional)""",
    )
    visualize: bool = Field(
        default=True,
        description=(
            "Whether this Text2SQL call should generate visualization DSL for UI display. "
            "Set to false for intermediate data retrieval used by downstream analysis tools."
        ),
    )


# Description for SQL tools
_TEXT2SQL_TOOL_DESCRIPTION = """Text2SQL tool to generate and execute SQL query and build visualization DSL for UI
based on user's question and context.

Returns:
    str: A formatted response containing SQL, data, and visualization status.

Important notes:
- Do not use this tool to explore database schema, list tables, or discover which tables/columns contain data.
  Use search_schema for schema discovery and show_schema for known table details before calling Text2SQL.
- If user want to change the visualization chart type or style, add the requirement in the question
- Set visualize=false when the call only fetches data for downstream analysis or when the user explicitly does not want a chart
- Make sure to provide question in English
- In staged workflows, pass subtask-scoped context only for this call and keep other stages under Deferred Tasks
- If this call depends on the previous SQL intent, use Carry-over with an explicit delta instead of resending the whole multi-stage ask
"""


def _normalize_text2sql_context(context: str | dict[str, Any] | list[dict[str, Any]] | list[str]) -> str:
    """Normalize tool context to the string payload expected by SQL graph."""
    if isinstance(context, str):
        return context
    try:
        return json.dumps(context, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(context)


def _requires_sql_execution_evidence() -> bool:
    try:
        return getattr(config.get(), "require_sql_execution_evidence", False) is True
    except ValueError:
        return False


def _format_sql_response(sql_graph_response: dict[str, Any], require_execution_evidence: bool = False) -> str:
    """Format SQL graph response into a standardized string format.

    Args:
        sql_graph_response: The response dictionary from the SQL graph

    Returns:
        str: Formatted response string
    """
    sql = sql_graph_response.get("sql", "")
    data = sql_graph_response.get("data", "")
    visualization_dsl = sql_graph_response.get("visualization_dsl", {})
    if require_execution_evidence and (not sql or sql_graph_response.get("sql_execution_result") != SQL_SUCCESS):
        return "SQL 执行未成功或没有执行证据，不能回答成没有数据。请明确说明查询失败，不能猜结果或重新委派同一任务。"

    response_parts: list[str] = []
    if sql:
        response_parts.append(f"SQL Query:\n```sql\n{sql}\n```")
    if data:
        response_parts.append(f"\nQuery Results (CSV format):\n```csv\n{data}\n```")

    # Include visualization status
    if visualization_dsl and "error" not in visualization_dsl:
        chart_type = visualization_dsl.get("chart_type", "unknown")
        response_parts.append(
            f"\nVisualization Created: {chart_type} chart has been automatically generated and will be displayed in the UI."
        )
    elif visualization_dsl and "error" in visualization_dsl:
        response_parts.append(f"\nVisualization Error: {visualization_dsl['error']}")

    return "\n\n".join(response_parts) if response_parts else "No results returned."


def get_sql_tools(sql_graph: CompiledStateGraph[Any, None, Any, Any], sync_mode: bool = False) -> StructuredTool:
    """Create SQL generation tool from compiled SQL graph.

    Args:
        sql_graph (CompiledStateGraph): The compiled SQL generation subgraph.
        sync_mode (bool): Whether to create synchronous or asynchronous tools

    Returns:
        function: Tool function for SQL generation.
    """

    require_evidence = _requires_sql_execution_evidence()

    def format_result(result):
        content = _format_sql_response(result, require_evidence)
        if require_evidence:
            return content, {"status": result.get("sql_execution_result"), "sql": result.get("sql"), "data": result.get("data")}
        return content

    def call_sql_graph_sync(
        context: str | dict[str, Any] | list[dict[str, Any]] | list[str],
        config: RunnableConfig,
        visualize: bool = True,
        reasoning: str = "",
    ) -> str:
        """Sync node function for Text2SQL tool"""
        normalized_context = _normalize_text2sql_context(context)
        log(f"Call SQL graph (sync) with reasoning: {reasoning}, visualize: {visualize}, context: {normalized_context}")
        try:
            sql_graph_response = sql_graph.invoke(
                {"messages": normalized_context, "visualize": visualize}, config=config
            )
            return format_result(sql_graph_response)
        except GraphInterrupt as e:
            log(f"Sql graph interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            log(f"Run sql graph error:\n{repr(e)}")
            traceback.print_exc()
        return format_result({}) if require_evidence else "Error occurred when calling Text2SQL tool."

    async def call_sql_graph_async(
        context: str | dict[str, Any] | list[dict[str, Any]] | list[str],
        config: RunnableConfig,
        visualize: bool = True,
        reasoning: str = "",
    ) -> str:
        """Async node function for Text2SQL tool"""
        normalized_context = _normalize_text2sql_context(context)
        log(
            f"Call SQL graph (async) with reasoning: {reasoning}, visualize: {visualize}, context: {normalized_context}"
        )
        try:
            sql_graph_response = await sql_graph.ainvoke(
                {"messages": normalized_context, "visualize": visualize}, config=config
            )
            return format_result(sql_graph_response)
        except GraphInterrupt as e:
            log(f"Sql graph interrupted:\n{repr(e)}")
            raise e
        except Exception as e:
            log(f"Run sql graph error:\n{repr(e)}")
            traceback.print_exc()
        return format_result({}) if require_evidence else "Error occurred when calling Text2SQL tool."

    if sync_mode:
        return StructuredTool.from_function(
            func=call_sql_graph_sync,
            name="text2sql",
            description=_TEXT2SQL_TOOL_DESCRIPTION,
            args_schema=CallSQLGraphInput,
            response_format="content_and_artifact" if require_evidence else "content",
            return_direct=False,
        )
    else:
        return StructuredTool.from_function(
            coroutine=call_sql_graph_async,
            name="text2sql",
            description=_TEXT2SQL_TOOL_DESCRIPTION,
            args_schema=CallSQLGraphInput,
            response_format="content_and_artifact" if require_evidence else "content",
            return_direct=False,
        )


def agent_llm_call(
    llm: BaseChatModel,
    tools: list[Any],
    context_manager: ContextManager | None = None,
) -> Callable[..., Any]:
    """Create llm call function to generate reasoning and determine next node based on tool calls in LLM response.

    Args:
        llm (BaseChatModel): The LLM for agent decision-making.
        tools: List of tools.
        context_manager: Optional context manager for handling long conversations.

    Returns:
        function: function that processes state and determines next node.
    """

    llm_with_tools = bind_tools_for_profile(llm, tools)

    def _call_model(state: AgentState) -> dict[str, Any]:
        # First, check and recover any incomplete tool calls
        recovery_ops = recover_incomplete_tool_calls(state)
        if recovery_ops:
            return {"messages": recovery_ops, "agent_next_node": "llm_node"}

        messages = state["messages"]
        final_messages: list[HumanMessage | AIMessage] = []
        if isinstance(messages[-1], HumanMessage):
            final_messages.append(messages[-1])

        # Apply context management if available (before processing)
        if context_manager:
            original_count = len(messages)
            context_manager.manage_context_messages(messages)  # type: ignore[arg-type]
            if len(messages) != original_count:
                logger.info(f"Context management: modified messages from {original_count} to {len(messages)}")

        system_prompt = get_agent_prompt_template().replace(
            "[time_field_placeholder]", datetime.datetime.now().strftime(datetime_format)
        )

        try:
            routing = getattr(config.get(), "support_task_routing", False) is True
        except ValueError:
            routing = False
        active_tools, mode = scoped_support_tools(tools, messages) if routing else (tools, None)
        if routing:
            report = render_knowledge_report(messages)
            report_kind = "extractive_knowledge_result"
            if report is None:
                report = render_support_report(messages)
                report_kind = "verified_metric_result"
            if report is None:
                report = append_sql_evidence("只读查询已执行，实际 SQL 和结果如下。", messages)
                if report == "只读查询已执行，实际 SQL 和结果如下。":
                    report = None
                report_kind = "verified_sql_result"
            if report is not None:
                reply = AIMessage(content=report, response_metadata={"program_report": report_kind})
                return {"messages": [reply], "final_answer": report,
                        "history_messages": final_messages + [reply], "agent_next_node": END}
        time_question = None
        if routing and mode != "knowledge":
            settings = getattr(config.get(), "bi_config", {}) or {}
            snapshot_value = settings.get("support_quality_skill", {}).get("snapshot_date")
            snapshot = datetime.date.fromisoformat(str(snapshot_value)) if snapshot_value else None
            time_question = relative_time_clarification(messages, snapshot)
        active_model = bind_tools_for_profile(llm, active_tools) if routing else llm_with_tools
        required_tool = {"metrics": "support_quality", "knowledge": "search_knowledge"}.get(mode)
        if required_tool and not tool_attempted_this_turn(messages, required_tool):
            names = {getattr(tool, "name", None) or getattr(tool, "__name__", None) for tool in active_tools}
            if required_tool in names:
                active_model = active_model.bind(tool_choice={"type": "function", "function": {"name": required_tool}})
        if time_question:
            response = AIMessage(
                content="" if AskHuman in active_tools else time_question,
                tool_calls=[{"name": "AskHuman", "args": {"question": time_question, "options": []},
                             "id": uuid.uuid4().hex}] if AskHuman in active_tools else [],
                response_metadata={"support_relative_time_guard": True},
            )
        else:
            response = call_llm_chat_model_with_retry(
                cast(BaseChatModel, active_model),
                ([SystemMessage(system_prompt)] + messages),
                streaming_tokens=True,
                bound_tools=active_tools,
                parallel_tool_call=True,
            )
        if (required_tool and isinstance(response, AIMessage) and not response.tool_calls
                and not time_question and not tool_attempted_this_turn(messages, required_tool)):
            label = "统计查询" if mode == "metrics" else "资料检索"
            response = AIMessage(
                content=f"本轮{label}尚未执行，没有可核对的新结果，不能把历史答案作为本轮结果。请确认请求条件后重试。",
                response_metadata={"support_missing_execution": required_tool},
            )
        if isinstance(response, AIMessage):
            if mode:
                response = response.model_copy(update={"response_metadata": {
                    **response.response_metadata, "support_tool_scope": {"mode": mode,
                        "tools": [getattr(tool, "name", None) or getattr(tool, "__name__", None) for tool in active_tools]},
                }})
            try:
                anchored = getattr(config.get(), "anchor_sql_user_request", False) is True
            except ValueError:
                anchored = False
            if anchored:
                response = anchor_sql_user_request(response, messages)
            response = apply_support_scope_guard(response, AskHuman in tools, messages)
            tool_calls = response.tool_calls
            print("Tool Call:", ", ".join(tool["name"] for tool in tool_calls))
            if tool_calls:
                # Group tool calls by type for parallel routing
                ask_human_calls = [call for call in tool_calls if call["name"] == "AskHuman"]
                normal_tool_calls = [call for call in tool_calls if call["name"] != "AskHuman"]

                # Create Send objects for parallel routing
                sends: list[Send] = []
                if ask_human_calls:
                    # Create message with only AskHuman calls
                    ask_human_msg = AIMessage(content=response.content, tool_calls=ask_human_calls)
                    sends.append(Send("ask_human", {"messages": [ask_human_msg]}))

                if normal_tool_calls:
                    # Create message with only normal tool calls
                    tool_msg = AIMessage(content=response.content, tool_calls=normal_tool_calls)
                    sends.append(Send("use_tool", {"messages": [tool_msg]}))

                return {"messages": [response], "history_messages": final_messages, "sends": sends}
            else:
                final_answer = get_text_from_content(response.content)
                with_evidence = append_sql_evidence(append_support_evidence(final_answer, messages), messages)
                if with_evidence != final_answer:
                    final_answer = with_evidence
                    response = response.model_copy(update={"content": final_answer})
                final_messages.append(AIMessage(response.content))
                return {
                    "messages": [response],
                    "final_answer": final_answer,
                    "history_messages": final_messages,
                    "agent_next_node": END,
                }
        elif response is None:
            return {
                "messages": [AIMessage("Sorry, the LLM service is currently unavailable.")],
                "history_messages": final_messages,
                "agent_next_node": END,
            }
        else:
            return {"messages": [response], "history_messages": final_messages, "agent_next_node": END}

    return _call_model


def _build_graph_core(
    catalog: CatalogStore,
    sync_mode: bool,
    checkpointer: Checkpointer | None,
    memory_store: BaseStore | None,
    memory_tools: list[Any] | None,
    mcp_tools: list[Any],
    enable_context_management: bool = True,
    llm_provider: str | None = None,
) -> CompiledStateGraph[AgentState, None, InputState, OutputState]:
    """Core graph building logic shared by both sync and async versions.

    Args:
        catalog: Catalog store containing schema information
        sync_mode: Whether to use synchronous mode for tools and operations
        checkpointer: The Checkpointer for state persistence
        memory_store: The BaseStore to use for long-term memory
        memory_tools: List of memory tools (manage_memory_tool, search_memory_tool)
        mcp_tools: Pre-initialized MCP tools
        enable_context_management: Whether to enable context management

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution
    """
    try:
        configured_allowlist = getattr(config.get(), "agent_tool_allowlist", None)
    except ValueError:
        configured_allowlist = None
    allowed = set(configured_allowlist) if configured_allowlist is not None else None

    def enabled(name: str) -> bool:
        return allowed is None or name in allowed

    normal_tools: list[Any] = [search_knowledge, search_schema, show_schema]
    if enabled("text2sql") or enabled("data_analysis"):
        sql_graph = build_sql_graph(catalog, checkpointer, cast(BaseStore, memory_store), llm_provider=llm_provider)
        if enabled("text2sql"):
            normal_tools.append(get_sql_tools(sql_graph=sql_graph, sync_mode=sync_mode))
        if enabled("data_analysis"):
            from openchatbi.analysis.agent import get_data_analysis_tool

            normal_tools.append(get_data_analysis_tool(
                sql_graph=sql_graph,
                sync_mode=sync_mode,
                llm_provider=llm_provider,
                checkpointer=cast(BaseCheckpointSaver[Any] | None, checkpointer if checkpointer is not False else None),
                memory_store=memory_store,
            ))

    # Use provided memory tools or create them
    memory_tools_for_graph: list[Any] | None = list(memory_tools) if memory_tools else None
    if not memory_tools_for_graph and (enabled("manage_memory") or enabled("search_memory")):
        memory_tools_for_graph = get_memory_tools(get_llm(llm_provider), sync_mode=sync_mode, store=memory_store)

    log(str(mcp_tools))
    normal_tools.extend([run_python_code, save_report])
    normal_tools.extend(get_support_quality_tools(catalog))
    if memory_tools_for_graph:
        normal_tools.extend(memory_tools_for_graph)
    # Forecasting/anomaly/drill-down analysis is delegated to the data analysis
    # agent (the `data_analysis` tool), which owns timeseries_forecast and the
    # related health check. It is intentionally not exposed directly here.
    normal_tools.extend(mcp_tools)

    if allowed is not None:
        available = {tool.name for tool in normal_tools} | {"AskHuman"}
        unknown = allowed - available
        if unknown:
            raise ValueError(f"Unknown or unavailable Agent tools: {sorted(unknown)}")
        normal_tools = [tool for tool in normal_tools if tool.name in allowed]

    # Initialize context manager if enabled
    context_manager = None
    if enable_context_management:
        context_manager = ContextManager(llm=get_llm(llm_provider), config=get_context_config())

    tool_node = ToolNode(normal_tools)

    # Define the agent graph
    graph = StateGraph(AgentState, input_schema=InputState, output_schema=OutputState)

    # Add nodes to the graph
    agent_tools: list[Any] = [*normal_tools, *([AskHuman] if enabled("AskHuman") else [])]
    graph.add_node("llm_node", agent_llm_call(get_llm(llm_provider), agent_tools, context_manager))
    graph.add_node("ask_human", ask_human)
    graph.add_node("use_tool", tool_node)

    # Add edges between nodes
    graph.add_edge(START, "llm_node")
    graph.add_edge("ask_human", "llm_node")
    graph.add_edge("use_tool", "llm_node")

    # Add conditional routing from llm node
    def route_tools(state: AgentState) -> str | list[Send]:
        # Only use sends if the last message came from the llm node (has tool_calls)
        last_message = state["messages"][-1] if state["messages"] else None
        if (
            last_message
            and isinstance(last_message, AIMessage)
            and last_message.tool_calls
            and "sends" in state
            and state["sends"]
        ):
            return state["sends"]  # Return Send objects for parallel execution
        next_node = state.get("agent_next_node")
        if next_node:
            return next_node  # Return single node name
        return END

    graph.add_conditional_edges(
        "llm_node",
        route_tools,
        # mapping of paths to node names (for single routing)
        {
            "llm_node": "llm_node",
            "ask_human": "ask_human",
            "use_tool": "use_tool",
            END: END,
        },
    )

    compiled_graph = graph.compile(name="agent_graph", checkpointer=checkpointer, store=memory_store)
    return compiled_graph


def build_agent_graph_sync(
    catalog: CatalogStore,
    checkpointer: Checkpointer | None = None,
    memory_store: BaseStore | None = None,
    enable_context_management: bool = True,
    llm_provider: str | None = None,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """Build the main agent graph with all nodes and edges (sync version).

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.
        enable_context_management: Whether to enable context management for long conversations.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    # Get MCP tools for sync context
    mcp_tools = create_mcp_tools_sync(_get_mcp_servers())

    return _build_graph_core(
        catalog=catalog,
        sync_mode=True,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=None,  # Always None for sync version - creates its own
        mcp_tools=mcp_tools,
        enable_context_management=enable_context_management,
        llm_provider=llm_provider,
    )


async def build_agent_graph_async(
    catalog: CatalogStore,
    checkpointer: Checkpointer | None = None,
    memory_store: BaseStore | None = None,
    memory_tools: list[Any] | None = None,
    enable_context_management: bool = True,
    llm_provider: str | None = None,
) -> CompiledStateGraph[Any, None, Any, Any]:
    """Build the main agent graph with all nodes and edges (async version).

    This function is identical to build_agent_graph_sync but properly handles
    async MCP tool initialization when called from async contexts.

    Args:
        catalog: Catalog store containing schema information.
        checkpointer: The Checkpointer for state persistence (short memory). If None, no short memory.
        memory_store: The BaseStore to use for long-term memory. If None, will auto assign according to sync_mode.
        memory_tools: List of memory tools (manage_memory_tool, search_memory_tool). If None, creates async tools.
        enable_context_management: Whether to enable context management for long conversations.

    Returns:
        CompiledStateGraph: Compiled agent graph ready for execution.
    """
    # Get MCP tools for async context
    mcp_tools = await get_mcp_tools_async(_get_mcp_servers())

    return _build_graph_core(
        catalog=catalog,
        sync_mode=False,
        checkpointer=checkpointer,
        memory_store=memory_store,
        memory_tools=memory_tools,
        mcp_tools=mcp_tools,
        enable_context_management=enable_context_management,
        llm_provider=llm_provider,
    )
