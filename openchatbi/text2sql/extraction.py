"""Information extraction module for text2sql processing."""

import traceback
from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from openchatbi import config
from openchatbi.constants import EXTRACTION_MAX_TOOL_ROUNDS
from openchatbi.graph_state import SQLGraphState
from openchatbi.llm.llm import call_llm_chat_model_with_retry
from openchatbi.prompts.system_prompt import get_basic_knowledge, get_extraction_prompt_template
from openchatbi.utils import extract_json_from_answer, get_text_from_content, log


def _get_max_extraction_tool_rounds() -> int:
    """Read the extraction tool-round limit from Config, defaulting to EXTRACTION_MAX_TOOL_ROUNDS."""
    try:
        cfg = config.get()
    except ValueError:
        return EXTRACTION_MAX_TOOL_ROUNDS
    max_rounds = getattr(cfg, "extraction_max_tool_rounds", EXTRACTION_MAX_TOOL_ROUNDS)
    if not isinstance(max_rounds, int) or max_rounds < 0:
        return EXTRACTION_MAX_TOOL_ROUNDS
    return max_rounds


def _generate_extraction_prompt() -> str:
    """Generate extraction prompt.

    Returns:
        str: Generated prompt with placeholders replaced.
    """
    try:
        compact = getattr(config.get(), "compact_sql_workflow", False) is True
    except ValueError:
        compact = False
    if compact:
        return (
            "把最新用户请求改写成明确的英文查询要求，保留字段、聚合方式、包含/排除条件。"
            "之前的问题仅作上下文，最新要求优先。用户明确全部记录/不作排除时 filter=[]，"
            "没有日期或明确不限制日期时 start_time/end_time=null；不要套业务指标过滤规则。"
            "原始记录、最大/最小值、行数是明确的基础查询，不需要查业务定义或追问日期。"
            "只输出紧凑 JSON，不输出解释或推理："
            '{"keywords":["requested concepts"],"dimensions":["requested fields"],'
            '"metrics":["requested aggregates"],"filter":[],"start_time":null,'
            '"end_time":null,"timezone":null,"rewrite_question":"complete English request"}.'
            "真正缺少必要条件才 AskHuman；不熟悉的业务词才 search_knowledge，"
            '且显式传 knowledge_bases=["business"] 和不超过 5 个 query_list 词。'
        )
    prompt = get_extraction_prompt_template()

    date_str = date.today().strftime("%Y-%m-%d")
    prompt = prompt.replace("[time_field_placeholder]", date_str)
    prompt = prompt.replace("[basic_knowledge_glossary]", get_basic_knowledge())
    return prompt


def _parse_extracted_info_json(llm_answer_content: Any) -> dict[str, Any]:
    """Extract and parse JSON from LLM response.

    Args:
        llm_answer_content: LLM response containing JSON.

    Returns:
        dict: Parsed JSON or empty dict if parsing fails.
    """
    try:
        text = get_text_from_content(llm_answer_content)
        result = extract_json_from_answer(text)
    except Exception:
        log(traceback.format_exc())
        result = {}
    return result


def information_extraction(llm: BaseChatModel) -> Callable:
    """Create function to extract information from questions.

    Args:
        llm (BaseChatModel): Language model for information extraction.

    Returns:
        function: Node function that extracts information from questions.
    """

    def _extract(state: SQLGraphState):
        """Extract information from question in state.

        Args:
            state (SQLGraphState): Current SQL graph state with question.

        Returns:
            dict: Updated state with extracted information.
        """
        messages = state["messages"]
        last_message = messages[-1]
        user_input = last_message.content
        log(f"information_extraction: {user_input}")
        system_prompt = _generate_extraction_prompt()
        prompt = "Please extract the information according to the context."
        response = call_llm_chat_model_with_retry(
            llm,
            ([SystemMessage(system_prompt)] + messages + [HumanMessage(prompt)]),
            bound_tools=["search_knowledge", "AskHuman"],
        )
        if response:
            log(response)
            if response.tool_calls:
                # A new question enters as a HumanMessage; tool results come back as ToolMessages.
                rounds = 0 if isinstance(last_message, HumanMessage) else state.get("extraction_tool_rounds", 0)
                return {"messages": [response], "extraction_tool_rounds": rounds + 1}
            else:
                llm_answer_content = response.content
                parsed_result = _parse_extracted_info_json(llm_answer_content)
                rewrite = parsed_result.get("rewrite_question")
                source = "model"
                try:
                    compact = getattr(config.get(), "compact_sql_workflow", False) is True
                except ValueError:
                    compact = False
                fields = ("keywords", "dimensions", "metrics", "filter")
                valid_fields = all(isinstance(parsed_result.get(key), list) and all(isinstance(item, str) for item in parsed_result[key]) for key in fields)
                if compact and not rewrite and valid_fields and any(parsed_result[key] for key in fields):
                    rewrite = user_input
                    source = "original_request_fallback"
                return {
                    "messages": [response],
                    "rewrite_question": rewrite,
                    "extraction_source": source,
                    "info_entities": parsed_result,
                }
        else:
            return {"messages": [AIMessage(role="system", content="{}")]}

    return _extract


def information_extraction_conditional_edges(state: SQLGraphState):
    """Determine next node after information extraction.

    Args:
        state (SQLGraphState): Current SQL graph state.

    Returns:
        str: Next node ('ask_human', 'search_knowledge', 'next', or 'end').
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = None
    if isinstance(last_message, AIMessage):
        tool_calls = last_message.tool_calls
        log(f"tool_calls: {tool_calls}")
    if tool_calls:
        max_rounds = _get_max_extraction_tool_rounds()
        if state.get("extraction_tool_rounds", 0) > max_rounds:
            log(f"information_extraction: tool round limit ({max_rounds}) reached")
            return "end"
        if tool_calls[0]["name"] == "AskHuman":
            return "ask_human"
        elif tool_calls[0]["name"] == "search_knowledge":
            return "search_knowledge"
        else:
            print(f"Unknown tool call: {tool_calls[0]['name']}")
            return "end"
    else:
        if state.get("rewrite_question"):
            return "next"
        else:
            return "end"
