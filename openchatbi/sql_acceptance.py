"""Shared evidence checks for the original main -> Text2SQL workflow."""

import csv
import io
import re
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import text

from openchatbi import config
from openchatbi.prompts.system_prompt import reset_cache

PARENT_TOOLS = frozenset({"text2sql", "search_schema", "show_schema", "AskHuman"})
REQUEST_TOOLS = PARENT_TOOLS | {"search_knowledge"}
QUESTION = "请查询合成工单原始表中的最大 id，包括测试、重复、已取消记录，不作业务排除、不限制日期。给出真实 SQL 和查询结果，不要图表。"
ROUTER_PROMPT = """你是只读工单审计的主助手，请始终用中文回答，保留工具名和 SQL 标识。
这些合成记录真实存放在 SQLite 中，可以执行只读 SELECT；合成数据不等于不能查询。
你的职责是发现表结构后，把查询委派给 text2sql 专项角色，它负责信息抽取、选表、生成并执行 SQL。
先看下方服务端提供的真实目录；若表和字段已明确，直接调用 text2sql，不要重复搜索或编造表名。
只有目录中信息不足时才 search_schema 或 show_schema。text2sql 的 context 使用完整英文查询要求，visualize=false。
仅查到表结构不算任务完成。禁止自己编写 SQL 后直接宣称完成，禁止根据表结构猜数字。
必须等 text2sql 返回实际 SQL 和 CSV 后，才能总结执行结果。执行失败就明确说失败，不能伪造结果。
用户明确要求查原始记录、包括测试/重复/取消记录时，不加业务排除条件；没有日期要求就不加日期条件。
只使用已绑定工具，不执行 Python。只有用户缺少必要查询条件时才 AskHuman，不对全量原始记录查询索要时间范围。
最后用中文总结真实查询结果，说明是合成演示数据，并保留实际 SQL。
"""


def csv_numbers(content: str) -> list[float]:
    matches = re.findall(r"```csv\s*\n(.*?)```", content, flags=re.DOTALL)
    if len(matches) != 1:
        return []
    rows = list(csv.DictReader(io.StringIO(matches[0].strip())))
    if len(rows) != 1 or len(rows[0]) != 1:
        return []
    try:
        return [float(next(iter(rows[0].values())))]
    except (TypeError, ValueError):
        return []


def run_sql_delegation(configuration: dict, record: dict, on_progress=None, use_configured_profile=False) -> dict:
    """The caller supplies a bounded client; all role decisions use the real model."""
    if not use_configured_profile:
        configuration["agent_tool_allowlist"] = sorted(PARENT_TOOLS)
        configuration["bi_config"]["agent_prompt_template"] = ROUTER_PROMPT
        configuration["bi_config"]["basic_knowledge_glossary"] = (
            "support_tickets contains fully synthetic raw records. A raw-record audit includes "
            "tests, duplicates and cancelled records; do not apply support-quality metric exclusions."
        )
    configuration["bi_config"]["include_catalog_overview"] = True
    configuration["visualization_mode"] = None
    configuration["sql_max_retries"] = 0
    configuration["extraction_max_tool_rounds"] = 1
    configuration["llm_max_attempts"] = 1
    config.set(configuration)
    catalog = config.get().catalog_store
    try:
        reset_cache()
        # Upstream SQL and schema indexes capture configuration at import time.
        from openchatbi.agent_graph import build_agent_graph_sync

        with catalog.get_sql_engine().connect() as connection:
            before = tuple(connection.execute(text("SELECT COUNT(*), MAX(id) FROM support_tickets")).one())
        record["independent_expected_max_id"] = before[1]
        graph = build_agent_graph_sync(catalog, checkpointer=MemorySaver(), enable_context_management=False)
        run_config = {"configurable": {"thread_id": f"sql-audit-{uuid.uuid4()}"}, "recursion_limit": 16}
        for namespace, update in graph.stream(
            {"messages": [HumanMessage(QUESTION)]},
            config=run_config,
            stream_mode="updates",
            subgraphs=True,
        ):
            for node, value in update.items():
                event = {"namespace": list(namespace), "node": node}
                if isinstance(value, dict):
                    event.update({key: value[key] for key in ("sql", "data", "sql_execution_result") if key in value})
                    if node == "table_selection":
                        event["selected_table_count"] = len(value.get("tables") or [])
                        event["selection_source"] = value.get("table_selection_source", "unknown")
                    elif node == "generate_sql":
                        event["generated_sql"] = bool(value.get("sql"))
                    elif node == "information_extraction":
                        event["extraction_source"] = value.get("extraction_source", "model_or_tool")
                record["events"].append(event)
            if on_progress:
                on_progress()
        messages = graph.get_state(run_config).values.get("messages", [])
        record["tool_calls"] = [call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls]
        record["tool_outputs"] = [
            {"name": m.name, "content": m.content} for m in messages if isinstance(m, ToolMessage)
        ]
        record["answer"] = next(
            (m.content for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), ""
        )
        with catalog.get_sql_engine().connect() as connection:
            after = tuple(connection.execute(text("SELECT COUNT(*), MAX(id) FROM support_tickets")).one())
        results = [item["content"] for item in record["tool_outputs"] if item["name"] == "text2sql"]
        checks = {
            "main_delegated_to_text2sql": any(call["name"] == "text2sql" for call in record["tool_calls"]),
            "nested_sql_execution_observed": any(
                e["node"] == "execute_sql" and e["namespace"] for e in record["events"]
            ),
            "csv_matches_independent_database_query": any(
                csv_numbers(value) == [float(before[1])] for value in results
            ),
            "data_unchanged": before == after,
            "has_final_answer": bool(record["answer"]),
        }
        record["checks"] = checks
        record["status"] = "passed" if all(checks.values()) else "checks_failed"
        return record
    finally:
        catalog.get_sql_engine().dispose()
