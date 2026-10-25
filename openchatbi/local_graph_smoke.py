"""Bounded CLI checks of the existing graph; never start any page or service."""

import argparse
import contextlib
import json
import re
import time
import uuid
from pathlib import Path

import httpx
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from openchatbi import config
from openchatbi.local_retrieval_smoke import EMBEDDING_MODEL, LocalEmbeddingBudget
from openchatbi.local_support_smoke import require_local_ollama
from openchatbi.prompts.system_prompt import reset_cache
from openchatbi.support_demo import LOCAL_MODEL_NAME
from openchatbi.tool.support_quality import EVIDENCE_MARKER

ALLOWED_TOOLS = frozenset({"support_quality", "search_knowledge", "AskHuman"})
CONFIGURED_TOOLS = ALLOWED_TOOLS | {"search_schema", "show_schema", "text2sql"}
SCENARIOS = {
    "overall": "统计 2026-10-01（含）到 2026-10-08（不含）新建工单的超时率，使用 UTC，按整体统计，不筛选类别。",
    "followup": "保持刚才的日期和超时率口径，按类别拆开看。",
    "definition": "超时率分母包括未解决的工单吗？请查规则，给出规则出处和 chunk_id。",
    "missing": "知识库规定退款审核需要几个工作日？请查资料给出处，资料没有就直说，不要猜。",
    "clarification": "帮我看看工单质量。",
    "average": "计算 2026-10-01（含）到 2026-10-08（不含）新建有效工单的平均处理时长，UTC，整体统计。",
    "future": "计算 2026-10-10（含）到 2026-10-11（不含）新建工单的超时率，UTC。",
    "newsession": "沿用刚才的日期和指标，按类别拆开看。",
    "raw": "请查询合成工单原始表中的最大 id，包括测试、重复、已取消记录，不作业务排除、不限制日期。给出真实 SQL 和查询结果，不要图表。",
}


class LocalRequestBudget:
    def __init__(self, max_requests=16, max_seconds=420, per_turn=4, allowed_tools=ALLOWED_TOOLS):
        self.max_requests = max_requests
        self.deadline = time.monotonic() + max_seconds
        self.per_turn = per_turn
        self.requests = 0
        self.turn_start = 0
        self.allowed_tools = frozenset(allowed_tools)

    def begin_turn(self):
        self.turn_start = self.requests

    def check(self, request: httpx.Request):
        require_local_ollama(request)
        body = json.loads(request.content)
        if body.get("model") != LOCAL_MODEL_NAME:
            raise ValueError("Only the approved local test model is permitted")
        names = {item["function"]["name"] for item in body.get("tools", [])}
        if not names.issubset(self.allowed_tools):
            raise ValueError("Non-allowlisted tools must not be exposed to the model")
        if (
            self.requests >= self.max_requests
            or self.requests - self.turn_start >= self.per_turn
            or time.monotonic() >= self.deadline
        ):
            raise RuntimeError("Local diagnostic request/time budget exhausted")
        self.requests += 1


def _turn_record(messages, result):
    calls, outputs, usage = [], [], []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls)
            if message.usage_metadata:
                usage.append(message.usage_metadata)
        elif isinstance(message, ToolMessage):
            try:
                content = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                content = message.content
            outputs.append({"name": message.name, "content": content, "artifact": message.artifact})
    final = next((m.content for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), "")
    guards = [
        m.response_metadata["support_scope_guard"]
        for m in messages
        if isinstance(m, AIMessage) and "support_scope_guard" in m.response_metadata
    ]
    return {
        "answer": final,
        "tool_calls": calls,
        "tool_outputs": outputs,
        "usage": usage,
        "runtime_scope_guards": guards,
        "runtime_tool_scopes": [
            m.response_metadata["support_tool_scope"]
            for m in messages
            if isinstance(m, AIMessage) and "support_tool_scope" in m.response_metadata
        ],
        "interrupts": [item.value for item in result.get("__interrupt__", [])],
    }


def check_scenario(name, turn):
    """Check observable evidence, never ask the tested model to grade itself."""
    successful = [
        item["content"]
        for item in turn["tool_outputs"]
        if item["name"] == "support_quality"
        and isinstance(item["content"], dict)
        and item["content"].get("status") == "succeeded"
    ]
    answer = str(turn["answer"])
    if name in {"overall", "clarification_resume", "followup"}:
        grouped = name == "followup"
        wanted = {"account": (3, 4), "billing": (0, 2), "delivery": (1, 4)} if grouped else {"all": (4, 10)}
        matched = any(
            data["contract"]["metric"] == "overdue_rate"
            and data["contract"]["start_date"] == "2026-10-01"
            and data["contract"]["end_date"] == "2026-10-08"
            and data["contract"]["category"] is None
            and data["contract"]["group_by"] == ("category" if grouped else "overall")
            and {row["group"]: (row["numerator"], row["denominator"]) for row in data["rows"]} == wanted
            for data in successful
        )
        return {
            "correct_scope_and_query": matched,
            "has_final_answer": bool(answer),
            "has_demo_label_and_evidence": "合成演示" in answer and "support-quality-v1#超时率" in answer,
            "single_evidence_footer": answer.count(EVIDENCE_MARKER) == 1,
            "answer_contains_expected_rates": all(
                value in answer.replace(" ", "") for value in (["75%", "0%", "25%"] if grouped else ["40%"])
            ),
        }
    if name == "clarification":
        return {"asked_user": bool(turn["interrupts"]), "no_successful_query": not successful}
    if name == "newsession":
        return {"asks_for_missing_previous_scope": bool(turn["interrupts"]), "no_invented_success": not successful}
    if name == "future":
        rejected = [
            item["content"]
            for item in turn["tool_outputs"]
            if item["name"] == "support_quality"
            and isinstance(item["content"], dict)
            and item["content"].get("status") == "needs_clarification"
        ]
        return {
            "date_range_rejected": any("date_range" in item.get("missing_fields", []) for item in rejected),
            "no_successful_query": not successful,
            "reports_issue_or_asks_user": bool(answer or turn["interrupts"]),
        }
    if name == "average":
        return {
            "actual_average_800": any(
                item["contract"]["metric"] == "avg_resolution_minutes" and item["rows"][0]["value"] == 800
                for item in successful
            ),
            "answer_contains_value": "800" in answer,
            "has_chinese_explanation": bool(re.search(r"[\u4e00-\u9fff]", answer)),
        }
    if name == "raw":
        from openchatbi.constants import SQL_SUCCESS
        from openchatbi.sql_acceptance import csv_numbers
        from openchatbi.tool.sql_handoff import SQL_EVIDENCE_MARKER

        executed = [
            item
            for item in turn["tool_outputs"]
            if item["name"] == "text2sql"
            and isinstance(item.get("artifact"), dict)
            and item["artifact"].get("status") == SQL_SUCCESS
        ]
        return {
            "executed_real_query": bool(executed),
            "actual_csv_matches_fixture": any(csv_numbers(item["content"]) == [15.0] for item in executed),
            "single_actual_sql_footer": answer.count(SQL_EVIDENCE_MARKER) == 1,
            "sql_and_csv_visible": "SELECT" in answer and "```csv" in answer,
        }
    business = [
        item["content"].get("business", {})
        for item in turn["tool_outputs"]
        if item["name"] == "search_knowledge" and isinstance(item["content"], dict)
    ]
    if name == "definition":
        passages = [p for data in business for p in data.get("passages", [])]
        cited = [
            p
            for p in passages
            if p["source_ref"] == "support-quality-v1#超时率" and p["source_ref"] in answer and p["chunk_id"] in answer
        ]
        return {"retrieved_and_cited_definition": bool(cited), "has_final_answer": bool(answer)}
    return {
        "searched_business_documents": bool(business),
        "explicitly_reports_missing_policy": any(
            word in answer for word in ("没有", "未找到", "无法", "不足", "未提供", "不包含")
        ),
        "does_not_invent_working_days": not any(f"{n}个工作日" in answer.replace(" ", "") for n in range(1, 31)),
    }


def run_local_graph_smoke(
    repository: Path, scenarios=tuple(SCENARIOS), *, use_configured_tools=False, vector_retrieval=False
) -> dict:
    root = repository.resolve(strict=True)
    if not scenarios or any(name not in SCENARIOS for name in scenarios):
        raise ValueError("Unknown diagnostic scenario")
    if "followup" in scenarios and "overall" not in scenarios:
        raise ValueError("The followup requires the overall scenario in the same run")
    configuration = yaml.safe_load((root / ".local/support-demo/config.local.yaml").read_text(encoding="utf-8"))
    expected_uri = f"sqlite:///file:{(root / '.local/support-demo/support.sqlite').as_posix()}?mode=ro&uri=true"
    if configuration["data_warehouse_config"]["uri"] != expected_uri:
        raise ValueError("Diagnostic requires the prepared local read-only SQLite dataset")
    if configuration.get("mcp_servers") or configuration.get("embedding_model") or configuration.get("llm_providers"):
        raise ValueError("No external services or embeddings are allowed in this local diagnostic")
    budget = LocalRequestBudget(allowed_tools=CONFIGURED_TOOLS if use_configured_tools else ALLOWED_TOOLS)
    started = time.monotonic()
    record = {
        "status": "not_completed",
        "model": LOCAL_MODEL_NAME,
        "turns": [],
        "paid_requests": 0,
        "page_started": False,
        "scope": "prepared six-tool profile" if use_configured_tools else "three-tool profile",
        "ui_acceptance": False,
    }
    report_name = (
        "local-graph-vector.json"
        if vector_retrieval
        else ("local-graph-configured.json" if use_configured_tools else "local-graph-smoke.json")
    )
    report = root / ".run" / report_name
    report.parent.mkdir(exist_ok=True)
    embedding_budget = None

    def save():
        record["requests"] = budget.requests
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if embedding_budget:
            record["embedding_requests"] = embedding_budget.requests
            record["embedding_model"] = EMBEDDING_MODEL
        report.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# 本地真实模型操作回放",
            "",
            "这是 CLI 验证记录，不是网页验收；数据均为合成，未调用付费 API。",
            "",
            f"状态：{record['status']}；本机模型请求：{budget.requests} 次。",
            "",
        ]
        for turn in record["turns"]:
            name = turn["scenario"]
            question = SCENARIOS["overall"] if name == "clarification_resume" else SCENARIOS[name]
            lines.extend(
                [
                    f"## {name}",
                    "",
                    f"输入：{question}",
                    "",
                    "实际工具：" + (", ".join(call["name"] for call in turn["tool_calls"]) or "无"),
                    "",
                    str(turn["answer"]) or "等待用户补充。",
                    "",
                ]
            )
            if turn["interrupts"]:
                lines.extend(
                    ["暂停信息：", "```json", json.dumps(turn["interrupts"], ensure_ascii=False, indent=2), "```", ""]
                )
            lines.extend(["检查：" + ("通过" if turn["passed"] else "未通过"), ""])
        report.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    catalog = None
    try:
        with contextlib.ExitStack() as stack:
            client = stack.enter_context(
                httpx.Client(timeout=60, trust_env=False, event_hooks={"request": [budget.check]})
            )
            if vector_retrieval:
                embedding_budget = LocalEmbeddingBudget()
                embedding_client = stack.enter_context(
                    httpx.Client(timeout=60, trust_env=False, event_hooks={"request": [embedding_budget.check]})
                )
                configuration["embedding_model"] = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    base_url="http://127.0.0.1:11434/v1",
                    api_key="ollama-local-not-a-secret",
                    max_retries=0,
                    request_timeout=60,
                    check_embedding_ctx_length=False,
                    http_client=embedding_client,
                )
                configuration["bi_config"]["business_documents"]["mode"] = "vector"
            configuration["default_llm"] = ChatOpenAI(
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama-local-not-a-secret",
                model=LOCAL_MODEL_NAME,
                temperature=0,
                max_tokens=384,
                timeout=60,
                max_retries=0,
                http_client=client,
                use_responses_api=False,
                http_socket_options=(),
            )
            if not use_configured_tools:
                configuration["agent_tool_allowlist"] = sorted(ALLOWED_TOOLS)
            config.set(configuration)
            reset_cache()
            # Upstream SQL/schema indexes initialize at import time from config.
            from openchatbi.agent_graph import build_agent_graph_sync

            catalog = config.get().catalog_store
            graph = build_agent_graph_sync(catalog, checkpointer=MemorySaver(), enable_context_management=False)
            overall_thread = f"local-overall-{uuid.uuid4()}"

            def run_turn(name, payload, thread):
                run_config = {"configurable": {"thread_id": thread}, "recursion_limit": 10}
                before = len(graph.get_state(run_config).values.get("messages", []))
                budget.begin_turn()
                turn_started = time.monotonic()
                result = graph.invoke(payload, config=run_config)
                turn = {"scenario": name, **_turn_record(result["messages"][before:], result)}
                turn["checks"] = check_scenario(name, turn)
                turn["passed"] = all(turn["checks"].values())
                turn["requests"] = budget.requests - budget.turn_start
                turn["elapsed_seconds"] = round(time.monotonic() - turn_started, 2)
                record["turns"].append(turn)
                save()
                return turn

            for name in scenarios:
                thread = overall_thread if name in {"overall", "followup"} else f"local-{uuid.uuid4()}"
                turn = run_turn(name, {"messages": [HumanMessage(SCENARIOS[name])]}, thread)
                if name == "clarification" and turn["interrupts"]:
                    run_turn("clarification_resume", Command(resume=SCENARIOS["overall"]), thread)
            record["status"] = "passed" if all(turn["passed"] for turn in record["turns"]) else "checks_failed"
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
    finally:
        if catalog is not None:
            catalog.get_sql_engine().dispose()
        save()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configured-tools", action="store_true", help="Use the prepared six-tool profile")
    parser.add_argument("--vector", action="store_true", help="Use the approved local embedding model")
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=tuple(SCENARIOS))
    args = parser.parse_args()
    result = run_local_graph_smoke(
        Path.cwd(), args.scenarios, use_configured_tools=args.configured_tools, vector_retrieval=args.vector
    )
    for turn in result["turns"]:
        print(f"{turn['scenario']}: {'PASS' if turn['passed'] else 'FAIL'} ({turn['requests']} local requests)")
    print(
        json.dumps({key: result.get(key) for key in ("status", "requests", "elapsed_seconds", "error_type")}, indent=2)
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
