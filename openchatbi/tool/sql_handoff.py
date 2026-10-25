"""Optional scope anchoring for a single-request read-only assistant profile."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from openchatbi.constants import SQL_SUCCESS

SQL_EVIDENCE_MARKER = "\n\n---\nSQL 执行依据（只读查询）："


def append_sql_evidence(answer: str, messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, ToolMessage) or message.name != "text2sql":
            continue
        artifact = message.artifact
        if not isinstance(artifact, dict) or artifact.get("status") != SQL_SUCCESS:
            return answer
        sql, data = artifact.get("sql"), artifact.get("data")
        if not isinstance(sql, str) or not sql.strip() or not isinstance(data, str):
            return answer
        note = SQL_EVIDENCE_MARKER
        if len(sql) <= 2000 and len(data) <= 4000:
            note += f"\n```sql\n{sql}\n```\n```csv\n{data}\n```"
        else:
            note += "\n查询已执行；SQL 或结果较长，请查看工具执行记录。"
        return answer.partition(SQL_EVIDENCE_MARKER)[0].rstrip() + note
    return answer


def anchor_sql_user_request(response: AIMessage, messages: list) -> AIMessage:
    requests = [message.content for message in messages if isinstance(message, HumanMessage)][-3:]
    if not any(call["name"] == "text2sql" for call in response.tool_calls) or not requests:
        return response
    if any(not isinstance(item, str) for item in requests) or sum(map(len, requests)) > 8000:
        raise ValueError("SQL handoff requires at most three text requests within 8000 characters")
    context = (
        "Shared Context (earlier user requests; latest changes take precedence):\n"
        + json.dumps(requests[:-1], ensure_ascii=False)
        + "\nCurrent Subtask (authoritative latest user request):\n"
        + requests[-1]
        + "\nPreserve the user's inclusion/exclusion and date scope exactly. "
        "Use the server catalog for schema. Do not invent filters. Deferred Tasks: none."
    )
    calls = [
        {**call, "args": {**call["args"], "context": context}} if call["name"] == "text2sql" else call
        for call in response.tool_calls
    ]
    return response.model_copy(
        update={
            "tool_calls": calls,
            "response_metadata": {
                **response.response_metadata,
                "sql_scope_anchor": {"source": "recent_user_messages", "original_tool_calls": response.tool_calls},
            },
        }
    )
