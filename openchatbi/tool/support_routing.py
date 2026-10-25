"""Conservative tool exposure for the explicitly scoped support-demo profile."""

import json
import re
from datetime import date

from langchain_core.messages import HumanMessage, ToolMessage

TOOLS_BY_MODE = {
    "audit": {"AskHuman", "search_schema", "show_schema", "text2sql"},
    "knowledge": {"AskHuman", "search_knowledge"},
    "metrics": {"AskHuman", "support_quality"},
    "clarification": {"AskHuman", "support_quality"},
}

RELATIVE_TIME = re.compile(
    r"这周|本周|上周|下周|这星期|本星期|上星期|下星期|这个月|本月|上个月|下个月|"
    r"今天|今日|昨天|昨日|明天|最近|近期|今年|去年|本季度|上季度|"
    r"\b(?:this|last|next)\s+(?:week|month|year|quarter)\b|"
    r"\b(?:today|yesterday|tomorrow|recently|past\s+\d+\s+days?)\b",
    re.IGNORECASE,
)


def relative_time_clarification(messages: list, snapshot: date | None = None) -> str | None:
    """Require explicit dates instead of resolving relative time from old results."""
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    text = messages[-1].content
    if not isinstance(text, str):
        return None
    match = RELATIVE_TIME.search(text)
    if not match:
        return None
    # Only a fully specified current-message range overrides a relative label.
    dates = re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", text)
    if len(dates) == 2:
        try:
            start, end = map(date.fromisoformat, dates)
            if start < end:
                return None
        except ValueError:
            pass
    question = (
        f"你说的“{match.group()}”是相对时间，不能直接沿用历史查询日期。"
        "请确认具体的 UTC 开始日期（包含）和结束日期（不包含）；"
        "本版暂不自动换算相对时间，也不会把历史结果当作本周数据。"
    )
    if snapshot is not None:
        question += (
            f"当前合成演示数据的观察截止为 {snapshot.isoformat()} 00:00:00 UTC（不含），"
            "不代表真实当前数据。如要沿用演示范围，请明确回复起止日期。"
        )
    return question


def support_task_mode(messages: list) -> str:
    latest = next((message.content for message in reversed(messages) if isinstance(message, HumanMessage)), "")
    if not isinstance(latest, str):
        return "clarification"
    text = latest.casefold()
    if any(term in text for term in ("原始", "raw record", "raw table")) or re.search(
        r"\bsql\b|\b(?:max|min)\s*\(|(?:最大|最小)\s*(?:id|编号)", text
    ):
        return "audit"
    if any(
        term in text
        for term in (
            "分母",
            "定义",
            "规则",
            "出处",
            "政策",
            "规定",
            "怎么算",
            "怎么计算",
            "如何计算",
            "为什么",
            "denominator",
            "definition",
            "policy",
            "formula",
        )
    ):
        return "knowledge"
    grouping = any(term in text for term in ("按类别", "按分类", "拆开", "分组", "by category"))
    if "口径" in text and not grouping and not any(term in text for term in ("统计", "计算", "查询")):
        return "knowledge"
    if any(
        term in text
        for term in (
            "超时率",
            "工单数量",
            "工单数",
            "平均处理",
            "平均耗时",
            "overdue",
            "ticket_count",
            "avg_resolution",
        )
    ):
        return "metrics"
    if grouping:
        for message in reversed(messages):
            if isinstance(message, ToolMessage) and message.name == "support_quality":
                try:
                    if json.loads(message.content).get("status") == "succeeded":
                        return "metrics"
                except (TypeError, ValueError, AttributeError):
                    pass
    return "clarification"


def scoped_support_tools(tools: list, messages: list) -> tuple[list, str]:
    mode = support_task_mode(messages)
    allowed = TOOLS_BY_MODE[mode]
    active = [tool for tool in tools if (getattr(tool, "name", None) or getattr(tool, "__name__", None)) in allowed]
    if not active:
        raise ValueError("Configured tools cannot serve the requested support task")
    return active, mode


def tool_attempted_this_turn(messages: list, name: str) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name == name:
            return True
    return False
