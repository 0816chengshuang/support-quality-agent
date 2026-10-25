"""Opt-in registration of the local support-quality Skill."""

import json
from datetime import date
from importlib.resources import files

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from openchatbi import config
from openchatbi.analysis.support_quality import (
    METRICS,
    SupportGrouping,
    SupportMetric,
    SupportQualityInput,
    analyze_support_quality,
)
from openchatbi.catalog import CatalogStore

EVIDENCE_MARKER = "\n\n---\n数据：合成演示工单。"


class SupportQualitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_date: date


def apply_support_scope_guard(response: AIMessage, ask_human_enabled: bool, messages: list | None = None) -> AIMessage:
    """Route missing scope to the existing interrupt node, recording the intervention."""
    latest = next((message.content for message in reversed(messages or []) if isinstance(message, HumanMessage)), "")
    if (
        isinstance(latest, str)
        and ("整体" in latest or "overall" in latest.casefold())
        and not any(term in latest.casefold() for term in ("类别", "分类", "category"))
    ):
        calls = [dict(call, args=dict(call["args"])) for call in response.tool_calls]
        corrected = False
        for call in calls:
            if call["name"] == "support_quality" and call["args"].get("group_by", "overall") is None:
                call["args"]["group_by"] = "overall"
                corrected = True
        if corrected:
            response = response.model_copy(
                update={
                    "tool_calls": calls,
                    "response_metadata": {
                        **response.response_metadata,
                        "support_explicit_group_guard": {
                            "group_by": "overall",
                            "source": "current_user_request",
                            "original_tool_calls": response.tool_calls,
                        },
                    },
                }
            )
    if not ask_human_enabled:
        return response
    labels = {"metric": "统计指标", "start_date": "UTC 开始日期（包含）", "end_date": "UTC 结束日期（不包含）"}
    for call in response.tool_calls:
        if call["name"] != "support_quality":
            continue
        missing = [key for key in labels if call["args"].get(key) is None]
        if missing:
            question = "请补充：" + "、".join(labels[key] for key in missing) + "。"
            if "metric" in missing:
                question += "支持工单数量、平均处理时长、超时率。"
            return response.model_copy(
                update={
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "AskHuman",
                            "args": {"question": question, "options": []},
                            "id": call["id"],
                            "type": "tool_call",
                        }
                    ],
                    "response_metadata": {
                        **response.response_metadata,
                        "support_scope_guard": {
                            "missing_fields": missing,
                            "original_tool_calls": response.tool_calls,
                        },
                    },
                }
            )
    return response


def support_evidence_note(messages: list) -> str:
    """Render only this turn's actual successful tool evidence, not model claims."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, ToolMessage) or message.name != "support_quality":
            continue
        try:
            result = json.loads(message.content)
            if result["status"] != "succeeded":
                return ""
            contract = result["contract"]
            metric = contract["metric"]
            if metric not in METRICS or contract["source"] != METRICS[metric]["source"]:
                return ""
            rows = result["rows"]
            if not rows:
                return ""
            details = "; ".join(
                f"{row['group']}: {row['numerator']}/{row['denominator']}"
                if row["denominator"] is not None
                else f"{row['group']}: {row['value']}"
                for row in rows
            )
            return (
                EVIDENCE_MARKER + "\n"
                f"UTC 范围：[{contract['start_date']}, {contract['end_date']})；"
                f"观察截止（不含）：{contract['snapshot_exclusive']}。\n"
                f"统计依据（分子/分母）：{details}。\n口径：`{contract['source']}`。"
            )
        except (KeyError, TypeError, ValueError):
            return ""
    return ""


def append_support_evidence(answer: str, messages: list) -> str:
    note = support_evidence_note(messages)
    if not note:
        return answer
    # A model may copy a prior footer; replace it with this query's evidence.
    return answer.partition(EVIDENCE_MARKER)[0].rstrip() + note


def render_support_report(messages: list) -> str | None:
    """Render computed values, rather than asking a model to recalculate them."""
    statuses = {
        "needs_clarification": "尚未执行统计：必需信息或日期范围不符合要求，请调整后重试。",
        "no_data": "查询范围内没有符合条件的数据，不能解释为 0% 或零耗时。",
        "execution_error": "统计查询执行失败，没有可用结果，不能解释为没有数据。",
        "unsupported_database": "当前统计 Skill 不支持该数据库，未执行查询。",
        "result_limit_exceeded": "分类数量超过限制，不能展示为完整结果；请缩小筛选范围。",
    }
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, ToolMessage) or message.name != "support_quality":
            continue
        try:
            result = json.loads(message.content)
            if isinstance(result, dict) and result.get("status") in statuses:
                detail = result.get("message", "")
                return statuses[result["status"]] + (f"\n校验说明：{detail}" if detail else "")
        except (TypeError, ValueError):
            pass
        break
    note = support_evidence_note(messages)
    if not note:
        return None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, ToolMessage) or message.name != "support_quality":
            continue
        result = json.loads(message.content)
        metric = result["contract"]["metric"]
        lines = ["合成演示工单统计完成："]
        for row in result["rows"]:
            label = "整体" if row["group"] == "all" else row["group"]
            value = row["value"]
            if value is None:
                lines.append(f"- {label}：暂无可计算值，不按零处理。")
            elif metric == "overdue_rate":
                lines.append(
                    f"- {label}超时率：{value * 100:.6g}%（{row['numerator']:.6g}/{row['denominator']:.6g}）。"
                )
            elif metric == "avg_resolution_minutes":
                lines.append(
                    f"- {label}平均处理时长：{value:.6g} 分钟（已观察解决 {row['denominator']:.6g} 条；合计约 {row['numerator']:.6g} 分钟）。"
                )
            else:
                lines.append(f"- {label}有效工单数量：{value:.6g} 条。")
        return "\n".join(lines) + note
    return None


def create_support_quality_tool(catalog: CatalogStore, snapshot_date: date) -> StructuredTool:
    """Use the existing catalog connection, without creating another agent runtime."""
    instructions = files("openchatbi").joinpath("skills/support_quality/SKILL.md").read_text(encoding="utf-8")

    def support_quality(
        metric: SupportMetric | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        group_by: SupportGrouping = "overall",
        category: str | None = None,
    ) -> dict:
        request = SupportQualityInput(
            metric=metric, start_date=start_date, end_date=end_date, group_by=group_by, category=category
        )
        return analyze_support_quality(catalog.get_sql_engine(), request, snapshot_date)

    return StructuredTool.from_function(
        func=support_quality,
        name="support_quality",
        description=instructions,
        args_schema=SupportQualityInput,
        return_direct=False,
    )


def get_support_quality_tools(catalog: CatalogStore) -> list[StructuredTool]:
    """Absent configuration preserves the upstream tool set."""
    try:
        bi_config = config.get().bi_config
    except ValueError:
        return []
    settings = bi_config.get("support_quality_skill") if isinstance(bi_config, dict) else None
    if settings is None:
        return []
    resolved = SupportQualitySettings.model_validate(settings)
    return [create_support_quality_tool(catalog, resolved.snapshot_date)]
