"""A bounded, deterministic support-quality Skill, not an LLM evaluator."""

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, text
from sqlalchemy.exc import SQLAlchemyError

SupportMetric = Literal["ticket_count", "avg_resolution_minutes", "overdue_rate"]
SupportGrouping = Literal["overall", "category"]

DEFINITION_VERSION = "support-quality-v1"
MAX_GROUPS = 20
MAX_WINDOW_DAYS = 93
METRICS = {
    "ticket_count": {
        "unit": "tickets",
        "formula": "count of eligible tickets created in [start_date, end_date)",
        "source": "support-quality-v1#ticket-count",
    },
    "avg_resolution_minutes": {
        "unit": "minutes",
        "formula": "resolution minutes / tickets resolved before the snapshot",
        "source": "support-quality-v1#average-resolution-minutes",
    },
    "overdue_rate": {
        "unit": "ratio",
        "formula": "overdue tickets / all eligible tickets, including unresolved tickets",
        "source": "support-quality-v1#overdue-rate",
    },
}


class SupportQualityInput(BaseModel):
    """Missing intent is returned for clarification, never silently inferred."""

    model_config = ConfigDict(extra="forbid")

    metric: SupportMetric | None = Field(
        default=None,
        description=(
            "Map the user's requested metric to an enum: overdue_rate = 超时率/逾期比例; "
            "ticket_count = 工单数量/新建工单量; avg_resolution_minutes = 平均处理时长/平均解决时间. "
            "Use null only when the requested metric is missing or ambiguous."
        ),
    )
    start_date: date | None = Field(default=None, description="Inclusive UTC date, YYYY-MM-DD; null if unspecified")
    end_date: date | None = Field(default=None, description="Exclusive UTC date, YYYY-MM-DD; null if unspecified")
    group_by: SupportGrouping = Field(default="overall", description="overall = 整体统计; category = 按类别拆分")
    category: str | None = Field(
        default=None, min_length=1, max_length=60, description="Optional exact category filter"
    )


def analyze_support_quality(engine: Engine, request: SupportQualityInput, snapshot_date: date) -> dict:
    """Validate scope, execute one parameterized SELECT, and return auditable values."""
    missing = [name for name in ("metric", "start_date", "end_date") if getattr(request, name) is None]
    if missing:
        return {
            "status": "needs_clarification",
            "missing_fields": missing,
            "message": "Ask the user for the missing metric and UTC date range; do not guess.",
            "rows": [],
        }
    start_date, end_date, metric = request.start_date, request.end_date, request.metric
    assert start_date is not None and end_date is not None and metric is not None
    if not (0 < (end_date - start_date).days <= MAX_WINDOW_DAYS) or end_date > snapshot_date:
        return {
            "status": "needs_clarification",
            "missing_fields": ["date_range"],
            "message": f"Use a nonempty range of at most {MAX_WINDOW_DAYS} days ending by {snapshot_date} (UTC).",
            "rows": [],
        }
    if engine.dialect.name != "sqlite":
        return {"status": "unsupported_database", "message": "This Skill currently supports SQLite only.", "rows": []}

    dimension = "category" if request.group_by == "category" else "'all'"
    grouping = "GROUP BY category" if request.group_by == "category" else ""
    # Only fixed enum-selected fragments enter SQL; all filter values are bound.
    sql = f"""SELECT {dimension} AS group_value,
        COUNT(*) AS ticket_count,
        SUM(CASE WHEN resolved_at < :snapshot THEN 1 ELSE 0 END) AS resolved_count,
        SUM(CASE WHEN resolved_at < :snapshot
            THEN (julianday(resolved_at) - julianday(created_at)) * 1440.0 ELSE 0 END) AS resolution_minutes,
        SUM(CASE WHEN due_at < :snapshot AND
            (resolved_at IS NULL OR resolved_at >= :snapshot OR resolved_at > due_at)
            THEN 1 ELSE 0 END) AS overdue_count
        FROM support_tickets
        WHERE created_at >= :start AND created_at < :end
            AND is_test = 0 AND duplicate_of IS NULL
            AND (cancelled_at IS NULL OR cancelled_at >= :snapshot)
            AND (:category IS NULL OR category = :category)
        {grouping}
        ORDER BY group_value
        LIMIT :group_limit"""
    params = {
        "start": f"{start_date} 00:00:00",
        "end": f"{end_date} 00:00:00",
        "snapshot": f"{snapshot_date} 00:00:00",
        "category": request.category,
        "group_limit": MAX_GROUPS + 1,
    }
    contract = {
        **request.model_dump(mode="json"),
        "timezone": "UTC",
        "snapshot_exclusive": params["snapshot"],
        "definition_version": DEFINITION_VERSION,
        **METRICS[metric],
        "cohort_source": "support-quality-v1#cohort",
    }
    fingerprint = hashlib.sha256(json.dumps({"sql": sql, "params": params}, sort_keys=True).encode()).hexdigest()
    evidence = {"sql": sql, "parameters": params, "query_fingerprint": fingerprint}
    try:
        with engine.connect() as connection:
            records = connection.execute(text(sql), params).mappings().all()
    except SQLAlchemyError:
        return {
            "status": "execution_error",
            "message": "The support dataset could not be queried. No result is available.",
            "contract": contract,
            "evidence": evidence,
            "rows": [],
        }
    if len(records) > MAX_GROUPS:
        return {
            "status": "result_limit_exceeded",
            "message": f"More than {MAX_GROUPS} categories; narrow the filter rather than interpreting a partial result.",
            "contract": contract,
            "evidence": evidence,
            "rows": [],
        }
    rows = []
    for record in records:
        count = int(record["ticket_count"])
        if not count:
            continue
        resolved = int(record["resolved_count"] or 0)
        overdue = int(record["overdue_count"] or 0)
        minutes = float(record["resolution_minutes"] or 0)
        if metric == "ticket_count":
            numerator, denominator, value = count, None, count
        elif metric == "avg_resolution_minutes":
            numerator, denominator = round(minutes, 6), resolved
            value = round(minutes / resolved, 2) if resolved else None
        else:
            numerator, denominator, value = overdue, count, round(overdue / count, 6)
        rows.append(
            {
                "group": record["group_value"],
                "value": value,
                "numerator": numerator,
                "denominator": denominator,
                "ticket_count": count,
                "resolved_count": resolved,
                "overdue_count": overdue,
            }
        )
    return {
        "status": "succeeded" if rows else "no_data",
        "contract": contract,
        "rows": rows,
        "evidence": evidence,
        "warnings": [
            "Synthetic/demo data is not evidence about a real organization.",
            "Correlations and group differences do not establish causes.",
        ],
    }
