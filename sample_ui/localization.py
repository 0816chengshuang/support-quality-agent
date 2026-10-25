"""Chinese UI labels for structured events; preserve SQL and source identifiers."""

TOOLS = {
    "AskHuman": "确认查询条件",
    "support_quality": "工单指标统计",
    "search_knowledge": "规则检索",
    "search_schema": "搜索数据表",
    "show_schema": "查看表结构",
    "text2sql": "SQL 查询助手",
    "data_analysis": "数据分析助手",
}


def step_text_zh(event) -> str:
    data = event.data
    if event.kind in {"tool", "sub_agent"}:
        names = data.get("tools", data.get("sub_agents", []))
        return "执行：" + "、".join(TOOLS.get(name, name) for name in names)
    if event.kind == "rewrite":
        return "整理本轮查询条件。"
    if event.kind == "tables":
        return f"选定数据表：{data.get('tables', [])}"
    if event.kind == "sql":
        return f"生成 SQL：\n```sql\n{data.get('sql', '')}\n```"
    if event.kind == "execute_sql":
        return "执行只读 SQL 查询。"
    if event.kind == "visualization":
        return "生成图表。"
    if event.kind == "confidence":
        score = data.get("sql_confidence")
        return f"SQL 检查评分：{score:.2f}" if score is not None else "SQL 检查评分暂不可用。"
    if event.kind == "tool_error":
        return f"{TOOLS.get(data.get('tool'), '工具')}执行失败，本步骤未获得有效结果。"
    return (
        event.text.replace("Using tool:", "调用工具：")
        .replace("Using tools:", "调用工具：")
        .replace("result：", "结果：")
    )
