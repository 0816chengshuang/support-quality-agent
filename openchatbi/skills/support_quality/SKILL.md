# Support Quality Analysis Skill

仅用于有效工单的三种业务指标：工单数量、平均处理时长、超时率。
原始表审计、MAX/MIN 字段值、包括全部测试/重复/取消记录的查询应使用 text2sql，
不能在本工具中伪造指标或索要无关日期。定义/政策问题应 search_knowledge 检索规则。

Use the support_quality tool for support-ticket volume, average resolution time
and overdue-rate questions against the configured synthetic support dataset.

1. Identify an explicitly requested metric and a UTC date range. Dates use an
   inclusive start and exclusive end. Do not invent missing dates or translate
   an ambiguous request such as "quality" into a metric without asking.
   Relative time such as "this week" is a new time constraint, not permission
   to reuse historical dates. This demo asks for explicit UTC start/end dates
   unless the current request already contains a complete YYYY-MM-DD range.
   Never relabel a historical result as current-week data.
2. Pass missing metric/dates as null. If the result is needs_clarification, use
   AskHuman to obtain the missing information before retrying. Never interpret
   needs_clarification, execution_error or result_limit_exceeded as success.
3. The tool validates scope and executes fixed, parameterized SQL. It includes
   overdue unresolved tickets, excludes test/duplicate/cancelled tickets, and
   uses the configured snapshot rather than the machine's current time.
4. Report the value, unit, time window, snapshot, numerator/denominator and
   returned definition source. A ratio of 0.4 means 40%. A null average means no
   observed resolutions, not zero minutes. no_data is not a quality score.
5. Use group_by=category for a breakdown. Never average category percentages
   without their denominators. Do not infer causes from a category difference.

Definitions: support-quality-v1 in METRICS.md in this Skill directory.
This is a curated task Skill exposed through one tool, not dynamic Skill
discovery, document RAG, an autonomous second agent or a general SQL sandbox.
