# support-quality-v1

## Cohort

统计范围、观察截止点、有效工单：时间为 UTC，开始日期包含、结束日期不包含。
测试工单、重复工单，以及截止点之前已取消的工单不计入统计。

One row is one ticket. Timestamps are canonical UTC strings
`YYYY-MM-DD HH:MM:SS`. The created-time cohort is `[start_date, end_date)`;
`end_date <= snapshot_date`. The snapshot is exclusive at UTC midnight.
Exclude test tickets, duplicates and cancellations strictly before the snapshot.
Cancellation/resolution events at or after the snapshot are not yet observed.
`due_at` is the preassigned deadline, not a deadline inferred by the model.
The first version assumes no reopening or changing SLA deadlines.

## Ticket Count

工单数量、新建工单量：统计区间内的有效工单总数，包括尚未解决的工单。

Count all eligible tickets, whether resolved or unresolved.

## Average Resolution Minutes

平均处理时长、平均解决时间：已解决工单的处理分钟数之和除以已解决数量。
尚未解决的工单不以零分钟混入平均值；没有已解决记录时均值为空，不是零。

Among cohort tickets with `resolved_at < snapshot`, divide total elapsed UTC
minutes from creation to resolution by the resolved count. An empty resolved
set has a null average. Unresolved tickets must not be treated as zero minutes.

## Overdue Rate

超时率、逾期比例：超时工单数除以全部有效工单数。
分子包括尚未解决但已过截止时间的工单，也包括超过处理期限才解决的工单。
分母包含未解决和还没到处理期限的有效工单，不是只统计已经解决的记录。
刚好在处理期限解决不算超时。没有有效工单时没有比例，不能解释为 0%。

The numerator counts tickets whose `due_at < snapshot` and that are either
unresolved as of the snapshot or resolved strictly after their deadline.
Resolution exactly at the deadline is on time. A deadline exactly at the
snapshot is not overdue yet. The denominator is all eligible tickets, including
unresolved tickets and tickets not yet due. This is deliberately not a
"late among resolved" ratio. No eligible tickets means no_data, not a 0% rate.

## Evidence and Limits

来源与限制：查询指纹标识查询和参数，不证明数据库快照真实或不可篡改。
类别比例不能不带分母直接平均；类别差异不能直接推断业务原因。

The result contains bound parameters, SQL and a query fingerprint. The
fingerprint identifies the query/parameters, not a signed database snapshot.
Return at most 20 groups; if more exist, fail explicitly rather than returning
a misleading partial breakdown. Limit the date range to 93 days. This does not
constitute a query timeout or full resource isolation.

Only SQLite is currently supported. The Skill cannot execute model-supplied
SQL, but other upstream tools still exist. Use a read-only database connection;
this feature is not an application-wide permission system or SQL sandbox.
