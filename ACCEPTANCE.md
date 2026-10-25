# Current Acceptance Gate

Updated: 2026-10-23. **User requires an interactive frontend; CLI is not an accepted delivery.**
The original frontend was verified locally at http://127.0.0.1:8501/ with the local
model and synthetic read-only database. Browser checks are recorded separately
from earlier offline tests; user acceptance is still pending.

## Portfolio Demo Check (2026-10-25)

- Re-ran 360 selected offline tests after local-code formatting: all passed in 22.10s. This includes mocks and is not a model-quality score.
- With the real local Ollama model in the browser, verified overall overdue 4/10, category follow-up 3/4, 0/2 and 1/4, cited rule text with a chunk ID, and read-only raw SQL `MAX(id)=15`. Screenshots of the metric and SQL answers are in `docs/images/`.
- Ruff lint and format checks passed for the locally added modules and tests. A full upstream-tree lint still reports two untouched upstream findings in `text2sql/errors.py` and `tool/memory.py`.
- No paid API calls were made in this release check. Latency remains CPU/model dependent; no request-level p95, concurrency, or production SLA was measured.

## Current Demonstrated Scope

One synthetic support-quality loop: scoped statistic -> category follow-up ->
rule lookup/citation -> raw SQL audit -> missing information/policy/error handling.
The local Agent-learning feature scope is now closed for project write-up. The
15-row fixture is sufficient for repeatable metric and boundary tests but not
evidence of analysis scale or statistical conclusions. The visible metric path
is one main agent calling a deterministic tool; the upstream main -> Text2SQL
subgraph demonstrates limited role delegation only on raw SQL questions. Neither
path is a locally built, general Multi-Agent system. The existing browser demo,
project case study and recorded limitations are the acceptance surface; more UI
nodes or hundreds of synthetic rows are not required for this scoped write-up.
User review of the explanation remains separate from this technical conclusion.

- [x] Latest baseline: 360 offline tests passed (18.63 seconds), including
  relative-time guards, Chinese labels, and persisted clickable history.
- [x] Original-page non-streamed report display and waiting-state fixes; ten UI-focused
  offline tests passed (5.31 seconds), using scripted graphs without serving a page.
- [x] Original configured page + real async graph/checkpoint/SQLite, scripted model:
  statistics, category follow-up, new-thread isolation, old-history restoration,
  restored interrupt question and resume; initialization-failure cleanup/retry.
- [x] Exit cleanup registered once without expired session-state access; final suite
  exited cleanly. These checks do not open the browser or satisfy the UI gate below.
- [x] Prepared six-tool original LangGraph profile has earlier ten-step diagnostic
  notes; its named output files were later overwritten by a four-scenario partial
  run with `not_completed`, so they are not current full-replay acceptance evidence.
- [x] Overall 4/10=40%, category 3/4, 0/2, 1/4; dates preserved.
- [x] Average 800 minutes from six observed resolutions; explicit numerator/denominator.
- [x] Actual business-document retrieval with source_ref / chunk_id.
- [x] Missing policy searched and reported as no matching passage, not invented.
- [x] Missing scope interrupt and user-supplied information resume via the original checkpoint.
- [x] Invalid date range rejected; new session does not invent inherited dates.
- [x] Raw SQL main -> original Text2SQL -> real read-only SQLite execution,
  actual MAX(id)=15; SQL/CSV artifact shown, no metric exclusions silently applied.
- [x] Real local embeddings/cache: six small retrieval cases, not a production benchmark.
- [x] Original graph vector retrieval/citation: one chat + six embedding requests, 19.51 seconds.
- [x] Reproducible CLI command and readable transcript in example/support_quality/README.md.

Rules are extractive citations; numeric/SQL results and error displays are program-rendered.
This is intentional: the small model previously contradicted a correctly cited rule.
No claim of evaluated free-form generative RAG answers or general Chinese answer quality.

## Recorded Follow-ups (Not Required for Local Scope Freeze)

- [x] Explicit user authorization to run the reviewed unchanged upstream entry;
  deleted script abandoned, protection unchanged/no exclusions, stop on new alert.
- [x] Original entry started with local config; health ok, loopback-only port 8501.
- [x] Real browser + qwen2.5:3b: overall 40% (4/10), category follow-up
  75% / 0% / 25%, rule passages with source_ref/chunk_id, and old-history restoration.
- [x] New conversation uses a fresh session; old history restored by session ID.
  Switching back to the pending task restores the question; user-supplied scope
  resumes that checkpoint and returns 40% (4/10) in the real browser.
- [x] Browser after historical statistic: "这周超时率多少？" asks explicit UTC dates
  and shows the demo data cutoff; no old percentage or model request for this turn.
  This is a clarification policy, not automatic relative-date resolution.
- [x] Chinese controls and structured execution labels; original SQL/source content
  retained. New dialogue keeps history visible; real browser click restores original
  conversation 138c7f67ea3f. Fresh-page persistence/user filtering also checked offline
  with real async graph/SQLite, not paid/model-quality evidence.
- [x] Historical clarification questions are restored from AskHuman calls; old
  messages are not deleted or silently rewritten. Following the null-group repair,
  a real explicit-scope browser query returned the verified 40% (4/10).
- [x] Browser raw SQL handoff to the original asynchronous SQL subgraph completed:
  SELECT max(id) AS max_id FROM support_tickets, no metric/date filters; real result
  max_id=15 displayed with SQL/CSV. Evidence: .run/browser-acceptance.md,
  browser-statistics.jpg / browser-sql.jpg and upstream-ui.*.log.
- [x] User has exercised statistics, free-date questioning and history in the
  interface, reporting issues that were fixed. Reading the project case study
  and learning the technical details remain separate personal learning work.
- [ ] Complete paid-provider SQL workflow: historical API transport/alias/tool calling
  passed, but SQL did not execute. Both prior budgets (6 + 5 slots) are exhausted;
  no further paid request or ledger reset is permitted without new explicit approval.
  This optional provider path is not part of the accepted local-model project scope.

The user explicitly narrowed the earlier whole-UI pause on 2026-10-23. The
unchanged upstream entry is independently authorized; this is not a false-positive
verdict for the deleted script. Do not recreate/rewrite/rename that script or
change protection. Stop if the original entry raises a new alert; do not reroute.
Vendor feedback on the abandoned script remains unsubmitted (forum 405, no receipt),
but is no longer a prerequisite to the authorized original UI startup.

## Explicitly Out of Scope

Production accuracy/latency, multi-tenant permissions, complete SQL sandbox,
full autonomous Multi-Agent, analysis/Python, charts/report downloads, advanced
memory, dynamic Skill/MCP platform, multimodal, new branding, commits/pushes and
GitHub packaging. Do not silently turn these into new freeze requirements.

Upstream and local contributions are separated in BASELINE.md. The backend project
was not changed or re-accepted in this round. All demo data is synthetic; local
diagnostics have no paid fallback.
