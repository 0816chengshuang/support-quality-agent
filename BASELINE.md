# Local Baseline

Updated: 2026-10-25. Sources/license: [UPSTREAM.md](./UPSTREAM.md).
This is an upstream-based local demonstration candidate, not production acceptance.
Detailed historical runs remain in the coordination LEARNING_LOG.md; older statements
are historical, not the current acceptance state.

## Current public presentation (2026-10-25)

The README and design note now use a project-specific SVG business flow in
place of the generic Mermaid convergence. Its upper branch means exactly one
task path is selected; its lower rail means any selected path must pass the
same current-turn evidence gate. The design note contains the complete local
classification-and-latency example inline, with the separate four-path script
retained for repetition. The original upstream README had linked its own
`example/demo.gif`; that historical upstream animation is not evidence for
this project's business or latency demonstration. This round changes only
public documentation and diagram assets; no new model run or test result is
claimed.

The Chinese portfolio now includes Mermaid user/system/latency flows and a
four-path local latency demonstration. The existing Streamlit page measures
per-turn in-process total and preparation time with a monotonic clock; it
shows time to the first parsed execution event only when one exists. Timing
persists in the active browser session, not in restored checkpoint history.
The full selected offline regression passed **360 tests in 14.61 seconds**.
A real local browser turn with the CPU model returned category overdue rates
3/4, 0/2 and 1/4 and displayed 27.55 s total, 27.51 s first event and
0.00 s preparation. This is one local sample, not p95, network latency or a
production target. No paid API request was made in this round. The four-path
demo script documents further questions; only the category turn was newly
repeated in a live browser for this timing change.

The public docs now include `docs/REQUIREMENTS.md`, a Chinese user-scenario and
acceptance document. It describes the ticket-review workflow and observable outcomes
without changing runtime behavior. The root README links to it. This was a
documentation-only update; previous test and model evidence remains the latest
runtime evidence until re-run.

The personal repository now has a Chinese-only root README, business-specific design
and demo narrative, and no embedded screenshots. The business rule corpus and Skill
instructions are Chinese; citations now use Chinese section anchors (for example,
`support-quality-v1#超时率`) while keeping the version ID. The full selected offline
regression passed **360 tests in 15.74 seconds** after these changes. A direct BM25
query for the unresolved-ticket denominator returned the Chinese overdue-rate section
with its source reference. The existing Streamlit service health endpoint returned
`ok`; this round did not repeat an end-to-end browser/model interaction or call a paid API.
The old English README remains accessible in Git history, with the original MIT
license and upstream commits preserved. The published root README credits upstream
once at the end; current contributions must still be described accurately.

The older sections below record historical development states. In particular, older
"no remote", "no UI", and "no commit" statements have been superseded by the
published repository and the authorized original Streamlit entrypoint.

## Environment and Isolation

- Independent repository, branch codex/baseline; upstream history and MIT retained.
- Python 3.12.14 in .venv, upstream locked dependency versions retained.
- markdown-it-py 3.0.0 was already locked/installed; declared as a direct dependency.
- Existing Ollama 0.40.4 service retained. Approved qwen2.5:3b is the local chat model;
  original qwen3:4b is retained, including its failed/timeout diagnostic evidence.
- User approved a local embedding download on 2026-10-23. Standard Ollama pull of
  embeddinggemma:300m completed (about 622 MB), CPU use, 768-dimensional outputs.
- .local/support-demo contains 15 synthetic records, a real schema catalog,
  read-only SQLite URI and loopback-only config. This is now the active UI config,
  selected through the original CONFIG_FILE environment setting.
- Secrets, databases, ledgers, logs, models, virtual environments and runtime reports
  are not Git content. No new commit, personal remote, push or backend edit this session.
- The real vector check initialized upstream .chroma_db indexes; this generated
  cache is ignored, retained locally and is not a source contribution.

## Current Scope and Contributions

Upstream supplies the LangGraph main loop, Text2SQL subgraph, ToolNode, checkpoint,
AskHuman and Streamlit UI. Our increment uses these mechanisms, not a replacement framework.

1. An opt-in support_quality Skill with versioned definitions, validated scope,
   fixed parameterized queries and explicit no-data/error/limit states.
2. Business-document retrieval in the existing search_knowledge tool: Markdown
   sections/chunks, BM25 default, optional LangChain InMemoryVectorStore, references,
   content hashes, bounded context and source-aware embedding cache.
3. An opt-in prepared six-tool profile: support_quality, search_knowledge,
   AskHuman, search_schema, show_schema and text2sql. Conservative literal task routing
   narrows model exposure; it never enables a tool absent from the allowlist.
4. Missing metric/date scope is recorded and routed to the original AskHuman interrupt.
   In this profile, resumed feedback is also an explicit HumanMessage, preserving
   authoritative user intent and the checkpoint flow.
5. Verified metric/SQL results are rendered by code; rule answers are extractive
   passages with actual citations. No additional model call recalculates numbers
   or paraphrases policy into an unsupported conclusion.
6. SQL handoff anchors the actual recent user requests. Explicit raw audits retain
   inclusion/date scope instead of inheriting metric exclusions. Compact extraction
   retains the original request when the rewrite field is omitted. A verified,
   complete single-table catalog can skip model-based table selection; provenance
   identifies program selection, not model reasoning.
7. SQL result artifacts distinguish executed SQL_SUCCESS from failed/unexecuted
   queries; final SQL/CSV evidence comes only from the current turn's real tool output.

The options default off outside the prepared profile. Logging-only reasoning is optional
for Text2SQL and knowledge search; required task/search parameters remain validated.

## Tests Actually Run

- Latest offline baseline: **360 passed in 18.63 seconds**, including relative-time,
  explicit null-group repair, Chinese labels and persisted history selection.
  Earlier 328/350/351-test results remain historical.
  Command: scripts/test-baseline.ps1; evidence: .run/baseline.xml / baseline.log.
- Covers selected upstream execution, SQLite, Skill, document retrieval, routing,
  scope/evidence, sync/async ToolNode flows, provider compatibility and budget guards.
- Offline tests prohibit external HTTP; mocks/scripted responses do not establish
  model quality. Streamlit checks cover the unconfigured-model state and offline
  execution of the original page's response handling with a scripted graph.
  New configured-page tests also use the real AsyncGraphManager, LangGraph and
  temporary read-only SQLite with scripted model decisions: statistics/follow-up,
  new-thread isolation, old-history restoration and interrupted-task resume.
  This remains in-process testing, not a running browser UI or model-quality result.
- An earlier prepared-profile replay recorded ten steps, 12 local chat requests
  and 133.92 seconds, but its output files were later overwritten by a four-scenario
  partial run. The current .run/local-graph-configured.json/.md report
  `not_completed`; do not cite them as evidence that the full replay passed.
  Real browser checks and the separate vector/SQL artifacts below remain available.
- Independent real SQL chain: .run/local-sql-configured.json passed with four local
  requests in 60.69 seconds; real MAX(id)=15 matched an independent database query,
  and the database was unchanged. Current combined replay also checks real SQL/CSV.
- Real embedding-only validation: .run/local-retrieval-smoke.json passed, six small
  Chinese/English cases Top-3 6/6 and Top-1 5/6, unit-norm finite 768-vectors and
  cache reuse. Nine local embedding requests / 13 texts, 3.55 seconds.
- Original graph with real vector retrieval: .run/local-graph-vector.json passed,
  one local chat request, six local embedding requests, 19.51 seconds; actual
  business retrieval mode=vector and correct rule source/chunk references.

These are curated local checks, not general accuracy, latency or scale claims.
The end-to-end vector case uses extractive reporting, not independently evaluated
generative RAG answer quality.

## Important Failures and Fixes

- User browser testing exposed a missed case: "this week" reused the old October
  1-8 scope/result. Missing-field checks alone cannot establish date provenance,
  and the compatible local model can ignore a required tool and answer from history.
  In the opt-in demo profile, relative time without a complete explicit current-message
  ISO range now goes to AskHuman before any model/query, with the configured data
  cutoff shown. Current-turn metric/retrieval responses without tool execution fail
  explicitly instead of presenting a model-generated result. This is conservative
  clarification, not an automatic calendar/timezone resolver or a complete intent parser.
  39 focused offline checks passed, then full baseline 350 passed; the real browser
  restored old history and confirmed the relative query asked dates without a model call.
  The first real resume exposed group_by=null from the 3B model; explicit current
  overall wording now repairs only that null argument, recording the intervention.
  Conflicting/category wording and other invalid arguments are not silently changed.

- The page is localized to Chinese, including structured tool steps, controls,
  placeholders, help and error labels. SQL/schema/source identifiers and quoted
  original documents are preserved. The framework toolbar uses minimal mode;
  the sidebar remains expandable on narrow windows.
- New conversation never deleted the stored checkpoint but lacked a discoverable
  return path. A sidebar now reads per-user checkpoint summaries with the existing
  alist API, deduplicates root threads and shows up to 50 recent conversations.
  Exact user metadata filters the list; this is not production authentication.
  An explicit aclosing releases the iterator/SQLite lock after the limit. No
  separate history database or destructive migration is introduced. Offline real
  async graph/SQLite page tests cover click-back, fresh-page restore and user switch.
  Restored history also reconstructs user-facing AskHuman questions from stored
  tool calls, instead of silently dropping the clarification between user turns.
  Real browser new conversation retained the history button and restored the user's
  original 138c7f67ea3f conversation by one click. The original failed model messages
  remain historical; after repair, a real new explicit-scope query returned 40%.
  Evidence: .run/browser-zh-history.jpg and .run/zh-history-acceptance.md.

- Earlier six-tool routing confused raw SQL and metrics; prompts alone did not
  reliably prevent wrong denominators or free-SQL metric calculations.
- The persisted catalog still excluded test/duplicate/cancelled records unconditionally;
  corrected both factory metadata and the actual ignored demo catalog.
- Small model omitted extraction rewrite and log-only reasoning; minimized unnecessary
  output, preserved user text and kept task-field validation.
- A generated rule explanation cited the correct passage but then contradicted it.
  A validation failure was also described as missing policy. The prepared profile
  now renders source passages / explicit failure states instead of another free summary.
- Resumed input was only a ToolMessage; the model lost the metric. Added authoritative
  user feedback in the opt-in profile and retested the real checkpoint flow.
- Evidence: .run/local-graph-before-required-actions.json and
  .run/local-graph-before-quote-resume.json preserve failed batches. Earlier isolated
  local-sql-smoke.json was overwritten by later diagnostics; it is not the current
  successful SQL evidence. Use local-sql-configured.json and the combined replay.

## Limits and Frozen Scope

This session converges on one closed loop, not an extensible analytics platform.
Literal routing is not general semantic classification. Required first-tool actions
and deterministic reports are program guarantees, not autonomous model reasoning.
Main -> Text2SQL is limited role collaboration, not full analysis/Python Multi-Agent.

Only the documented SQLite schema, three metrics, at most 93 days and 20 groups.
Read-only connections, document path checks and scoped tools are not a complete
SQL sandbox, multi-tenant authorization or global query timeout.
Curated Skill text is not dynamic Skill discovery; a query fingerprint is not a
signed database snapshot. Graph diagnostics disable context compression to avoid
unbudgeted model calls; full long-dialog behavior is not accepted.
No multimodal, advanced memory, MCP platform, forecasting, report/chart download
or production deployment is part of this round.

## Paid API State

No paid request was made on 2026-10-23. The two previous authorizations are exhausted:
- Original ledger .run/paid-acceptance-v1.sqlite: six occupied slots, four HTTP 200,
  one HTTP 400, one unknown reserved outcome. Known input/output: 7122 / 1287 tokens.
- Additional ledger .run/paid-acceptance-additional5-v1.sqlite: five HTTP 200,
  input/output 11434 / 2022. Table selection returned no usable result; SQL did not execute.
- Transport/model alias and main-to-Text2SQL tool calling were verified, not complete
  API SQL execution. Currency cost is unknown; unknown reservations are not released.
- Explicit compatibility options omit strict and incompatible response_format parsing
  in the paid profile; compact workflow and newest guards have not been paid-retested.
  Local replay uses strict=true and cannot certify the non-strict provider profile.
- Preserve both ledgers. No further provider calls, larger caps or historical key
  recovery. Fresh paid testing requires new explicit authorization and current local config.

Diagnostic request/time budgets apply to diagnostics, not the entire application.
No automatic paid fallback or API credential is configured for the local demo.

## Current Browser Startup

The user explicitly narrowed the earlier workspace condition on 2026-10-23:
abandon the deleted script, keep protection enabled/no exclusions, but review and
run the unchanged original entry with local configuration. Stop if this entry
raises an alert; do not reroute. This is not a false-positive verdict for the script.

run_streamlit_ui.py has no difference from the fixed upstream commit. It is now
running with CONFIG_FILE=.local/support-demo/config.local.yaml, local qwen2.5:3b,
BM25 and read-only synthetic SQLite. No paid provider or credential is configured.
URL: http://127.0.0.1:8501/ ; health returned ok. Restarted for the relative-time
fix through the same unchanged entry: current entry PID 12996, Streamlit PID 30240;
loopback IPv4/IPv6 only. Current logs: .run/upstream-ui-zh.*.log.
Earlier startup evidence remains in .run/upstream-ui.*.log (old PIDs no longer current).
Actual browser interaction with the real model displayed 40% (4/10), category
follow-up 75% / 0% / 25%, and retrieved rule passages with source_ref/chunk_id.
New conversation, old-history restoration and switching back to a pending task
then resuming to 40% also passed. Raw SQL delegated to the original async subgraph
and displayed SELECT max(id) AS max_id FROM support_tickets with result 15, without
inheriting metric exclusions/dates. Six real submitted chat turns, no paid API;
evidence is .run/browser-acceptance.md plus statistics/SQL screenshots and startup
logs. No new offline suite was run for this configuration/startup/docs-only round.
User acceptance is still separate. Restart instructions are in the scenario README.

## UI Review History (Superseded Startup Condition)

The following records describe the earlier paused state, not the current server.
The latest explicit authorization above supersedes the old whole-UI pause only;
the deleted script remains abandoned and has no vendor verification result.

Read-only cause review after the user's follow-up: run_streamlit_ui.py is unchanged
against fixed upstream 97818e4fdcea85535cdfa53d8ae4054878342af1; Streamlit 1.54.0 is
installed. Port 8501 has zero listening connections. The default config.yaml is
absent, and CONFIG_FILE is unset in the inspected process; the prepared demo YAML
exists separately. The diagnostic module calls config.set inside its own process,
which neither starts a page nor configures a different process. These observations
show incomplete page startup/configuration, not an observed crash of the original
launcher or evidence that Huorong blocked port 8501. No startup was attempted in
this review. Vendor review is a workspace security condition, not a software
dependency of upstream OpenChatBI; the precise heuristic trigger remains unknown.

The latest user requires an actual interactive frontend and rejects CLI as delivery.
Static review found an integration bug: streamlit_ui.py accumulated LLM tokens but
could omit program-generated metric/SQL/citation reports. It now reads the current
completed AIMessage from checkpoint state. Pending turns cannot reuse old answers;
interruptions show waiting, not completion or an empty options list. Ten UI-focused
offline tests passed in 5.31 seconds, including in-process AppTest response rendering.
They start no server and use no model; they are not browser/end-to-end acceptance.
The prepared config remains inactive until the independent security gate is resolved.

Further frontend fixes on 2026-10-23: replace display-only Clear Chat with a new
conversation/thread ID while retaining old history; restore the real interrupt
state and pending question when changing/refreshing a thread. Graph initialization
failure now releases its checkpoint. Exit cleanup is registered once per session
and captures manager/loop directly instead of reading an expired Streamlit context.
The configured-page tests initially exposed repeated exit logging errors; the final
328-test run exited cleanly after the fix. The prepared YAML/factory use the existing
context_config.enabled=false option, avoiding extra summarization calls in this
small demo. No default config was activated, server/launcher run or model called.

Upstream includes a Streamlit UI, but no interactive Agent server is running.
Huorong deleted newly authored scripts/local-ui.ps1 on 2026-10-22 11:46:55 with
HEUR:TrojanDownloader/PS.NetLoader.ae. On 2026-10-23 the user reconfirmed no official
verification result. The cause is unconfirmed, not a verified false positive.

Do not recreate/rewrite/rename the script, change protection or switch launcher/tool
to evade the same block. Do not upload evidence without user authorization.
See the coordination SECURITY_FEEDBACK.md for recorded evidence/vendor review.
The user has now authorized sending only sanitized detection/time/purpose text.
The official forum returned 405 in web access and the actual in-app tab; no submission
form or receipt was obtained, and no feedback text/files were transmitted. Do not
claim the report has been filed or that permission to file resolves the alert.
No working UI URL is claimed. CLI replay is not browser/manual acceptance.
After legitimate verification, connect the existing UI normally and run the
[scenario guide](./example/support_quality/README.md). Do not add a second frontend.
