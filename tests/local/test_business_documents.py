"""Deterministic retrieval checks, not semantic-model or answer-quality claims."""

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

import openchatbi.business_knowledge as knowledge
from openchatbi.business_knowledge import (
    BusinessKnowledgeSettings,
    load_knowledge_chunks,
    search_business_documents,
)
from openchatbi.tool import search_knowledge as tool_module

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "openchatbi/skills/support_quality"


@pytest.fixture(autouse=True)
def fresh_index_cache(monkeypatch):
    monkeypatch.setattr(knowledge, "_INDEX_CACHE", [])


def settings(**overrides):
    return BusinessKnowledgeSettings.model_validate(
        {"root": str(METRICS), "sources": [{"id": "support-quality-v1", "path": "METRICS.md"}], **overrides}
    )


class CountingEmbeddings(Embeddings):
    """Tiny, manually assigned vectors; NOT a real semantic embedding model."""

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    @staticmethod
    def vector(text):
        lowered = text.casefold()
        if any(word in lowered for word in ("overdue", "late", "超时")):
            return [1.0, 0.0, 0.0]
        if any(word in lowered for word in ("average", "mean", "平均")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self.vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self.vector(text)


@pytest.mark.parametrize(
    "query,source_ref",
    [
        ("超时率怎么算", "support-quality-v1#overdue-rate"),
        ("未解决工单超时率分母", "support-quality-v1#overdue-rate"),
        ("平均处理时长", "support-quality-v1#average-resolution-minutes"),
        ("新建工单量", "support-quality-v1#ticket-count"),
        ("overdue rate denominator unresolved", "support-quality-v1#overdue-rate"),
        ("UTC snapshot cancelled test duplicate", "support-quality-v1#cohort"),
    ],
)
def test_small_independent_retrieval_cases(query, source_ref):
    result = search_business_documents(settings(top_k=1), [query])
    assert result["status"] == "matched"
    assert result["passages"][0]["source_ref"] == source_ref


def test_citations_point_to_actual_source_lines_and_versions():
    config = settings()
    chunks = load_knowledge_chunks(config)
    lines = (METRICS / "METRICS.md").read_text(encoding="utf-8").splitlines(keepends=True)
    assert chunks
    for chunk in chunks:
        metadata = chunk.metadata
        quoted_lines = "".join(lines[metadata["line_start"] - 1 : metadata["line_end"]])
        assert chunk.page_content in quoted_lines
        assert len(chunk.page_content) <= config.chunk_chars
        assert len(metadata["content_sha256"]) == 64
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in load_knowledge_chunks(config)]


def test_calculation_definition_references_resolve_to_the_corpus():
    from openchatbi.analysis.support_quality import METRICS as DEFINITIONS

    references = {chunk.metadata["source_ref"] for chunk in load_knowledge_chunks(settings())}
    assert {definition["source"] for definition in DEFINITIONS.values()} <= references
    assert "support-quality-v1#cohort" in references


def test_commonmark_parser_does_not_treat_fenced_code_as_heading(tmp_path):
    (tmp_path / "rules.md").write_text("# Rules\n\n## Valid\nReal policy.\n```md\n## Fake\n```\n", encoding="utf-8")
    config = settings(root=str(tmp_path), sources=[{"id": "rules", "path": "rules.md"}])
    chunks = load_knowledge_chunks(config)
    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "Rules / Valid"


def test_long_sections_have_bounded_overlapping_chunks(tmp_path):
    (tmp_path / "rules.md").write_text("# Rules\n\n" + "sample words " * 100, encoding="utf-8")
    config = settings(
        root=str(tmp_path), sources=[{"id": "rules", "path": "rules.md"}], chunk_chars=256, chunk_overlap=64
    )
    chunks = load_knowledge_chunks(config)
    assert len(chunks) > 1
    assert chunks[0].page_content[-64:] == chunks[1].page_content[:64]
    assert all(len(chunk.page_content) <= 256 for chunk in chunks)


@pytest.mark.parametrize("query", ["量子纠缠波函数", "zzzznonexistentterm"])
def test_no_lexical_match_does_not_return_arbitrary_top_k(query):
    result = search_business_documents(settings(), [query])
    assert result["status"] == "no_match"
    assert result["passages"] == []


def test_context_budget_does_not_clip_a_source_and_hide_it():
    config = settings(max_context_chars=512)
    result = search_business_documents(config, ["overdue"])
    assert result["context_chars"] <= config.max_context_chars
    assert result["status"] == "context_budget_exceeded"
    assert result["passages"] == []
    assert result["omitted_due_to_budget"] > 0


def test_top_k_deduplication_and_serialized_context_budget():
    config = settings(top_k=2, max_context_chars=3000)
    result = search_business_documents(config, ["overdue", "overdue", "average"])
    passages = result["passages"]
    assert 1 <= len(passages) <= 2
    assert len({passage["chunk_id"] for passage in passages}) == len(passages)
    assert len(json.dumps(passages, ensure_ascii=False)) == result["context_chars"] <= 3000


@pytest.mark.parametrize("path", ["../private.md", "rules.txt"])
def test_unlisted_or_non_markdown_path_is_rejected(tmp_path, path):
    config = settings(root=str(tmp_path), sources=[{"id": "rules", "path": path}])
    with pytest.raises(ValueError):
        load_knowledge_chunks(config)


def test_absolute_source_path_is_rejected():
    config = settings(sources=[{"id": "rules", "path": str(METRICS / "METRICS.md")}])
    with pytest.raises(ValueError):
        load_knowledge_chunks(config)


def test_oversized_source_is_rejected_before_indexing(tmp_path):
    (tmp_path / "large.md").write_text("x" * (knowledge.MAX_FILE_BYTES + 1), encoding="utf-8")
    config = settings(root=str(tmp_path), sources=[{"id": "large", "path": "large.md"}])
    with pytest.raises(ValueError, match="size budget"):
        load_knowledge_chunks(config)


def test_invalid_queries_do_not_embed_documents():
    embedding = CountingEmbeddings()
    with pytest.raises(ValueError):
        search_business_documents(settings(mode="vector"), ["x" * 501], embedding)
    assert embedding.document_calls == 0


def test_vector_mode_requires_explicit_embedding():
    with pytest.raises(ValueError, match="explicitly configured"):
        search_business_documents(settings(mode="vector"), ["overdue"])


def test_empty_corpus_does_not_spend_embedding_calls(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    config = settings(root=str(tmp_path), sources=[{"id": "empty", "path": "empty.md"}], mode="vector")
    embedding = CountingEmbeddings()
    result = search_business_documents(config, ["late"], embedding)
    assert result["status"] == "no_match"
    assert embedding.document_calls == embedding.query_calls == 0


def test_vector_similarity_threshold_can_return_no_match(tmp_path):
    (tmp_path / "rules.md").write_text("# Overdue\nLate means past the deadline.\n", encoding="utf-8")
    config = settings(root=str(tmp_path), sources=[{"id": "rules", "path": "rules.md"}], mode="vector")
    result = search_business_documents(config, ["average"], CountingEmbeddings())
    assert result["status"] == "no_match"
    assert result["passages"] == []


def test_vector_store_embedding_and_cache_path():
    embedding = CountingEmbeddings()
    config = settings(mode="vector", top_k=1, min_similarity=0.9)
    first = search_business_documents(config, ["late", "late"], embedding)
    second = search_business_documents(config, ["late"], embedding)
    assert first["passages"] == second["passages"]
    assert first["passages"][0]["source_ref"] == "support-quality-v1#overdue-rate"
    assert embedding.document_calls == 1
    assert embedding.query_calls == 2


def test_source_edit_invalidates_chunk_ids_and_embedding_cache(tmp_path):
    source = tmp_path / "rules.md"
    source.write_text("# Overdue\nLate means past the deadline.\n", encoding="utf-8")
    config = settings(root=str(tmp_path), sources=[{"id": "rules", "path": "rules.md"}], mode="vector")
    embedding = CountingEmbeddings()
    first = search_business_documents(config, ["late"], embedding)
    source.write_text("# Overdue\nLate includes unresolved tickets past the deadline.\n", encoding="utf-8")
    second = search_business_documents(config, ["late"], embedding)
    assert first["passages"][0]["chunk_id"] != second["passages"][0]["chunk_id"]
    assert first["passages"][0]["content_sha256"] != second["passages"][0]["content_sha256"]
    assert embedding.document_calls == 2


def test_line_ending_changes_do_not_change_content_ids(tmp_path):
    path = tmp_path / "rules.md"
    path.write_bytes(b"# Rules\n\nA stable overdue policy.\n")
    config = settings(root=str(tmp_path), sources=[{"id": "rules", "path": "rules.md"}])
    first = load_knowledge_chunks(config)
    path.write_bytes(b"# Rules\r\n\r\nA stable overdue policy.\r\n")
    assert first[0].id == load_knowledge_chunks(config)[0].id


def test_distinct_configured_corpora_do_not_share_passages(tmp_path):
    for name, text in [("alpha", "Alpha overdue policy."), ("beta", "Beta overdue policy.")]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "rules.md").write_text(text, encoding="utf-8")
    alpha = settings(root=str(tmp_path / "alpha"), sources=[{"id": "rules", "path": "rules.md"}])
    beta = settings(root=str(tmp_path / "beta"), sources=[{"id": "rules", "path": "rules.md"}])
    first = search_business_documents(alpha, ["overdue"])
    second = search_business_documents(beta, ["overdue"])
    assert "Alpha" in first["passages"][0]["text"]
    assert "Beta" in second["passages"][0]["text"]
    assert "Alpha" not in second["passages"][0]["text"]


def configure_tool(monkeypatch, config):
    monkeypatch.setattr(
        tool_module.config,
        "get",
        lambda: SimpleNamespace(
            bi_config={
                "business_documents": config.model_dump(mode="json"),
                "basic_knowledge_glossary": "Legacy text must not mask a configured retrieval failure.",
            }
        ),
    )


def test_existing_tool_uses_retrieval_without_calling_embedding(monkeypatch):
    configure_tool(monkeypatch, settings(top_k=1))
    embedding_loader = Mock(side_effect=AssertionError("BM25 must not load an embedding model"))
    monkeypatch.setattr(tool_module, "get_embedding_model", embedding_loader)
    result = tool_module.search_knowledge.invoke(
        {"reasoning": "Find the definition", "query_list": ["超时率"], "knowledge_bases": ["business"]}
    )
    assert result["business"]["status"] == "matched"
    assert result["business"]["passages"][0]["source_ref"] == "support-quality-v1#overdue-rate"
    assert result["warnings"] == []
    embedding_loader.assert_not_called()


def test_search_trace_explanation_is_optional_but_search_conditions_are_not(monkeypatch):
    configure_tool(monkeypatch, settings(top_k=1))
    result = tool_module.search_knowledge.invoke({"query_list": ["超时率"], "knowledge_bases": ["business"]})
    assert result["business"]["status"] == "matched"
    with pytest.raises(ValidationError):
        tool_module.SearchKnowledgeInput.model_validate({"reasoning": "not a query"})


@pytest.mark.parametrize("content", ["Error invoking tool", '{"business":{"status":"retrieval_error"}}'])
def test_failed_retrieval_is_not_reported_as_missing_policy(content):
    evidence = ToolMessage(name="search_knowledge", tool_call_id="failed", content=content)
    assert "检索失败" in tool_module.render_knowledge_report([evidence])
    from langchain_core.messages import HumanMessage

    assert tool_module.render_knowledge_report([evidence, HumanMessage("new question")]) is None


def test_successful_no_match_is_not_a_retrieval_failure():
    evidence = ToolMessage(
        name="search_knowledge", tool_call_id="searched", content='{"business":{"status":"no_match","passages":[]}}'
    )
    assert "未找到匹配片段" in tool_module.render_knowledge_report([evidence])
    assert "检索失败" not in tool_module.render_knowledge_report([evidence])


def test_extractive_report_uses_only_actual_passages_and_citations(monkeypatch):
    configure_tool(monkeypatch, settings(top_k=1))
    result = tool_module.search_knowledge.invoke({"query_list": ["超时率分母"], "knowledge_bases": ["business"]})
    passage = result["business"]["passages"][0]
    report = tool_module.render_knowledge_report(
        [ToolMessage(name="search_knowledge", tool_call_id="actual", content=json.dumps(result))]
    )
    assert passage["text"].strip() in report
    assert passage["source_ref"] in report and passage["chunk_id"] in report


def test_legacy_glossary_behavior_is_unchanged_without_document_config(monkeypatch):
    monkeypatch.setattr(
        tool_module.config, "get", lambda: SimpleNamespace(bi_config={"basic_knowledge_glossary": "Legacy"})
    )
    assert tool_module._search_business_knowledge(["anything"]) == ("Legacy", None)


def test_no_match_returns_explicit_warning(monkeypatch):
    configure_tool(monkeypatch, settings())
    result, warning = tool_module._search_business_knowledge(["zzzznonexistentterm"])
    assert result["status"] == "no_match"
    assert "no_match" in warning


def test_embedding_failure_is_not_hidden_by_glossary_fallback(monkeypatch):
    configure_tool(monkeypatch, settings(mode="vector"))
    monkeypatch.setattr(tool_module, "get_embedding_model", Mock(side_effect=RuntimeError("secret-api-key")))
    result, warning = tool_module._search_business_knowledge(["overdue"])
    assert result["status"] == "retrieval_error"
    assert result["passages"] == []
    assert "fallback" in warning
    assert "secret" not in json.dumps(result)
    assert "Legacy" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_original_search_tool_returns_citable_passages_through_toolnode(monkeypatch, sync_mode):
    configure_tool(monkeypatch, settings(top_k=1))
    workflow = StateGraph(MessagesState)
    workflow.add_node("tools", ToolNode([tool_module.search_knowledge]))
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    node = workflow.compile()
    payload = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "id": "knowledge-test-1",
                        "args": {
                            "reasoning": "Read definition",
                            "query_list": ["超时率"],
                            "knowledge_bases": ["business"],
                        },
                    }
                ],
            )
        ]
    }
    result = node.invoke(payload) if sync_mode else await node.ainvoke(payload)
    tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    business = json.loads(tool_messages[0].content)["business"]
    assert business["status"] == "matched"
    assert business["passages"][0]["source_ref"] == "support-quality-v1#overdue-rate"
    assert len(business["passages"][0]["chunk_id"]) == 20


def test_markdown_parser_is_a_direct_dependency_and_uses_existing_locked_version():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    assert "markdown-it-py>=3.0.0,<4.0.0" in project["project"]["dependencies"]
    package = next(package for package in lock["package"] if package["name"] == "openchatbi")
    assert {"name": "markdown-it-py"} in package["dependencies"]
    assert next(package for package in lock["package"] if package["name"] == "markdown-it-py")["version"] == "3.0.0"


def test_settings_reject_duplicate_ids():
    with pytest.raises(ValidationError):
        settings(sources=[{"id": "rules", "path": "one.md"}, {"id": "rules", "path": "two.md"}])
