"""Catalog-driven shortcuts, not model reasoning or hardcoded query results."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage

from openchatbi.text2sql.schema_linking import _single_catalog_selection, _valid_compact_selection, schema_linking


def candidates(table="audit_rows"):
    return {
        table: (
            {"description": "Synthetic rows"},
            [
                {
                    "column_name": "record_id",
                    "category": "dimension",
                    "display_name": "record_id",
                    "description": "Identifier",
                },
            ],
        )
    }


def test_single_verified_table_is_derived_from_catalog_not_demo_names():
    result = _single_catalog_selection(candidates(), ["audit_rows"], ["record_id"])
    assert result == {
        "tables": [{"table": "audit_rows", "columns": ["record_id"]}],
        "table_selection_source": "single_verified_catalog_table",
    }


@pytest.mark.parametrize(
    "tables,dimensions",
    [
        (["audit_rows", "other_rows"], ["record_id"]),
        (["other_rows"], ["record_id"]),
        (["audit_rows"], ["unknown_field"]),
        (["audit_rows"], [None]),
    ],
)
def test_ambiguous_or_unknown_schema_does_not_take_shortcut(tables, dimensions):
    assert _single_catalog_selection(candidates(), tables, dimensions) is None


@pytest.mark.parametrize(
    "selection",
    [
        [],
        [{"table": "invented", "columns": ["record_id"]}],
        [{"table": "audit_rows", "columns": ["invented"]}],
        [{"table": "audit_rows", "columns": []}],
        [{"table": "audit_rows", "columns": [None]}],
        "not a list",
    ],
)
def test_compact_output_rejects_invented_or_missing_tables_and_columns(selection):
    assert not _valid_compact_selection(selection, candidates())


def test_valid_compact_selection_is_accepted():
    assert _valid_compact_selection([{"table": "audit_rows", "columns": ["record_id"]}], candidates())


def test_compact_extraction_prompt_has_no_demo_answer_and_does_not_request_reasoning(monkeypatch):
    import openchatbi.text2sql.extraction as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(compact_sql_workflow=True))
    prompt = module._generate_extraction_prompt()
    assert len(prompt) < 1500
    assert "support_tickets" not in prompt and "MAX(id)" not in prompt
    assert '"reasoning"' not in prompt
    assert "knowledge_bases" in prompt


@pytest.mark.parametrize("compact,expected", [(True, "Keep all purchase records"), (False, None)])
def test_missing_rewrite_preserves_original_input_only_with_valid_compact_structure(monkeypatch, compact, expected):
    from langchain_core.messages import HumanMessage

    import openchatbi.text2sql.extraction as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(compact_sql_workflow=compact))
    monkeypatch.setattr(module, "_generate_extraction_prompt", lambda: "Offline extraction")
    monkeypatch.setattr(
        module,
        "call_llm_chat_model_with_retry",
        lambda *args, **kwargs: AIMessage(content='{"keywords":["purchase"],"dimensions":[],"metrics":[],"filter":[]}'),
    )
    result = module.information_extraction(Mock())({"messages": [HumanMessage("Keep all purchase records")]})
    assert result["rewrite_question"] == expected
    if compact:
        assert result["extraction_source"] == "original_request_fallback"


@pytest.mark.parametrize("compact,table_count,expected_calls", [(True, 1, 0), (False, 1, 1), (True, 2, 1)])
def test_original_schema_node_uses_fast_path_only_when_opted_in_and_unambiguous(
    monkeypatch, compact, table_count, expected_calls
):
    import openchatbi.text2sql.schema_linking as module

    monkeypatch.setattr(module.config, "get", lambda: SimpleNamespace(compact_sql_workflow=compact, llm_max_attempts=1))
    monkeypatch.setattr(module, "get_relevant_columns", lambda *args: ["record_id"])
    monkeypatch.setattr(module, "column_tables_mapping", {"record_id": ["audit_rows"]})
    monkeypatch.setattr(module, "col_dict", {"record_id": candidates()["audit_rows"][1][0]})
    monkeypatch.setattr(module, "table_selection_retriever", SimpleNamespace(invoke=lambda query: []))
    monkeypatch.setattr(module, "get_table_selection_prompt_template", lambda: "Offline catalog: [tables]")
    model = Mock()
    model.invoke.return_value = AIMessage(content='{"tables":[{"table":"audit_rows","columns":["record_id"]}]}')
    catalog = Mock()
    catalog.get_table_list.return_value = ["audit_rows"] if table_count == 1 else ["audit_rows", "another"]
    catalog.get_table_information.return_value = {"description": "Synthetic rows"}
    result = schema_linking(model, catalog)(
        {
            "messages": [],
            "rewrite_question": "maximum record_id",
            "info_entities": {"keywords": ["record_id"], "dimensions": ["record_id"], "metrics": []},
        }
    )
    assert result["tables"][0]["table"] == "audit_rows"
    assert model.invoke.call_count == expected_calls
    if compact and expected_calls:
        assert "No reasoning or prose" in model.invoke.call_args.args[0][0].content
