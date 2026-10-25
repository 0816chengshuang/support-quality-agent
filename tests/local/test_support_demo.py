"""Verify real demo catalog/configuration without any model or page startup."""

import json
import shutil
from datetime import date
from pathlib import Path

import httpx
import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from openchatbi.analysis.support_quality import SupportQualityInput, analyze_support_quality
from openchatbi.catalog.store.file_system import FileSystemCatalogStore
from openchatbi.config_loader import ConfigLoader
from openchatbi.support_demo import prepare_support_demo
from openchatbi.tool.support_quality import get_support_quality_tools

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def demo_repository(tmp_path):
    for relative in (
        "pyproject.toml",
        "example/support_quality/fixture.sql",
        "example/support_quality/bi.yaml",
        "openchatbi/skills/support_quality/METRICS.md",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


def test_preparation_does_not_read_or_enable_a_paid_key(demo_repository, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-read-this-paid-key")
    result = prepare_support_demo(demo_repository)
    contents = Path(result["config_file"]).read_text(encoding="utf-8")
    config = yaml.safe_load(contents)
    assert "do-not-read-this-paid-key" not in contents
    assert "gptgod" not in contents
    assert result["paid_provider_enabled"] is False
    assert result["page_started"] is False
    assert config["default_llm"]["params"]["base_url"] == "http://127.0.0.1:11434/v1"
    assert config["default_llm"]["params"]["model"] == "qwen2.5:3b"
    assert "reasoning_effort" not in config["default_llm"]["params"]
    assert "embedding_model" not in config
    assert set(config["agent_tool_allowlist"]) == {
        "AskHuman",
        "search_knowledge",
        "support_quality",
        "search_schema",
        "show_schema",
        "text2sql",
    }
    assert config["llm_max_attempts"] == 1
    assert config["context_config"]["enabled"] is False


def test_preparation_does_not_overwrite_existing_data(demo_repository):
    result = prepare_support_demo(demo_repository)
    before = Path(result["database"]).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_support_demo(demo_repository)
    assert Path(result["database"]).read_bytes() == before


def test_demo_uses_real_catalog_and_read_only_database(demo_repository):
    state = prepare_support_demo(demo_repository)
    config = yaml.safe_load(Path(state["config_file"]).read_text(encoding="utf-8"))
    catalog = FileSystemCatalogStore(config["catalog_store"]["data_path"], config["data_warehouse_config"])
    try:
        assert catalog.get_table_list() == ["support_tickets"]
        assert len(catalog.get_column_list("support_tickets")) == 8
        assert len(catalog.get_column_list()) == 8
        assert "synthetic" in catalog.get_table_information("support_tickets")["description"]
        assert (
            "Raw-record audits preserve explicit user scope"
            in catalog.get_table_information("support_tickets")["sql_rule"]
        )
        engine = catalog.get_sql_engine()
        with engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM support_tickets")).scalar_one() == 15
            with pytest.raises(OperationalError, match="readonly"):
                connection.execute(text("DELETE FROM support_tickets"))
        request = SupportQualityInput(metric="overdue_rate", start_date=date(2026, 10, 1), end_date=date(2026, 10, 8))
        result = analyze_support_quality(engine, request, date(2026, 10, 8))
        assert result["rows"][0]["value"] == 0.4
    finally:
        catalog.get_sql_engine().dispose()


def test_prepared_config_loads_real_clients_but_never_calls_them(demo_repository, monkeypatch):
    import openchatbi.tool.support_quality as tool_module

    state = prepare_support_demo(demo_repository)
    loader = ConfigLoader()
    loader.load(state["config_file"])
    config = loader.get()
    # The upstream fixture resets the singleton, so bind consumers to this loader.
    monkeypatch.setattr(tool_module.config, "get", loader.get)
    try:
        assert str(config.default_llm.openai_api_base) == "http://127.0.0.1:11434/v1"
        assert config.default_llm.max_tokens == 384
        assert config.default_llm.max_retries == 0
        assert config.embedding_model is None
        assert config.bi_config["support_quality_skill"]["snapshot_date"] == "2026-10-08"
        tools = get_support_quality_tools(config.catalog_store)
        result = tools[0].invoke({"metric": "overdue_rate", "start_date": "2026-10-01", "end_date": "2026-10-08"})
        assert result["rows"][0]["value"] == 0.4
        assert not (demo_repository / "openchatbi/config.yaml").exists()
    finally:
        config.catalog_store.get_sql_engine().dispose()


def test_preparation_manifest_is_only_data(demo_repository):
    state = prepare_support_demo(demo_repository)
    manifest = json.loads((demo_repository / ".local/support-demo/prepared.json").read_text(encoding="utf-8"))
    assert manifest == state
    assert manifest["status"] == "prepared_not_activated"
    assert not (demo_repository / "scripts/local-ui.ps1").exists()


def test_invalid_repository_does_not_create_output(tmp_path):
    with pytest.raises(ValueError):
        prepare_support_demo(tmp_path)
    assert not (tmp_path / ".local").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://api.gptgod.online/v1/chat/completions",
        "http://127.0.0.1:11434/api/pull",
        "http://localhost:11434/v1/chat/completions",
        "http://127.0.0.1:11435/v1/chat/completions",
    ],
)
def test_local_diagnostic_rejects_every_nonapproved_endpoint(url):
    from openchatbi.local_support_smoke import require_local_ollama

    with pytest.raises(ValueError):
        require_local_ollama(httpx.Request("POST", url))


def test_local_diagnostic_accepts_only_the_fixed_loopback_chat_path():
    from openchatbi.local_support_smoke import require_local_ollama

    require_local_ollama(httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions"))
