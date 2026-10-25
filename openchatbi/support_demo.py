"""Prepare local demo data/config only; never launch a page or model server."""

import json
import sqlite3
from pathlib import Path

import yaml

from openchatbi.catalog.store.file_system import FileSystemCatalogStore

LOCAL_MODEL_NAME = "qwen2.5:3b"
SUPPORT_COLUMNS = [
    ("id", "INTEGER", "dimension", "Ticket identifier; one row per ticket"),
    ("category", "TEXT", "dimension", "Support category: account, billing or delivery"),
    ("created_at", "TEXT", "dimension", "UTC creation time; inclusive start, exclusive end cohort"),
    ("due_at", "TEXT", "dimension", "UTC SLA deadline assigned to the ticket"),
    ("resolved_at", "TEXT", "dimension", "UTC resolution time; null means unresolved"),
    ("cancelled_at", "TEXT", "dimension", "UTC cancellation time; events after snapshot are not yet observed"),
    (
        "is_test",
        "INTEGER",
        "dimension",
        "Test flag; eligibility exclusion applies only to defined quality metrics, not raw audits",
    ),
    (
        "duplicate_of",
        "INTEGER",
        "dimension",
        "Duplicate reference; eligibility exclusion applies only to defined quality metrics, not raw audits",
    ),
]


def prepare_support_demo(repository: Path) -> dict:
    """Create a fresh ignored demo directory, refusing to overwrite existing work."""
    root = repository.resolve(strict=True)
    fixture = root / "example/support_quality/fixture.sql"
    bi_file = root / "example/support_quality/bi.yaml"
    if not (root / "pyproject.toml").is_file() or not fixture.is_file() or not bi_file.is_file():
        raise ValueError("Expected the OpenChatBI repository and support-quality fixtures")
    output = root / ".local/support-demo"
    if not output.resolve().is_relative_to(root):
        raise ValueError("Demo directory must remain inside the repository")
    output.mkdir(parents=True, exist_ok=False)
    database = output / "support.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(fixture.read_text(encoding="utf-8"))
    warehouse = {
        "uri": f"sqlite:///file:{database.as_posix()}?mode=ro&uri=true",
        "database_name": "",
        "include_tables": ["support_tickets"],
    }
    catalog_directory = output / "catalog"
    catalog = FileSystemCatalogStore(str(catalog_directory), warehouse)
    try:
        columns = [
            {
                "column_name": name,
                "display_name": name,
                "alias": "",
                "type": type_name,
                "category": category,
                "tag": "synthetic-support",
                "description": description,
                "is_common": True,
            }
            for name, type_name, category, description in SUPPORT_COLUMNS
        ]
        saved = catalog.save_table_information(
            "support_tickets",
            {
                "description": "Fully synthetic support tickets; fixed snapshot 2026-10-08 00:00:00 UTC",
                "selection_rule": "Use for support-ticket questions, not real organizational claims",
                "sql_rule": "Read-only queries. Use support_quality for the three defined quality metrics. Raw-record audits preserve explicit user scope; do not add eligibility exclusions unless requested.",
                "derived_metric": "Definitions: support-quality-v1; overdue denominator includes unresolved tickets",
            },
            columns,
            database="",
        )
        if not saved:
            raise RuntimeError("Failed to prepare the local schema catalog")
        if not catalog.save_table_sql_examples(
            "support_tickets",
            [
                {
                    "question": "Show raw synthetic ticket row count, including excluded examples",
                    "answer": "SELECT COUNT(*) AS raw_count FROM support_tickets",
                }
            ],
            database="",
        ):
            raise RuntimeError("Failed to prepare the local SQL example")
        if not catalog.save_table_selection_examples([("support tickets", ["support_tickets"])]):
            raise RuntimeError("Failed to prepare the table-selection example")
    finally:
        catalog.get_sql_engine().dispose()

    configuration = {
        "organization": "Synthetic Support Demo",
        "dialect": "sqlite",
        "agent_tool_allowlist": [
            "AskHuman",
            "search_knowledge",
            "support_quality",
            "search_schema",
            "show_schema",
            "text2sql",
        ],
        "llm_max_attempts": 1,
        "openai_strict_tools": True,
        "compact_sql_workflow": True,
        "anchor_sql_user_request": True,
        "require_sql_execution_evidence": True,
        "support_task_routing": True,
        "context_config": {"enabled": False},
        "bi_config_file": str(bi_file),
        "bi_config": {
            "business_documents": {
                **yaml.safe_load(bi_file.read_text(encoding="utf-8"))["business_documents"],
                "root": str(root / "openchatbi/skills/support_quality"),
            }
        },
        "default_llm": {
            "class": "langchain_openai.ChatOpenAI",
            "params": {
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama-local-not-a-secret",
                "model": LOCAL_MODEL_NAME,
                "temperature": 0,
                "max_tokens": 384,
                "timeout": 90,
                "max_retries": 0,
                "use_responses_api": False,
            },
        },
        "catalog_store": {"store_type": "file_system", "data_path": str(catalog_directory), "auto_load": False},
        "data_warehouse_config": warehouse,
        "report_directory": str(output / "reports"),
        "python_executor": "restricted_local",
        "visualization_mode": None,
        "enable_fail_closed_sql_guard": True,
        "enable_sql_result_limit": True,
        "sql_result_limit": 100,
        "sql_max_retries": 0,
    }
    config_path = output / "config.local.yaml"
    config_path.write_text(yaml.safe_dump(configuration, allow_unicode=True, sort_keys=False), encoding="utf-8")
    state = {
        "status": "prepared_not_activated",
        "database": str(database),
        "config_file": str(config_path),
        "snapshot_exclusive_utc": "2026-10-08 00:00:00",
        "model": LOCAL_MODEL_NAME,
        "model_endpoint": "http://127.0.0.1:11434/v1",
        "paid_provider_enabled": False,
        "page_started": False,
    }
    (output / "prepared.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main() -> None:
    """Prepare the synthetic demo from the repository root; do not start services."""
    state = prepare_support_demo(Path(__file__).resolve().parents[1])
    print(f"演示配置已准备：{state['config_file']}")
    print("数据仅为合成样本；此命令不会启动页面、模型或付费接口。")


if __name__ == "__main__":
    main()
