"""Bounded local delegation check of upstream roles; never start any UI/service."""

import json
import time
from pathlib import Path

import httpx
import yaml
from langchain_openai import ChatOpenAI

from openchatbi.local_graph_smoke import LocalRequestBudget
from openchatbi.sql_acceptance import QUESTION, REQUEST_TOOLS, run_sql_delegation
from openchatbi.support_demo import LOCAL_MODEL_NAME


def run_local_sql_smoke(repository: Path, use_configured_profile=False) -> dict:
    root = repository.resolve(strict=True)
    configuration = yaml.safe_load((root / ".local/support-demo/config.local.yaml").read_text(encoding="utf-8"))
    expected_uri = f"sqlite:///file:{(root / '.local/support-demo/support.sqlite').as_posix()}?mode=ro&uri=true"
    if configuration["data_warehouse_config"]["uri"] != expected_uri:
        raise ValueError("Requires the prepared local read-only dataset")
    if any(
        configuration.get(key)
        for key in ("mcp_servers", "embedding_model", "llm_providers", "text2sql_llm", "analysis_llm")
    ):
        raise ValueError("No external or separately configured models are allowed")
    allowed = REQUEST_TOOLS | {"support_quality"} if use_configured_profile else REQUEST_TOOLS
    budget = LocalRequestBudget(max_requests=10, per_turn=10, max_seconds=360, allowed_tools=allowed)
    started = time.monotonic()
    record = {
        "status": "not_completed",
        "model": LOCAL_MODEL_NAME,
        "paid_requests": 0,
        "page_started": False,
        "question": QUESTION,
        "events": [],
        "scope": "configured demo" if use_configured_profile else "isolated audit",
        "ui_acceptance": False,
    }
    report = root / (".run/local-sql-configured.json" if use_configured_profile else ".run/local-sql-smoke.json")
    report.parent.mkdir(exist_ok=True)

    def save():
        record["requests"] = budget.requests
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        with httpx.Client(timeout=90, trust_env=False, event_hooks={"request": [budget.check]}) as client:
            configuration["default_llm"] = ChatOpenAI(
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama-local-not-a-secret",
                model=LOCAL_MODEL_NAME,
                temperature=0,
                max_tokens=512,
                timeout=90,
                max_retries=0,
                http_client=client,
                use_responses_api=False,
                http_socket_options=(),
            )
            run_sql_delegation(configuration, record, on_progress=save, use_configured_profile=use_configured_profile)
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
    finally:
        save()
    return record


if __name__ == "__main__":
    result = run_local_sql_smoke(Path.cwd())
    print(
        json.dumps(
            {key: result.get(key) for key in ("status", "requests", "checks", "elapsed_seconds", "error_type")},
            indent=2,
        )
    )
