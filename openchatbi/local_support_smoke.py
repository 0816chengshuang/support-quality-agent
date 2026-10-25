"""Two-call local-model diagnostic; no server startup, downloads or paid fallback."""

import json
import time
from datetime import date
from pathlib import Path

import httpx
import yaml
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

from openchatbi.catalog.store.file_system import FileSystemCatalogStore
from openchatbi.support_demo import LOCAL_MODEL_NAME
from openchatbi.tool.support_quality import create_support_quality_tool


class LocalNumericSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    numerator: int
    denominator: int
    ratio: float
    source_ref: str


def require_local_ollama(request: httpx.Request) -> None:
    if not (
        request.url.scheme == "http"
        and request.url.host == "127.0.0.1"
        and request.url.port == 11434
        and request.url.path == "/v1/chat/completions"
    ):
        raise ValueError("This diagnostic may only call the fixed loopback Ollama chat endpoint")


def run_local_support_smoke(repository: Path) -> dict:
    root = repository.resolve(strict=True)
    prepared = root / ".local/support-demo/config.local.yaml"
    configuration = yaml.safe_load(prepared.read_text(encoding="utf-8"))
    warehouse = configuration["data_warehouse_config"]
    expected_uri = f"sqlite:///file:{(root / '.local/support-demo/support.sqlite').as_posix()}?mode=ro&uri=true"
    if warehouse.get("uri") != expected_uri:
        raise ValueError("Diagnostic requires the prepared local read-only SQLite dataset")
    catalog = FileSystemCatalogStore(configuration["catalog_store"]["data_path"], warehouse)
    tool = create_support_quality_tool(catalog, date(2026, 10, 8))
    started = time.monotonic()
    record = {
        "status": "not_completed",
        "model": LOCAL_MODEL_NAME,
        "endpoint": "http://127.0.0.1:11434/v1",
        "model_calls": 0,
        "paid_requests": 0,
        "page_started": False,
        "scope": "staged diagnostic, not full Agent/UI acceptance",
    }
    try:
        with httpx.Client(timeout=90, trust_env=False, event_hooks={"request": [require_local_ollama]}) as client:
            model = ChatOpenAI(
                base_url=record["endpoint"],
                api_key="ollama-local-not-a-secret",
                model=LOCAL_MODEL_NAME,
                temperature=0,
                max_tokens=384,
                timeout=90,
                max_retries=0,
                http_client=client,
                use_responses_api=False,
                http_socket_options=(),
            )
            messages = [
                SystemMessage("Use the supplied tool to compute facts. Do not guess data. Be brief."),
                HumanMessage(
                    "Compute overdue_rate for tickets created from 2026-10-01 inclusive to 2026-10-08 exclusive, UTC. Use overall grouping and no category filter."
                ),
            ]
            record["model_calls"] += 1
            decision = model.bind_tools([tool]).invoke(messages)
            record["first_usage"] = decision.usage_metadata
            record["tool_calls"] = decision.tool_calls
            if len(decision.tool_calls) != 1 or decision.tool_calls[0]["name"] != tool.name:
                record["status"] = "tool_selection_failed"
                record["model_text"] = decision.content
                return record
            call = decision.tool_calls[0]
            result = tool.invoke(call["args"])
            record["tool_result"] = result
            if result["status"] != "succeeded" or result["rows"][0]["value"] != 0.4:
                record["status"] = "tool_result_failed"
                return record
            parser = PydanticOutputParser(pydantic_object=LocalNumericSummary)
            compact = {"contract": result["contract"], "rows": result["rows"]}
            record["model_calls"] += 1
            answer = model.invoke(
                messages
                + [
                    decision,
                    ToolMessage(content=json.dumps(compact), tool_call_id=call["id"]),
                    HumanMessage(
                        "Return only the JSON summary from the tool evidence. ratio is 0 to 1, not percentage.\n"
                        + parser.get_format_instructions()
                    ),
                ]
            )
            record["second_usage"] = answer.usage_metadata
            record["model_text"] = answer.content
            summary = parser.parse(answer.content)
            record["summary"] = summary.model_dump()
            record["status"] = (
                "passed"
                if summary.model_dump()
                == {"numerator": 4, "denominator": 10, "ratio": 0.4, "source_ref": "support-quality-v1#超时率"}
                else "summary_verification_failed"
            )
            return record
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        return record
    finally:
        catalog.get_sql_engine().dispose()
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report = root / ".run/local-support-smoke.json"
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
