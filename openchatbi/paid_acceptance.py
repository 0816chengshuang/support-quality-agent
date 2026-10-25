"""Explicit six-request API acceptance; no UI startup, hidden fallback or key files."""

import contextlib
import io
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import httpx
import yaml
from langchain_openai import ChatOpenAI

from openchatbi.sql_acceptance import QUESTION, REQUEST_TOOLS, run_sql_delegation

MODEL = "deepseek-v4.1-flash"
BASE_URL = "https://api.gptgod.online/v1"
MAX_REQUESTS = 6
MAX_OUTPUT_TOKENS = 512
MAX_REQUEST_BYTES = 48000
MAX_TOTAL_BYTES = 120000
MIN_SQL_REQUESTS = 5


class AcceptanceStop(BaseException):
    """Control-flow stop must not be swallowed by an upstream retry handler."""

    def __init__(self, reason: str, *, provider_error: dict | None = None):
        super().__init__(reason)
        self.provider_error = provider_error


class AcceptanceChatOpenAI(ChatOpenAI):
    """Stop the bounded acceptance on SDK errors instead of asking again."""

    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as error:
            raise AcceptanceStop(f"model_sdk_{type(error).__name__}_no_retry") from None


def provider_error_summary(response: httpx.Response, request: httpx.Request) -> dict:
    """Keep only bounded error fields, never headers or the raw response body."""
    if len(response.content) > 16384:
        return {"detail": "error_body_exceeds_diagnostic_limit"}
    try:
        payload = response.json()
    except ValueError:
        return {"detail": "non_json_error_body_omitted"}
    if not isinstance(payload, dict):
        return {"detail": "unrecognized_error_body_omitted"}
    error = payload.get("error", payload)
    if isinstance(error, str):
        error = {"message": error}
    if not isinstance(error, dict):
        return {"detail": "unrecognized_error_body_omitted"}
    authorization = request.headers.get("authorization", "")
    key = authorization.partition(" ")[2].strip()
    result = {}
    for field in ("type", "code", "param", "message"):
        value = error.get(field)
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            continue
        text = str(value)
        if key:
            text = text.replace(key, "[REDACTED]")
        text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
        text = re.sub(r"\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text, flags=re.I)
        text = " ".join(
            "".join(character for character in text if character.isprintable() or character.isspace()).split()
        )
        result[field] = text[:512]
    return result or {"detail": "no_standard_error_fields"}


def validate_sql_stage(event: dict):
    """Do not spend another model turn retrying a failed diagnostic stage."""
    if event.get("node") == "table_selection" and not event.get("selected_table_count"):
        raise AcceptanceStop("sql_table_selection_failed_no_retry")
    if event.get("node") == "generate_sql" and not event.get("generated_sql"):
        raise AcceptanceStop("sql_generation_failed_no_retry")


class AcceptanceBudget:
    """Reserve before sending; a persistent SQLite ledger survives reruns."""

    def __init__(self, path: Path, max_requests: int = MAX_REQUESTS):
        if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
            raise ValueError("A ledger must have an explicit limit between one and six")
        self.path = path
        self.max_requests = max_requests
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY, body_bytes INTEGER NOT NULL,
                reserved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL, http_status INTEGER,
                input_tokens INTEGER, output_tokens INTEGER)""")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS budget_settings (id INTEGER PRIMARY KEY CHECK (id=1), max_requests INTEGER NOT NULL)"
            )
            connection.execute("INSERT OR IGNORE INTO budget_settings VALUES (1, ?)", (max_requests,))
            if connection.execute("SELECT max_requests FROM budget_settings WHERE id=1").fetchone()[0] != max_requests:
                raise ValueError("An existing ledger's request limit cannot be changed")

    def reserve(self, request: httpx.Request) -> int:
        url = request.url
        if not (
            request.method == "POST"
            and url.scheme == "https"
            and url.host == "api.gptgod.online"
            and url.port in (None, 443)
            and url.path == "/v1/chat/completions"
            and not url.query
            and not url.username
            and not url.password
        ):
            raise AcceptanceStop("unapproved_endpoint")
        try:
            body = json.loads(request.content)
            limits = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
            names = {item["function"]["name"] for item in body.get("tools", [])}
        except (KeyError, TypeError, ValueError):
            raise AcceptanceStop("invalid_request_schema") from None
        if (
            body.get("model") != MODEL
            or not limits
            or any(type(value) is not int or not 1 <= value <= MAX_OUTPUT_TOKENS for value in limits)
            or body.get("n", 1) != 1
            or body.get("stream", False)
            or not names.issubset(REQUEST_TOOLS)
        ):
            raise AcceptanceStop("unapproved_model_tokens_or_tools")
        size = len(request.content)
        if size > MAX_REQUEST_BYTES:
            raise AcceptanceStop("request_context_too_large")
        with sqlite3.connect(self.path, timeout=5) as connection:
            connection.execute("BEGIN IMMEDIATE")
            used, total_size = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(body_bytes), 0) FROM requests"
            ).fetchone()
            if used >= self.max_requests or total_size + size > MAX_TOTAL_BYTES:
                raise AcceptanceStop("authorized_budget_exhausted")
            if connection.execute(
                "SELECT 1 FROM requests WHERE status = 'provider_output_limit_violation' LIMIT 1"
            ).fetchone():
                raise AcceptanceStop("provider_output_limit_violation")
            cursor = connection.execute("INSERT INTO requests (body_bytes, status) VALUES (?, 'reserved')", (size,))
            return cursor.lastrowid

    def finish(self, identifier: int, status: str, response=None):
        usage = {}
        if response is not None and response.status_code < 300:
            try:
                usage = response.json().get("usage") or {}
            except (TypeError, ValueError):
                pass
        counts = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        counts = [value if type(value) is int and value >= 0 else None for value in counts]
        if counts[1] is not None and counts[1] > MAX_OUTPUT_TOKENS:
            status = "provider_output_limit_violation"
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """UPDATE requests SET status=?, http_status=?, input_tokens=?, output_tokens=?
                WHERE id=?""",
                (status, response.status_code if response is not None else None, *counts, identifier),
            )
        if status == "provider_output_limit_violation":
            raise AcceptanceStop(status)

    def snapshot(self) -> dict:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT id, status, http_status, input_tokens, output_tokens FROM requests ORDER BY id"
            ).fetchall()
        return {
            "authorized_requests": self.max_requests,
            "reserved_requests": len(rows),
            "remaining_requests": max(0, self.max_requests - len(rows)),
            "input_tokens_reported": sum(row[3] or 0 for row in rows),
            "output_tokens_reported": sum(row[4] or 0 for row in rows),
            "requests": [
                {"id": r[0], "status": r[1], "http_status": r[2], "input_tokens": r[3], "output_tokens": r[4]}
                for r in rows
            ],
            "currency_cost": None,
        }


class AcceptanceHTTPClient(httpx.Client):
    def __init__(self, budget: AcceptanceBudget, **kwargs):
        super().__init__(timeout=90, follow_redirects=False, **kwargs)
        self.budget = budget
        self.deadline = time.monotonic() + 360

    def send(self, request, **kwargs):
        if time.monotonic() >= self.deadline:
            raise AcceptanceStop("acceptance_deadline_reached")
        identifier = self.budget.reserve(request)
        try:
            response = super().send(request, **kwargs)
            response.read()
        except httpx.HTTPError:
            self.budget.finish(identifier, "transport_error")
            raise AcceptanceStop("transport_error_no_retry") from None
        if response.status_code >= 300:
            self.budget.finish(identifier, "provider_error", response)
            raise AcceptanceStop(
                f"provider_http_{response.status_code}_no_retry",
                provider_error=provider_error_summary(response, request),
            )
        self.budget.finish(identifier, "received", response)
        try:
            choices = response.json().get("choices") or []
            truncated = any(choice.get("finish_reason") == "length" for choice in choices)
        except (AttributeError, TypeError, ValueError):
            truncated = False
        if truncated:
            raise AcceptanceStop("model_output_truncated_no_retry")
        return response


def run_paid_acceptance(repository: Path, *, additional_five: bool = False) -> dict:
    root = repository.resolve(strict=True)
    name = "paid-acceptance-additional5" if additional_five else "paid-acceptance"
    report = root / f".run/{name}.json"
    report.parent.mkdir(exist_ok=True)
    key = os.environ.get("OPENCHATBI_ACCEPTANCE_API_KEY", "").strip()
    record = {
        "status": "needs_local_key",
        "model": MODEL,
        "endpoint": BASE_URL,
        "page_started": False,
        "question": QUESTION,
        "events": [],
        "requests_this_run": 0,
        "scope": "original main -> SQL role acceptance only, not UI acceptance",
    }
    if not key:
        report.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record
    configuration = yaml.safe_load((root / ".local/support-demo/config.local.yaml").read_text(encoding="utf-8"))
    expected_uri = f"sqlite:///file:{(root / '.local/support-demo/support.sqlite').as_posix()}?mode=ro&uri=true"
    if configuration["data_warehouse_config"]["uri"] != expected_uri:
        raise ValueError("Only the prepared read-only synthetic dataset is allowed")
    if any(
        configuration.get(name)
        for name in ("mcp_servers", "embedding_model", "llm_providers", "text2sql_llm", "analysis_llm")
    ):
        raise ValueError("Additional providers/embedding services must be disabled")
    if additional_five:
        previous = root / ".run/paid-acceptance-v1.sqlite"
        if not previous.is_file():
            raise ValueError("The separately authorized extension requires the original ledger")
        record["previous_budget"] = AcceptanceBudget(previous).snapshot()
        record["authorization_scope"] = "user-approved additional five; original ledger retained"
    ledger = "paid-acceptance-additional5-v1.sqlite" if additional_five else "paid-acceptance-v1.sqlite"
    budget = AcceptanceBudget(root / ".run" / ledger, max_requests=5 if additional_five else MAX_REQUESTS)
    initial = budget.snapshot()["reserved_requests"]
    started = time.monotonic()
    captured = io.StringIO()
    for logger_name in ("openai", "httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    def save():
        record["budget"] = budget.snapshot()
        record["cumulative_reserved_requests"] = record["budget"]["reserved_requests"] + record.get(
            "previous_budget", {}
        ).get("reserved_requests", 0)
        record["requests_this_run"] = record["budget"]["reserved_requests"] - initial
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        contents = json.dumps(record, ensure_ascii=False, indent=2).replace(key, "[REDACTED]")
        report.write_text(contents, encoding="utf-8")

    def save_progress():
        save()
        if record["events"]:
            validate_sql_stage(record["events"][-1])

    try:
        if budget.snapshot()["remaining_requests"] < MIN_SQL_REQUESTS:
            raise AcceptanceStop("insufficient_budget_for_sql_workflow")
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            with AcceptanceHTTPClient(budget) as client:
                configuration["openai_strict_tools"] = False
                configuration["default_llm"] = AcceptanceChatOpenAI(
                    base_url=BASE_URL,
                    api_key=key,
                    model=MODEL,
                    temperature=0,
                    organization="",
                    extra_body={"max_tokens": MAX_OUTPUT_TOKENS},
                    timeout=90,
                    max_retries=0,
                    http_client=client,
                    use_responses_api=False,
                    http_socket_options=(),
                )
                record["status"] = "running"
                save()
                run_sql_delegation(configuration, record, on_progress=save_progress)
    except AcceptanceStop as error:
        record["status"] = "stopped"
        record["stop_reason"] = str(error)
        if error.provider_error:
            record["provider_error"] = error.provider_error
    except KeyboardInterrupt:
        record["status"] = "interrupted"
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
    finally:
        save()
        (root / f".run/{name}.log").write_text(captured.getvalue().replace(key, "[REDACTED]"), encoding="utf-8")
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--additional-five",
        action="store_true",
        help="Use the separately user-authorized five-request extension, never reset the original ledger",
    )
    outcome = run_paid_acceptance(Path.cwd(), additional_five=parser.parse_args().additional_five)
    print(
        json.dumps(
            {
                name: outcome.get(name)
                for name in (
                    "status",
                    "stop_reason",
                    "provider_error",
                    "requests_this_run",
                    "budget",
                    "cumulative_reserved_requests",
                    "checks",
                )
            },
            ensure_ascii=True,
            indent=2,
        )
    )
