"""Offline budget/transport tests; no paid or local model endpoint is contacted."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from openchatbi.paid_acceptance import (
    BASE_URL,
    MODEL,
    AcceptanceBudget,
    AcceptanceChatOpenAI,
    AcceptanceHTTPClient,
    AcceptanceStop,
    provider_error_summary,
    run_paid_acceptance,
    validate_sql_stage,
)


def request(url=BASE_URL + "/chat/completions", method="POST", **changes):
    return httpx.Request(method, url, json={"model": MODEL, "max_tokens": 512, "messages": [], **changes})


def test_six_request_budget_persists_across_reruns(tmp_path):
    path = tmp_path / "ledger.sqlite"
    for _ in range(6):
        AcceptanceBudget(path).reserve(request())
    budget = AcceptanceBudget(path)
    with pytest.raises(AcceptanceStop, match="exhausted"):
        budget.reserve(request())
    assert budget.snapshot()["reserved_requests"] == 6
    assert budget.snapshot()["remaining_requests"] == 0


def test_additional_ledger_stays_at_five_without_changing_original(tmp_path):
    original = AcceptanceBudget(tmp_path / "original.sqlite")
    for _ in range(6):
        original.reserve(request())
    path = tmp_path / "additional.sqlite"
    budget = AcceptanceBudget(path, max_requests=5)
    for _ in range(5):
        budget.reserve(request())
    with pytest.raises(AcceptanceStop, match="exhausted"):
        AcceptanceBudget(path, max_requests=5).reserve(request())
    with pytest.raises(ValueError, match="cannot be changed"):
        AcceptanceBudget(path, max_requests=6)
    assert budget.snapshot()["authorized_requests"] == 5
    assert budget.snapshot()["reserved_requests"] == 5
    assert original.snapshot()["reserved_requests"] == 6


def test_concurrent_process_style_reservations_do_not_exceed_six(tmp_path):
    path = tmp_path / "ledger.sqlite"
    AcceptanceBudget(path)

    def attempt(_):
        try:
            AcceptanceBudget(path).reserve(request())
            return True
        except AcceptanceStop:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(10))) == 6


@pytest.mark.parametrize(
    "url,method",
    [
        ("http://api.gptgod.online/v1/chat/completions", "POST"),
        ("https://example.com/v1/chat/completions", "POST"),
        (BASE_URL + "/models", "GET"),
        (BASE_URL + "/chat/completions?extra=1", "POST"),
    ],
)
def test_wrong_endpoint_does_not_spend_budget(tmp_path, url, method):
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with pytest.raises(AcceptanceStop):
        budget.reserve(request(url, method))
    assert budget.snapshot()["reserved_requests"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "not-approved"},
        {"max_tokens": 513},
        {"max_tokens": None},
        {"max_tokens": True},
        {"n": 2},
        {"stream": True},
        {"tools": [{"function": {"name": "run_python_code"}}]},
    ],
)
def test_unapproved_request_is_blocked_before_network(tmp_path, changes):
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with pytest.raises(AcceptanceStop):
        budget.reserve(request(**changes))
    assert budget.snapshot()["reserved_requests"] == 0


def test_request_and_aggregate_context_byte_limits(tmp_path):
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with pytest.raises(AcceptanceStop, match="too_large"):
        budget.reserve(request(messages=[{"role": "user", "content": "x" * 48000}]))
    for _ in range(2):
        budget.reserve(request(messages=[{"role": "user", "content": "x" * 45000}]))
    with pytest.raises(AcceptanceStop, match="exhausted"):
        budget.reserve(request(messages=[{"role": "user", "content": "x" * 45000}]))
    assert budget.snapshot()["reserved_requests"] == 2


def test_http_error_stops_without_retry_or_error_body_in_ledger(tmp_path, monkeypatch):
    calls = []

    def fake_send(self, call, **kwargs):
        calls.append(call)
        return httpx.Response(503, json={"error": "sensitive-error-body"}, request=call)

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    path = tmp_path / "ledger.sqlite"
    budget = AcceptanceBudget(path)
    with AcceptanceHTTPClient(budget) as client, pytest.raises(AcceptanceStop, match="503"):
        client.send(request())
    assert len(calls) == 1
    assert budget.snapshot()["reserved_requests"] == 1
    assert b"sensitive-error-body" not in path.read_bytes()
    assert not issubclass(AcceptanceStop, Exception)


def test_transport_error_is_not_retried(tmp_path, monkeypatch):
    def fake_send(self, call, **kwargs):
        raise httpx.ConnectError("sensitive-connect-details", request=call)

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with AcceptanceHTTPClient(budget) as client, pytest.raises(AcceptanceStop, match="transport_error_no_retry"):
        client.send(request())
    assert budget.snapshot()["reserved_requests"] == 1


def test_provider_error_fields_are_bounded_and_credentials_are_redacted():
    key = "offline-secret-without-standard-prefix"
    call = request()
    call.headers["authorization"] = f"Bearer {key}"
    response = httpx.Response(
        400,
        json={
            "error": {
                "type": "invalid_request_error",
                "code": "unsupported_parameter",
                "param": "tools",
                "message": f"Rejected {key}; sk-offline-other-secret; Bearer another-token\n" + "x" * 600,
                "headers": {"authorization": key},
            },
            "unrelated": key,
        },
    )
    result = provider_error_summary(response, call)
    assert set(result) == {"type", "code", "param", "message"}
    assert result["param"] == "tools"
    assert len(result["message"]) == 512
    assert all(secret not in json.dumps(result) for secret in (key, "sk-offline-other-secret", "another-token"))
    assert "\n" not in result["message"]


@pytest.mark.parametrize(
    "body,detail",
    [
        (b"<html>private upstream details</html>", "non_json_error_body_omitted"),
        (b"x" * 16385, "error_body_exceeds_diagnostic_limit"),
        (b"[]", "unrecognized_error_body_omitted"),
    ],
)
def test_nonstandard_error_body_is_not_dumped(body, detail):
    assert provider_error_summary(httpx.Response(400, content=body), request()) == {"detail": detail}


def test_http_400_preserves_diagnostic_but_does_not_retry_or_store_body(tmp_path, monkeypatch):
    calls = []

    def fake_send(self, call, **kwargs):
        calls.append(call)
        return httpx.Response(
            400, request=call, json={"error": {"code": "invalid_model", "message": "offline rejection"}}
        )

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with AcceptanceHTTPClient(budget) as client, pytest.raises(AcceptanceStop) as stopped:
        client.send(request())
    assert stopped.value.provider_error["code"] == "invalid_model"
    assert len(calls) == 1
    assert budget.snapshot()["remaining_requests"] == 5
    assert b"offline rejection" not in budget.path.read_bytes()


def test_provider_reported_overrun_blocks_further_calls(tmp_path):
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    identifier = budget.reserve(request())
    response = httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 700}})
    with pytest.raises(AcceptanceStop, match="output_limit"):
        budget.finish(identifier, "received", response)
    with pytest.raises(AcceptanceStop, match="output_limit"):
        budget.reserve(request())
    assert budget.snapshot()["reserved_requests"] == 1


def test_actual_sdk_payload_has_512_token_cap_without_a_real_request(tmp_path, monkeypatch):
    import openchatbi.llm.llm as llm_module
    from openchatbi.llm.llm import bind_tools_for_profile
    from openchatbi.tool.ask_human import AskHuman

    monkeypatch.setattr(llm_module.config, "get", lambda: SimpleNamespace(openai_strict_tools=False))
    seen = []

    def fake_send(self, call, **kwargs):
        seen.append(json.loads(call.content))
        return httpx.Response(
            200,
            request=call,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "offline"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with AcceptanceHTTPClient(budget) as client:
        model = ChatOpenAI(
            base_url=BASE_URL,
            model=MODEL,
            api_key="offline-not-a-key",
            extra_body={"max_tokens": 512},
            max_retries=0,
            http_client=client,
            use_responses_api=False,
            http_socket_options=(),
        )
        assert (
            bind_tools_for_profile(model, [AskHuman], json_output=True).invoke([HumanMessage("offline")]).content
            == "offline"
        )
    assert seen[0]["max_tokens"] == 512
    assert seen[0]["model"] == MODEL
    assert all("strict" not in tool["function"] for tool in seen[0]["tools"])
    assert "response_format" not in seen[0]
    assert budget.snapshot()["output_tokens_reported"] == 1


def test_sdk_error_stops_acceptance_instead_of_becoming_an_empty_tool_result(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("offline-sensitive-diagnostic")

    monkeypatch.setattr(ChatOpenAI, "invoke", fail)
    model = AcceptanceChatOpenAI(model=MODEL, api_key="offline-placeholder-only", max_retries=0)
    with pytest.raises(AcceptanceStop, match="model_sdk_ValueError_no_retry") as stopped:
        model.invoke([HumanMessage("offline")])
    assert "sensitive" not in str(stopped.value)


def test_truncated_response_stops_before_sdk_json_parsing_or_another_call(tmp_path, monkeypatch):
    calls = []

    def fake_send(self, call, **kwargs):
        calls.append(call)
        return httpx.Response(
            200,
            request=call,
            json={
                "choices": [{"finish_reason": "length", "message": {"content": "incomplete JSON"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 512},
            },
        )

    monkeypatch.setattr(httpx.Client, "send", fake_send)
    budget = AcceptanceBudget(tmp_path / "ledger.sqlite")
    with AcceptanceHTTPClient(budget) as client, pytest.raises(AcceptanceStop, match="model_output_truncated_no_retry"):
        client.send(request())
    assert len(calls) == 1
    assert budget.snapshot()["reserved_requests"] == 1
    assert budget.snapshot()["output_tokens_reported"] == 512


@pytest.mark.parametrize(
    "event,reason",
    [
        ({"node": "table_selection", "selected_table_count": 0}, "sql_table_selection_failed_no_retry"),
        ({"node": "generate_sql", "generated_sql": False}, "sql_generation_failed_no_retry"),
    ],
)
def test_failed_sql_stage_does_not_fall_back_to_parent_model_retry(event, reason):
    with pytest.raises(AcceptanceStop, match=reason):
        validate_sql_stage(event)


def test_successful_sql_stages_and_parent_events_are_not_blocked():
    for event in (
        {"node": "llm_node"},
        {"node": "table_selection", "selected_table_count": 1},
        {"node": "generate_sql", "generated_sql": True},
    ):
        validate_sql_stage(event)


def test_incomplete_remaining_budget_is_rejected_before_constructing_model(tmp_path, monkeypatch):
    import yaml

    import openchatbi.paid_acceptance as module

    monkeypatch.setenv("OPENCHATBI_ACCEPTANCE_API_KEY", "offline-placeholder-only")
    folder = tmp_path / ".local/support-demo"
    folder.mkdir(parents=True)
    uri = f"sqlite:///file:{(folder / 'support.sqlite').as_posix()}?mode=ro&uri=true"
    (folder / "config.local.yaml").write_text(yaml.safe_dump({"data_warehouse_config": {"uri": uri}}), encoding="utf-8")
    budget = AcceptanceBudget(tmp_path / ".run/paid-acceptance-v1.sqlite")
    for _ in range(2):
        budget.reserve(request())

    def forbidden(*args, **kwargs):
        raise AssertionError("Insufficient budget must not construct a model or start a workflow")

    monkeypatch.setattr(module, "AcceptanceChatOpenAI", forbidden)
    monkeypatch.setattr(module, "run_sql_delegation", forbidden)
    result = run_paid_acceptance(tmp_path)
    assert result["stop_reason"] == "insufficient_budget_for_sql_workflow"
    assert result["requests_this_run"] == 0
    assert budget.snapshot()["reserved_requests"] == 2


def test_missing_key_never_constructs_a_budget_or_contacts_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCHATBI_ACCEPTANCE_API_KEY", raising=False)
    result = run_paid_acceptance(tmp_path)
    assert result["status"] == "needs_local_key"
    assert result["requests_this_run"] == 0
    assert not (tmp_path / ".run/paid-acceptance-v1.sqlite").exists()


def test_key_is_redacted_from_records_and_captured_logs(tmp_path, monkeypatch):
    import yaml

    import openchatbi.paid_acceptance as module

    key = "offline-sensitive-key-never-send"
    monkeypatch.setenv("OPENCHATBI_ACCEPTANCE_API_KEY", key)
    folder = tmp_path / ".local/support-demo"
    folder.mkdir(parents=True)
    uri = f"sqlite:///file:{(folder / 'support.sqlite').as_posix()}?mode=ro&uri=true"
    (folder / "config.local.yaml").write_text(yaml.safe_dump({"data_warehouse_config": {"uri": uri}}), encoding="utf-8")

    def fake_workflow(configuration, record, on_progress):
        print(key)
        record.update(status="offline_test_only", answer=key)
        on_progress()

    monkeypatch.setattr(module, "run_sql_delegation", fake_workflow)
    result = run_paid_acceptance(tmp_path)
    assert result["requests_this_run"] == 0
    assert key not in (tmp_path / ".run/paid-acceptance.json").read_text(encoding="utf-8")
    assert key not in (tmp_path / ".run/paid-acceptance.log").read_text(encoding="utf-8")
    assert key.encode() not in (tmp_path / ".run/paid-acceptance-v1.sqlite").read_bytes()
