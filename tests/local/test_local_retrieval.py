"""Offline endpoint/budget checks; not real embedding-quality evidence."""

import httpx
import pytest

from openchatbi.local_retrieval_smoke import EMBEDDING_MODEL, LocalEmbeddingBudget


def request(url="http://127.0.0.1:11434/v1/embeddings", **overrides):
    return httpx.Request("POST", url, json={"model": EMBEDDING_MODEL, "input": ["synthetic text"], **overrides})


@pytest.mark.parametrize(
    "url",
    ["https://example.com/v1/embeddings", "http://127.0.0.1:11434/api/pull", "http://localhost:11434/v1/embeddings"],
)
def test_remote_or_download_endpoint_is_blocked(url):
    budget = LocalEmbeddingBudget()
    with pytest.raises(ValueError):
        budget.check(request(url))
    assert budget.requests == 0


@pytest.mark.parametrize("body", [{"model": "other"}, {"input": [1, 2]}, {"input": []}, {"input": ["x" * 2001]}])
def test_wrong_model_or_unbounded_input_is_blocked(body):
    budget = LocalEmbeddingBudget()
    with pytest.raises(ValueError):
        budget.check(request(**body))
    assert budget.requests == 0


def test_sixteen_request_limit_and_deadline():
    budget = LocalEmbeddingBudget()
    for _ in range(16):
        budget.check(request())
    with pytest.raises(RuntimeError):
        budget.check(request())
    assert budget.requests == 16
    expired = LocalEmbeddingBudget()
    expired.deadline = 0
    with pytest.raises(RuntimeError):
        expired.check(request())
    assert expired.requests == 0
