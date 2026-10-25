"""Bounded real local embedding checks, never a model download or page launcher."""

import json
import math
import time
from pathlib import Path

import httpx
from langchain_openai import OpenAIEmbeddings

from openchatbi.business_knowledge import BusinessKnowledgeSettings, search_business_documents

EMBEDDING_MODEL = "embeddinggemma:300m"
CASES = (
    ("到观察时间还没办完的单子，会计入超时占比的基数吗", "超时率"),
    ("测试单、重复单和已经取消的单子会参加统计吗", "统计对象"),
    ("已经完成的单子平均用了多长时间，怎么算", "平均处理时长"),
    ("How many eligible tickets were newly created within the range?", "工单数量"),
    ("还没有解决的请求，会被当成零分钟计入平均耗时吗", "平均处理时长"),
    ("Which clock and boundary define the observation cutoff?", "统计对象"),
)


class LocalEmbeddingBudget:
    def __init__(self):
        self.requests = 0
        self.texts = 0
        self.characters = 0
        self.deadline = time.monotonic() + 240

    def check(self, request: httpx.Request):
        if request.method != "POST" or str(request.url) != "http://127.0.0.1:11434/v1/embeddings":
            raise ValueError("Only the fixed local embedding endpoint is permitted")
        body = json.loads(request.content)
        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if body.get("model") != EMBEDDING_MODEL or not isinstance(inputs, list) or not 1 <= len(inputs) <= 32:
            raise ValueError("Unapproved embedding model or input batch")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 2000 for value in inputs):
            raise ValueError("Embedding inputs must be bounded plain text")
        size = sum(map(len, inputs))
        if (
            self.requests >= 16
            or self.texts + len(inputs) > 64
            or self.characters + size > 64000
            or time.monotonic() >= self.deadline
        ):
            raise RuntimeError("Local embedding budget exhausted")
        self.requests += 1
        self.texts += len(inputs)
        self.characters += size


def run_local_retrieval_smoke(repository: Path) -> dict:
    root = repository.resolve(strict=True)
    budget = LocalEmbeddingBudget()
    record = {
        "status": "not_completed",
        "model": EMBEDDING_MODEL,
        "paid_requests": 0,
        "page_started": False,
        "cases": [],
        "scope": "six small retrieval cases, not production recall or answer accuracy",
    }
    started = time.monotonic()
    try:
        with httpx.Client(timeout=60, trust_env=False, event_hooks={"request": [budget.check]}) as client:
            embedding = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama-local-not-a-secret",
                max_retries=0,
                request_timeout=60,
                check_embedding_ctx_length=False,
                http_client=client,
            )
            vector = embedding.embed_query("验证本地向量维度")
            record["dimension"] = len(vector)
            record["finite_unit_vector"] = (
                bool(vector)
                and all(math.isfinite(value) for value in vector)
                and abs(sum(value * value for value in vector) - 1) < 0.001
            )
            settings = BusinessKnowledgeSettings.model_validate(
                {
                    "root": str(root / "openchatbi/skills/support_quality"),
                    "sources": [{"id": "support-quality-v1", "path": "METRICS.md"}],
                    "mode": "vector",
                    "top_k": 3,
                    "min_similarity": 0.35,
                }
            )
            for query, section in CASES:
                result = search_business_documents(settings, [query], embedding)
                refs = [item["source_ref"] for item in result["passages"]]
                expected = f"support-quality-v1#{section}"
                record["cases"].append(
                    {
                        "query": query,
                        "expected": expected,
                        "actual": refs,
                        "top1_match": bool(refs) and refs[0] == expected,
                        "top3_match": expected in refs,
                        "status": result["status"],
                        "context_chars": result["context_chars"],
                    }
                )
            before = budget.requests
            search_business_documents(settings, [CASES[0][0]], embedding)
            record["cached_index_one_query_request"] = budget.requests - before == 1
            record["status"] = (
                "passed"
                if record["finite_unit_vector"]
                and record["cached_index_one_query_request"]
                and all(item["top3_match"] for item in record["cases"])
                else "checks_failed"
            )
    except Exception as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
    record.update(
        requests=budget.requests, embedded_texts=budget.texts, elapsed_seconds=round(time.monotonic() - started, 2)
    )
    output = root / ".run/local-retrieval-smoke.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    print(json.dumps(run_local_retrieval_smoke(Path.cwd()), ensure_ascii=True, indent=2))
