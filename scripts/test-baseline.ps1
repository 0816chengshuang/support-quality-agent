param()

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Create the Python environment and install the frozen upstream test dependencies first."
}
$variables = @("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "ANONYMIZED_TELEMETRY", "OPENAI_API_KEY")
$previous = @{}
foreach ($name in $variables) { $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
Push-Location $root
try {
    $env:LANGSMITH_TRACING = "false"
    $env:LANGCHAIN_TRACING_V2 = "false"
    $env:ANONYMIZED_TELEMETRY = "false"
    $env:OPENAI_API_KEY = ""
    New-Item -ItemType Directory -Path ".run" -Force | Out-Null
    $tests = @(
        "tests/test_langgraph_dependency_resolution.py",
        "tests/test_langgraph_graph_runtime.py",
        "tests/test_langgraph_persistence_contract.py",
        "tests/test_text2sql_confidence_gate.py",
        "tests/test_tools_search_knowledge.py",
        "tests/test_config_loader.py",
        "tests/test_history_loader.py",
        "tests/test_streaming.py",
        "tests/test_langgraph_streaming_contract.py",
        "tests/local/test_sqlite_baseline.py",
        "tests/local/test_streamlit_unconfigured.py",
        "tests/local/test_streamlit_answers.py",
        "tests/local/test_streamlit_flow.py",
        "tests/local/test_support_quality.py",
        "tests/local/test_business_documents.py",
        "tests/local/test_support_demo.py",
        "tests/local/test_agent_allowlist.py",
        "tests/local/test_sql_acceptance.py",
        "tests/local/test_schema_efficiency.py",
        "tests/local/test_sql_handoff.py",
        "tests/local/test_local_retrieval.py",
        "tests/local/test_support_routing.py",
        "tests/local/test_paid_acceptance.py",
        "tests/test_llm.py",
        "tests/test_text2sql_schema_linking.py"
    )
    & $python -m pytest @tests -o addopts= --junitxml=.run/baseline.xml -q 2>&1 |
        Tee-Object -FilePath ".run/baseline.log"
    if ($LASTEXITCODE -ne 0) { throw "Baseline tests failed. See .run/baseline.log." }
}
finally {
    foreach ($name in $variables) { [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process") }
    Pop-Location
}
