"""Offline checks of upstream SQL execution against synthetic ticket data.

These checks do not evaluate LLM generation or prove production SQL isolation.
"""

import csv
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, text

from openchatbi.constants import SQL_SECURITY_ERROR, SQL_SUCCESS
from openchatbi.text2sql import generate_sql


@pytest.fixture
def ticket_executor(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'tickets.db').as_posix()}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE tickets (id INTEGER PRIMARY KEY, category TEXT, minutes INTEGER)"))
        conn.execute(
            text("INSERT INTO tickets VALUES (:id, :category, :minutes)"),
            [
                {"id": 1, "category": "delivery", "minutes": 120},
                {"id": 2, "category": "delivery", "minutes": 60},
                {"id": 3, "category": "account", "minutes": 30},
                {"id": 4, "category": "account", "minutes": None},
            ],
        )
    settings = SimpleNamespace(
        enable_fail_closed_sql_guard=True,
        enable_sql_result_limit=True,
        sql_result_limit=2,
        enable_empty_result_error=False,
    )
    monkeypatch.setattr(generate_sql.config, "get", lambda: settings)
    catalog = Mock()
    catalog.get_sql_engine.return_value = engine
    _, execute, _, _, _, _ = generate_sql.create_sql_nodes(Mock(), catalog, "sqlite", None)
    yield execute, engine
    engine.dispose()


def test_ticket_aggregate_matches_known_answer(ticket_executor):
    execute, _ = ticket_executor
    result = execute(
        {
            "messages": [],
            "sql": (
                "SELECT category, AVG(minutes) AS average_minutes FROM tickets "
                "WHERE minutes IS NOT NULL GROUP BY category ORDER BY category"
            ),
        }
    )
    assert result["sql_execution_result"] == SQL_SUCCESS
    rows = list(csv.DictReader(io.StringIO(result["data"])))
    assert [(row["category"], float(row["average_minutes"])) for row in rows] == [("account", 30.0), ("delivery", 90.0)]


def test_query_output_is_bounded(ticket_executor):
    execute, _ = ticket_executor
    result = execute({"messages": [], "sql": "SELECT id FROM tickets ORDER BY id"})
    assert result["sql_execution_result"] == SQL_SUCCESS
    assert [row["id"] for row in csv.DictReader(io.StringIO(result["data"]))] == ["1", "2"]


@pytest.mark.parametrize("sql", ["DROP TABLE tickets", "DELETE FROM tickets", "PRAGMA table_info(tickets)"])
def test_write_or_unsupported_operation_is_rejected(ticket_executor, sql):
    execute, engine = ticket_executor
    result = execute({"messages": [], "sql": sql})
    assert result["sql_execution_result"] == SQL_SECURITY_ERROR
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tickets")).scalar_one() == 4
