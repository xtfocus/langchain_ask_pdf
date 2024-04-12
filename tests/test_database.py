import logging
import os
import sqlite3

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine.base import Connection

from backend.database import (create_history_table, delete_history_table,
                              insert_into_session_table)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

HISTORY_TABLE = "test_history_table"
HISTORY_DB = "test_history.db"

SQLITEDB_PATH = "sqlite:///" + HISTORY_DB

HistoryDBEngine = create_engine(SQLITEDB_PATH)


@pytest.fixture
def db_connection():
    """
    Setup a database connection for the tests, and teardown by deleting the database file after tests.
    """

    with HistoryDBEngine.connect() as conn:
        yield conn


def test_create_history_table(db_connection):
    """
    Test that the history table can be created successfully.
    """
    create_history_table(db_connection, HISTORY_TABLE)
    print(f"created {HISTORY_DB} for tests")
    table_exists = db_connection.execute(
        text(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{HISTORY_TABLE}';"
        )
    ).fetchone()

    assert table_exists, f"{HISTORY_TABLE} table should exist after creation."


def test_insert_into_session_table(db_connection):
    """
    Test that data can be inserted into the history table.
    """
    create_history_table(db_connection, HISTORY_TABLE)
    test_data = ("test_file.txt", "This is a test chat history.")
    insert_into_session_table(db_connection, HISTORY_TABLE, test_data)
    data_row = db_connection.execute(text(f"SELECT * FROM {HISTORY_TABLE}")).fetchone()
    logger.info(f"{data_row}")

    assert data_row, "Data should be inserted into the session_table."


def test_delete_history_table(db_connection):
    """
    Test that the history table can be deleted successfully.
    """
    create_history_table(db_connection, HISTORY_TABLE)

    delete_history_table(db_connection, HISTORY_TABLE)
    table_exists = db_connection.execute(
        text(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{HISTORY_TABLE}';"
        )
    ).fetchone()
    assert not table_exists, f"{HISTORY_TABLE} table should not exist after deletion."
