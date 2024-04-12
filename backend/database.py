"""
Managing sessions's history data

Sessions are identified by unique ids
Each session only consume one pdf
"""

import logging
import sqlite3
from datetime import datetime
from sqlite3 import Error
from typing import Tuple

from sqlalchemy import text
from sqlalchemy.engine.base import Connection

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def create_history_table(conn: Connection, table_name: str) -> None:
    """
    Create new history table with the given name if it does not already exist.
    """

    try:
        # Check if the table already exists
        existing_table = conn.execute(
            text(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            )
        ).fetchone()

        if existing_table:
            logger.info(f"Table '{table_name}' already exists. Skipping creation.")
            return  # If table already exists, do nothing

        # If the table does not exist, create it
        create_table_sql = text(
            f"""CREATE TABLE {table_name} (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        chat_history TEXT,
        created_time TEXT
        );"""
        )

        conn.execute(create_table_sql)
        conn.commit()
        logger.info(f"Created a new {table_name}")
        print(f"Created a new {table_name}")

    except Error as e:
        logger.error(e)


def delete_history_table(conn: Connection, table_name: str) -> None:
    # SQL statement to drop the table
    drop_table_sql = f"DROP TABLE IF EXISTS {table_name};"

    try:
        conn.execute(text(drop_table_sql))
        conn.commit()
        logger.info(f"Dropped {table_name}")

    except Error as e:
        logger.error(e)


def insert_into_session_table(conn: Connection, table_name: str, data: Tuple[str]):
    """
    Inserts a new row into the session_table.

    param:
        data: A tuple containing (file_name, chat_history).
    """

    created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = (*data, created_time)
    insert_sql = f"""INSERT INTO {table_name} (file_name, chat_history, created_time)
                    VALUES {row};"""

    try:
        conn.execute(text(insert_sql))
        conn.commit()
        logger.info("New row has been inserted into 'session_table'.")
    except Error as e:
        logger.error(e)
