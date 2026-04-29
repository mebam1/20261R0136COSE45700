from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime
from typing import Any

from app.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TEXT NOT NULL,
                store_name TEXT NOT NULL,
                cctv_id TEXT NOT NULL,
                cctv_nickname TEXT NOT NULL,
                roi_name TEXT NOT NULL,
                item_type TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence REAL NOT NULL,
                visible_ratio REAL NOT NULL,
                occlusion_duration REAL NOT NULL,
                brightness_mismatch_duration REAL NOT NULL,
                summary TEXT NOT NULL,
                source_path TEXT NOT NULL
            )
            """
        )
        connection.commit()


def truncate_to_hour(timestamp: datetime) -> str:
    return timestamp.replace(minute=0, second=0, microsecond=0).isoformat(timespec="minutes")


def insert_result(record: dict[str, Any]) -> None:
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO analysis_results (
                analyzed_at, store_name, cctv_id, cctv_nickname, roi_name, item_type,
                decision, confidence, visible_ratio, occlusion_duration,
                brightness_mismatch_duration, summary, source_path
            )
            VALUES (
                :analyzed_at, :store_name, :cctv_id, :cctv_nickname, :roi_name, :item_type,
                :decision, :confidence, :visible_ratio, :occlusion_duration,
                :brightness_mismatch_duration, :summary, :source_path
            )
            """,
            record,
        )
        connection.commit()


def fetch_results(filters: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
    filters = filters or {}
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for key in ("store_name", "cctv_id", "roi_name", "decision", "item_type"):
        value = filters.get(key)
        if value:
            clauses.append(f"{key} = :{key}")
            params[key] = value

    query = "SELECT * FROM analysis_results"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY analyzed_at DESC, id DESC"

    with closing(get_connection()) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def fetch_filter_options() -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    with closing(get_connection()) as connection:
        for column in ("store_name", "cctv_id", "roi_name", "decision", "item_type"):
            rows = connection.execute(
                f"SELECT DISTINCT {column} AS value FROM analysis_results ORDER BY value"
            ).fetchall()
            options[column] = [row["value"] for row in rows if row["value"]]
    return options


def fetch_latest_by_roi(roi_name: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    roi_clause = ""
    if roi_name:
        roi_clause = "WHERE roi_name = :roi_name"
        params["roi_name"] = roi_name

    query = f"""
        SELECT r1.*
        FROM analysis_results r1
        JOIN (
            SELECT cctv_id, roi_name, MAX(id) AS max_id
            FROM analysis_results
            {roi_clause}
            GROUP BY cctv_id, roi_name
        ) latest
        ON r1.id = latest.max_id
        ORDER BY
            CASE
                WHEN r1.decision = 'Absent' THEN 0
                WHEN r1.decision = 'Unknown' THEN 1
                ELSE 2
            END,
            r1.store_name,
            r1.cctv_nickname
    """
    with closing(get_connection()) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def clear_results() -> None:
    with closing(get_connection()) as connection:
        connection.execute("DELETE FROM analysis_results")
        connection.commit()


def bulk_insert(records: Iterable[dict[str, Any]]) -> None:
    with closing(get_connection()) as connection:
        connection.executemany(
            """
            INSERT INTO analysis_results (
                analyzed_at, store_name, cctv_id, cctv_nickname, roi_name, item_type,
                decision, confidence, visible_ratio, occlusion_duration,
                brightness_mismatch_duration, summary, source_path
            )
            VALUES (
                :analyzed_at, :store_name, :cctv_id, :cctv_nickname, :roi_name, :item_type,
                :decision, :confidence, :visible_ratio, :occlusion_duration,
                :brightness_mismatch_duration, :summary, :source_path
            )
            """,
            list(records),
        )
        connection.commit()
