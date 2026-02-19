"""SQLite-backed cache for issue metadata and embeddings."""

from __future__ import annotations

import json
import sqlite3
import time


class IssueCache:
    """SQLite cache with TTL-based invalidation for issues."""

    def __init__(self, db_path: str = ":memory:", ttl_hours: int = 24):
        self.db_path = db_path
        self.ttl_seconds = ttl_hours * 3600
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS issue_cache (
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                issue_number INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT,
                fetched_at REAL NOT NULL,
                PRIMARY KEY (owner, repo, issue_number)
            )
        """)
        self._conn.commit()

    def get_issue(self, owner: str, repo: str, issue_number: int) -> dict | None:
        """Get cached issue metadata, or None if missing/stale."""
        row = self._conn.execute(
            "SELECT metadata_json, fetched_at FROM issue_cache WHERE owner=? AND repo=? AND issue_number=?",
            (owner, repo, issue_number),
        ).fetchone()

        if row is None:
            return None

        if time.time() - row["fetched_at"] > self.ttl_seconds:
            return None

        return json.loads(row["metadata_json"])

    def put_issue(self, owner: str, repo: str, issue_number: int, metadata: dict) -> None:
        """Store issue metadata in cache."""
        self._conn.execute(
            """INSERT OR REPLACE INTO issue_cache (owner, repo, issue_number, metadata_json, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            (owner, repo, issue_number, json.dumps(metadata), time.time()),
        )
        self._conn.commit()

    def get_embedding(self, owner: str, repo: str, issue_number: int) -> list[float] | None:
        """Get cached embedding vector, or None if missing."""
        row = self._conn.execute(
            "SELECT embedding_json FROM issue_cache WHERE owner=? AND repo=? AND issue_number=?",
            (owner, repo, issue_number),
        ).fetchone()

        if row is None or row["embedding_json"] is None:
            return None

        return json.loads(row["embedding_json"])

    def put_embedding(self, owner: str, repo: str, issue_number: int, embedding: list[float]) -> None:
        """Store embedding vector for a cached issue."""
        self._conn.execute(
            "UPDATE issue_cache SET embedding_json=? WHERE owner=? AND repo=? AND issue_number=?",
            (json.dumps(embedding), owner, repo, issue_number),
        )
        self._conn.commit()

    def get_all_issues(self, owner: str, repo: str) -> list[dict]:
        """Get all non-stale cached issues for a repo."""
        cutoff = time.time() - self.ttl_seconds
        rows = self._conn.execute(
            "SELECT metadata_json FROM issue_cache WHERE owner=? AND repo=? AND fetched_at>?",
            (owner, repo, cutoff),
        ).fetchall()
        return [json.loads(row["metadata_json"]) for row in rows]

    def clear_stale(self) -> int:
        """Remove stale entries. Returns count of deleted rows."""
        cutoff = time.time() - self.ttl_seconds
        cursor = self._conn.execute(
            "DELETE FROM issue_cache WHERE fetched_at<?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
