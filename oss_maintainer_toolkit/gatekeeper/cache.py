"""SQLite-backed cache for PR metadata and embeddings."""

from __future__ import annotations

import json
import sqlite3
import time


class PRCache:
    """SQLite cache with TTL-based invalidation."""

    def __init__(self, db_path: str = ":memory:", ttl_hours: int = 24):
        self.db_path = db_path
        self.ttl_seconds = ttl_hours * 3600
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pr_cache (
                owner TEXT NOT NULL,
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT,
                fetched_at REAL NOT NULL,
                PRIMARY KEY (owner, repo, pr_number)
            )
        """)
        self._conn.commit()

    def get_pr(self, owner: str, repo: str, pr_number: int) -> dict | None:
        """Get cached PR metadata, or None if missing/stale."""
        row = self._conn.execute(
            "SELECT metadata_json, fetched_at FROM pr_cache WHERE owner=? AND repo=? AND pr_number=?",
            (owner, repo, pr_number),
        ).fetchone()

        if row is None:
            return None

        if time.time() - row["fetched_at"] > self.ttl_seconds:
            return None

        return json.loads(row["metadata_json"])

    def put_pr(self, owner: str, repo: str, pr_number: int, metadata: dict) -> None:
        """Store PR metadata in cache."""
        self._conn.execute(
            """INSERT OR REPLACE INTO pr_cache (owner, repo, pr_number, metadata_json, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            (owner, repo, pr_number, json.dumps(metadata), time.time()),
        )
        self._conn.commit()

    def get_embedding(self, owner: str, repo: str, pr_number: int) -> list[float] | None:
        """Get cached embedding vector, or None if missing."""
        row = self._conn.execute(
            "SELECT embedding_json FROM pr_cache WHERE owner=? AND repo=? AND pr_number=?",
            (owner, repo, pr_number),
        ).fetchone()

        if row is None or row["embedding_json"] is None:
            return None

        return json.loads(row["embedding_json"])

    def put_embedding(self, owner: str, repo: str, pr_number: int, embedding: list[float]) -> None:
        """Store embedding vector for a cached PR."""
        self._conn.execute(
            "UPDATE pr_cache SET embedding_json=? WHERE owner=? AND repo=? AND pr_number=?",
            (json.dumps(embedding), owner, repo, pr_number),
        )
        self._conn.commit()

    def get_all_prs(self, owner: str, repo: str) -> list[dict]:
        """Get all non-stale cached PRs for a repo."""
        cutoff = time.time() - self.ttl_seconds
        rows = self._conn.execute(
            "SELECT metadata_json FROM pr_cache WHERE owner=? AND repo=? AND fetched_at>?",
            (owner, repo, cutoff),
        ).fetchall()
        return [json.loads(row["metadata_json"]) for row in rows]

    def clear_stale(self) -> int:
        """Remove stale entries. Returns count of deleted rows."""
        cutoff = time.time() - self.ttl_seconds
        cursor = self._conn.execute(
            "DELETE FROM pr_cache WHERE fetched_at<?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
