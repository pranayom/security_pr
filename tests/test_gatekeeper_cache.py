"""Tests for the Gatekeeper SQLite cache."""

import time

import pytest

from mcp_ai_auditor.gatekeeper.cache import PRCache


class TestPRCache:
    def setup_method(self):
        self.cache = PRCache(db_path=":memory:", ttl_hours=1)

    def teardown_method(self):
        self.cache.close()

    def test_put_and_get_pr(self):
        metadata = {"number": 42, "title": "Test PR"}
        self.cache.put_pr("owner", "repo", 42, metadata)
        result = self.cache.get_pr("owner", "repo", 42)
        assert result == metadata

    def test_get_pr_missing(self):
        result = self.cache.get_pr("owner", "repo", 999)
        assert result is None

    def test_get_pr_stale(self):
        """TTL-expired entries return None."""
        cache = PRCache(db_path=":memory:", ttl_hours=0)  # 0 hours = immediately stale
        metadata = {"number": 42, "title": "Test PR"}
        cache.put_pr("owner", "repo", 42, metadata)
        time.sleep(0.01)  # Ensure time passes
        result = cache.get_pr("owner", "repo", 42)
        assert result is None
        cache.close()

    def test_put_pr_upsert(self):
        """Putting same key twice overwrites."""
        self.cache.put_pr("owner", "repo", 42, {"title": "v1"})
        self.cache.put_pr("owner", "repo", 42, {"title": "v2"})
        result = self.cache.get_pr("owner", "repo", 42)
        assert result["title"] == "v2"

    def test_put_and_get_embedding(self):
        self.cache.put_pr("owner", "repo", 42, {"number": 42})
        embedding = [0.1, 0.2, 0.3]
        self.cache.put_embedding("owner", "repo", 42, embedding)
        result = self.cache.get_embedding("owner", "repo", 42)
        assert result == embedding

    def test_get_embedding_missing_pr(self):
        result = self.cache.get_embedding("owner", "repo", 999)
        assert result is None

    def test_get_embedding_no_embedding(self):
        self.cache.put_pr("owner", "repo", 42, {"number": 42})
        result = self.cache.get_embedding("owner", "repo", 42)
        assert result is None

    def test_get_all_prs(self):
        self.cache.put_pr("owner", "repo", 1, {"number": 1})
        self.cache.put_pr("owner", "repo", 2, {"number": 2})
        self.cache.put_pr("owner", "other", 3, {"number": 3})

        results = self.cache.get_all_prs("owner", "repo")
        assert len(results) == 2
        numbers = {r["number"] for r in results}
        assert numbers == {1, 2}

    def test_clear_stale(self):
        cache = PRCache(db_path=":memory:", ttl_hours=0)
        cache.put_pr("owner", "repo", 1, {"number": 1})
        time.sleep(0.01)
        deleted = cache.clear_stale()
        assert deleted == 1
        cache.close()

    def test_different_repos_isolated(self):
        self.cache.put_pr("owner1", "repo1", 42, {"title": "PR in repo1"})
        self.cache.put_pr("owner2", "repo2", 42, {"title": "PR in repo2"})

        r1 = self.cache.get_pr("owner1", "repo1", 42)
        r2 = self.cache.get_pr("owner2", "repo2", 42)

        assert r1["title"] == "PR in repo1"
        assert r2["title"] == "PR in repo2"
