"""Tests for the Gatekeeper issue SQLite cache."""

import time

import pytest

from oss_maintainer_toolkit.gatekeeper.issue_cache import IssueCache


class TestIssueCache:
    def setup_method(self):
        self.cache = IssueCache(db_path=":memory:", ttl_hours=1)

    def teardown_method(self):
        self.cache.close()

    def test_put_and_get_issue(self):
        metadata = {"number": 101, "title": "Test Issue"}
        self.cache.put_issue("owner", "repo", 101, metadata)
        result = self.cache.get_issue("owner", "repo", 101)
        assert result == metadata

    def test_get_issue_missing(self):
        result = self.cache.get_issue("owner", "repo", 999)
        assert result is None

    def test_get_issue_stale(self):
        """TTL-expired entries return None."""
        cache = IssueCache(db_path=":memory:", ttl_hours=0)
        metadata = {"number": 101, "title": "Test Issue"}
        cache.put_issue("owner", "repo", 101, metadata)
        time.sleep(0.01)
        result = cache.get_issue("owner", "repo", 101)
        assert result is None
        cache.close()

    def test_put_issue_upsert(self):
        """Putting same key twice overwrites."""
        self.cache.put_issue("owner", "repo", 101, {"title": "v1"})
        self.cache.put_issue("owner", "repo", 101, {"title": "v2"})
        result = self.cache.get_issue("owner", "repo", 101)
        assert result["title"] == "v2"

    def test_put_and_get_embedding(self):
        self.cache.put_issue("owner", "repo", 101, {"number": 101})
        embedding = [0.1, 0.2, 0.3]
        self.cache.put_embedding("owner", "repo", 101, embedding)
        result = self.cache.get_embedding("owner", "repo", 101)
        assert result == embedding

    def test_get_embedding_missing_issue(self):
        result = self.cache.get_embedding("owner", "repo", 999)
        assert result is None

    def test_get_embedding_no_embedding(self):
        self.cache.put_issue("owner", "repo", 101, {"number": 101})
        result = self.cache.get_embedding("owner", "repo", 101)
        assert result is None

    def test_get_all_issues(self):
        self.cache.put_issue("owner", "repo", 1, {"number": 1})
        self.cache.put_issue("owner", "repo", 2, {"number": 2})
        self.cache.put_issue("owner", "other", 3, {"number": 3})

        results = self.cache.get_all_issues("owner", "repo")
        assert len(results) == 2
        numbers = {r["number"] for r in results}
        assert numbers == {1, 2}

    def test_clear_stale(self):
        cache = IssueCache(db_path=":memory:", ttl_hours=0)
        cache.put_issue("owner", "repo", 1, {"number": 1})
        time.sleep(0.01)
        deleted = cache.clear_stale()
        assert deleted == 1
        cache.close()

    def test_different_repos_isolated(self):
        self.cache.put_issue("owner1", "repo1", 101, {"title": "Issue in repo1"})
        self.cache.put_issue("owner2", "repo2", 101, {"title": "Issue in repo2"})

        r1 = self.cache.get_issue("owner1", "repo1", 101)
        r2 = self.cache.get_issue("owner2", "repo2", 101)

        assert r1["title"] == "Issue in repo1"
        assert r2["title"] == "Issue in repo2"
