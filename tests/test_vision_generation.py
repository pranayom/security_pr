"""Tests for vision document generation."""

import httpx
import pytest
import respx

from oss_maintainer_toolkit.gatekeeper.github_client import GitHubClient
from oss_maintainer_toolkit.gatekeeper.models import VisionDocument
from oss_maintainer_toolkit.gatekeeper.vision_generation import (
    SYSTEM_PROMPT,
    VISION_DOC_SCHEMA,
    _dispatch_for_generation,
    _parse_vision_response,
    build_generation_prompt,
    fetch_repo_context,
    vision_document_to_yaml,
)

BASE_URL = "https://api.github.com"

# Sample LLM response matching the VISION_DOC_SCHEMA
SAMPLE_LLM_RESPONSE = {
    "project": "TestProject",
    "principles": [
        {"name": "Code Quality", "description": "All code must be tested"},
        {"name": "Security First", "description": "Security is paramount"},
    ],
    "anti_patterns": [
        "Submitting untested code",
        "Modifying CI without approval",
    ],
    "focus_areas": [
        "auth/",
        ".github/workflows",
        "config/",
    ],
    "label_taxonomy": [
        {"name": "bug", "description": "Bug reports", "keywords": ["bug", "error", "crash"]},
        {"name": "feature", "description": "New features", "keywords": ["feature", "add", "new"]},
    ],
}


class TestFetchRepoContext:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetches_readme_and_contributing(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/README.md").mock(
            return_value=httpx.Response(200, text="# Test Project\nA description.")
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/CONTRIBUTING.md").mock(
            return_value=httpx.Response(200, text="# Contributing\nPlease follow the rules.")
        )
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(200, json=[])
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            context = await fetch_repo_context("owner", "repo", client)

        assert "Test Project" in context["readme"]
        assert "Contributing" in context["contributing"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_falls_back_to_lowercase_readme(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/README.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/readme.md").mock(
            return_value=httpx.Response(200, text="# lowercase readme")
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/CONTRIBUTING.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/contributing.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(200, json=[])
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            context = await fetch_repo_context("owner", "repo", client)

        assert "lowercase readme" in context["readme"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_handles_missing_files_gracefully(self):
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/README.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/readme.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/CONTRIBUTING.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/contributing.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(200, json=[])
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            context = await fetch_repo_context("owner", "repo", client)

        assert context["readme"] == ""
        assert context["contributing"] == ""

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetches_merged_and_rejected_prs(self):
        merged_pr = {
            "number": 1,
            "title": "Good PR",
            "body": "Merged",
            "merged_at": "2026-02-15T00:00:00Z",
        }
        rejected_pr = {
            "number": 2,
            "title": "Bad PR",
            "body": "Rejected",
            "merged_at": None,
        }

        calls = []

        def pr_handler(request):
            calls.append(str(request.url))
            # Diff requests have diff Accept header
            if "diff" in request.headers.get("accept", ""):
                return httpx.Response(200, text="diff content here")
            # Closed PRs list
            return httpx.Response(200, json=[merged_pr, rejected_pr])

        respx.get(f"{BASE_URL}/repos/owner/repo/contents/README.md").mock(
            return_value=httpx.Response(200, text="# Readme")
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/CONTRIBUTING.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(f"{BASE_URL}/repos/owner/repo/contents/contributing.md").mock(
            return_value=httpx.Response(404)
        )
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            side_effect=pr_handler
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            context = await fetch_repo_context("owner", "repo", client, max_merged=5, max_rejected=5)

        assert len(context["merged_prs"]) == 1
        assert context["merged_prs"][0]["title"] == "Good PR"
        assert len(context["rejected_prs"]) == 1
        assert context["rejected_prs"][0]["title"] == "Bad PR"


class TestBuildGenerationPrompt:
    def test_includes_repo_info(self):
        context = {
            "readme": "# My Project",
            "contributing": "# Contributing guide",
            "merged_prs": [],
            "rejected_prs": [],
        }
        prompt = build_generation_prompt("owner", "repo", context)

        assert "owner/repo" in prompt
        assert "My Project" in prompt
        assert "Contributing guide" in prompt

    def test_includes_merged_prs(self):
        context = {
            "readme": "",
            "contributing": "",
            "merged_prs": [
                {"number": 1, "title": "Add tests", "body": "Tests added", "diff_summary": "+test code"},
            ],
            "rejected_prs": [],
        }
        prompt = build_generation_prompt("owner", "repo", context)

        assert "Merged PR #1: Add tests" in prompt
        assert "+test code" in prompt

    def test_includes_rejected_prs(self):
        context = {
            "readme": "",
            "contributing": "",
            "merged_prs": [],
            "rejected_prs": [
                {"number": 2, "title": "Bad change", "body": "Nope", "diff_summary": "-deleted"},
            ],
        }
        prompt = build_generation_prompt("owner", "repo", context)

        assert "Rejected PR #2: Bad change" in prompt
        assert "-deleted" in prompt

    def test_handles_empty_context(self):
        context = {
            "readme": "",
            "contributing": "",
            "merged_prs": [],
            "rejected_prs": [],
        }
        prompt = build_generation_prompt("owner", "repo", context)

        assert "no README found" in prompt
        assert "no CONTRIBUTING.md found" in prompt


class TestParseVisionResponse:
    def test_parses_complete_response(self):
        doc = _parse_vision_response(SAMPLE_LLM_RESPONSE)

        assert isinstance(doc, VisionDocument)
        assert doc.project == "TestProject"
        assert len(doc.principles) == 2
        assert doc.principles[0].name == "Code Quality"
        assert len(doc.anti_patterns) == 2
        assert len(doc.focus_areas) == 3
        assert len(doc.label_taxonomy) == 2
        assert doc.label_taxonomy[0].name == "bug"

    def test_handles_missing_fields(self):
        doc = _parse_vision_response({"project": "Minimal"})

        assert doc.project == "Minimal"
        assert doc.principles == []
        assert doc.anti_patterns == []
        assert doc.focus_areas == []
        assert doc.label_taxonomy == []

    def test_label_source_is_generated(self):
        doc = _parse_vision_response(SAMPLE_LLM_RESPONSE)
        for label in doc.label_taxonomy:
            assert label.source == "generated"


class TestVisionDocumentToYaml:
    def test_generates_valid_yaml(self):
        doc = _parse_vision_response(SAMPLE_LLM_RESPONSE)
        yaml_str = vision_document_to_yaml(doc, "owner", "repo")

        assert "project: TestProject" in yaml_str
        assert "Code Quality" in yaml_str
        assert "auth/" in yaml_str

    def test_includes_header(self):
        doc = _parse_vision_response(SAMPLE_LLM_RESPONSE)
        yaml_str = vision_document_to_yaml(doc, "owner", "repo")

        assert "# Vision Document: TestProject" in yaml_str
        assert "# Repo: owner/repo" in yaml_str
        assert "# Status: draft" in yaml_str

    def test_includes_label_taxonomy(self):
        doc = _parse_vision_response(SAMPLE_LLM_RESPONSE)
        yaml_str = vision_document_to_yaml(doc, "owner", "repo")

        assert "label_taxonomy" in yaml_str
        assert "bug" in yaml_str
        assert "feature" in yaml_str


class TestDispatchForGeneration:
    @pytest.mark.asyncio
    async def test_raises_on_claude_cli(self, monkeypatch):
        """claude_cli is not supported for generation."""
        from oss_maintainer_toolkit.gatekeeper import vision as vision_mod
        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        monkeypatch.setattr(
            vision_mod, "_resolve_effective_provider",
            lambda p="", a="", o="": ("claude_cli", ""),
        )

        with pytest.raises(ProviderError, match="claude_cli is not supported"):
            await _dispatch_for_generation("test prompt")

    @pytest.mark.asyncio
    async def test_raises_on_unknown_provider(self, monkeypatch):
        from oss_maintainer_toolkit.gatekeeper import vision_generation as vg_mod
        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        monkeypatch.setattr(
            vg_mod, "_resolve_effective_provider",
            lambda p="", a="": ("unknown_provider", "key123"),
        )

        with pytest.raises(ProviderError, match="Unknown LLM provider"):
            await _dispatch_for_generation("test prompt")

    @pytest.mark.asyncio
    async def test_dispatches_to_openrouter(self, monkeypatch):
        from oss_maintainer_toolkit.gatekeeper import vision_generation as vg_mod

        monkeypatch.setattr(
            vg_mod, "_resolve_effective_provider",
            lambda p="", a="": ("openrouter", "sk-or-test"),
        )

        captured = {}

        async def mock_openai_compat(**kwargs):
            captured.update(kwargs)
            return SAMPLE_LLM_RESPONSE

        monkeypatch.setattr(vg_mod, "call_openai_compatible", mock_openai_compat)

        result = await _dispatch_for_generation("test prompt")

        assert result == SAMPLE_LLM_RESPONSE
        assert captured["api_key"] == "sk-or-test"
        assert captured["system_prompt"] == SYSTEM_PROMPT
        assert captured["json_schema"] == VISION_DOC_SCHEMA

    @pytest.mark.asyncio
    async def test_dispatches_to_anthropic(self, monkeypatch):
        from oss_maintainer_toolkit.gatekeeper import vision_generation as vg_mod

        monkeypatch.setattr(
            vg_mod, "_resolve_effective_provider",
            lambda p="", a="": ("anthropic", "sk-ant-test"),
        )

        captured = {}

        async def mock_anthropic(**kwargs):
            captured.update(kwargs)
            return SAMPLE_LLM_RESPONSE

        monkeypatch.setattr(vg_mod, "call_anthropic", mock_anthropic)

        result = await _dispatch_for_generation("test prompt")

        assert result == SAMPLE_LLM_RESPONSE
        assert captured["api_key"] == "sk-ant-test"
        # Anthropic gets schema instruction appended to prompt
        assert "MUST respond with ONLY valid JSON" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_dispatches_to_gemini(self, monkeypatch):
        from oss_maintainer_toolkit.gatekeeper import vision_generation as vg_mod

        monkeypatch.setattr(
            vg_mod, "_resolve_effective_provider",
            lambda p="", a="": ("gemini", "AIza-test"),
        )

        captured = {}

        async def mock_gemini(**kwargs):
            captured.update(kwargs)
            return SAMPLE_LLM_RESPONSE

        monkeypatch.setattr(vg_mod, "call_gemini", mock_gemini)

        result = await _dispatch_for_generation("test prompt")

        assert result == SAMPLE_LLM_RESPONSE
        assert captured["api_key"] == "AIza-test"
        assert "MUST respond with ONLY valid JSON" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_raises_on_missing_openrouter_key(self, monkeypatch):
        from oss_maintainer_toolkit.gatekeeper import vision_generation as vg_mod
        from oss_maintainer_toolkit.gatekeeper.providers import ProviderError

        monkeypatch.setattr(
            vg_mod, "_resolve_effective_provider",
            lambda p="", a="": ("openrouter", ""),
        )

        with pytest.raises(ProviderError, match="No API key"):
            await _dispatch_for_generation("test prompt")


class TestGitHubClientClosedUnmerged:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_closed_unmerged_prs(self):
        items = [
            {"number": 1, "merged_at": "2026-01-01T00:00:00Z"},
            {"number": 2, "merged_at": None},
            {"number": 3, "merged_at": None},
            {"number": 4, "merged_at": "2026-01-02T00:00:00Z"},
            {"number": 5, "merged_at": None},
        ]
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(200, json=items)
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            rejected = await client.list_closed_unmerged_prs("owner", "repo", max_results=2)

        assert len(rejected) == 2
        assert rejected[0]["number"] == 2
        assert rejected[1]["number"] == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_returns_empty_when_all_merged(self):
        items = [
            {"number": 1, "merged_at": "2026-01-01T00:00:00Z"},
            {"number": 2, "merged_at": "2026-01-02T00:00:00Z"},
        ]
        respx.get(url__startswith=f"{BASE_URL}/repos/owner/repo/pulls").mock(
            return_value=httpx.Response(200, json=items)
        )

        async with GitHubClient(api_url=BASE_URL) as client:
            rejected = await client.list_closed_unmerged_prs("owner", "repo")

        assert rejected == []


class TestSchemaStructure:
    def test_vision_doc_schema_has_required_fields(self):
        required = VISION_DOC_SCHEMA["schema"]["required"]
        assert "project" in required
        assert "principles" in required
        assert "anti_patterns" in required
        assert "focus_areas" in required
        assert "label_taxonomy" in required

    def test_vision_doc_schema_is_strict(self):
        assert VISION_DOC_SCHEMA["strict"] is True
        assert VISION_DOC_SCHEMA["schema"]["additionalProperties"] is False
