"""Tests for AdapterRouter — keyword-overlap routing logic."""

import pytest

from juiceplug.router import AdapterRouter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def router() -> AdapterRouter:
    """Pre-configured router with three domain routes."""
    r = AdapterRouter(default_adapter="fallback-adapter")
    r.add_route("adapter-code", ["code", "bug", "compile", "stack trace", "debug"])
    r.add_route("adapter-legal", ["contract", "clause", "liability", "statute"])
    r.add_route("adapter-general", ["reason", "think", "step by step", "explain"])
    return r


# ------------------------------------------------------------------
# Basic routing
# ------------------------------------------------------------------


class TestAdapterRouter:
    def test_exact_keyword_match(self, router: AdapterRouter) -> None:
        assert router.route("There's a bug in my code") == "adapter-code"

    def test_multiple_keyword_match(self, router: AdapterRouter) -> None:
        # "code" + "bug" → score 2 for adapter-code
        result = router.route("This code has a bug, help me debug it")
        assert result == "adapter-code"

    def test_legal_routing(self, router: AdapterRouter) -> None:
        result = router.route("Is this liability clause enforceable?")
        assert result == "adapter-legal"

    def test_general_reasoning(self, router: AdapterRouter) -> None:
        result = router.route("Explain step by step how LoRA works")
        assert result == "adapter-general"

    def test_no_match_returns_default(self, router: AdapterRouter) -> None:
        result = router.route("What is the weather today?")
        assert result == "fallback-adapter"

    def test_no_match_returns_none_when_no_default(self) -> None:
        r = AdapterRouter(default_adapter=None)
        r.add_route("adapter-code", ["code"])
        assert r.route("weather forecast") is None

    def test_case_insensitive(self, router: AdapterRouter) -> None:
        result = router.route("There's a BUG in my CODE")
        assert result == "adapter-code"

    def test_substring_matching(self, router: AdapterRouter) -> None:
        # "contract" should match even in "contractor"
        result = router.route("The contractor signed the contract")
        assert result == "adapter-legal"

    def test_empty_query(self, router: AdapterRouter) -> None:
        # No keywords match → default
        result = router.route("")
        assert result == "fallback-adapter"


# ------------------------------------------------------------------
# Route management
# ------------------------------------------------------------------


class TestRouteManagement:
    def test_add_route(self) -> None:
        r = AdapterRouter()
        r.add_route("a", ["x", "y"])
        assert "a" in r.routes

    def test_remove_route(self, router: AdapterRouter) -> None:
        router.remove_route("adapter-code")
        assert "adapter-code" not in router.routes
        # "code" query should now not match code adapter
        result = router.route("fix this code bug")
        assert result != "adapter-code"

    def test_remove_nonexistent_route(self, router: AdapterRouter) -> None:
        # Should not raise
        router.remove_route("nonexistent")

    def test_routes_property_is_copy(self, router: AdapterRouter) -> None:
        routes = router.routes
        routes["fake"] = ["x"]
        assert "fake" not in router.routes
