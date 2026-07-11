"""Tests for the tool registry — decorator, get, list, reset."""

import pytest

from juiceplug.tools import (
    _reset_registry,
    get_tool,
    list_tools,
    register_tool,
    _TOOL_REGISTRY,
)


# ------------------------------------------------------------------
# Fixtures — isolate each test from the global registry
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the tool registry around each test."""
    snapshot = dict(_TOOL_REGISTRY)
    yield
    _TOOL_REGISTRY.clear()
    _TOOL_REGISTRY.update(snapshot)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------


class TestRegisterTool:
    def test_register_and_retrieve(self) -> None:
        @register_tool("my_tool")
        def my_tool(query: str) -> str:
            return f"echo: {query}"

        assert get_tool("my_tool") is my_tool

    def test_duplicate_raises(self) -> None:
        @register_tool("dup_tool")
        def _first(q: str) -> str:
            return q

        with pytest.raises(ValueError, match="already registered"):

            @register_tool("dup_tool")
            def _second(q: str) -> str:
                return q

    def test_list_tools_includes_registered(self) -> None:
        @register_tool("alpha_tool")
        def _alpha(q: str) -> str:
            return q

        assert "alpha_tool" in list_tools()

    def test_list_tools_sorted(self) -> None:
        @register_tool("zzz_tool")
        def _z(q: str) -> str:
            return q

        @register_tool("aaa_tool")
        def _a(q: str) -> str:
            return q

        tools = list_tools()
        assert tools == sorted(tools)

    def test_get_missing_tool_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            get_tool("nonexistent_tool")


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestResetRegistry:
    def test_reset_clears(self) -> None:
        @register_tool("temp_tool")
        def _t(q: str) -> str:
            return q

        _reset_registry()
        assert "temp_tool" not in _TOOL_REGISTRY


# ------------------------------------------------------------------
# Built-in web_search is registered
# ------------------------------------------------------------------


class TestBuiltinTools:
    def test_web_search_registered(self) -> None:
        # web_search should have been registered when juiceplug.tools was
        # first imported — check the snapshot (the fixture restores it).
        assert "web_search" in list_tools()
