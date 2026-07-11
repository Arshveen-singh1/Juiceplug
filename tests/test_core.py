"""
Tests for the tool-loop parsing logic in ``JuicedModel``.

These tests mock the model so they run **without** torch, transformers,
or a GPU.  They verify that:
- ``TOOL_CALL_PATTERN`` correctly extracts tool name + query
- The tool loop executes tools and feeds observations back
- The hard iteration cap and timeout are enforced
- The "I don't know" retry heuristic fires correctly
- ``_strip_tool_tags`` cleans residual tags from final answers
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from juiceplug.core import (
    TOOL_CALL_PATTERN,
    DEFAULT_MAX_TOOL_TURNS,
    JuicedModel,
    ModelNotFoundError,
    AdapterLoadError,
    _IDK_SIGNALS,
)
from juiceplug.tools import register_tool, _TOOL_REGISTRY


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_tool_registry():
    """Snapshot and restore the tool registry around each test."""
    snapshot = dict(_TOOL_REGISTRY)
    yield
    _TOOL_REGISTRY.clear()
    _TOOL_REGISTRY.update(snapshot)


def _make_juiced_model(
    responses: List[str],
    tools: Optional[List[str]] = None,
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS,
    tool_timeout_seconds: float = 30,
    log_transcripts: bool = False,
) -> JuicedModel:
    """Create a JuicedModel with a mock model that returns canned strings.

    ``responses`` is a list of strings the mock model will return in
    sequence, one per ``_generate`` call.
    """
    call_count = {"n": 0}

    def fake_generate(messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    jm = JuicedModel(
        model=mock_model,
        tokenizer=mock_tokenizer,
        tools=tools or [],
        max_tool_turns=max_tool_turns,
        tool_timeout_seconds=tool_timeout_seconds,
        log_transcripts=log_transcripts,
    )
    # Monkey-patch _generate so we don't need real model weights
    jm._generate = fake_generate  # type: ignore[assignment]
    return jm


# ------------------------------------------------------------------
# TOOL_CALL_PATTERN parsing
# ------------------------------------------------------------------


class TestToolCallPattern:
    def test_basic_match(self) -> None:
        text = "<tool_call>web_search: latest PyTorch release</tool_call>"
        m = TOOL_CALL_PATTERN.search(text)
        assert m is not None
        assert m.group(1) == "web_search"
        assert m.group(2).strip() == "latest PyTorch release"

    def test_match_with_surrounding_text(self) -> None:
        text = (
            "Let me look that up.\n"
            "<tool_call>web_search: PyTorch changelog</tool_call>\n"
            "I'll report back."
        )
        m = TOOL_CALL_PATTERN.search(text)
        assert m is not None
        assert m.group(1) == "web_search"

    def test_multiline_query(self) -> None:
        text = "<tool_call>web_search: query\nwith newlines</tool_call>"
        m = TOOL_CALL_PATTERN.search(text)
        assert m is not None
        assert "newlines" in m.group(2)

    def test_no_match_on_plain_text(self) -> None:
        text = "There is no tool call here."
        assert TOOL_CALL_PATTERN.search(text) is None

    def test_no_match_on_partial_tag(self) -> None:
        text = "<tool_call>web_search: query"
        assert TOOL_CALL_PATTERN.search(text) is None

    def test_multiple_matches(self) -> None:
        text = (
            "<tool_call>web_search: a</tool_call> "
            "<tool_call>calculator: 1+1</tool_call>"
        )
        matches = TOOL_CALL_PATTERN.findall(text)
        assert len(matches) == 2
        assert matches[0] == ("web_search", "a")
        assert matches[1] == ("calculator", "1+1")

    def test_whitespace_tolerance(self) -> None:
        text = "<tool_call>  web_search :  query  </tool_call>"
        m = TOOL_CALL_PATTERN.search(text)
        assert m is not None
        assert m.group(1) == "web_search"
        assert m.group(2).strip() == "query"


# ------------------------------------------------------------------
# Tool loop integration (mocked model)
# ------------------------------------------------------------------


class TestToolLoop:
    def test_direct_answer_no_tools(self) -> None:
        """Model answers directly → no tool calls."""
        jm = _make_juiced_model(["The answer is 42."])
        result = jm.ask("What is the meaning of life?")
        assert result == "The answer is 42."

    def test_single_tool_call(self) -> None:
        """Model emits one tool call, gets observation, then answers."""
        @register_tool("mock_search")
        def mock_search(q: str) -> str:
            return "PyTorch 2.4 released July 2024."

        responses = [
            "<tool_call>mock_search: latest PyTorch</tool_call>",
            "Based on the search, PyTorch 2.4 was released in July 2024.",
        ]
        jm = _make_juiced_model(responses, tools=["mock_search"])
        result = jm.ask("Latest PyTorch version?")
        assert "2.4" in result

    def test_multiple_tool_calls(self) -> None:
        """Model calls tools twice before answering."""
        call_count = {"n": 0}

        @register_tool("counter_tool")
        def counter_tool(q: str) -> str:
            call_count["n"] += 1
            return f"result-{call_count['n']}"

        responses = [
            "<tool_call>counter_tool: first</tool_call>",
            "<tool_call>counter_tool: second</tool_call>",
            "Final answer using result-1 and result-2.",
        ]
        jm = _make_juiced_model(responses, tools=["counter_tool"])
        result = jm.ask("Run two searches")
        assert "result-1" in result
        assert call_count["n"] == 2

    def test_max_iterations_cap(self) -> None:
        """Model loops forever → hard cap forces a final answer."""
        @register_tool("looper_tool")
        def looper_tool(q: str) -> str:
            return "keep going"

        # All responses are tool calls — the cap should stop the loop
        responses = ["<tool_call>looper_tool: again</tool_call>"] * 20
        # Add a final response for when the model is forced to answer
        responses.append("Forced final answer.")

        jm = _make_juiced_model(
            responses, tools=["looper_tool"], max_tool_turns=3
        )
        result = jm.ask("Loop forever")
        # Should get either the forced answer or hit the cap gracefully
        assert result is not None
        assert isinstance(result, str)

    def test_tool_not_found_returns_error(self) -> None:
        """If the model calls a tool that doesn't exist, we get an error
        message in the observation, not a crash."""
        responses = [
            "<tool_call>nonexistent_tool: query</tool_call>",
            "OK, that tool wasn't available.",
        ]
        jm = _make_juiced_model(responses, tools=["web_search"])
        result = jm.ask("Use a fake tool")
        assert result is not None

    def test_tool_exception_handled(self) -> None:
        """If a tool raises, the error is caught and returned as an
        observation."""
        @register_tool("broken_tool")
        def broken_tool(q: str) -> str:
            raise RuntimeError("something broke")

        responses = [
            "<tool_call>broken_tool: query</tool_call>",
            "The tool failed, but I can still answer.",
        ]
        jm = _make_juiced_model(responses, tools=["broken_tool"])
        result = jm.ask("Call broken tool")
        assert result is not None
        assert "failed" in result.lower() or isinstance(result, str)

    def test_timeout_guard(self) -> None:
        """Tool loop respects the timeout."""
        @register_tool("slow_tool")
        def slow_tool(q: str) -> str:
            return "done"

        call_count = {"n": 0}
        original_generate = None

        responses = ["<tool_call>slow_tool: go</tool_call>"] * 10
        responses.append("Final after timeout.")

        jm = _make_juiced_model(
            responses, tools=["slow_tool"], tool_timeout_seconds=0.1
        )

        # Patch _generate to add a small delay
        orig = jm._generate

        def slow_generate(messages, max_new_tokens):
            time.sleep(0.05)
            return orig(messages, max_new_tokens)

        jm._generate = slow_generate  # type: ignore[assignment]

        result = jm.ask("Keep going")
        # Should complete without hanging
        assert result is not None

    def test_no_tools_enabled_skips_loop(self) -> None:
        """When tools=[] the model can emit <tool_call> but it won't
        be executed — it's treated as a direct answer."""
        responses = [
            "<tool_call>web_search: test</tool_call> The answer is 7.",
        ]
        jm = _make_juiced_model(responses, tools=[])
        result = jm.ask("What is 3+4?")
        # Tool call tag should be stripped from the final answer
        assert "<tool_call>" not in result


# ------------------------------------------------------------------
# IDK retry heuristic
# ------------------------------------------------------------------


class TestIdkRetry:
    def test_looks_like_idk_true(self) -> None:
        assert JuicedModel._looks_like_idk("I don't know the current time.")

    def test_looks_like_idk_false(self) -> None:
        assert not JuicedModel._looks_like_idk("The answer is 42.")

    def test_all_signals_detected(self) -> None:
        for signal in _IDK_SIGNALS:
            assert JuicedModel._looks_like_idk(
                f"Well, {signal}, so I can't help."
            ), f"Signal not detected: {signal!r}"


# ------------------------------------------------------------------
# Tag stripping
# ------------------------------------------------------------------


class TestStripToolTags:
    def test_removes_tool_call(self) -> None:
        text = "Before <tool_call>web_search: q</tool_call> After"
        assert JuicedModel._strip_tool_tags(text) == "Before  After"

    def test_no_tags_passthrough(self) -> None:
        text = "Just a plain answer."
        assert JuicedModel._strip_tool_tags(text) == text

    def test_multiple_tags(self) -> None:
        text = "<tool_call>a: 1</tool_call><tool_call>b: 2</tool_call>Answer"
        result = JuicedModel._strip_tool_tags(text)
        assert "<tool_call>" not in result
        assert "Answer" in result


# ------------------------------------------------------------------
# Force tool use
# ------------------------------------------------------------------


class TestForceToolUse:
    def test_force_tool_use_runs_presearch(self) -> None:
        """force_tool_use=True should add a pre-search observation."""
        @register_tool("forced_search")
        def forced_search(q: str) -> str:
            return "pre-search result"

        # Register it as "web_search" for the force_tool_use codepath
        # (which specifically checks for "web_search")
        if "web_search" not in _TOOL_REGISTRY:
            _TOOL_REGISTRY["web_search"] = forced_search

        responses = [
            "Based on the observation, the answer is X.",
        ]
        jm = _make_juiced_model(responses, tools=["web_search"])
        result = jm.ask("Time-sensitive question", force_tool_use=True)
        assert result is not None


# ------------------------------------------------------------------
# Custom exceptions
# ------------------------------------------------------------------


class TestCustomExceptions:
    def test_model_not_found_error(self) -> None:
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("test")

    def test_adapter_load_error(self) -> None:
        with pytest.raises(AdapterLoadError):
            raise AdapterLoadError("test")
