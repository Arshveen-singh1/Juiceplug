"""
Tool registry for JuicePlug.

Tools are ``(query: str) -> str`` callables registered via the
``@register_tool("name")`` decorator.  The registry is a plain dict; tools
registered before ``JuicedModel.from_pretrained(…, tools=[…])`` is called
will be available in the tool loop.

Built-in tools
--------------
- ``web_search`` — DuckDuckGo search (no API key required).

Example — adding a custom tool
-------------------------------
>>> from juiceplug.tools import register_tool
>>> @register_tool("calculator")
... def calculator(query: str) -> str:
...     return str(eval(query, {"__builtins__": {}}))
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global tool registry
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: Dict[str, Callable[[str], str]] = {}


def register_tool(name: str) -> Callable:
    """Decorator that registers a tool function under *name*.

    Parameters
    ----------
    name : str
        The name the model should use in ``<tool_call>name: query</tool_call>``.

    Returns
    -------
    Callable
        The original function, unmodified.

    Raises
    ------
    ValueError
        If *name* is already registered.
    """

    def _decorator(fn: Callable[[str], str]) -> Callable[[str], str]:
        if name in _TOOL_REGISTRY:
            raise ValueError(
                f"Tool {name!r} is already registered — "
                f"pointing at {_TOOL_REGISTRY[name]!r}"
            )
        _TOOL_REGISTRY[name] = fn
        logger.info("Registered tool %r -> %s", name, fn.__qualname__)
        return fn

    return _decorator


def get_tool(name: str) -> Callable[[str], str]:
    """Return the tool registered under *name*, or raise ``KeyError``."""
    if name not in _TOOL_REGISTRY:
        raise KeyError(
            f"Tool {name!r} is not registered. "
            f"Available tools: {list(_TOOL_REGISTRY)}"
        )
    return _TOOL_REGISTRY[name]


def list_tools() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(_TOOL_REGISTRY)


def _reset_registry() -> None:
    """Clear all registered tools.  **For testing only.**"""
    _TOOL_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Auto-register built-in tools on import
# ---------------------------------------------------------------------------
from juiceplug.tools import web_search as _web_search_module  # noqa: F401, E402
