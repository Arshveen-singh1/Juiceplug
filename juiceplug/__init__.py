"""
JuicePlug — plug-and-play reasoning adapters + live web tools for any LLM.

Public API
----------
JuicedModel     : Load a causal LM, optionally with a LoRA reasoning adapter
                  and pluggable tools.
AdapterRouter   : Keyword-overlap router that picks an adapter per query.

Heavy ML dependencies (torch, transformers, peft) are **lazy-imported** —
``import juiceplug`` works without them.  They are only required when you
call ``JuicedModel.from_pretrained(…)``.
"""

__version__ = "0.1.0"

from juiceplug.router import AdapterRouter

# Lazy import: JuicedModel pulls in torch/transformers/peft on first access.
# We re-export it here so ``from juiceplug import JuicedModel`` works, but
# the actual import of heavy deps happens inside core.py.


def __getattr__(name: str):
    if name == "JuicedModel":
        from juiceplug.core import JuicedModel

        return JuicedModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["JuicedModel", "AdapterRouter", "__version__"]
