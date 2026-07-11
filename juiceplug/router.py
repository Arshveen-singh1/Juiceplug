"""
AdapterRouter — picks a PEFT reasoning adapter per query.

v1 uses keyword-overlap scoring (zero external dependencies).
``EmbeddingAdapterRouter`` (Phase 4) will add cosine-similarity routing
via ``sentence-transformers`` as an opt-in extra.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AdapterRouter:
    """Keyword-overlap adapter router.

    Register ``{adapter_id: [keywords]}`` routes, then call
    ``router.route(query)`` — returns the adapter whose keywords overlap
    most with the tokenized query, or ``default_adapter`` if nothing matches.

    Parameters
    ----------
    default_adapter : str | None
        Adapter id to return when no route matches.  ``None`` means
        "use the base model with no adapter".
    """

    def __init__(self, default_adapter: Optional[str] = None) -> None:
        self.default_adapter: Optional[str] = default_adapter
        self._routes: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_route(self, adapter_id: str, keywords: List[str]) -> None:
        """Register an adapter with its associated *keywords*.

        Parameters
        ----------
        adapter_id : str
            Hugging Face hub id or local path of the LoRA adapter.
        keywords : list[str]
            Lowercased words or phrases.  The query is tokenized and
            scored by overlap with these.
        """
        self._routes[adapter_id] = [kw.lower() for kw in keywords]
        logger.info(
            "AdapterRouter: registered %r with %d keywords",
            adapter_id,
            len(keywords),
        )

    def remove_route(self, adapter_id: str) -> None:
        """Remove a previously registered route."""
        self._routes.pop(adapter_id, None)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, query: str) -> Optional[str]:
        """Return the best-matching adapter id for *query*.

        Scoring: for each registered adapter, count how many of its
        keywords appear (as substrings) in the lowercased query.  The
        adapter with the highest count wins.  Ties are broken by
        insertion order; if all counts are 0 the ``default_adapter`` is
        returned.

        Parameters
        ----------
        query : str
            The user's question.

        Returns
        -------
        str | None
            Adapter id, or ``default_adapter`` (which may be ``None``).
        """
        query_lower = query.lower()
        best_id: Optional[str] = None
        best_score: int = 0

        for adapter_id, keywords in self._routes.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_id = adapter_id

        chosen = best_id if best_score > 0 else self.default_adapter
        logger.info(
            "AdapterRouter: query=%r -> adapter=%r (score=%d)",
            query[:60],
            chosen,
            best_score,
        )
        return chosen

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def routes(self) -> Dict[str, List[str]]:
        """Return a copy of the current route table."""
        return dict(self._routes)

    def __repr__(self) -> str:  # pragma: no cover
        adapters = list(self._routes)
        return (
            f"AdapterRouter(default={self.default_adapter!r}, "
            f"routes={adapters!r})"
        )
