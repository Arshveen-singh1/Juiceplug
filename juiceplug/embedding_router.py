"""
EmbeddingAdapterRouter — routing via sentence embeddings.

Uses `sentence-transformers` to compute cosine similarity between the
user's query and the registered adapter keywords/descriptions. This is
more robust than exact keyword matching.

Requires the `router` extra: `pip install juiceplug[router]`
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from juiceplug.router import AdapterRouter

logger = logging.getLogger(__name__)


class EmbeddingAdapterRouter(AdapterRouter):
    """Embedding-based adapter router.

    Registers routes similarly to ``AdapterRouter``, but uses a
    sentence-transformer model to route queries based on semantic
    similarity to the registered keywords/descriptions.

    Parameters
    ----------
    default_adapter : str | None
        Adapter id to return when no route meets the similarity threshold.
    embedding_model : str
        HF model id for sentence-transformers (default: ``all-MiniLM-L6-v2``).
    threshold : float
        Minimum cosine similarity score (0.0 to 1.0) to consider a match.
        If the best match is below this, returns ``default_adapter``.
    """

    def __init__(
        self,
        default_adapter: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.3,
    ) -> None:
        super().__init__(default_adapter)
        self.embedding_model_id = embedding_model
        self.threshold = threshold
        self._model = None
        self._route_embeddings: Dict[str, Any] = {}

    def _lazy_load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingAdapterRouter. "
                    "Install it via: pip install juiceplug[router]"
                ) from exc
            
            logger.info("Loading embedding model: %s", self.embedding_model_id)
            self._model = SentenceTransformer(self.embedding_model_id)
        return self._model

    def add_route(self, adapter_id: str, keywords: List[str]) -> None:
        """Register an adapter and pre-compute embeddings for its keywords."""
        super().add_route(adapter_id, keywords)
        
        # We pre-compute embeddings for all keywords of this route.
        model = self._lazy_load_model()
        # Create a single descriptive string from keywords, or embed them individually.
        # Embedding as a single block often works well for short descriptions.
        description = " ".join(keywords)
        self._route_embeddings[adapter_id] = model.encode(description, convert_to_tensor=True)
        logger.info("Computed embedding for route %r", adapter_id)

    def remove_route(self, adapter_id: str) -> None:
        """Remove a route and its cached embedding."""
        super().remove_route(adapter_id)
        self._route_embeddings.pop(adapter_id, None)

    def route(self, query: str) -> Optional[str]:
        """Route the query using cosine similarity.

        Returns the adapter with the highest similarity score, provided
        it meets the configured ``threshold``. Otherwise returns
        ``default_adapter``.
        """
        if not self._route_embeddings:
            return self.default_adapter

        try:
            from sentence_transformers.util import cos_sim
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for EmbeddingAdapterRouter. "
                "Install it via: pip install juiceplug[router]"
            ) from exc

        model = self._lazy_load_model()
        query_emb = model.encode(query, convert_to_tensor=True)

        best_id: Optional[str] = None
        best_score: float = -1.0

        for adapter_id, emb in self._route_embeddings.items():
            # cos_sim returns a 2D tensor, we want the single float value
            score = cos_sim(query_emb, emb).item()
            if score > best_score:
                best_score = score
                best_id = adapter_id

        if best_score >= self.threshold:
            chosen = best_id
        else:
            chosen = self.default_adapter

        logger.info(
            "EmbeddingRouter: query=%r -> adapter=%r (score=%.3f, threshold=%.3f)",
            query[:60],
            chosen,
            best_score,
            self.threshold,
        )
        return chosen
