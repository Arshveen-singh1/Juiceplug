"""Tests for EmbeddingAdapterRouter using a mocked embedding model."""

import pytest
from unittest.mock import MagicMock, patch

from juiceplug.embedding_router import EmbeddingAdapterRouter

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_sentence_transformers():
    """Mock the sentence-transformers module to return fake embeddings and similarities."""
    with patch("juiceplug.embedding_router.SentenceTransformer") as mock_st, \
         patch("juiceplug.embedding_router.cos_sim") as mock_cos_sim:
        
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        # We don't actually need to return real tensors, just unique strings or ints
        # since we're mocking cos_sim as well.
        def mock_encode(text, convert_to_tensor=False):
            return f"emb_{text}"
            
        mock_model.encode.side_effect = mock_encode
        
        # We need to simulate cosine similarity. We'll use a simple dictionary
        # mapping pairs of strings to scores.
        def mock_cos_sim_fn(query_emb, route_emb):
            # query_emb is like "emb_query_string"
            # route_emb is like "emb_route_description"
            
            # Extract the original text from the dummy embedding string
            query = query_emb.replace("emb_", "")
            route = route_emb.replace("emb_", "")
            
            # Simple heuristic for testing: if query word is in route, high score
            score = 0.1
            if "code" in query and "code" in route:
                score = 0.9
            elif "legal" in query and "contract" in route:
                score = 0.85
            elif "reason" in query and "think step by step" in route:
                score = 0.95
                
            mock_tensor = MagicMock()
            mock_tensor.item.return_value = score
            return mock_tensor
            
        mock_cos_sim.side_effect = mock_cos_sim_fn
        
        # We have to mock the module imports directly inside EmbeddingAdapterRouter
        import sys
        sys.modules["sentence_transformers"] = MagicMock(SentenceTransformer=mock_st)
        sys.modules["sentence_transformers.util"] = MagicMock(cos_sim=mock_cos_sim)
        
        yield mock_st, mock_cos_sim
        
        # Cleanup
        del sys.modules["sentence_transformers"]
        del sys.modules["sentence_transformers.util"]

@pytest.fixture
def router(mock_sentence_transformers):
    r = EmbeddingAdapterRouter(default_adapter="fallback-adapter", threshold=0.3)
    r.add_route("adapter-code", ["debug python code"])
    r.add_route("adapter-legal", ["review contract"])
    r.add_route("adapter-general", ["think step by step"])
    return r

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestEmbeddingAdapterRouter:
    def test_routing_code(self, router):
        assert router.route("help me write some code") == "adapter-code"

    def test_routing_legal(self, router):
        assert router.route("is this legal?") == "adapter-legal"
        
    def test_routing_general(self, router):
        assert router.route("what is the reason for this?") == "adapter-general"
        
    def test_routing_fallback_below_threshold(self, router):
        # "weather" won't match any heuristic -> score 0.1 -> below 0.3 threshold
        assert router.route("weather today") == "fallback-adapter"

    def test_no_routes(self, mock_sentence_transformers):
        r = EmbeddingAdapterRouter(default_adapter="fallback")
        assert r.route("anything") == "fallback"
        
    def test_remove_route(self, router):
        assert router.route("help me write some code") == "adapter-code"
        router.remove_route("adapter-code")
        # Now it should fall back to the default because score will be 0.1
        assert router.route("help me write some code") == "fallback-adapter"

    def test_import_error_without_sentence_transformers(self):
        # Don't use the mock fixture here
        r = EmbeddingAdapterRouter()
        r.add_route("a", ["b"])
        # We need to simulate the module being missing, but since we didn't patch it,
        # if it's not installed, it will raise ImportError natively.
        # However, it might be installed in this environment. Let's just mock the 
        # import failure to be safe.
        with patch("builtins.__import__", side_effect=ImportError("mocked failure")):
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                r._lazy_load_model()
