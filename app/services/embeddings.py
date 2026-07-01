"""
============================================================
embeddings.py — Embedding Service (Local, No API Key Needed)
============================================================

PURPOSE:
    Convert assessment text into vectors using a local model
    (sentence-transformers) and build a FAISS index.

WHY LOCAL EMBEDDINGS:
    Gemini's embedding API was blocked. sentence-transformers
    runs entirely on your CPU — no API key, no rate limits,
    no network calls. The model downloads once (~80MB) and
    runs locally forever.

MODEL: all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Very fast on CPU
    - Good quality for semantic search
    - Used by millions of developers
    - Perfect for our 377-assessment catalog

INTERVIEW QUESTION:
    Q: "Why local embeddings instead of an API?"
    A: "Local embeddings have zero latency (no network call),
       zero cost, and zero rate limits. For a small catalog
       of 377 items, the quality of all-MiniLM-L6-v2 is more
       than sufficient. In production with millions of items,
       I'd consider a larger model or an API."
============================================================
"""

import logging
import numpy as np
from typing import List

from app.config import settings
from app.services.catalog import Assessment

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Creates text embeddings using sentence-transformers (local)
    and manages the FAISS index for similarity search.
    """
    
    def __init__(self):
        """Initialize — model is loaded when build_index() is called."""
        self._model = None
        self._index = None
        self._assessments = []
        self._dimension = None
    
    def _load_model(self):
        """
        Load the sentence-transformer model.
        
        This downloads the model on first run (~80MB) and
        caches it locally. Subsequent runs use the cache.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text strings into embedding vectors.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            numpy array of shape (len(texts), 384)
        """
        self._load_model()
        
        # sentence-transformers handles batching internally
        embeddings = self._model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # Pre-normalize for cosine similarity
            batch_size=64,
        )
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Created {len(embeddings)} embeddings, dimension={embeddings.shape[1]}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for searching.
        
        Args:
            query: The user's search text
        
        Returns:
            numpy array of shape (1, 384)
        """
        self._load_model()
        
        embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        )
        
        return np.array(embedding, dtype=np.float32)
    
    def build_index(self, assessments: List[Assessment]) -> None:
        """
        Build the FAISS index from a list of assessments.
        Called ONCE at startup.
        
        Args:
            assessments: List of Assessment objects from the catalog
        """
        import faiss
        
        logger.info(f"Building FAISS index for {len(assessments)} assessments...")
        
        self._assessments = assessments
        
        # Get embedding texts
        texts = [a.embedding_text for a in assessments]
        
        # Create embeddings (local, no API call)
        embeddings = self.embed_texts(texts)
        
        # Build FAISS index (Inner Product for cosine similarity)
        self._dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)
        
        logger.info(
            f"FAISS index built: {self._index.ntotal} vectors, "
            f"dimension={self._dimension}"
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Assessment]:
        """
        Find the top-K most similar assessments to a query.
        
        Args:
            query_embedding: The query vector from embed_query()
            top_k: Number of results to return
        
        Returns:
            List of top-K most similar Assessment objects
        """
        if self._index is None:
            raise RuntimeError("FAISS index not built. Call build_index() first.")
        
        # Search the index
        distances, indices = self._index.search(query_embedding, top_k)
        
        # Convert indices to Assessment objects
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._assessments):
                results.append(self._assessments[idx])
        
        logger.debug(f"FAISS search returned {len(results)} results")
        return results
