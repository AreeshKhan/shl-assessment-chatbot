"""
============================================================
embeddings.py — Embedding Service (Memory Optimized)
============================================================

PURPOSE:
    Manage the FAISS index for similarity search.
    Since Render free tier has only 512MB RAM, loading
    PyTorch and sentence-transformers crashes it.
    
    Instead, we pre-computed the embeddings into a tiny
    580KB numpy array (embeddings.npy) and load that!

INTERVIEW QUESTION:
    Q: "Why did your initial Render deploy fail with OOM?"
    A: "The sentence-transformers library loads PyTorch, which 
       requires ~400MB RAM just to initialize. Render's free 
       tier is 512MB. I optimized it by pre-computing the 
       embeddings into a numpy array, reducing memory usage
       from ~500MB to ~50MB."
============================================================
"""

import os
import logging
import numpy as np
from typing import List

from app.config import settings
from app.services.catalog import Assessment

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages the FAISS index for similarity search.
    Loads pre-computed embeddings to save memory.
    """
    
    def __init__(self):
        self._index = None
        self._assessments = []
        self._dimension = 384
        self._model = None
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            except ImportError:
                raise ImportError("sentence-transformers not installed (Render memory optimization)")
            
        embedding = self._model.encode([query], normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)
    
    def build_index(self, assessments: List[Assessment]) -> None:
        """
        Build the FAISS index from PRE-COMPUTED embeddings.
        """
        import faiss
        
        self._assessments = assessments
        
        npy_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "embeddings.npy"
        )
        
        if os.path.exists(npy_path):
            logger.info(f"Loading PRE-COMPUTED embeddings from {npy_path}")
            embeddings = np.load(npy_path)
            self._dimension = embeddings.shape[1]
        else:
            logger.warning("Pre-computed embeddings NOT FOUND. Building in memory...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(settings.EMBEDDING_MODEL)
            texts = [a.embedding_text for a in assessments]
            embeddings = model.encode(texts, normalize_embeddings=True)
            
        self._index = faiss.IndexFlatL2(self._dimension)
        self._index.add(embeddings)
        logger.info(f"FAISS index built with {self._index.ntotal} assessments.")

    def search(self, query: str, top_k: int = 5) -> List[Assessment]:
        """
        Search the FAISS index for the most similar assessments.
        Falls back to fast keyword search if embedding model is not available.
        """
        if not self._index or not self._assessments:
            logger.warning("Search called before index was built!")
            return []
            
        try:
            query_embedding = self._get_query_embedding(query)
            
            # Search FAISS
            distances, indices = self._index.search(query_embedding, top_k)
            
            # Map indices back to Assessment objects
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._assessments):
                    results.append(self._assessments[idx])
            
            return results
            
        except (MemoryError, ImportError) as e:
            logger.warning(f"Embedding query failed: {e}. Falling back to keyword search.")
            # Ultra-fast keyword fallback (0 memory)
            query_terms = set(query.lower().split())
            scored = []
            for a in self._assessments:
                text = (a.name + " " + a.description + " " + " ".join(a.skills)).lower()
                score = sum(1 for term in query_terms if term in text)
                scored.append((score, a))
            
            # Sort by keyword matches
            scored.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in scored[:top_k]]
