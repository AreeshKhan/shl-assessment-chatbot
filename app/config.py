"""
============================================================
config.py — Application Configuration (Updated for Groq)
============================================================

PURPOSE:
    Centralize ALL configuration in one place.

CHANGE LOG:
    - Switched from Gemini to Groq for LLM (free tier works!)
    - Using sentence-transformers for embeddings (fully local, no API needed)
    - Kept Gemini as optional fallback
============================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # ---- API Keys ----
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # ---- LLM Configuration (Groq) ----
    # Using Groq's Llama 3.3 70B — fast, free, excellent quality
    # WHY llama-3.3-70b-versatile:
    #   - 70B parameters = very capable for reasoning
    #   - "versatile" variant = good at following instructions
    #   - Groq's LPU makes it extremely fast (~300 tokens/sec)
    #   - Free tier: 30 RPM, 6000 tokens/min
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # ---- Embedding Configuration ----
    # Using sentence-transformers locally — no API key needed!
    # "all-MiniLM-L6-v2" is a small but effective model:
    #   - 384-dimensional embeddings
    #   - Only ~80MB download
    #   - Runs on CPU in milliseconds
    #   - Good quality for semantic search
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # ---- Retrieval Configuration ----
    RETRIEVAL_TOP_K: int = 20
    
    # ---- Conversation Limits ----
    MAX_TURNS: int = 8
    MAX_RECOMMENDATIONS: int = 10
    
    # ---- Catalog Configuration ----
    CATALOG_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "shl_product_catalog.json"
    )
    CATALOG_URL: str = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
    
    def validate(self) -> None:
        """Check that all required settings are present."""
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set! "
                "Get a free key at: https://console.groq.com/keys"
            )


settings = Settings()
