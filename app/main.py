"""
============================================================
main.py — FastAPI Application Entry Point
============================================================

PURPOSE:
    This is where everything comes together. This file:
    1. Creates the FastAPI application
    2. Registers all route handlers
    3. Initializes all services at startup
    4. Configures CORS (Cross-Origin Resource Sharing)

THEORY — What is FastAPI?
    FastAPI is a modern Python web framework for building APIs.
    It combines:
    - Speed: One of the fastest Python frameworks (async)
    - Validation: Automatic request validation via Pydantic
    - Documentation: Auto-generated Swagger UI at /docs
    - Type safety: Full Python type hint support
    
    REAL-WORLD ANALOGY:
    FastAPI is like a restaurant's front-of-house system:
    - It takes customer orders (HTTP requests)
    - Validates them (Pydantic schemas)
    - Routes them to the kitchen (our services)
    - Serves the response back (HTTP responses)
    - And keeps a menu posted at the door (Swagger docs at /docs)

WHAT HAPPENS WHEN THE SERVER STARTS:
    1. Python runs this file
    2. FastAPI app is created
    3. Routes are registered (/health, /chat)
    4. The startup event fires:
       a. Settings are validated (API key present?)
       b. Catalog is loaded (377 assessments)
       c. FAISS index is built (embeddings created)
       d. Agent is assembled (all services wired together)
    5. Server starts listening for requests

INTERVIEW QUESTION:
    Q: "What's the startup sequence? What if one step fails?"
    A: "Catalog → Embeddings → FAISS Index → Agent. If any step
       fails, the server logs the error and exits. This is 
       'fail fast' — better to crash at startup than serve
       broken responses that fail the evaluator."
============================================================
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import health, chat
from app.services.catalog import CatalogService
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.services.agent import Agent

# ============================================================
# Configure Logging
# ============================================================
# WHY: Without logging, debugging production issues is impossible.
# FORMAT: timestamp — module — level — message
# LEVEL: INFO shows important events without excessive noise
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Application Lifespan (Startup & Shutdown)
# ============================================================
# This runs ONCE when the server starts and ONCE when it stops.
# We use it to initialize all our services.
#
# WHY asynccontextmanager:
#   FastAPI's modern way to handle startup/shutdown events.
#   The code before 'yield' runs at startup.
#   The code after 'yield' runs at shutdown.
#   This replaces the older @app.on_event("startup") pattern.
#
# WHY INITIALIZE HERE (not in each service):
#   1. Control the initialization ORDER (catalog before embeddings)
#   2. Fail fast if any service can't initialize
#   3. Log the full startup sequence
#   4. Services are created ONCE, shared across all requests
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    STARTUP SEQUENCE:
        1. Validate settings (API key present?)
        2. Load catalog (377 assessments from JSON)
        3. Build FAISS index (embed all assessments)
        4. Create Agent (wire everything together)
    """
    logger.info("=" * 60)
    logger.info("SHL Assessment Recommender — Starting up...")
    logger.info("=" * 60)
    
    try:
        # ---- Step 1: Validate settings ----
        logger.info("Step 1/4: Validating settings...")
        settings.validate()
        logger.info(f"  Model: {settings.LLM_MODEL}")
        logger.info(f"  Embedding: {settings.EMBEDDING_MODEL}")
        logger.info(f"  Top-K retrieval: {settings.RETRIEVAL_TOP_K}")
        
        # ---- Step 2: Load catalog ----
        logger.info("Step 2/4: Loading SHL product catalog...")
        catalog_service = CatalogService()
        catalog_service.load(settings.CATALOG_PATH)
        logger.info(f"  Loaded {len(catalog_service.assessments)} assessments")
        
        # ---- Step 3: Build FAISS index ----
        logger.info("Step 3/4: Building FAISS index (this may take 30-60 seconds)...")
        embedding_service = EmbeddingService()
        embedding_service.build_index(catalog_service.assessments)
        logger.info("  FAISS index built successfully")
        
        # ---- Step 4: Create Agent ----
        logger.info("Step 4/4: Creating agent...")
        llm_service = LLMService()
        agent_instance = Agent(
            catalog_service=catalog_service,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )
        
        # Wire the agent into the chat route
        # This is the connection between the HTTP layer and the brain
        chat.agent = agent_instance
        
        logger.info("=" * 60)
        logger.info("SHL Assessment Recommender — Ready!")
        logger.info("  Swagger docs: http://localhost:8000/docs")
        logger.info("  Health check: http://localhost:8000/health")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"STARTUP FAILED: {e}", exc_info=True)
        raise
    
    # ---- yield = server is running ----
    yield
    
    # ---- Shutdown ----
    logger.info("SHL Assessment Recommender — Shutting down...")


# ============================================================
# Create the FastAPI Application
# ============================================================
# WHAT EACH PARAMETER DOES:
#   title: Shows in the Swagger UI header
#   description: Shows in the Swagger UI description
#   version: Shows in the Swagger UI
#   lifespan: Our startup/shutdown handler
# ============================================================
app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "A conversational AI agent that recommends SHL assessments "
        "based on hiring needs. Powered by RAG (FAISS + Gemini)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Configure CORS (Cross-Origin Resource Sharing)
# ============================================================
# THEORY — What is CORS?
#   By default, browsers block requests from one domain to another.
#   If our frontend is on "localhost:3000" and our API is on
#   "localhost:8000", the browser blocks the request.
#   
#   CORS headers tell the browser: "It's okay, this origin
#   is allowed to call my API."
#
# WHY allow_origins=["*"]:
#   For this assignment, we allow all origins because:
#   1. The evaluator might call from any domain
#   2. We might add a frontend on a different port
#   3. It's a demo, not production security
#
#   IN PRODUCTION: You'd restrict this to specific domains.
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)


# ============================================================
# Register Route Handlers
# ============================================================
# Each router handles a group of related endpoints.
# The order doesn't matter — FastAPI matches by URL pattern.
# ============================================================
app.include_router(health.router)  # GET /health
app.include_router(chat.router)    # POST /chat

@app.get("/")
def read_root():
    """Root endpoint for browser visitors."""
    return {
        "message": "Welcome to the SHL Assessment Chatbot API! 🚀",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs"
        },
        "status": "online"
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Ignore favicon requests to prevent 404s in logs."""
    return {}
