"""
============================================================
health.py — Health Check Endpoint
============================================================

PURPOSE:
    Provide a GET /health endpoint that returns {"status": "ok"}.
    
FROM PDF:
    "GET /health returns {"status": "ok"} with HTTP 200.
     For cold start hosting services, the first /health call
     will allow up to 2 minutes for service to wake up."

WHY A SEPARATE FILE:
    Keeps the health check isolated from the chat logic.
    If /health breaks, it's immediately clear where to look.

THIS IS SIMPLE BY DESIGN:
    The health endpoint is just a heartbeat. It confirms the
    server is running. The evaluator calls this first to check
    if your deployment is alive.
============================================================
"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse

# Create a router for health-related endpoints
# prefix="" because /health is at the root level
router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Returns service health status. Used by the evaluator to check deployment.",
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        {"status": "ok"} with HTTP 200
    
    FROM PDF: "The first /health call will allow up to 2 minutes
    for service to wake up." — Render free tier sleeps after
    inactivity, so cold starts are expected.
    """
    return HealthResponse(status="ok")
