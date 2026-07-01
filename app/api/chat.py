"""
============================================================
chat.py — Chat Endpoint (POST /chat)
============================================================

PURPOSE:
    Handle the main POST /chat endpoint. This is where the
    evaluator sends conversation messages and receives the
    agent's response.

FROM PDF (EXACT SPEC):
    Request:  {"messages": [{"role": "user", "content": "..."}]}
    Response: {"reply": "...", "recommendations": [...], "end_of_conversation": false}

WHAT THIS FILE DOES:
    1. Receives the request (validated by Pydantic)
    2. Passes messages to the Agent
    3. Returns the structured response

WHAT THIS FILE DOES NOT DO:
    - No business logic (that's in agent.py)
    - No LLM calls (that's in llm.py)
    - No catalog access (that's in catalog.py)
    
    This file is a THIN LAYER between HTTP and business logic.
    This is called the "Controller" pattern in MVC architecture.

INTERVIEW QUESTION:
    Q: "Why is the route handler so simple?"
    A: "Separation of concerns. The route handler only handles
       HTTP — request parsing, response formatting, error handling.
       All business logic lives in agent.py. This makes both
       independently testable."
============================================================
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, Recommendation

logger = logging.getLogger(__name__)

# Create a router for chat-related endpoints
router = APIRouter(tags=["Chat"])

# This will be set during app startup (in main.py)
# WHY MODULE-LEVEL VARIABLE: The agent is created once at startup
# and shared across all requests. Each request uses the same
# catalog, FAISS index, and LLM configuration.
agent = None


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with the SHL Assessment Recommender",
    description="Send conversation messages and receive assessment recommendations.",
)
async def chat(request: ChatRequest):
    """
    Process a chat message and return recommendations.
    
    The API is STATELESS — every call includes the full
    conversation history. The last message should be from "user".
    
    Args:
        request: ChatRequest with messages array
    
    Returns:
        ChatResponse with reply, recommendations, end_of_conversation
    
    Raises:
        HTTPException 500: If the agent encounters an internal error
        HTTPException 422: If the request doesn't match the schema (Pydantic)
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Agent not initialized."
        )
    
    # Convert Pydantic Message objects to plain dicts for the agent
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    logger.info(
        f"Processing chat request: {len(messages)} messages, "
        f"last user message: {messages[-1]['content'][:80]}..."
    )
    
    # Process through the agent
    result = agent.process_message(messages)
    
    # Convert to response schema
    recommendations = [
        Recommendation(
            name=rec["name"],
            url=rec["url"],
            test_type=rec["test_type"],
        )
        for rec in result.get("recommendations", [])
    ]
    
    response = ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recommendations,
        end_of_conversation=result.get("end_of_conversation", False),
    )
    
    logger.info(
        f"Response: {len(recommendations)} recommendations, "
        f"end_of_conversation={response.end_of_conversation}"
    )
    
    return response
