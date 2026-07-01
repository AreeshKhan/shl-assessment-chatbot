"""
============================================================
schemas.py — Pydantic Data Models (Request/Response Schemas)
============================================================

PURPOSE:
    Define the EXACT JSON structure for API requests and responses.
    These models are dictated by the assignment PDF and are
    NON-NEGOTIABLE — deviating breaks the automated evaluator.

THEORY — What is Pydantic?
    Pydantic is a data validation library that uses Python type
    hints. When FastAPI receives a request, Pydantic automatically:
    1. Parses the JSON body
    2. Validates every field (correct type? required? present?)
    3. Returns a clear error if validation fails (HTTP 422)
    
    REAL-WORLD ANALOGY:
    Imagine a bouncer at a club. Pydantic is the bouncer for your API.
    It checks every request's "ID" (schema) before letting it in.
    If the request doesn't have the right "ID" (wrong fields), 
    it gets rejected at the door.

WHY THIS FILE EXISTS:
    The PDF says: "The schema is non-negotiable. Deviating breaks
    our automated evaluator, and your submission will not score."
    These models GUARANTEE our API matches exactly.

WHAT HAPPENS IF WE REMOVE THIS FILE:
    - No request validation → invalid requests crash the server
    - No response structure → evaluator can't parse our output
    - Instant zero score on the assignment

INTERVIEW QUESTION:
    Q: "Why Pydantic instead of raw dictionaries?"
    A: "Pydantic gives us automatic validation, type safety,
       and documentation. If someone sends the wrong JSON,
       they get a clear error message. With raw dicts, we'd
       need to manually check every field ourselves."
============================================================
"""

from pydantic import BaseModel, Field
from typing import List


# ============================================================
# Message — One message in the conversation
# ============================================================
# This represents a single turn in the conversation.
# The conversation history is a list of these.
#
# FROM PDF:
#   {"role": "user", "content": "Hiring a Java developer"}
#   {"role": "assistant", "content": "Sure. What is seniority?"}
# ============================================================
class Message(BaseModel):
    """
    A single message in the conversation history.
    
    Attributes:
        role: Who sent the message. Either "user" or "assistant".
              - "user" = the hiring manager/recruiter
              - "assistant" = our chatbot
        content: The text content of the message.
    
    Example:
        {"role": "user", "content": "I need a test for Java developers"}
    """
    role: str = Field(
        ...,  # ... means "required"
        description="Who sent this message: 'user' or 'assistant'"
    )
    content: str = Field(
        ...,
        description="The text content of the message"
    )


# ============================================================
# ChatRequest — What the client sends to POST /chat
# ============================================================
# FROM PDF (page 3):
#   {
#     "messages": [
#       {"role": "user", "content": "Hiring a Java developer..."},
#       {"role": "assistant", "content": "Sure. What is seniority?"},
#       {"role": "user", "content": "Mid-level, around 4 years"}
#     ]
#   }
#
# KEY INSIGHT: The API is STATELESS. The client sends the FULL
# conversation history every time. We store nothing between calls.
# ============================================================
class ChatRequest(BaseModel):
    """
    Request body for POST /chat endpoint.
    
    The API is STATELESS — every call includes the complete
    conversation history. The last message is always from "user".
    
    Attributes:
        messages: The full conversation history as a list of Messages.
                  Must contain at least one message.
    
    Example:
        {
            "messages": [
                {"role": "user", "content": "I need assessments for hiring Java developers"}
            ]
        }
    """
    messages: List[Message] = Field(
        ...,
        description="Full conversation history. Last message must be from 'user'.",
        min_length=1  # At least one message required
    )


# ============================================================
# Recommendation — One assessment in the recommendation list
# ============================================================
# FROM PDF (page 4):
#   {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
#
# ONLY 3 FIELDS. The sample conversations show richer tables
# in the reply text, but the structured recommendations array
# has exactly these 3 fields.
#
# Test type codes (derived from catalog 'keys' field):
#   K = Knowledge & Skills
#   P = Personality & Behavior
#   A = Ability & Aptitude
#   S = Simulations
#   B = Biodata & Situational Judgment
#   C = Competencies
#   D = Development & 360
#   E = Assessment Exercises
# ============================================================
class Recommendation(BaseModel):
    """
    A single assessment recommendation.
    
    Attributes:
        name: The exact assessment name from the SHL catalog.
              Must match a real entry in the catalog JSON.
        url: The catalog URL for this assessment.
             Must be a real URL from the catalog's 'link' field.
        test_type: Single-letter code(s) for the assessment type.
                   Derived from the catalog's 'keys' field.
    
    Example:
        {"name": "Core Java (Advanced Level) (New)", 
         "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
         "test_type": "K"}
    """
    name: str = Field(
        ...,
        description="Assessment name from the SHL catalog"
    )
    url: str = Field(
        ...,
        description="Catalog URL for this assessment"
    )
    test_type: str = Field(
        ...,
        description="Test type code(s): K, P, A, S, B, C, D, E"
    )


# ============================================================
# ChatResponse — What our API returns from POST /chat
# ============================================================
# FROM PDF (page 4):
#   {
#     "reply": "Got it. Here are 5 assessments...",
#     "recommendations": [
#       {"name": "Java 8 (New)", "url": "...", "test_type": "K"}
#     ],
#     "end_of_conversation": false
#   }
#
# RULES:
# - "recommendations" is EMPTY [] when gathering context or refusing
# - "recommendations" has 1-10 items when agent has a shortlist
# - "end_of_conversation" is true ONLY when task is complete
# ============================================================
class ChatResponse(BaseModel):
    """
    Response body from POST /chat endpoint.
    
    Attributes:
        reply: The agent's natural language response text.
               This is where the detailed explanation goes.
        recommendations: List of recommended assessments.
                        Empty [] when still gathering context or refusing.
                        1-10 items when agent has committed to a shortlist.
        end_of_conversation: Whether the agent considers the task complete.
                            true only when the user has confirmed the shortlist.
    
    Example (gathering context):
        {"reply": "What seniority level?", "recommendations": [], "end_of_conversation": false}
    
    Example (recommending):
        {"reply": "Here are my recommendations...", 
         "recommendations": [{"name": "...", "url": "...", "test_type": "K"}],
         "end_of_conversation": false}
    
    Example (task complete):
        {"reply": "Great, shortlist confirmed.", 
         "recommendations": [{"name": "...", "url": "...", "test_type": "K"}],
         "end_of_conversation": true}
    """
    reply: str = Field(
        ...,
        description="The agent's natural language response"
    )
    recommendations: List[Recommendation] = Field(
        default_factory=list,  # Default to empty list
        description="Recommended assessments (empty when gathering context)"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )


# ============================================================
# HealthResponse — What GET /health returns
# ============================================================
# FROM PDF: GET /health returns {"status": "ok"} with HTTP 200
# ============================================================
class HealthResponse(BaseModel):
    """
    Response for the health check endpoint.
    
    FROM PDF: 'GET /health returns "status": "ok"} with HTTP 200'
    """
    status: str = Field(
        default="ok",
        description="Service health status"
    )
"""
============================================================
AFTER-FILE EXPLANATION
============================================================

WHAT THIS FILE DOES:
    Defines 5 Pydantic models that match the exact JSON schema
    required by the SHL automated evaluator.

HOW EXECUTION REACHES HERE:
    FastAPI imports these models in the route handlers (chat.py, 
    health.py). When a request arrives, FastAPI uses these models
    to validate the input and structure the output.

WHO CALLS IT:
    - chat.py: Uses ChatRequest and ChatResponse
    - health.py: Uses HealthResponse
    - agent.py: Uses Recommendation to build responses

POSSIBLE ERRORS:
    - Client sends {"message": "..."} instead of {"messages": [...]}
      → Pydantic returns 422 with "messages field required"
    - Client sends messages without 'role' field
      → Pydantic returns 422 with "role field required"

HOW TO DEBUG:
    - FastAPI auto-generates docs at /docs (Swagger UI)
    - Test with: curl -X POST /chat -d '{"messages":[...]}'
    - Check the 422 error response for validation details

COMMON INTERVIEW QUESTIONS:
    Q: "Why is recommendations a list and not null?"
    A: "The PDF says 'recommendations are EMPTY when gathering
       context'. Empty list [] is clearer than null for the
       evaluator to parse. null vs [] is a common API design
       debate — for machine consumption, [] is more consistent."
    
    Q: "Why only 3 fields in Recommendation?"
    A: "That's what the PDF specifies. The sample conversations
       show richer tables in the reply text, but the structured
       JSON only has name, url, test_type. Extra fields would
       be ignored by the evaluator or could break it."
============================================================
"""
