"""
============================================================
agent.py — The Agent Brain (Main Orchestrator)
============================================================

PURPOSE:
    This is the "brain" of the chatbot. It orchestrates the
    entire pipeline for each /chat request:
    
    1. Parse the conversation history
    2. Build a search query from the conversation
    3. Retrieve relevant assessments (FAISS)
    4. Build the prompt with context
    5. Call the LLM (Gemini)
    6. Validate the response (hallucination check)
    7. Return the structured response

THEORY — What is an Agent?
    In AI, an "agent" is a system that:
    - Perceives its environment (reads user messages)
    - Reasons about what to do (decides: clarify? recommend? refuse?)
    - Takes action (generates a response)
    - Observes the result (tracks conversation state)
    
    Our agent is a RAG (Retrieval-Augmented Generation) agent:
    it RETRIEVES relevant data before GENERATING a response.

WHY THIS IS THE MOST IMPORTANT FILE:
    Every other service is a tool. This file decides WHEN and
    HOW to use those tools. The quality of recommendations
    depends on how well this orchestration works.

INTERVIEW QUESTION:
    Q: "Walk me through what happens when a user sends a message."
    A: "1. The agent extracts a search query from the conversation.
        2. FAISS retrieves the top-20 similar assessments.
        3. Those assessments are formatted as context for the LLM.
        4. The LLM generates a response with recommendations.
        5. We validate every recommendation against the catalog.
        6. We return the validated response with the exact schema."
============================================================
"""

import logging
from typing import List, Dict, Any

from app.services.catalog import CatalogService, Assessment
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMService
from app.prompts.templates import SYSTEM_PROMPT, USER_TURN_TEMPLATE
from app.config import settings

logger = logging.getLogger(__name__)


class Agent:
    """
    The main orchestrator that processes chat messages and
    produces assessment recommendations.
    
    ARCHITECTURE:
        This class follows the "Facade Pattern" — it provides
        a simple interface (process_message) that hides the
        complexity of the underlying services.
    
    DEPENDENCIES:
        - CatalogService: Provides assessment data and validation
        - EmbeddingService: Handles FAISS search
        - LLMService: Handles Gemini API calls
    
    LIFECYCLE:
        1. __init__(): Store service references
        2. process_message(): Called for each /chat request
    """
    
    def __init__(
        self,
        catalog_service: CatalogService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
    ):
        """
        Initialize the agent with its dependent services.
        
        WHY DEPENDENCY INJECTION:
            Instead of creating services inside the agent, we
            receive them as parameters. This means:
            1. We can test the agent with mock services
            2. Services are created once and shared
            3. The agent doesn't know about config details
        
        INTERVIEW QUESTION:
            Q: "What is dependency injection?"
            A: "Instead of a class creating its own dependencies,
               they're provided ('injected') from outside. This
               makes the code more testable and flexible."
        
        Args:
            catalog_service: Loaded catalog with all assessments
            embedding_service: Initialized with FAISS index
            llm_service: Configured with Gemini API key
        """
        self.catalog = catalog_service
        self.embeddings = embedding_service
        self.llm = llm_service
    
    def process_message(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process a chat request and return the agent's response.
        
        This is the main entry point — called by the /chat route.
        
        EXECUTION FLOW:
        ┌──────────────────────────────┐
        │ 1. Extract search query      │
        │    from conversation history │
        ├──────────────────────────────┤
        │ 2. Retrieve candidates       │
        │    from FAISS (top-K)        │
        ├──────────────────────────────┤
        │ 3. Build prompt with         │
        │    context + history         │
        ├──────────────────────────────┤
        │ 4. Call LLM (Gemini)         │
        ├──────────────────────────────┤
        │ 5. Validate recommendations  │
        │    against catalog           │
        ├──────────────────────────────┤
        │ 6. Return response           │
        └──────────────────────────────┘
        
        Args:
            messages: List of message dicts [{"role": "...", "content": "..."}]
        
        Returns:
            Dict with "reply", "recommendations", "end_of_conversation"
        """
        try:
            # ---- Step 1: Extract search query ----
            search_query = self._build_search_query(messages)
            logger.info(f"Search query: {search_query[:100]}...")
            
            # ---- Step 2: Retrieve candidates from FAISS ----
            candidates = self.embeddings.search(
                search_query, 
                top_k=settings.RETRIEVAL_TOP_K
            )
            logger.info(f"Retrieved {len(candidates)} candidates")
            
            # ---- Step 3: Build the prompt ----
            user_prompt = self._build_prompt(messages, candidates)
            
            # ---- Step 4: Call the LLM ----
            response = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            
            # ---- Step 5: Validate recommendations ----
            response = self._validate_recommendations(response)
            
            # ---- Step 6: Return the response ----
            return response
            
        except Exception as e:
            import traceback
            err_trace = traceback.format_exc()
            logger.error(f"Agent error: {e}\n{err_trace}", exc_info=True)
            # Return a safe fallback — never crash the API
            return {
                "reply": f"ERROR FOR DEBUG: {str(e)} | Trace: {err_trace}",
                "recommendations": [],
                "end_of_conversation": False,
            }
    
    def _build_search_query(self, messages: List[Dict[str, str]]) -> str:
        """
        Extract a search query from the conversation history.
        
        PURPOSE:
            The FAISS index needs a text query to search with.
            We can't just use the last user message — we need
            to consider the ENTIRE conversation to understand
            what the user really needs.
        
        STRATEGY:
            Combine ALL user messages into one search query.
            This captures the evolving requirements across turns.
            
            Example:
                Turn 1: "I need a Java test" 
                Turn 2: "For senior developers"
                Turn 3: "Add personality testing"
                
                Search query: "I need a Java test For senior developers Add personality testing"
            
            This ensures FAISS retrieves assessments matching ALL
            aspects of the user's needs, not just the last message.
        
        Args:
            messages: Full conversation history
        
        Returns:
            Combined text from all user messages
        """
        # Collect all user messages
        user_messages = [
            msg["content"] 
            for msg in messages 
            if msg.get("role") == "user"
        ]
        
        # Combine into one query string
        # WHY join all: The user may describe requirements across
        # multiple turns. We need to capture all of them.
        query = " ".join(user_messages)
        
        # If somehow empty, use a generic query
        if not query.strip():
            query = "SHL assessment recommendation"
        
        return query
    
    def _build_prompt(
        self, 
        messages: List[Dict[str, str]], 
        candidates: List[Assessment]
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        STRUCTURE:
            The prompt has two main parts:
            1. RETRIEVED ASSESSMENTS: The FAISS candidates (context)
            2. CONVERSATION HISTORY: The full message history
        
        WHY THIS ORDER:
            Assessments first because the LLM needs to "read"
            the available options before processing the conversation.
            Like giving a consultant a product catalog BEFORE
            asking them for a recommendation.
        
        Args:
            messages: Full conversation history
            candidates: Retrieved Assessment objects from FAISS
        
        Returns:
            Formatted prompt string ready for the LLM
        """
        # ---- Format retrieved assessments ----
        assessment_texts = []
        for i, assessment in enumerate(candidates, 1):
            text = self.catalog.get_assessment_details_text(assessment)
            assessment_texts.append(f"--- Assessment {i} ---\n{text}")
        
        retrieved_section = "\n\n".join(assessment_texts)
        
        # ---- Format conversation history ----
        history_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        
        history_section = "\n".join(history_parts)
        
        # ---- Fill in the template ----
        prompt = USER_TURN_TEMPLATE.format(
            retrieved_assessments=retrieved_section,
            conversation_history=history_section,
        )
        
        return prompt
    
    def _validate_recommendations(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that every recommendation exists in the catalog.
        
        PURPOSE:
            This is our HALLUCINATION PREVENTION gate.
            The LLM might:
            1. Invent an assessment name that doesn't exist
            2. Generate a URL that's not in the catalog
            3. Use a slightly wrong name (typo, different version)
            
            FROM PDF: "Items from catalog only in recommendations"
            is a HARD EVAL (must pass). Hallucinated items = fail.
        
        STRATEGY:
            1. For each recommendation, check if it exists in catalog
            2. If it does, use the EXACT name and URL from catalog
               (in case the LLM changed capitalization or wording)
            3. If it doesn't, try fuzzy matching by name
            4. If still no match, REMOVE it from recommendations
        
        Args:
            response: Dict from the LLM with recommendations
        
        Returns:
            Same dict with validated (and corrected) recommendations
        """
        if not response.get("recommendations"):
            return response
        
        validated_recs = []
        
        for rec in response["recommendations"]:
            name = rec.get("name", "")
            url = rec.get("url", "")
            test_type = rec.get("test_type", "K")
            
            # Strategy 1: Exact match by URL
            assessment = self.catalog.find_by_url(url)
            
            # Strategy 2: Exact match by name
            if not assessment:
                assessment = self.catalog.find_by_name(name)
            
            # Strategy 3: Try partial name matching
            if not assessment:
                assessment = self._fuzzy_find(name)
            
            if assessment:
                # Use the EXACT catalog data (not LLM's version)
                validated_recs.append({
                    "name": assessment.name,
                    "url": assessment.url,
                    "test_type": assessment.test_type,
                })
            else:
                # Remove this recommendation — it's hallucinated
                logger.warning(
                    f"Hallucination detected! Removing: "
                    f"name='{name}', url='{url}'"
                )
        
        # Ensure no duplicates (by URL)
        seen_urls = set()
        unique_recs = []
        for rec in validated_recs:
            if rec["url"] not in seen_urls:
                seen_urls.add(rec["url"])
                unique_recs.append(rec)
        
        # Limit to MAX_RECOMMENDATIONS
        response["recommendations"] = unique_recs[:settings.MAX_RECOMMENDATIONS]
        
        return response
    
    def _fuzzy_find(self, name: str) -> Any:
        """
        Try to find an assessment by partial name matching.
        
        WHY: The LLM might say "Core Java Advanced" instead of
        "Core Java (Advanced Level) (New)". Fuzzy matching catches
        these near-misses.
        
        STRATEGY: Check if the LLM's name is a substring of any
        catalog name, or vice versa.
        
        Args:
            name: The assessment name from the LLM
        
        Returns:
            Assessment if found, None otherwise
        """
        if not name:
            return None
        
        name_lower = name.lower().strip()
        
        for assessment in self.catalog.assessments:
            catalog_name_lower = assessment.name.lower().strip()
            
            # Check both directions of substring matching
            if name_lower in catalog_name_lower or catalog_name_lower in name_lower:
                return assessment
        
        return None
