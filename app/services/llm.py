"""
============================================================
llm.py — LLM Service (Groq API Integration)
============================================================

PURPOSE:
    Handle all communication with the Groq LLM API.
    Groq runs open-source models (Llama 3.3 70B) at blazing
    speed using their custom LPU chips.

WHY GROQ INSTEAD OF GEMINI:
    Gemini's free tier was blocked for this account (limit: 0).
    Groq provides a generous free tier with no credit card:
    - 30 requests/minute
    - 6000 tokens/minute
    - Extremely fast inference (~300 tokens/sec)
    
    The PDF explicitly allows this: "Free LLM tiers (Gemini, Groq, OpenRouter)"

INTERVIEW QUESTION:
    Q: "Why did you switch from Gemini to Groq?"
    A: "Gemini's free tier was unavailable in my region. The 
       assignment allows any free LLM tier. Groq runs Llama 3.3
       70B which is comparable to GPT-4 quality, and its LPU
       hardware makes inference extremely fast — well within
       the 30-second timeout requirement."
============================================================
"""

import json
import logging
import time
from typing import Optional, Dict, Any

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles LLM interactions with Groq's API.
    
    Uses Llama 3.3 70B for high-quality text generation.
    """
    
    def __init__(self):
        """Initialize the Groq client."""
        self._client = Groq(api_key=settings.GROQ_API_KEY)
    
    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Send a prompt to Groq and parse the JSON response.
        
        Args:
            system_prompt: The system prompt (agent personality/rules)
            user_prompt: The per-turn prompt (context + history)
            max_retries: How many times to retry on failure
        
        Returns:
            Dict with keys: "reply", "recommendations", "end_of_conversation"
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Call Groq API using the chat completions format
                response = self._client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                    # response_format={"type": "json_object"},  # Force JSON output
                )
                
                # Extract text from response
                raw_text = response.choices[0].message.content
                
                # Parse JSON from the response
                parsed = self._extract_json(raw_text)
                
                if parsed:
                    return self._validate_response(parsed)
                else:
                    logger.warning(
                        f"Attempt {attempt + 1}: Could not parse JSON. "
                        f"Raw: {raw_text[:200]}..."
                    )
                    last_error = "JSON parse failed"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                last_error = str(e)
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed — return a safe fallback
        logger.error(f"All {max_retries + 1} attempts failed. Last error: {last_error}")
        return {
            "reply": "I apologize, but I'm having trouble processing your request right now. Could you please try again?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extract a JSON object from the LLM's text response.
        Handles markdown code blocks and extra text around JSON.
        """
        # Strategy 1: Try parsing the whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Look for JSON in markdown code blocks
        import re
        json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if json_block_match:
            try:
                return json.loads(json_block_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find the first { ... } block
        brace_start = text.find('{')
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
        
        return None
    
    def _validate_response(self, parsed: Dict) -> Dict:
        """Ensure the parsed response has all required fields."""
        if "reply" not in parsed or not isinstance(parsed["reply"], str):
            parsed["reply"] = parsed.get("reply", parsed.get("response", ""))
        
        if "recommendations" not in parsed or not isinstance(parsed["recommendations"], list):
            parsed["recommendations"] = []
        
        if "end_of_conversation" not in parsed:
            parsed["end_of_conversation"] = False
        
        # Validate each recommendation
        valid_recs = []
        for rec in parsed["recommendations"]:
            if isinstance(rec, dict) and "name" in rec and "url" in rec:
                valid_recs.append({
                    "name": rec["name"],
                    "url": rec["url"],
                    "test_type": rec.get("test_type", "K"),
                })
        parsed["recommendations"] = valid_recs
        
        return parsed
